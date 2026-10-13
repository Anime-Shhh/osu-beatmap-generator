"""Constraint-aware active-event geometry and slider-path diffusion.

Diffusion belongs on the continuous, multimodal geometry problem—not on the
entire `.osu` syntax.  Unlike the original v2 refiner, this module receives an
ordered active-event sequence, object types, anchors, and masks.  It never
diffuses inactive grid lanes or asks one coordinate to stand in for a slider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .events import EventKind, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .grammar import MAX_ANCHORS


GEOMETRY_POINTS = MAX_ANCHORS + 1  # object head plus up to eight slider anchors.


@dataclass(frozen=True)
class GeometryDiffusionConfig:
    context_dim: int = 768
    hidden_dim: int = 512
    heads: int = 8
    layers: int = 6
    dropout: float = 0.1
    max_anchors: int = MAX_ANCHORS
    # Retained as a corpus-audit advisory limit. Positions themselves use a
    # non-periodic sinusoidal representation, so a long song is never silently
    # aliased to event 2047 during windowed inference.
    max_events: int = 2048
    event_feature_dim: int = 4  # prev/next gap, duration, target path length.

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.max_events < 1 or self.event_feature_dim < 1:
            raise ValueError("max_events and event_feature_dim must be positive")


def coordinates_to_logits(coordinates: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Move bounded normalized coordinates to unconstrained diffusion space."""

    return torch.logit(coordinates.clamp(eps, 1.0 - eps))


def logits_to_coordinates(logits: torch.Tensor) -> torch.Tensor:
    """Map sampled geometry back to the legal normalized playfield domain."""

    return torch.sigmoid(logits)


def cosine_schedule(noise_level: torch.Tensor, *, s: float = 0.008) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous cosine schedule with bounded terminal SNR to avoid endpoint singularity."""
    level = noise_level.clamp(0.0, 1.0)
    # Cosine schedule with offset s = 0.008
    f_t = torch.cos(((level + s) / (1.0 + s)) * (math.pi / 2.0))
    f_0 = math.cos((s / (1.0 + s)) * (math.pi / 2.0))
    alpha_bar = (f_t / f_0).square().clamp(min=1e-4, max=1.0)
    alpha = alpha_bar.sqrt()
    sigma = (1.0 - alpha_bar).clamp(min=0.0).sqrt()
    return alpha, sigma


def geometry_point_mask(
    event_mask: torch.Tensor,
    event_kinds: torch.Tensor,
    anchor_counts: torch.Tensor,
    *,
    max_anchors: int = MAX_ANCHORS,
) -> torch.Tensor:
    """Return `[B,N,1+K]` validity for heads and variable slider anchors."""

    if event_mask.dim() != 2 or event_kinds.shape != event_mask.shape or anchor_counts.shape != event_mask.shape:
        raise ValueError("event_mask, event_kinds, and anchor_counts must all be [B,N]")
    positions = torch.arange(max_anchors + 1, device=event_mask.device).view(1, 1, -1)
    is_slider = event_kinds.eq(int(EventKind.SLIDER)).unsqueeze(-1)
    counts = anchor_counts.clamp(0, max_anchors).unsqueeze(-1)
    # Head always exists for each active event. Anchor positions begin at 1.
    valid = positions.eq(0) | (is_slider & positions.le(counts))
    return valid & event_mask.unsqueeze(-1)


class ActiveEventGeometryDiffuser(nn.Module):
    """Joint sequence denoiser for circle heads and variable slider anchors."""

    def __init__(self, config: GeometryDiffusionConfig = GeometryDiffusionConfig()):
        super().__init__()
        self.config = config
        self.coordinate = nn.Linear(2, config.hidden_dim)
        self.context = nn.Linear(config.context_dim, config.hidden_dim)
        self.event_features = nn.Linear(config.event_feature_dim, config.hidden_dim)
        self.kind = nn.Embedding(len(EventKind), config.hidden_dim)
        self.event_position = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.anchor_index = nn.Embedding(config.max_anchors + 1, config.hidden_dim)
        self.time = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, 2))

    @staticmethod
    def _time_embedding(noise_level: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=noise_level.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = noise_level.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedding.shape[-1] < dim:
            embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
        return embedding[..., :dim]

    @staticmethod
    def _event_position_embedding(event_positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Encode absolute event order without a fixed learned-table ceiling.

        Geometry is sampled in local windows, but a song can readily contain
        more than a few thousand objects. Clamping an embedding lookup makes
        all later events indistinguishable and creates a hidden long-song
        failure. A standard sinusoidal basis keeps absolute positions usable
        for arbitrary non-negative record indices while the learned MLP adapts
        that basis to this task.
        """

        if event_positions.dim() != 2:
            raise ValueError("event_positions must be [B,N]")
        if (event_positions < 0).any():
            raise ValueError("event_positions must be non-negative")
        half = dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=event_positions.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = event_positions.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedding.shape[-1] < dim:
            embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
        return embedding[..., :dim]

    def forward(
        self,
        context: torch.Tensor,
        event_kinds: torch.Tensor,
        anchor_counts: torch.Tensor,
        noisy_geometry_logits: torch.Tensor,
        noise_level: torch.Tensor,
        *,
        event_mask: Optional[torch.Tensor] = None,
        event_features: Optional[torch.Tensor] = None,
        event_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict diffusion epsilon for `[B, events, 1+anchors, 2]`."""

        if context.dim() != 3:
            raise ValueError("context must be [B,N,D]")
        batch, events, _ = context.shape
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context feature dimension must be {self.config.context_dim}, got {context.shape[-1]}"
            )
        expected = (batch, events, self.config.max_anchors + 1, 2)
        if noisy_geometry_logits.shape != expected:
            raise ValueError(f"noisy_geometry_logits must be {expected}")
        if event_kinds.shape != (batch, events) or anchor_counts.shape != (batch, events):
            raise ValueError("event metadata must be [B,N]")
        if noise_level.shape not in {(batch,), (batch, 1)}:
            raise ValueError("noise_level must be [B] or [B,1]")
        if event_mask is None:
            event_mask = torch.ones((batch, events), device=context.device, dtype=torch.bool)
        if event_features is None:
            event_features = torch.zeros(
                (batch, events, self.config.event_feature_dim),
                device=context.device,
                dtype=context.dtype,
            )
        if event_features.shape != (batch, events, self.config.event_feature_dim):
            raise ValueError("event_features must be [B,N,event_feature_dim]")
        if event_positions is None:
            event_positions = torch.arange(events, device=context.device).unsqueeze(0).expand(batch, -1)
        if event_positions.shape != (batch, events):
            raise ValueError("event_positions must be [B,N]")
        point_mask = geometry_point_mask(
            event_mask, event_kinds, anchor_counts, max_anchors=self.config.max_anchors
        )
        anchor_ids = torch.arange(self.config.max_anchors + 1, device=context.device).view(1, 1, -1)
        tokens = (
            self.coordinate(noisy_geometry_logits)
            + self.context(context).unsqueeze(2)
            + self.event_features(event_features).unsqueeze(2)
            + self.kind(event_kinds.clamp(0, len(EventKind) - 1)).unsqueeze(2)
            + self.event_position(
                self._event_position_embedding(event_positions, self.config.hidden_dim).to(context.dtype)
            ).unsqueeze(2)
            + self.anchor_index(anchor_ids)
            + self.time(self._time_embedding(noise_level.reshape(batch), self.config.hidden_dim)).view(batch, 1, 1, -1)
        )
        tokens = tokens.reshape(batch, events * (self.config.max_anchors + 1), self.config.hidden_dim)
        flat_mask = point_mask.reshape(batch, -1)
        tokens = tokens * flat_mask.unsqueeze(-1).to(tokens.dtype)
        # A collated batch can contain an empty bar beside a populated one.
        # Transformer attention with every key masked for that row is undefined
        # on some PyTorch versions, so reserve a zero dummy token internally
        # and remove it again from the returned epsilon.
        attention_mask = flat_mask.clone()
        empty_rows = ~attention_mask.any(dim=1)
        if empty_rows.any() and attention_mask.shape[1] > 0:
            attention_mask[empty_rows, 0] = True
        encoded = self.sequence(tokens, src_key_padding_mask=~attention_mask)
        epsilon = self.out(encoded).reshape_as(noisy_geometry_logits)
        return epsilon * point_mask.unsqueeze(-1).to(epsilon.dtype)


def geometry_diffusion_loss(
    diffuser: ActiveEventGeometryDiffuser,
    context: torch.Tensor,
    event_kinds: torch.Tensor,
    anchor_counts: torch.Tensor,
    clean_geometry: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    noise_level: Optional[torch.Tensor] = None,
    event_features: Optional[torch.Tensor] = None,
    event_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked epsilon loss in logit coordinate space."""

    batch = clean_geometry.shape[0]
    if noise_level is None:
        noise_level = torch.rand(batch, device=clean_geometry.device, dtype=clean_geometry.dtype)
    clean_logits = coordinates_to_logits(clean_geometry)
    alpha, sigma = cosine_schedule(noise_level.reshape(batch))
    noise = torch.randn_like(clean_logits)
    noisy = alpha[:, None, None, None] * clean_logits + sigma[:, None, None, None] * noise
    predicted = diffuser(
        context,
        event_kinds,
        anchor_counts,
        noisy,
        noise_level,
        event_mask=event_mask,
        event_features=event_features,
        event_positions=event_positions,
    )
    mask = geometry_point_mask(
        event_mask, event_kinds, anchor_counts, max_anchors=diffuser.config.max_anchors
    ).unsqueeze(-1)
    squared = (predicted - noise).square() * mask.to(predicted.dtype)
    return squared.sum() / mask.sum().clamp_min(1).to(predicted.dtype) / 2.0


@torch.no_grad()
def sample_geometry(
    diffuser: ActiveEventGeometryDiffuser,
    context: torch.Tensor,
    event_kinds: torch.Tensor,
    anchor_counts: torch.Tensor,
    *,
    event_mask: Optional[torch.Tensor] = None,
    steps: int = 32,
    generator: Optional[torch.Generator] = None,
    event_features: Optional[torch.Tensor] = None,
    event_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deterministically sample bounded event geometry with DDIM-style steps.

    The denoiser predicts epsilon under :func:`cosine_schedule`.  This sampler
    deliberately has no classifier-free or model-download side effect: callers
    supply a trained diffuser and an optional seeded generator.  Invalid event
    and anchor slots are zeroed at every step so their random noise cannot
    influence active geometry through the sequence transformer.
    """

    if steps < 2 or steps > 256:
        raise ValueError("geometry sampling steps must be between 2 and 256")
    if context.dim() != 3:
        raise ValueError("context must be [B,N,D]")
    batch, event_count, _ = context.shape
    if event_kinds.shape != (batch, event_count) or anchor_counts.shape != (batch, event_count):
        raise ValueError("event metadata must be [B,N]")
    if event_mask is None:
        event_mask = torch.ones((batch, event_count), dtype=torch.bool, device=context.device)
    point_mask = geometry_point_mask(
        event_mask, event_kinds, anchor_counts, max_anchors=diffuser.config.max_anchors
    ).unsqueeze(-1)
    shape = (batch, event_count, diffuser.config.max_anchors + 1, 2)
    state = torch.randn(shape, device=context.device, dtype=context.dtype, generator=generator)
    state = state * point_mask.to(state.dtype)
    # Descend from pure noise (t=1) to data (t=0).  Eta is intentionally zero
    # for reproducibility; candidate diversity comes from the initial noise and
    # the symbolic sampler seed.
    schedule = torch.linspace(1.0, 0.0, steps, device=context.device, dtype=context.dtype)
    for current, following in zip(schedule[:-1], schedule[1:]):
        level = current.expand(batch)
        epsilon = diffuser(
            context,
            event_kinds,
            anchor_counts,
            state,
            level,
            event_mask=event_mask,
            event_features=event_features,
            event_positions=event_positions,
        )
        alpha, sigma = cosine_schedule(level)
        x0 = (state - sigma[:, None, None, None] * epsilon) / alpha[:, None, None, None].clamp_min(1e-5)
        next_level = following.expand(batch)
        next_alpha, next_sigma = cosine_schedule(next_level)
        state = next_alpha[:, None, None, None] * x0 + next_sigma[:, None, None, None] * epsilon
        state = state * point_mask.to(state.dtype)
    return logits_to_coordinates(state) * point_mask.to(state.dtype)


@torch.no_grad()
def sample_geometry_windows(
    diffuser: ActiveEventGeometryDiffuser,
    context: torch.Tensor,
    event_kinds: torch.Tensor,
    anchor_counts: torch.Tensor,
    *,
    event_mask: Optional[torch.Tensor] = None,
    event_features: Optional[torch.Tensor] = None,
    event_positions: Optional[torch.Tensor] = None,
    steps: int = 32,
    window_events: int = 192,
    overlap_events: int = 32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample long maps in overlapping active-event windows and reconcile seams.

    Full-song geometry attention grows quadratically with object count. Windows
    retain the model's local flow context; shared overlap events are sampled in
    both neighboring windows and averaged deterministically, avoiding an
    arbitrary cut point. Absolute event positions remain part of the conditioner.
    """

    if window_events < 2 or not 0 <= overlap_events < window_events:
        raise ValueError("window_events must be >=2 and overlap_events in [0, window_events)")
    if context.dim() != 3:
        raise ValueError("context must be [B,N,D]")
    event_count = context.shape[1]
    if event_count <= window_events:
        return sample_geometry(
            diffuser,
            context,
            event_kinds,
            anchor_counts,
            event_mask=event_mask,
            event_features=event_features,
            event_positions=event_positions,
            steps=steps,
            generator=generator,
        )
    if event_mask is None:
        event_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
    output = context.new_zeros((
        context.shape[0], event_count, diffuser.config.max_anchors + 1, 2
    ))
    weights = context.new_zeros((context.shape[0], event_count, 1, 1))
    stride = window_events - overlap_events
    start = 0
    while start < event_count:
        end = min(event_count, start + window_events)
        sampled = sample_geometry(
            diffuser,
            context[:, start:end],
            event_kinds[:, start:end],
            anchor_counts[:, start:end],
            event_mask=event_mask[:, start:end],
            event_features=None if event_features is None else event_features[:, start:end],
            event_positions=None if event_positions is None else event_positions[:, start:end],
            steps=steps,
            generator=generator,
        )
        mask = event_mask[:, start:end].unsqueeze(-1).unsqueeze(-1).to(output.dtype)
        output[:, start:end] += sampled * mask
        weights[:, start:end] += mask
        if end == event_count:
            break
        start += stride
    return output / weights.clamp_min(1.0)


def project_slider_paths(
    geometry: torch.Tensor,
    event_kinds: torch.Tensor,
    anchor_counts: torch.Tensor,
    target_lengths_px: torch.Tensor,
    *,
    event_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project slider anchors to canvas and approximately requested path length.

    The projection is deterministic and conservative: it rescales the anchor
    polyline around the slider head, clamps to the canvas, and leaves exact
    curve/ranking validation to the compiler. It gives sampling a feasibility
    repair rather than silently trusting unconstrained diffusion output.
    """

    if geometry.dim() != 4 or geometry.shape[-1] != 2:
        raise ValueError("geometry must be [B,N,P,2]")
    if event_mask is None:
        event_mask = torch.ones(geometry.shape[:2], dtype=torch.bool, device=geometry.device)
    output = geometry.clamp(0.0, 1.0).clone()
    width_height = output.new_tensor((PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT))
    for batch_index in range(output.shape[0]):
        for event_index in range(output.shape[1]):
            if not bool(event_mask[batch_index, event_index]) or int(event_kinds[batch_index, event_index]) != int(EventKind.SLIDER):
                continue
            count = int(anchor_counts[batch_index, event_index].clamp(0, output.shape[2] - 1).item())
            if count < 1:
                continue
            points = output[batch_index, event_index, : count + 1]
            pixels = points * width_height
            current = torch.linalg.vector_norm(pixels[1:] - pixels[:-1], dim=-1).sum()
            desired = target_lengths_px[batch_index, event_index].clamp_min(1.0)
            scale = (desired / current.clamp_min(1e-4)).clamp(0.25, 4.0)
            projected = pixels[0] + (pixels - pixels[0]) * scale
            output[batch_index, event_index, : count + 1] = (projected / width_height).clamp(0.0, 1.0)
    return output
