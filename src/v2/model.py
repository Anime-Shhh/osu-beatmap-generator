"""Hierarchical beat-grid planner and coordinate-diffusion refiner.

This is a deliberately different decomposition from the legacy autoregressive
token decoder and the experimental dense-signal latent flow model:

* high-rate rhythm features establish event timing;
* cached music-SSL features provide semantic/tonal context;
* a bar/section-scale Transformer supplies long-range structure without full
  song O(T^2) attention; and
* continuous diffusion is reserved for ambiguous geometry, after a discrete
  rhythm/object plan is known.

The module is a trainable foundation, not a claim that an untrained model can
produce maps.  It has no hidden model downloads and does not instantiate a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BeatGridModelConfig:
    """Configuration sized for initial single-A100 ablations, not maximal scale."""

    rhythm_channels: int = 16
    semantic_dim: int = 768
    map_setting_dim: int = 4  # e.g. AR, OD, CS, slider multiplier
    hidden_dim: int = 192
    local_layers: int = 4
    global_layers: int = 3
    num_heads: int = 6
    events_per_slot: int = 2
    subdivisions_per_beat: int = 16
    section_slots: int = 64  # 4 beats at 1/16 resolution
    max_slider_repeats: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm")
        if self.events_per_slot < 1 or self.subdivisions_per_beat < 1:
            raise ValueError("events_per_slot and subdivisions_per_beat must be positive")


def sinusoidal_positions(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous sine/cosine position features for beat-grid indices.

    ``positions`` can contain absolute song slot indices, so adjacent chunks
    preserve a shared temporal coordinate instead of each resetting to zero.
    """

    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    if positions.dim() != 2:
        raise ValueError("positions must have shape [S] or [B, S]")
    half_dim = max(1, dim // 2)
    frequencies = torch.exp(
        torch.arange(half_dim, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if encoding.shape[-1] < dim:
        encoding = F.pad(encoding, (0, dim - encoding.shape[-1]))
    return encoding[..., :dim]


class TemporalResidualBlock(nn.Module):
    """Local O(T) temporal context, appropriate for long song chunks."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.depthwise(x))
        x = self.pointwise(x)
        value, gate = x.chunk(2, dim=1)
        x = value * torch.sigmoid(gate)
        x = self.norm2(x)
        x = self.dropout(F.gelu(self.out(x)))
        return residual + x


class MultiRateAudioFusion(nn.Module):
    """Fuse high-rate rhythm features with cached lower-rate music embeddings.

    A frozen MERT encoder belongs in preprocessing/caching for initial
    experiments.  This adapter deliberately accepts its features rather than
    downloading or fine-tuning a huge encoder inside every train worker.
    """

    def __init__(self, rhythm_channels: int, semantic_dim: int, hidden_dim: int):
        super().__init__()
        self.rhythm_proj = nn.Conv1d(rhythm_channels, hidden_dim, kernel_size=5, padding=2)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        self.gate = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)

    def forward(
        self,
        rhythm_features: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return fused features as ``[B, hidden, high_rate_frames]``."""

        if rhythm_features.dim() != 3:
            raise ValueError("rhythm_features must have shape [B, C, T]")
        rhythm = self.rhythm_proj(rhythm_features)
        if semantic_features is None:
            return rhythm
        if semantic_features.dim() != 3:
            raise ValueError("semantic_features must have shape [B, T_semantic, D]")
        if semantic_features.shape[0] != rhythm.shape[0]:
            raise ValueError("rhythm and semantic batches must have the same size")
        semantic = self.semantic_proj(semantic_features).transpose(1, 2)
        semantic = F.interpolate(
            semantic,
            size=rhythm.shape[-1],
            mode="linear",
            align_corners=False,
        )
        gate = torch.sigmoid(self.gate(torch.cat((rhythm, semantic), dim=1)))
        return rhythm + gate * semantic


class GlobalConditioner(nn.Module):
    """Continuous difficulty/settings conditioning without coarse-only bins."""

    def __init__(self, map_setting_dim: int, hidden_dim: int):
        super().__init__()
        self.difficulty = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.settings = nn.Sequential(
            nn.Linear(map_setting_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        difficulty: torch.Tensor,
        map_settings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if difficulty.dim() == 1:
            difficulty = difficulty.unsqueeze(-1)
        if difficulty.dim() != 2 or difficulty.shape[-1] != 1:
            raise ValueError("difficulty must have shape [B] or [B, 1]")
        if map_settings is None:
            map_settings = torch.zeros(
                difficulty.shape[0],
                self.settings[0].in_features,
                device=difficulty.device,
                dtype=difficulty.dtype,
            )
        if map_settings.dim() != 2 or map_settings.shape[0] != difficulty.shape[0]:
            raise ValueError("map_settings must have shape [B, setting_dim]")
        return self.norm(self.difficulty(difficulty) + self.settings(map_settings))


class HierarchicalBeatGridPlanner(nn.Module):
    """Plan sparse event occupancy/type at beat-relative candidate slots."""

    def __init__(self, config: BeatGridModelConfig = BeatGridModelConfig()):
        super().__init__()
        self.config = config
        self.fusion = MultiRateAudioFusion(
            config.rhythm_channels, config.semantic_dim, config.hidden_dim
        )
        self.local_blocks = nn.ModuleList(
            TemporalResidualBlock(
                config.hidden_dim,
                dilation=2 ** (index % 4),
                dropout=config.dropout,
            )
            for index in range(config.local_layers)
        )
        self.phase_proj = nn.Linear(2, config.hidden_dim)
        self.conditioner = GlobalConditioner(config.map_setting_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.section_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.global_layers, enable_nested_tensor=False
        )
        self.detail_blocks = nn.ModuleList(
            TemporalResidualBlock(config.hidden_dim, dilation=2 ** (index % 3), dropout=config.dropout)
            for index in range(2)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

        lanes = config.events_per_slot
        self.occupancy_head = nn.Linear(config.hidden_dim, lanes)
        self.kind_head = nn.Linear(config.hidden_dim, lanes * 3)
        self.time_residual_head = nn.Linear(config.hidden_dim, lanes)
        self.base_coordinate_head = nn.Linear(config.hidden_dim, lanes * 2)
        self.spinner_duration_head = nn.Linear(config.hidden_dim, lanes)
        self.slider_length_head = nn.Linear(config.hidden_dim, lanes)
        self.slider_repeat_head = nn.Linear(config.hidden_dim, lanes * config.max_slider_repeats)

    def _pool_sections(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool slots to bar-scale tokens while honoring padding masks."""

        batch, num_slots, hidden = slots.shape
        section_size = self.config.section_slots
        sections = math.ceil(num_slots / section_size)
        padded_slots = sections * section_size
        pad = padded_slots - num_slots
        if pad:
            slots = F.pad(slots, (0, 0, 0, pad))
            slot_mask = F.pad(slot_mask, (0, pad), value=False)
        slots = slots.view(batch, sections, section_size, hidden)
        mask = slot_mask.view(batch, sections, section_size)
        denominator = mask.sum(dim=2, keepdim=True).clamp_min(1).to(slots.dtype)
        pooled = (slots * mask.unsqueeze(-1).to(slots.dtype)).sum(dim=2) / denominator
        section_mask = mask.any(dim=2)
        return pooled, section_mask

    def forward(
        self,
        rhythm_features: torch.Tensor,
        difficulty: torch.Tensor,
        *,
        num_slots: Optional[int] = None,
        semantic_features: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
        slot_positions: Optional[torch.Tensor] = None,
        beat_phase: Optional[torch.Tensor] = None,
        map_settings: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return discrete planning heads and contextual slot embeddings.

        ``slot_mask`` is true for valid (non-padding) candidate slots.  Model
        losses must consume it; it exists specifically to avoid the unmasked
        full-song padding behavior in the prior flow experiment.
        """

        fused = self.fusion(rhythm_features, semantic_features)
        for block in self.local_blocks:
            fused = block(fused)

        batch, hidden, frame_count = fused.shape
        num_slots = int(num_slots or frame_count)
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        slots = F.interpolate(fused, size=num_slots, mode="linear", align_corners=False).transpose(1, 2)

        if slot_mask is None:
            slot_mask = torch.ones(batch, num_slots, device=slots.device, dtype=torch.bool)
        if slot_mask.shape != (batch, num_slots):
            raise ValueError("slot_mask must have shape [B, num_slots]")
        if slot_positions is None:
            slot_positions = torch.arange(num_slots, device=slots.device).expand(batch, -1)
        if slot_positions.shape != (batch, num_slots):
            raise ValueError("slot_positions must have shape [B, num_slots]")

        slots = slots + sinusoidal_positions(slot_positions, hidden).to(slots.dtype)
        if beat_phase is not None:
            if beat_phase.shape != (batch, num_slots, 2):
                raise ValueError("beat_phase must have shape [B, num_slots, 2] as sin/cos phase")
            slots = slots + self.phase_proj(beat_phase.to(slots.dtype))
        slots = slots + self.conditioner(difficulty, map_settings).unsqueeze(1)

        sections, section_mask = self._pool_sections(slots, slot_mask)
        section_positions = torch.arange(sections.shape[1], device=slots.device).expand(batch, -1)
        sections = sections + sinusoidal_positions(section_positions, hidden).to(sections.dtype)
        global_context = self.section_encoder(
            sections,
            src_key_padding_mask=~section_mask,
        )
        global_context = global_context.repeat_interleave(self.config.section_slots, dim=1)[:, :num_slots]
        slots = slots + global_context

        detail = slots.transpose(1, 2)
        for block in self.detail_blocks:
            detail = block(detail)
        context = self.output_norm(detail.transpose(1, 2))

        lanes = self.config.events_per_slot
        residual_bound = 0.5 / self.config.subdivisions_per_beat
        return {
            "context": context,
            "slot_mask": slot_mask,
            "occupancy_logits": self.occupancy_head(context),
            "kind_logits": self.kind_head(context).view(batch, num_slots, lanes, 3),
            "time_residual_beats": torch.tanh(self.time_residual_head(context)) * residual_bound,
            "base_coordinates": torch.sigmoid(
                self.base_coordinate_head(context).view(batch, num_slots, lanes, 2)
            ),
            "spinner_duration_beats": F.softplus(self.spinner_duration_head(context)),
            "slider_length": F.softplus(self.slider_length_head(context)),
            "slider_repeat_logits": self.slider_repeat_head(context).view(
                batch, num_slots, lanes, self.config.max_slider_repeats
            ),
        }


class CoordinateDiffusionRefiner(nn.Module):
    """Conditional continuous denoiser for circles and slider anchors.

    The discrete planner decides whether and what an event is.  This component
    only resolves the many plausible coordinates for that plan, which is the
    part of osu generation where diffusion/flow-style sampling is useful.
    """

    def __init__(self, config: BeatGridModelConfig = BeatGridModelConfig(), layers: int = 4):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.coordinate_proj = nn.Linear(2, hidden)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(hidden, dilation=2 ** (index % 4), dropout=config.dropout)
            for index in range(layers)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.GELU(), nn.Conv1d(hidden, 2, kernel_size=1)
        )

    def _time_embedding(self, noise_level: torch.Tensor, dim: int) -> torch.Tensor:
        return sinusoidal_positions(noise_level.unsqueeze(-1), dim).squeeze(1)

    def forward(
        self,
        context: torch.Tensor,
        noisy_coordinates: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian noise for ``[B, slots, lanes, 2]`` coordinates."""

        if context.dim() != 3:
            raise ValueError("context must have shape [B, slots, hidden]")
        batch, slots, hidden = context.shape
        lanes = self.config.events_per_slot
        if noisy_coordinates.shape != (batch, slots, lanes, 2):
            raise ValueError("noisy_coordinates must have shape [B, slots, lanes, 2]")
        if noise_level.shape not in {(batch,), (batch, 1)}:
            raise ValueError("noise_level must have shape [B] or [B, 1]")
        noise_level = noise_level.reshape(batch)

        coordinate = self.coordinate_proj(noisy_coordinates)
        time = self.time_proj(self._time_embedding(noise_level, hidden).to(context.dtype))
        x = coordinate + context.unsqueeze(2) + time[:, None, None, :]
        # Each lane receives a local temporal denoiser.  Shared planner context
        # carries global section information without flattening a whole song
        # into an infeasible all-pairs attention matrix.
        x = x.permute(0, 2, 3, 1).reshape(batch * lanes, hidden, slots)
        for block in self.blocks:
            x = block(x)
        epsilon = self.output(x).reshape(batch, lanes, 2, slots).permute(0, 3, 1, 2)
        return epsilon


def cosine_noise_schedule(noise_level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return alpha/sigma for a stable continuous coordinate diffusion target."""

    t = noise_level.clamp(0.0, 1.0)
    alpha = torch.cos(t * math.pi / 2.0)
    sigma = torch.sin(t * math.pi / 2.0)
    return alpha, sigma


def add_coordinate_noise(
    clean_coordinates: torch.Tensor,
    noise_level: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a diffusion training pair from normalized clean coordinates."""

    if clean_coordinates.dim() != 4 or clean_coordinates.shape[-1] != 2:
        raise ValueError("clean_coordinates must have shape [B, slots, lanes, 2]")
    batch = clean_coordinates.shape[0]
    if noise_level.shape not in {(batch,), (batch, 1)}:
        raise ValueError("noise_level must have shape [B] or [B, 1]")
    if noise is None:
        noise = torch.randn_like(clean_coordinates)
    alpha, sigma = cosine_noise_schedule(noise_level.reshape(batch))
    noisy = (
        alpha[:, None, None, None] * clean_coordinates
        + sigma[:, None, None, None] * noise
    )
    return noisy, noise
