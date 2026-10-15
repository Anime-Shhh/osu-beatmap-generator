"""Timestamp-aligned hierarchical sparse-event planner.

This is the trainable v2 planner used instead of the earlier independent
two-lane beat-grid heads.  It deliberately accepts real feature timestamps and
bar boundaries, so a variable-BPM song is never aligned by stretching a frame
axis to a guessed number of beat slots.

The default configuration is sized for one 48 GB RTX A6000. It is intentionally
not instantiated by module import and has no model-download or GPU side effect.
Generation uses an evaluation-only pre-norm decoder cache so a long song does
not repeatedly recompute every bar-token prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import DecodePhase, EventGrammar, EventVocabulary, GrammarState


@dataclass(frozen=True)
class SparsePlannerConfig:
    """A single-A6000 symbolic planner configuration (about 75--90M params)."""

    rhythm_dim: int = 101  # 96 mel + RMS/flux/transient/beat confidence features.
    semantic_dim: int = 128
    profile_dim: int = 6  # target star, AR, CS, OD/HP, slider multiplier/tick-rate.
    clock_dim: int = 4  # tempo, meter/bar length, confidence, tempo-change summaries.
    model_dim: int = 768
    local_layers: int = 4
    global_layers: int = 3
    decoder_layers: int = 7
    heads: int = 12
    ff_dim: int = 2560
    dropout: float = 0.1
    boundary_context_layers: int = 1
    max_bars: int = 512
    max_decoder_tokens: int = 2048
    profile_dropout: float = 0.15
    local_audio_tokens: int = 0  # 0 preserves trained legacy checkpoint behavior.

    def __post_init__(self) -> None:
        if self.local_audio_tokens != 0 and not 2 <= self.local_audio_tokens <= 512:
            raise ValueError("local_audio_tokens must be zero or in [2,512]")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.profile_dropout < 0.0 or self.profile_dropout >= 1.0:
            raise ValueError("profile_dropout must be in [0, 1)")
        if self.boundary_context_layers < 1:
            raise ValueError("boundary_context_layers must be positive")


def _gather_time_axis(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather `[B, T, C]` values at `[B, Q]` indices without flattening batches."""

    if values.dim() != 3 or indices.dim() != 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("values must be [B,T,C] and indices [B,Q]")
    expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return values.gather(1, expanded)


def sample_features_at_times(
    values: torch.Tensor,
    times_ms: torch.Tensor,
    query_times_ms: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linearly sample timestamped features at real query times.

    `torch.searchsorted` is per-batch here.  This may be slightly less compact
    than `F.interpolate`, but it preserves the correct mapping when beat slots
    are nonuniform in milliseconds because tempo changes.
    """

    if values.dim() != 3 or times_ms.dim() != 2 or query_times_ms.dim() != 2:
        raise ValueError("values=[B,T,C], times_ms=[B,T], query_times_ms=[B,Q] required")
    if values.shape[:2] != times_ms.shape or values.shape[0] != query_times_ms.shape[0]:
        raise ValueError("timestamp/value batch shapes are incompatible")
    if values.shape[1] == 0:
        raise ValueError("cannot sample an empty feature stream")
    if valid_mask is not None:
        if valid_mask.shape != times_ms.shape:
            raise ValueError("valid_mask must match times_ms")
        # Invalid padding must never become an arbitrary endpoint.  Compact
        # valid entries into the beginning for each row before lookup.
        compact_values = []
        compact_times = []
        for row in range(values.shape[0]):
            count = int(valid_mask[row].sum().item())
            if count < 1:
                raise ValueError("each feature row needs at least one valid timestamp")
            row_values = values[row, valid_mask[row]]
            row_times = times_ms[row, valid_mask[row]]
            if count < values.shape[1]:
                row_values = F.pad(row_values, (0, 0, 0, values.shape[1] - count))
                row_times = F.pad(row_times, (0, values.shape[1] - count), value=float(row_times[-1]))
            compact_values.append(row_values)
            compact_times.append(row_times)
        values = torch.stack(compact_values, dim=0)
        times_ms = torch.stack(compact_times, dim=0)

    # ``searchsorted`` interpolation needs two endpoints.  A one-frame stream
    # can occur in very short clips or a deliberately pooled test fixture; its
    # only sensible interpolation is a constant feature.
    if values.shape[1] == 1:
        return values[:, :1, :].expand(-1, query_times_ms.shape[1], -1)

    # searchsorted operates over the final dimension and supports batched
    # sorted sequences. Clamp produces sensible endpoint extrapolation.
    upper = torch.searchsorted(times_ms.contiguous(), query_times_ms.contiguous(), right=False)
    upper = upper.clamp(1, values.shape[1] - 1)
    lower = upper - 1
    lower_time = times_ms.gather(1, lower)
    upper_time = times_ms.gather(1, upper)
    lower_values = _gather_time_axis(values, lower)
    upper_values = _gather_time_axis(values, upper)
    weight = ((query_times_ms - lower_time) / (upper_time - lower_time).clamp_min(1e-6)).unsqueeze(-1)
    sampled = lower_values.lerp(upper_values, weight.to(values.dtype))

    before = query_times_ms <= times_ms[:, :1]
    after = query_times_ms >= times_ms[:, -1:]
    sampled = torch.where(before.unsqueeze(-1), values[:, :1, :].expand_as(sampled), sampled)
    return torch.where(after.unsqueeze(-1), values[:, -1:, :].expand_as(sampled), sampled)


class MaskedChannelNorm(nn.Module):
    """Per-channel temporal normalization that excludes padded frames.

    ``GroupNorm`` computes statistics over the temporal axis, so merely
    zeroing padding before and after it still changes valid-frame statistics
    whenever two songs have different lengths. This small masked alternative
    keeps the local encoder invariant to right-padding content.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            mean = values.mean(dim=-1, keepdim=True)
            variance = values.var(dim=-1, keepdim=True, unbiased=False)
        else:
            weight = mask.unsqueeze(1).to(values.dtype)
            count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean = (values * weight).sum(dim=-1, keepdim=True) / count
            variance = ((values - mean).square() * weight).sum(dim=-1, keepdim=True) / count
        normalized = (values - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class MaskedTemporalBlock(nn.Module):
    """A local temporal block that zeroes padding at every residual boundary."""

    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = MaskedChannelNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=5, padding=2 * dilation, dilation=dilation, groups=dim
        )
        self.pointwise = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            values = values * mask.unsqueeze(1).to(values.dtype)
        residual = values
        values = self.norm(values, mask)
        if mask is not None:
            values = values * mask.unsqueeze(1).to(values.dtype)
        values = F.gelu(self.depthwise(values))
        left, gate = self.pointwise(values).chunk(2, dim=1)
        values = self.out(left * torch.sigmoid(gate))
        values = residual + self.dropout(F.gelu(values))
        if mask is not None:
            values = values * mask.unsqueeze(1).to(values.dtype)
        return values


class TimestampedAudioEncoder(nn.Module):
    """Fuse high-rate rhythmic and cached semantic streams at bar timestamps."""

    def __init__(self, config: SparsePlannerConfig):
        super().__init__()
        self.config = config
        local_dim = config.model_dim // 2
        self.rhythm_in = nn.Linear(config.rhythm_dim, local_dim)
        self.semantic_in = nn.Linear(config.semantic_dim, local_dim)
        self.rhythm_blocks = nn.ModuleList(
            MaskedTemporalBlock(local_dim, 2**index, config.dropout) for index in range(config.local_layers)
        )
        self.fuse = nn.Sequential(
            nn.Linear(local_dim * 2, config.model_dim), nn.GELU(), nn.LayerNorm(config.model_dim)
        )

    def forward(
        self,
        rhythm_features: torch.Tensor,
        rhythm_times_ms: torch.Tensor,
        semantic_features: torch.Tensor,
        semantic_times_ms: torch.Tensor,
        query_times_ms: torch.Tensor,
        *,
        rhythm_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
        return_local_frames: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if rhythm_features.dim() != 3 or semantic_features.dim() != 3:
            raise ValueError("audio features must be [B,T,C]")
        rhythm = self.rhythm_in(rhythm_features).transpose(1, 2)
        for block in self.rhythm_blocks:
            rhythm = block(rhythm, rhythm_mask)
        rhythm = rhythm.transpose(1, 2)
        rhythm_at_bars = sample_features_at_times(
            rhythm, rhythm_times_ms, query_times_ms, valid_mask=rhythm_mask
        )
        semantic = self.semantic_in(semantic_features)
        semantic_at_bars = sample_features_at_times(
            semantic, semantic_times_ms, query_times_ms, valid_mask=semantic_mask
        )
        bar_fused = self.fuse(torch.cat((rhythm_at_bars, semantic_at_bars), dim=-1))
        if return_local_frames:
            # Local high-rate frame representations for fine-time cross-attention
            return bar_fused, rhythm
        return bar_fused


class SparseEventPlanner(nn.Module):
    """Whole-song bar context plus grammar-constrained causal event decoder."""

    def __init__(
        self,
        config: SparsePlannerConfig = SparsePlannerConfig(),
        vocabulary: Optional[EventVocabulary] = None,
    ):
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary or EventVocabulary()
        self.audio_encoder = TimestampedAudioEncoder(config)
        self.profile = nn.Sequential(
            nn.Linear(config.profile_dim + config.clock_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.bar_position = nn.Embedding(config.max_bars, config.model_dim)
        bar_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bar_encoder = nn.TransformerEncoder(bar_layer, config.global_layers, enable_nested_tensor=False)
        self.token_embedding = nn.Embedding(len(self.vocabulary), config.model_dim)
        self.token_position = nn.Embedding(config.max_decoder_tokens, config.model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        # A bar-only target sequence otherwise has no symbolic knowledge of
        # the immediately preceding generated bar. A compact recurrent summary
        # carries that boundary state without turning full-song decoding into
        # one quadratic 10k-token attention problem.
        self.boundary_encoder = nn.GRU(
            config.model_dim,
            config.model_dim,
            num_layers=config.boundary_context_layers,
            batch_first=True,
        )
        self.no_boundary = nn.Parameter(torch.zeros(config.model_dim))
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, len(self.vocabulary))
        if config.local_audio_tokens:
            self.local_audio_projection = nn.Linear(config.model_dim // 2 + config.rhythm_dim, config.model_dim)
            self.local_audio_norm = nn.LayerNorm(config.model_dim)
            self.alignment_head = nn.Linear(config.model_dim, 4)

    def _profile_embedding(
        self,
        profile: torch.Tensor,
        clock_features: torch.Tensor,
        *,
        drop_profile: bool | torch.Tensor,
    ) -> torch.Tensor:
        if profile.dim() != 2 or profile.shape[-1] != self.config.profile_dim:
            raise ValueError("profile must be [B, profile_dim]")
        if clock_features.dim() != 3 or clock_features.shape[-1] != self.config.clock_dim:
            raise ValueError("clock_features must be [B,bars,clock_dim]")
        profile = profile.unsqueeze(1).expand(-1, clock_features.shape[1], -1)
        if isinstance(drop_profile, torch.Tensor):
            if drop_profile.shape != (profile.shape[0],):
                raise ValueError("profile dropout mask must be [B]")
            profile = profile * (~drop_profile).view(-1, 1, 1).to(profile.dtype)
        elif drop_profile:
            profile = torch.zeros_like(profile)
        return self.profile(torch.cat((profile, clock_features), dim=-1))

    def encode_bars(
        self,
        rhythm_features: torch.Tensor,
        rhythm_times_ms: torch.Tensor,
        semantic_features: torch.Tensor,
        semantic_times_ms: torch.Tensor,
        bar_mid_times_ms: torch.Tensor,
        profile: torch.Tensor,
        clock_features: torch.Tensor,
        *,
        bar_mask: Optional[torch.Tensor] = None,
        rhythm_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
        drop_profile: bool | torch.Tensor = False,
        return_local_frames: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if bar_mid_times_ms.dim() != 2:
            raise ValueError("bar_mid_times_ms must be [B,bars]")
        bars = self.audio_encoder(
            rhythm_features,
            rhythm_times_ms,
            semantic_features,
            semantic_times_ms,
            bar_mid_times_ms,
            rhythm_mask=rhythm_mask,
            semantic_mask=semantic_mask,
            return_local_frames=return_local_frames,
        )
        if return_local_frames:
            bars, local_frames = bars
        if bars.shape[1] > self.config.max_bars:
            raise ValueError("bar count exceeds configured max_bars")
        positions = torch.arange(bars.shape[1], device=bars.device).unsqueeze(0)
        bars = bars + self.bar_position(positions) + self._profile_embedding(
            profile, clock_features, drop_profile=drop_profile
        )
        if bar_mask is None:
            bar_mask = torch.ones(bars.shape[:2], device=bars.device, dtype=torch.bool)
        if bar_mask.shape != bars.shape[:2]:
            raise ValueError("bar_mask must be [B,bars]")
        bars = bars * bar_mask.unsqueeze(-1).to(bars.dtype)
        encoded = self.bar_encoder(bars, src_key_padding_mask=~bar_mask)
        encoded = encoded * bar_mask.unsqueeze(-1).to(encoded.dtype)
        return (encoded, local_frames) if return_local_frames else encoded

    def local_audio_memory(self, bar_memory, local_frames, rhythm_features, rhythm_times_ms,
                           query_times_ms, target_bar_index, *, rhythm_mask=None,
                           bar_mask=None, query_mask=None):
        """Append ordered target-bar audio frames; keep bar indices unchanged.

        Chart labels never enter this path. The auxiliary head sees only audio
        plus clock/profile-conditioned bar context, not token-prefix history.
        """
        if not self.config.local_audio_tokens:
            raise ValueError("local audio memory is disabled in this checkpoint")
        if query_times_ms.shape != (bar_memory.shape[0], self.config.local_audio_tokens):
            raise ValueError("query grid does not match checkpoint local_audio_tokens")
        if not torch.isfinite(query_times_ms).all() or (query_times_ms[:,1:] <= query_times_ms[:,:-1]).any():
            raise ValueError("local query times must be finite and strictly increasing")
        sampled = sample_features_at_times(local_frames, rhythm_times_ms, query_times_ms, valid_mask=rhythm_mask)
        raw = sample_features_at_times(rhythm_features, rhythm_times_ms, query_times_ms, valid_mask=rhythm_mask)
        # Raw features bypass temporal normalization, preserving relative energy
        # and sharp attacks alongside the learned receptive-field features.
        audio = self.local_audio_projection(torch.cat((sampled, raw), dim=-1))
        phase = torch.arange(query_times_ms.shape[1], device=audio.device).float()
        frequencies = torch.exp(torch.arange((audio.shape[-1]+1)//2, device=audio.device).float()
                                * (-math.log(10000.) / max(audio.shape[-1]//2-1, 1)))
        angles = phase[:,None] * frequencies[None,:]
        position = torch.cat((angles.sin(), angles.cos()), dim=-1)[:,:audio.shape[-1]]
        context = bar_memory.gather(1, target_bar_index[:,None,None].expand(-1,1,bar_memory.shape[-1]))
        audio = self.local_audio_norm(audio + position.unsqueeze(0) + context)
        if bar_mask is None:
            bar_mask = torch.ones(bar_memory.shape[:2], device=audio.device, dtype=torch.bool)
        if rhythm_mask is None:
            rhythm_mask = torch.ones(rhythm_times_ms.shape, device=audio.device, dtype=torch.bool)
        first = rhythm_times_ms.masked_fill(~rhythm_mask, float('inf')).min(dim=1).values
        last = rhythm_times_ms.masked_fill(~rhythm_mask, -float('inf')).max(dim=1).values
        valid = (query_times_ms >= first[:,None]) & (query_times_ms <= last[:,None])
        if query_mask is not None:
            valid = valid & query_mask
        audio = audio * valid.unsqueeze(-1)
        return (torch.cat((bar_memory, audio), dim=1), torch.cat((bar_mask, valid), dim=1),
                self.alignment_head(audio), valid)

    def _boundary_embedding(
        self,
        previous_bar_tokens: Optional[torch.Tensor],
        previous_bar_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        drop_boundary: bool | torch.Tensor,
    ) -> torch.Tensor:
        """Summarize the prior symbolic bar or emit a learned start state.

        Training receives the preceding reference bar, while generation feeds
        back the preceding sampled bar. Condition dropout prevents the model
        from requiring a perfect predecessor and makes the first-bar contract
        explicit. Masks must be right-padded: accepting holes would make a
        serial boundary state depend on an arbitrary padding convention.
        """

        baseline = self.no_boundary.unsqueeze(0).expand(batch_size, -1)
        if previous_bar_tokens is None:
            summary = baseline
        else:
            if previous_bar_tokens.dim() != 2 or previous_bar_tokens.shape[0] != batch_size:
                raise ValueError("previous_bar_tokens must be [B,L]")
            if previous_bar_mask is None:
                previous_bar_mask = torch.ones_like(previous_bar_tokens, dtype=torch.bool)
            if previous_bar_mask.shape != previous_bar_tokens.shape:
                raise ValueError("previous_bar_mask must match previous_bar_tokens")
            if previous_bar_tokens.shape[1] == 0:
                summary = baseline
            else:
                lengths = previous_bar_mask.sum(dim=1)
                positions = torch.arange(
                    previous_bar_tokens.shape[1], device=previous_bar_tokens.device
                ).unsqueeze(0)
                expected = positions < lengths.unsqueeze(1)
                if not torch.equal(previous_bar_mask, expected):
                    raise ValueError("previous_bar_mask must be right-padded without holes")
                valid_rows = lengths.gt(0)
                summary = baseline.clone()
                if valid_rows.any():
                    embedded = self.token_embedding(previous_bar_tokens[valid_rows])
                    packed = nn.utils.rnn.pack_padded_sequence(
                        embedded,
                        lengths[valid_rows].detach().cpu(),
                        batch_first=True,
                        enforce_sorted=False,
                    )
                    _, hidden = self.boundary_encoder(packed)
                    summary = summary.index_copy(
                        0, valid_rows.nonzero(as_tuple=False).squeeze(1), hidden[-1].to(summary.dtype)
                    )
        if isinstance(drop_boundary, torch.Tensor):
            if drop_boundary.shape != (batch_size,):
                raise ValueError("boundary dropout mask must be [B]")
            return torch.where(drop_boundary.unsqueeze(-1), baseline, summary)
        return baseline if drop_boundary else summary

    def _decode_conditions(
        self,
        bar_memory: torch.Tensor,
        *,
        target_bar_index: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        previous_bar_tokens: Optional[torch.Tensor],
        previous_bar_mask: Optional[torch.Tensor],
        drop_boundary: bool | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve reusable target-bar and prior-bar conditions for decoding."""

        batch_size = bar_memory.shape[0]
        if target_bar_index is None:
            target_bar_index = torch.zeros(batch_size, dtype=torch.long, device=bar_memory.device)
        if target_bar_index.shape != (batch_size,):
            raise ValueError("target_bar_index must be [B]")
        if (target_bar_index < 0).any() or (target_bar_index >= bar_memory.shape[1]).any():
            raise ValueError("target_bar_index contains an out-of-range bar")
        if memory_mask is None:
            memory_mask = torch.ones(bar_memory.shape[:2], device=bar_memory.device, dtype=torch.bool)
        if memory_mask.shape != bar_memory.shape[:2]:
            raise ValueError("memory_mask must be [B,bars]")
        if not memory_mask.gather(1, target_bar_index.view(-1, 1)).all():
            raise ValueError("target_bar_index must select an unpadded bar")
        target_context = bar_memory.gather(
            1,
            target_bar_index.view(-1, 1, 1).expand(-1, 1, bar_memory.shape[-1]),
        )
        boundary_context = self._boundary_embedding(
            previous_bar_tokens,
            previous_bar_mask,
            batch_size=batch_size,
            drop_boundary=drop_boundary,
        )
        return target_context, boundary_context, memory_mask, target_bar_index

    def decode_hidden(
        self,
        decoder_tokens: torch.Tensor,
        bar_memory: torch.Tensor,
        *,
        target_bar_index: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        previous_bar_tokens: Optional[torch.Tensor] = None,
        previous_bar_mask: Optional[torch.Tensor] = None,
        drop_boundary: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        if decoder_tokens.dim() != 2 or bar_memory.dim() != 3:
            raise ValueError("decoder_tokens=[B,L], bar_memory=[B,bars,D] required")
        if decoder_tokens.shape[0] != bar_memory.shape[0]:
            raise ValueError("decoder and memory batch sizes must agree")
        if decoder_tokens.shape[1] > self.config.max_decoder_tokens:
            raise ValueError("decoder sequence exceeds max_decoder_tokens")
        length = decoder_tokens.shape[1]
        positions = torch.arange(length, device=decoder_tokens.device).unsqueeze(0)
        target = self.token_embedding(decoder_tokens) + self.token_position(positions)
        # Each record contains one grammar sequence for one meter-aware bar,
        # while cross-attention sees the complete song's bar context.  Without
        # this explicit target-bar identity the decoder can attend to a song
        # but has no way to know which bar it is meant to emit.
        target_context, boundary_context, memory_mask, _ = self._decode_conditions(
            bar_memory,
            target_bar_index=target_bar_index,
            memory_mask=memory_mask,
            previous_bar_tokens=previous_bar_tokens,
            previous_bar_mask=previous_bar_mask,
            drop_boundary=drop_boundary,
        )
        target = target + target_context + boundary_context.unsqueeze(1)
        causal = torch.full((length, length), float("-inf"), device=target.device)
        causal = torch.triu(causal, diagonal=1)
        if token_mask is None:
            token_mask = torch.ones_like(decoder_tokens, dtype=torch.bool)
        decoded = self.decoder(
            target,
            bar_memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~token_mask,
            memory_key_padding_mask=~memory_mask,
        )
        return self.output_norm(decoded)

    def _decode_step_with_conditions(
        self,
        decoder_token: torch.Tensor,
        bar_memory: torch.Tensor,
        *,
        position: int,
        target_context: torch.Tensor,
        boundary_context: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Incrementally decode one token using pre-norm self-attention cache.

        ``nn.TransformerDecoder`` does not expose a KV-cache API. Re-running
        it over every prefix makes autoregressive full-song inference cubic in
        tokens per bar. In evaluation mode, this mirrors its pre-norm layer
        equations while caching each layer's normalized self-attention keys and
        values. Cross-attention still uses the complete song-bar memory, but
        all already-decoded target positions avoid repeated decoder passes.
        """

        if decoder_token.dim() != 2 or decoder_token.shape[1] != 1:
            raise ValueError("decoder_token must be [B,1]")
        if decoder_token.shape[0] != bar_memory.shape[0]:
            raise ValueError("decoder token and bar memory batch sizes must agree")
        if not 0 <= position < self.config.max_decoder_tokens:
            raise ValueError("decoder position exceeds max_decoder_tokens")
        if target_context.shape != (bar_memory.shape[0], 1, self.config.model_dim):
            raise ValueError("target_context must be [B,1,model_dim]")
        if boundary_context.shape != (bar_memory.shape[0], self.config.model_dim):
            raise ValueError("boundary_context must be [B,model_dim]")
        if memory_mask.shape != bar_memory.shape[:2]:
            raise ValueError("memory_mask must be [B,bars]")
        histories = tuple(cache or ())
        if histories and len(histories) != len(self.decoder.layers):
            raise ValueError("decoder cache layer count does not match planner")
        if not histories:
            histories = (None,) * len(self.decoder.layers)  # type: ignore[assignment]
        target = self.token_embedding(decoder_token) + self.token_position(
            torch.full((decoder_token.shape[0], 1), position, device=decoder_token.device, dtype=torch.long)
        )
        target = target + target_context + boundary_context.unsqueeze(1)
        next_cache: list[torch.Tensor] = []
        for layer, history in zip(self.decoder.layers, histories):
            # The planner deliberately constructs TransformerDecoderLayer with
            # norm_first=True. Keeping this assertion close to the cache math
            # prevents a future model-config edit from silently changing
            # generation semantics.
            if not layer.norm_first:
                raise RuntimeError("incremental decoder cache requires norm_first=True")
            normalized = layer.norm1(target)
            if history is not None:
                if history.dim() != 3 or history.shape[0] != normalized.shape[0] or history.shape[2] != normalized.shape[2]:
                    raise ValueError("decoder cache tensor has incompatible shape")
                keys_values = torch.cat((history, normalized), dim=1)
            else:
                keys_values = normalized
            self_attention, _ = layer.self_attn(
                normalized,
                keys_values,
                keys_values,
                need_weights=False,
            )
            target = target + layer.dropout1(self_attention)
            cross_input = layer.norm2(target)
            cross_attention, _ = layer.multihead_attn(
                cross_input,
                bar_memory,
                bar_memory,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )
            target = target + layer.dropout2(cross_attention)
            feedforward_input = layer.norm3(target)
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(feedforward_input))))
            target = target + layer.dropout3(feedforward)
            next_cache.append(keys_values)
        return self.output_norm(target), tuple(next_cache)

    def decode_step(
        self,
        decoder_token: torch.Tensor,
        bar_memory: torch.Tensor,
        *,
        position: int,
        target_bar_index: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        previous_bar_tokens: Optional[torch.Tensor] = None,
        previous_bar_mask: Optional[torch.Tensor] = None,
        drop_boundary: bool | torch.Tensor = False,
        cache: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Decode one token incrementally; intended for evaluation generation."""

        if self.training:
            raise RuntimeError("decode_step cache is evaluation-only; call planner.eval() before generation")
        target_context, boundary_context, resolved_memory_mask, _ = self._decode_conditions(
            bar_memory,
            target_bar_index=target_bar_index,
            memory_mask=memory_mask,
            previous_bar_tokens=previous_bar_tokens,
            previous_bar_mask=previous_bar_mask,
            drop_boundary=drop_boundary,
        )
        return self._decode_step_with_conditions(
            decoder_token,
            bar_memory,
            position=position,
            target_context=target_context,
            boundary_context=boundary_context,
            memory_mask=resolved_memory_mask,
            cache=cache,
        )

    def decode_tokens(
        self,
        decoder_tokens: torch.Tensor,
        bar_memory: torch.Tensor,
        *,
        target_bar_index: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        previous_bar_tokens: Optional[torch.Tensor] = None,
        previous_bar_mask: Optional[torch.Tensor] = None,
        drop_boundary: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        """Return next-token logits while retaining ``decode_hidden`` for diffusion."""

        return self.output(
            self.decode_hidden(
                decoder_tokens,
                bar_memory,
                target_bar_index=target_bar_index,
                token_mask=token_mask,
                memory_mask=memory_mask,
                previous_bar_tokens=previous_bar_tokens,
                previous_bar_mask=previous_bar_mask,
                drop_boundary=drop_boundary,
            )
        )

    def forward(
        self,
        *,
        rhythm_features: torch.Tensor,
        rhythm_times_ms: torch.Tensor,
        semantic_features: torch.Tensor,
        semantic_times_ms: torch.Tensor,
        bar_mid_times_ms: torch.Tensor,
        profile: torch.Tensor,
        clock_features: torch.Tensor,
        decoder_tokens: torch.Tensor,
        target_bar_index: Optional[torch.Tensor] = None,
        bar_mask: Optional[torch.Tensor] = None,
        rhythm_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        drop_profile: bool | torch.Tensor = False,
        previous_bar_tokens: Optional[torch.Tensor] = None,
        previous_bar_mask: Optional[torch.Tensor] = None,
        drop_boundary: bool | torch.Tensor = False,
        local_query_times_ms: Optional[torch.Tensor] = None,
        local_query_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_bars(
            rhythm_features,
            rhythm_times_ms,
            semantic_features,
            semantic_times_ms,
            bar_mid_times_ms,
            profile,
            clock_features,
            bar_mask=bar_mask,
            rhythm_mask=rhythm_mask,
            semantic_mask=semantic_mask,
            drop_profile=drop_profile,
            return_local_frames=bool(self.config.local_audio_tokens),
        )
        extra = {}
        decoder_memory_mask = bar_mask
        if self.config.local_audio_tokens:
            if local_query_times_ms is None or target_bar_index is None:
                raise ValueError("alignment planner requires timestamped local queries and target bar")
            memory, local_frames = memory
            memory, decoder_memory_mask, logits, mask = self.local_audio_memory(
                memory, local_frames, rhythm_features, rhythm_times_ms, local_query_times_ms,
                target_bar_index, rhythm_mask=rhythm_mask, bar_mask=bar_mask, query_mask=local_query_mask)
            extra = {"alignment_logits": logits, "alignment_mask": mask}
        hidden = self.decode_hidden(
            decoder_tokens,
            memory,
            target_bar_index=target_bar_index,
            token_mask=token_mask,
            memory_mask=decoder_memory_mask,
            previous_bar_tokens=previous_bar_tokens,
            previous_bar_mask=previous_bar_mask,
            drop_boundary=drop_boundary,
        )
        return {**extra, "logits": self.output(hidden), "bar_memory": memory, "decoder_hidden": hidden}

    @torch.no_grad()
    def generate_bar(
        self,
        bar_memory: torch.Tensor,
        *,
        memory_mask: Optional[torch.Tensor] = None,
        target_bar_index: int = 0,
        bar_length_beats: float | None = None,
        minimum_start_gap_beats: float = 0.0,
        previous_event_relative_beat: float | None = None,
        previous_bar_tokens: Optional[Sequence[int] | torch.Tensor] = None,
        max_tokens: int = 256,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> list[int]:
        """Generate one valid grammar sequence for a batch-size-one bar memory.

        ``previous_bar_tokens`` is the immediately preceding *sampled* bar at
        inference (or the reference bar in teacher-forced training). It is a
        compact boundary condition, not a hidden oracle timing map.
        """

        if bar_memory.shape[0] != 1:
            raise ValueError("generate_bar currently supports one candidate at a time")
        if self.training:
            raise RuntimeError("generate_bar requires planner.eval() so cached decoding matches teacher forcing")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        grammar = EventGrammar(self.vocabulary)
        if bar_length_beats is not None and not 0.0 < float(bar_length_beats) <= self.vocabulary.max_bar_beats:
            raise ValueError("bar_length_beats must fit the sparse vocabulary when supplied")
        if previous_bar_tokens is None:
            boundary_tokens = None
            boundary_mask = None
        else:
            if isinstance(previous_bar_tokens, torch.Tensor):
                raw_boundary = previous_bar_tokens.to(device=bar_memory.device, dtype=torch.long)
                if raw_boundary.dim() == 1:
                    raw_boundary = raw_boundary.unsqueeze(0)
                if raw_boundary.shape[0] != 1:
                    raise ValueError("previous_bar_tokens tensor must be [L] or [1,L]")
            else:
                raw_boundary = torch.tensor(
                    tuple(previous_bar_tokens), dtype=torch.long, device=bar_memory.device
                ).unsqueeze(0)
            if raw_boundary.shape[1] == 0:
                boundary_tokens = None
                boundary_mask = None
            else:
                if (raw_boundary < 0).any() or (raw_boundary >= len(self.vocabulary)).any():
                    raise ValueError("previous_bar_tokens contain unknown vocabulary IDs")
                boundary_tokens = raw_boundary
                boundary_mask = torch.ones_like(raw_boundary, dtype=torch.bool)
        target_index = torch.tensor([target_bar_index], dtype=torch.long, device=bar_memory.device)
        target_context, boundary_context, resolved_memory_mask, _ = self._decode_conditions(
            bar_memory,
            target_bar_index=target_index,
            memory_mask=memory_mask,
            previous_bar_tokens=boundary_tokens,
            previous_bar_mask=boundary_mask,
            drop_boundary=False,
        )
        tokens = [self.vocabulary.bos_id]
        if minimum_start_gap_beats and bar_length_beats is None:
            raise ValueError("minimum start gap requires explicit bar_length_beats")
        from .grammar import RESIDUAL_TICKS_PER_BEAT
        initial = GrammarState(last_time_units=round(previous_event_relative_beat * RESIDUAL_TICKS_PER_BEAT)) \
            if previous_event_relative_beat is not None else GrammarState()
        state = grammar.transition(initial, tokens[0])
        cache: tuple[torch.Tensor, ...] | None = None
        for _ in range(max_tokens):
            input_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=bar_memory.device)
            hidden, cache = self._decode_step_with_conditions(
                input_token,
                bar_memory,
                position=len(tokens) - 1,
                target_context=target_context,
                boundary_context=boundary_context,
                memory_mask=resolved_memory_mask,
                cache=cache,
            )
            logits = self.output(hidden)[:, -1, :]
            allowed = (
                grammar.allowed_ids(state)
                if bar_length_beats is None
                else grammar.allowed_ids_for_bar(state, meter=float(bar_length_beats),
                                                 minimum_start_gap_beats=minimum_start_gap_beats)
            )
            if not allowed:
                raise RuntimeError("meter-aware grammar has no legal continuation")
            allowed_mask = torch.full_like(logits, float("-inf"))
            allowed_mask[:, list(allowed)] = 0.0
            next_logits = logits / temperature + allowed_mask
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probabilities[0], 1, generator=generator).item())
            tokens.append(next_token)
            state = grammar.transition(state, next_token)
            if state.phase == DecodePhase.START:
                return tokens
        raise RuntimeError("grammar-constrained generation exceeded max_tokens without EOS")


def count_trainable_parameters(module: nn.Module) -> int:
    """Small utility used by dry-run commands and tests; never allocates a GPU."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
