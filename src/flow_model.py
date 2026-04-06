"""
Latent flow-matching model for full-song beatmap generation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_ae import ResidualConvBlock
from .model import AudioEncoder, ConditioningEncoder


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class GlobalDifficultyConditioner(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.legacy_cond = ConditioningEncoder(d_model=d_model)
        self.scalar_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        difficulty_id: torch.Tensor,
        difficulty_value: torch.Tensor,
        bpm: torch.Tensor,
        *,
        drop_difficulty: bool = False,
    ) -> torch.Tensor:
        if drop_difficulty:
            difficulty_id = torch.zeros_like(difficulty_id)
            difficulty_value = torch.zeros_like(difficulty_value)

        legacy = self.legacy_cond(difficulty_id, bpm).squeeze(1)
        scalar = self.scalar_proj(difficulty_value)
        return self.norm(legacy + scalar)


class ChunkedASTAudioConditioner(nn.Module):
    """
    Reuses the existing AST frontend by chunking full-song mel/onset features.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        chunk_frames: int = 301,
        overlap_frames: int = 96,
        encoder_pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        encoder_unfreeze_last_n: int = 2,
        encoder_dim: int = 768,
    ):
        super().__init__()
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.audio_encoder = AudioEncoder(
            pretrained=encoder_pretrained,
            d_model=encoder_dim,
            unfreeze_last_n=encoder_unfreeze_last_n,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, cond_dim),
        )

    def _encode_single_chunk(self, chunk: torch.Tensor, original_len: int) -> torch.Tensor:
        encoded = self.audio_encoder(chunk)
        encoded = F.interpolate(
            encoded.transpose(1, 2),
            size=original_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return encoded

    def forward(self, mel: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        if mel.dim() != 4:
            raise ValueError(f"Expected mel shape [B, 1, C, T], got {tuple(mel.shape)}")

        batch, _, _, total_frames = mel.shape
        if total_frames <= self.chunk_frames:
            encoded = self._encode_single_chunk(mel, total_frames)
        else:
            step = max(self.chunk_frames - self.overlap_frames, 1)
            accum = mel.new_zeros(batch, total_frames, self.audio_encoder.d_model)
            counts = mel.new_zeros(batch, total_frames, 1)

            start = 0
            while start < total_frames:
                end = min(total_frames, start + self.chunk_frames)
                chunk = mel[:, :, :, start:end]
                chunk_len = end - start
                if chunk_len < self.chunk_frames:
                    chunk = F.pad(chunk, (0, self.chunk_frames - chunk_len))

                chunk_encoded = self._encode_single_chunk(chunk, chunk_len)
                accum[:, start:end, :] += chunk_encoded
                counts[:, start:end, :] += 1.0
                if end >= total_frames:
                    break
                start += step

            encoded = accum / counts.clamp(min=1.0)

        encoded = self.out_proj(encoded)
        if target_len is not None and encoded.shape[1] != target_len:
            encoded = F.interpolate(
                encoded.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return encoded.transpose(1, 2)  # [B, cond_dim, T]


class FlowResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_channels: int, global_dim: int):
        super().__init__()
        self.block = ResidualConvBlock(channels, cond_channels=cond_channels)
        self.global_proj = nn.Linear(global_dim, channels)

    def forward(self, x: torch.Tensor, cond_seq: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = x + self.global_proj(global_cond).unsqueeze(-1)
        return self.block(x, cond=cond_seq)


class LatentFlowMatcher(nn.Module):
    def __init__(
        self,
        latent_dim: int = 96,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        encoder_pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        encoder_unfreeze_last_n: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.audio_conditioner = ChunkedASTAudioConditioner(
            cond_dim=cond_dim,
            encoder_pretrained=encoder_pretrained,
            encoder_unfreeze_last_n=encoder_unfreeze_last_n,
        )
        self.global_conditioner = GlobalDifficultyConditioner(d_model=cond_dim)
        self.time_embedding = SinusoidalTimeEmbedding(cond_dim)

        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.audio_proj = nn.Conv1d(cond_dim, hidden_dim, kernel_size=1)

        self.down1 = FlowResidualBlock(hidden_dim, cond_channels=cond_dim, global_dim=cond_dim)
        self.pool1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.down2 = FlowResidualBlock(hidden_dim, cond_channels=cond_dim, global_dim=cond_dim)
        self.pool2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.mid = FlowResidualBlock(hidden_dim, cond_channels=cond_dim, global_dim=cond_dim)

        self.up1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.up_block1 = FlowResidualBlock(hidden_dim, cond_channels=cond_dim, global_dim=cond_dim)
        self.up2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.up_block2 = FlowResidualBlock(hidden_dim, cond_channels=cond_dim, global_dim=cond_dim)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mel: torch.Tensor,
        difficulty_id: torch.Tensor,
        difficulty_value: torch.Tensor,
        bpm: torch.Tensor,
        *,
        drop_difficulty: bool = False,
    ) -> dict[str, torch.Tensor]:
        audio_cond = self.audio_conditioner(mel, target_len=z_t.shape[-1])
        global_cond = self.global_conditioner(
            difficulty_id=difficulty_id,
            difficulty_value=difficulty_value,
            bpm=bpm,
            drop_difficulty=drop_difficulty,
        ) + self.time_embedding(t)

        x = self.input_proj(z_t) + self.audio_proj(audio_cond)
        skip1 = self.down1(x, cond_seq=audio_cond, global_cond=global_cond)
        x = self.pool1(skip1)

        cond_half = F.interpolate(audio_cond, size=x.shape[-1], mode="linear", align_corners=False)
        skip2 = self.down2(x, cond_seq=cond_half, global_cond=global_cond)
        x = self.pool2(skip2)

        cond_quarter = F.interpolate(audio_cond, size=x.shape[-1], mode="linear", align_corners=False)
        x = self.mid(x, cond_seq=cond_quarter, global_cond=global_cond)

        x = self.up1(x)
        if x.shape[-1] != skip2.shape[-1]:
            x = F.interpolate(x, size=skip2.shape[-1], mode="linear", align_corners=False)
        x = x + skip2
        x = self.up_block1(x, cond_seq=cond_half, global_cond=global_cond)

        x = self.up2(x)
        if x.shape[-1] != skip1.shape[-1]:
            x = F.interpolate(x, size=skip1.shape[-1], mode="linear", align_corners=False)
        x = x + skip1
        x = self.up_block2(x, cond_seq=audio_cond, global_cond=global_cond)

        velocity = self.output_proj(x)
        return {
            "velocity": velocity,
            "audio_condition": audio_cond,
            "global_condition": global_cond,
        }


def sample_flow_inputs(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Straight-line flow matching in latent space.
    """
    z1 = latent
    z0 = torch.randn_like(z1)
    t = torch.rand(z1.shape[0], device=z1.device)
    t_expanded = t.view(-1, 1, 1)
    zt = (1.0 - t_expanded) * z0 + t_expanded * z1
    velocity_target = z1 - z0
    return z0, t, zt, velocity_target


@torch.no_grad()
def integrate_flow(
    flow_model: LatentFlowMatcher,
    latent_shape: tuple[int, int, int],
    mel: torch.Tensor,
    difficulty_id: torch.Tensor,
    difficulty_value: torch.Tensor,
    bpm: torch.Tensor,
    *,
    steps: int = 32,
    guidance_scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = mel.device

    z = torch.randn(latent_shape, device=device)
    dt = 1.0 / max(steps, 1)
    use_autocast = device.type == "cuda"

    for step in range(steps):
        t = torch.full((latent_shape[0],), step / max(steps, 1), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            conditioned = flow_model(
                z_t=z,
                t=t,
                mel=mel,
                difficulty_id=difficulty_id,
                difficulty_value=difficulty_value,
                bpm=bpm,
                drop_difficulty=False,
            )["velocity"]

        if guidance_scale != 1.0:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                unconditioned = flow_model(
                    z_t=z,
                    t=t,
                    mel=mel,
                    difficulty_id=difficulty_id,
                    difficulty_value=difficulty_value,
                    bpm=bpm,
                    drop_difficulty=True,
                )["velocity"]
            velocity = unconditioned + guidance_scale * (conditioned - unconditioned)
        else:
            velocity = conditioned

        z = z + velocity * dt

    return z
