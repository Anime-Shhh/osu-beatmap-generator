"""
Latent autoencoder and losses for continuous beatmap signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representation import NUM_SIGNAL_CHANNELS
from .tokenizer import TOTAL_VOCAB


@dataclass
class AutoencoderLossBreakdown:
    total: torch.Tensor
    l1: torch.Tensor
    l2: torch.Tensor
    multi_scale: torch.Tensor
    adversarial: torch.Tensor
    feature_matching: torch.Tensor


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, cond_channels: int = 0):
        super().__init__()
        self.cond_proj = nn.Conv1d(cond_channels, channels, kernel_size=1) if cond_channels > 0 else None
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        if self.cond_proj is not None and cond is not None:
            if cond.shape[-1] != x.shape[-1]:
                cond = F.interpolate(cond, size=x.shape[-1], mode="linear", align_corners=False)
            x = x + self.cond_proj(cond)
        return residual + self.block(x)


class LatentTokenHead(nn.Module):
    """Optional low-weight token supervision head on top of the latent sequence."""

    def __init__(self, latent_dim: int, vocab_size: int = TOTAL_VOCAB):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, vocab_size),
        )

    def forward(self, latent: torch.Tensor, target_len: int) -> torch.Tensor:
        if latent.shape[-1] != target_len:
            latent = F.interpolate(latent, size=target_len, mode="linear", align_corners=False)
        return self.proj(latent.transpose(1, 2))


class SignalAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_SIGNAL_CHANNELS,
        hidden_dim: int = 256,
        latent_dim: int = 96,
        downsample_factor: int = 8,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if downsample_factor != 8:
            raise ValueError("SignalAutoencoder currently expects downsample_factor=8")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor

        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualConvBlock(hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            )
            for _ in range(3)
        ])
        self.to_latent = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.latent_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.token_head = LatentTokenHead(latent_dim)

        self.from_latent = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualConvBlock(hidden_dim),
            )
            for _ in range(3)
        ])
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(signal)
        for block in self.encoder_blocks:
            x = block(x)
        latent = self.to_latent(x)
        latent = self.latent_transformer(latent.transpose(1, 2)).transpose(1, 2)
        return latent

    def decode(self, latent: torch.Tensor, output_len: Optional[int] = None) -> torch.Tensor:
        x = self.from_latent(latent)
        for block in self.decoder_blocks:
            x = block(x)
        if output_len is not None and x.shape[-1] != output_len:
            x = F.interpolate(x, size=output_len, mode="linear", align_corners=False)
        return self.output_proj(x)

    def forward(self, signal: torch.Tensor, token_len: int = 0) -> dict[str, torch.Tensor]:
        latent = self.encode(signal)
        reconstruction = self.decode(latent, output_len=signal.shape[-1])
        outputs = {
            "latent": latent,
            "reconstruction": reconstruction,
        }
        if token_len > 0:
            outputs["token_logits"] = self.token_head(latent, token_len)
        return outputs


class SignalCritic(nn.Module):
    def __init__(self, in_channels: int = NUM_SIGNAL_CHANNELS, hidden_dim: int = 128):
        super().__init__()
        channels = [in_channels, hidden_dim, hidden_dim * 2, hidden_dim * 4]
        layers = []
        for idx in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[idx], channels[idx + 1], kernel_size=5, stride=2, padding=2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, signal: torch.Tensor, return_features: bool = False):
        feature_maps = []
        x = signal
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feature_maps.append(x)
        logits = self.head(x)
        if return_features:
            return logits, feature_maps
        return logits


def multi_scale_l1_loss(pred: torch.Tensor, target: torch.Tensor, scales: tuple[int, ...] = (1, 2, 4, 8)) -> torch.Tensor:
    losses = []
    for scale in scales:
        if scale == 1:
            pred_scaled = pred
            target_scaled = target
        else:
            pred_scaled = F.avg_pool1d(pred, kernel_size=scale, stride=scale, ceil_mode=False)
            target_scaled = F.avg_pool1d(target, kernel_size=scale, stride=scale, ceil_mode=False)
        losses.append(F.l1_loss(pred_scaled, target_scaled))
    return sum(losses) / max(len(losses), 1)


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def feature_matching_loss(fake_features: list[torch.Tensor], real_features: list[torch.Tensor]) -> torch.Tensor:
    if not fake_features or not real_features:
        return torch.zeros((), device=fake_features[0].device if fake_features else real_features[0].device)
    loss = torch.zeros((), device=fake_features[0].device)
    for fake, real in zip(fake_features, real_features):
        loss = loss + F.l1_loss(fake, real.detach())
    return loss / len(fake_features)


def compute_autoencoder_losses(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    fake_logits: Optional[torch.Tensor] = None,
    real_features: Optional[list[torch.Tensor]] = None,
    fake_features: Optional[list[torch.Tensor]] = None,
) -> AutoencoderLossBreakdown:
    l1 = F.l1_loss(reconstruction, target)
    l2 = F.mse_loss(reconstruction, target)
    ms = multi_scale_l1_loss(reconstruction, target)

    adv = torch.zeros((), device=target.device)
    fm = torch.zeros((), device=target.device)
    if fake_logits is not None:
        adv = hinge_generator_loss(fake_logits)
    if fake_features is not None and real_features is not None:
        fm = feature_matching_loss(fake_features, real_features)

    total = (1.0 * l1) + (0.5 * l2) + (0.5 * ms) + (0.05 * adv) + (0.1 * fm)
    return AutoencoderLossBreakdown(
        total=total,
        l1=l1,
        l2=l2,
        multi_scale=ms,
        adversarial=adv,
        feature_matching=fm,
    )
