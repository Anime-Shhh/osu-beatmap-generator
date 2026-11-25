"""Shared transformer building blocks (all randomly initialized).

DiT-style blocks: RMSNorm, QK-normalized attention with rotary embeddings over
*float* positions (beat index for rhythm, musical time for placement, frame
index for timing), SwiGLU feed-forward, and AdaLN-zero global conditioning.
Attention uses torch SDPA (flash/mem-efficient kernels when available).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return y * self.weight if self.weight is not None else y


def rope_cos_sin(pos: torch.Tensor, dim: int, base: float = 10000.0):
    """pos: [B, L] float -> cos/sin [B, 1, L, dim]."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    ang = pos.float()[..., None] * inv
    ang = torch.cat([ang, ang], -1)
    return ang.cos()[:, None], ang.sin()[:, None]


def apply_rope(x, cos, sin):
    a, b = x.chunk(2, -1)
    rot = torch.cat([-b, a], -1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, rope_base: float = 10000.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim, self.rope_base = heads, dim // heads, rope_base
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, pos, key_mask=None):
        b, l, _ = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(pos, self.head_dim, self.rope_base)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, l, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        hidden = 64 * ((hidden + 63) // 64)
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(self.drop(F.silu(a) * b))


class DiTBlock(nn.Module):
    """Pre-norm block with AdaLN-zero modulation from a global condition vector."""

    def __init__(self, dim: int, heads: int, cond_dim: int, dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = Attention(dim, heads, rope_base)
        self.norm2 = RMSNorm(dim, affine=False)
        self.ffn = SwiGLU(dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond, pos, key_mask=None):
        s1, c1, g1, s2, c2, g2 = self.ada(F.silu(cond))[:, None].chunk(6, -1)
        h = self.norm1(x) * (1 + c1) + s1
        x = x + g1 * self.drop(self.attn(h, pos, key_mask))
        h = self.norm2(x) * (1 + c2) + s2
        return x + g2 * self.drop(self.ffn(h))


class Backbone(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, cond_dim: int, dropout: float = 0.0,
                 rope_base: float = 10000.0):
        super().__init__()
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, cond_dim, dropout, rope_base) for _ in range(depth))
        self.norm = RMSNorm(dim, affine=False)
        self.final_ada = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)

    def forward(self, x, cond, pos, key_mask=None):
        for block in self.blocks:
            x = block(x, cond, pos, key_mask)
        shift, scale = self.final_ada(F.silu(cond))[:, None].chunk(2, -1)
        return self.norm(x) * (1 + scale) + shift


class FourierScalar(nn.Module):
    """Continuous scalar -> embedding with a learned null value for dropout/CFG."""

    def __init__(self, out_dim: int, n_freq: int = 16, max_freq: float = 64.0, nullable: bool = True):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(max_freq), n_freq)) * math.pi,
                             persistent=False)
        self.proj = nn.Sequential(nn.Linear(2 * n_freq + 1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
        # Only scalars that are actually dropped get a null: a parameter that never receives a
        # gradient is harmless on one GPU but aborts DDP (find_unused_parameters=False).
        self.null = nn.Parameter(torch.zeros(out_dim)) if nullable else None

    def forward(self, value, keep=None):
        """value: [...] already scaled to roughly [0, 1]; keep: bool [...] (False -> null)."""
        ang = value.float()[..., None] * self.freqs
        emb = self.proj(torch.cat([value.float()[..., None], ang.sin(), ang.cos()], -1))
        if keep is not None:
            if self.null is None:
                raise ValueError("this FourierScalar was built with nullable=False")
            emb = torch.where(keep[..., None], emb, self.null.to(emb.dtype))
        return emb


class ConvFrontend(nn.Module):
    """Audio features [B, T, F] at 5 ms -> [B, T, out] with a ~+-80 ms receptive field."""

    def __init__(self, in_dim: int, width: int, out_dim: int, depth: int = 4):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, width, 5, padding=2)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.GroupNorm(1, width), nn.Conv1d(width, width, 5, padding=2 * d, dilation=d, groups=width),
                nn.Conv1d(width, 2 * width, 1), nn.GLU(dim=1)))
        self.out = nn.Conv1d(width, out_dim, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        return self.out(h).transpose(1, 2)


def gather_frames(feats: torch.Tensor, frame_pos: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of [B, T, C] features at fractional frame positions [B, ...]."""
    b, t, c = feats.shape
    shape = frame_pos.shape
    pos = frame_pos.reshape(b, -1).clamp(0, t - 1.001)
    lo = pos.floor().long()
    w = (pos - lo).to(feats.dtype)[..., None]
    idx_lo = lo[..., None].expand(-1, -1, c)
    a = torch.gather(feats, 1, idx_lo)
    z = torch.gather(feats, 1, (idx_lo + 1).clamp(max=t - 1))
    return (a * (1 - w) + z * w).reshape(*shape, c)
