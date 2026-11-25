"""[T] Timing: beat / downbeat activations at 5 ms plus a song-level tempo class.

Conv front end at 5 ms -> stride-4 to 20 ms -> transformer (RoPE over frames) ->
transposed conv back to 5 ms with a skip from the front end. Labels come from
osu! red lines (mapper BPM conventions), so the tempo head learns which octave a
mapper would choose. Precision comes from the tempo fit (src/v3/infer/tempo.py),
which least-squares fits a grid to many peaks at once.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Backbone, ConvFrontend

TEMPO_BINS = 80                 # 0.05-octave bins from 40 BPM (40 .. ~640 BPM)
TEMPO_MIN = 40.0


def tempo_bin(bpm: float) -> int:
    return int(np.clip(round(math.log2(max(bpm, TEMPO_MIN) / TEMPO_MIN) / 0.05), 0, TEMPO_BINS - 1))


def bin_bpm(b: int) -> float:
    return TEMPO_MIN * 2 ** (b * 0.05)


class TimingModel(nn.Module):
    def __init__(self, *, d: int = 384, depth: int = 8, heads: int = 6, audio_dim: int = 130, width: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.frontend = ConvFrontend(audio_dim, width, width, depth=5)
        self.down = nn.Conv1d(width, d, 4, stride=4)
        self.cond = nn.Parameter(torch.zeros(1, 64))
        self.backbone = Backbone(d, depth, heads, 64, dropout, rope_base=10000.0)
        self.up = nn.ConvTranspose1d(d, width, 4, stride=4)
        self.head = nn.Sequential(nn.Conv1d(2 * width, width, 5, padding=2), nn.GELU(), nn.Conv1d(width, 2, 1))
        self.pool = nn.Linear(d, 1)
        self.tempo = nn.Linear(d, TEMPO_BINS)

    def forward(self, audio, frame_mask):
        """audio [B, T, F] (T multiple of 4), frame_mask [B, T] -> beat/downbeat logits [B, T, 2], tempo [B, bins]."""
        b, t, _ = audio.shape
        f = self.frontend(audio)                                         # [B, T, w]
        h = self.down(f.transpose(1, 2)).transpose(1, 2)                 # [B, T/4, d]
        m4 = frame_mask[:, ::4][:, : h.shape[1]]
        pos = torch.arange(h.shape[1], device=h.device, dtype=torch.float32)[None].expand(b, -1)
        h = self.backbone(h, self.cond.expand(b, -1), pos, m4)
        u = self.up(h.transpose(1, 2))[..., :t]
        logits = self.head(torch.cat([u, f.transpose(1, 2)], 1)).transpose(1, 2)
        w = self.pool(h).squeeze(-1).masked_fill(~m4, -1e4).softmax(-1)
        tempo = self.tempo((w[..., None] * h).sum(1))
        return logits, tempo

    @torch.no_grad()
    def activations(self, audio_np: np.ndarray, device, chunk_frames: int = 12000, overlap: int = 2000):
        """Full-song inference in overlapping chunks -> beat, downbeat probs [T], tempo probs [bins]."""
        t = len(audio_np)
        beat = np.zeros(t, np.float32)
        down = np.zeros(t, np.float32)
        weight = np.zeros(t, np.float32)
        tempo_logits = []
        step = chunk_frames - overlap
        for s in range(0, max(1, t - overlap), step):
            e = min(t, s + chunk_frames)
            n = e - s
            pad = (-n) % 4
            x = torch.from_numpy(np.pad(audio_np[s:e], ((0, pad), (0, 0)))).to(device)[None]
            m = torch.zeros(1, n + pad, dtype=torch.bool, device=device)
            m[:, :n] = True
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg, tp = self(x, m)
            p = torch.sigmoid(lg.float())[0, :n].cpu().numpy()
            win = np.minimum(1.0, np.minimum(np.arange(n) + 1, n - np.arange(n)) / max(1, overlap / 2)).astype(np.float32)
            beat[s:e] += p[:, 0] * win
            down[s:e] += p[:, 1] * win
            weight[s:e] += win
            tempo_logits.append(tp.float()[0].cpu().numpy() * n)
            if e == t:
                break
        weight = np.maximum(weight, 1e-6)
        tl = np.sum(tempo_logits, 0) / t
        return beat / weight, down / weight, np.exp(tl - tl.max()) / np.exp(tl - tl.max()).sum()
