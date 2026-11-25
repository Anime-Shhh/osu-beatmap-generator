"""One GPU pass over cached features -> small per-song artifacts.

* ``env``: float16 [T] onset envelope (z-scored mel flux + short-STFT flux), for
  the offset audit.
* ``fp``: int32 [T10] 31-bit binary audio fingerprints at 10 ms (Haitsma-Kalker
  style: signs of time-differences of adjacent band-energy differences), for
  duplicate search. Same-recording bit error rate is ~0.1-0.27 even with heavy
  added noise, unrelated audio sits at ~0.5 (measured on corpus songs).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .audio import N_MELS, SR

FP_BANDS = 32
FP_POOL = 2               # 5 ms feature frames -> 10 ms fingerprint frames
FP_FRAME_MS = 10.0


@lru_cache(maxsize=1)
def mel_centers_hz() -> np.ndarray:
    import librosa
    return librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=20.0, fmax=SR / 2)[1:-1]


def onset_envelope(feats, device: str = "cuda"):
    import torch
    x = torch.as_tensor(np.array(feats, dtype=np.float32), device=device)
    mel_flux = torch.relu(torch.diff(x[:, :N_MELS], dim=0, prepend=x[:1, :N_MELS])).mean(1)

    def z(v):
        k = 101  # ~0.5 s moving average removes loudness drift
        pad = torch.nn.functional.pad(v[None, None], (k // 2, k // 2), mode="replicate")
        base = torch.nn.functional.avg_pool1d(pad, k, 1)[0, 0]
        d = v - base
        return d / (d.std() + 1e-6)

    return (z(mel_flux) + z(x[:, N_MELS])).clamp(-5, 20)


def fingerprint(feats, device: str = "cuda", freq_scale: float = 1.0, time_scale: float = 1.0):
    """int64 [T10] 31-bit codes; optional pitch (freq) and tempo (time) warps of the input."""
    import torch
    x = torch.as_tensor(np.array(feats[:, :N_MELS], dtype=np.float32), device=device)
    if freq_scale != 1.0:
        centers = mel_centers_hz()
        src = np.interp(centers / freq_scale, centers, np.arange(N_MELS), left=0, right=N_MELS - 1)
        lo = torch.as_tensor(np.floor(src).astype(np.int64), device=device)
        frac = torch.as_tensor(src - np.floor(src), device=device, dtype=torch.float32)
        hi = (lo + 1).clamp(max=N_MELS - 1)
        x = x[:, lo] * (1 - frac) + x[:, hi] * frac
    power = x.exp()
    t = power.shape[0] // FP_POOL
    e = torch.log(power[: t * FP_POOL].reshape(t, FP_POOL, FP_BANDS, N_MELS // FP_BANDS).sum((1, 3)) + 1e-8)
    if time_scale != 1.0 and t > 2:
        n = max(2, int(round(t * time_scale)))
        e = torch.nn.functional.interpolate(e.T[None], size=n, mode="linear", align_corners=True)[0].T
    d = e[:, :-1] - e[:, 1:]
    b = (d[1:] - d[:-1]) > 0
    weights = (2 ** torch.arange(FP_BANDS - 1, device=device, dtype=torch.int64))
    return (b.to(torch.int64) * weights).sum(1)


def bit_error_rate(a, b) -> float:
    import torch
    x = torch.bitwise_xor(a, b)
    bits = sum(((x >> k) & 1).sum() for k in range(FP_BANDS - 1))
    return float(bits) / max(1, (FP_BANDS - 1) * x.numel())
