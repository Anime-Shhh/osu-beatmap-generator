"""Audio decoding (CPU, ffmpeg) and feature extraction (GPU, torch).

Features per song: float16 array [T, 130] at 22,050 Hz with hop 110 (4.9887 ms):
  0..127  natural-log mel power (n_fft 1024, Hann, 20 Hz..11.025 kHz, Slaney)
  128     spectral flux of a short 256-point STFT (sharp transient timing)
  129     log RMS of the short STFT frame
Frame i is centered on sample i * HOP (torch.stft center=True), i.e. at
``i * HOP / SR`` seconds of the decoded stream.

Container sniffing is by content, never by extension: ~10% of the corpus'
``.mp3`` tar members are Ogg, a few MP3s carry leading junk, OZUT is Opus.
The LAME/Xing tag presence is recorded because it controls encoder-delay
trimming, a stratum in the offset audit.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

SR = 22050
HOP = 110
N_FFT = 1024
N_MELS = 128
SHORT_FFT = 256
N_FEATURES = N_MELS + 2
FRAME_MS = 1000.0 * HOP / SR
FFMPEG = os.environ.get("OSU_FFMPEG", "ffmpeg")
MAX_SECONDS = 20 * 60


@dataclass(frozen=True)
class AudioInfo:
    container: str
    lame_tag: bool
    leading_zero_bytes: int
    nbytes: int


def read_locator(audio: dict[str, Any]) -> bytes:
    """Bytes for an index ``audio`` record: tar member (offset/bytes) or plain file."""
    kind = audio.get("kind")
    if kind == "tar-member":
        with open(audio["archive"], "rb") as handle:
            handle.seek(int(audio["offset"]))
            data = handle.read(int(audio["bytes"]))
        if len(data) != int(audio["bytes"]):
            raise IOError("short read from tar member")
        return data
    if kind == "file":
        with open(audio["path"], "rb") as handle:
            return handle.read()
    raise ValueError(f"unknown audio locator kind {kind!r}")


def sniff(data: bytes) -> AudioInfo:
    zeros = len(data) - len(data.lstrip(b"\x00"))
    body = data[zeros:]
    head = body[:64]
    if head.startswith(b"OggS"):
        container = "opus" if b"OpusHead" in body[:256] else "ogg"
    elif head.startswith(b"fLaC"):
        container = "flac"
    elif head.startswith(b"RIFF"):
        container = "wav"
    elif head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        container = "mp3"
    else:
        container = "unknown"
    lame = False
    if container == "mp3":
        start = 0
        if body.startswith(b"ID3") and len(body) >= 10:
            size = (body[6] << 21) | (body[7] << 14) | (body[8] << 7) | body[9]
            start = 10 + size
        window = body[start:start + 4096]
        lame = (b"Xing" in window or b"Info" in window) and b"LAME" in window
    return AudioInfo(container, lame, zeros, len(data))


def decode(data: bytes, timeout: float = 600.0) -> np.ndarray:
    """Decode any supported container to float32 mono at SR via ffmpeg (stdin pipe)."""
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-v", "error", "-i", "pipe:0",
           "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-300:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size < SR:
        raise RuntimeError("decoded audio shorter than one second")
    if audio.size > MAX_SECONDS * SR:
        raise RuntimeError("decoded audio longer than budget")
    return audio


@lru_cache(maxsize=4)
def _mel_basis(device: str):
    import librosa
    import torch
    basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=SR / 2, norm="slaney")
    return torch.from_numpy(basis.astype(np.float32)).to(device)


def features(audio: np.ndarray, device: str = "cuda") -> np.ndarray:
    """[T, 130] float16 feature matrix; T = 1 + len(audio) // HOP."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(audio)).to(device, torch.float32)
        long = torch.stft(x, N_FFT, HOP, N_FFT, torch.hann_window(N_FFT, device=device),
                          center=True, pad_mode="constant", return_complex=True)
        mel = _mel_basis(device) @ long.abs().pow(2)
        logmel = torch.log(mel.clamp_min(1e-10))
        short = torch.stft(x, SHORT_FFT, HOP, SHORT_FFT, torch.hann_window(SHORT_FFT, device=device),
                           center=True, pad_mode="constant", return_complex=True).abs()
        logmag = torch.log1p(100.0 * short)
        flux = torch.relu(torch.diff(logmag, dim=1, prepend=logmag[:, :1])).sum(0)
        rms = torch.log(short.pow(2).mean(0).clamp_min(1e-10)) * 0.5
        out = torch.cat([logmel, flux[None], rms[None]], 0).T.contiguous()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite features")
        return out.to(torch.float16).cpu().numpy()


def frame_times_ms(n_frames: int) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) * FRAME_MS
