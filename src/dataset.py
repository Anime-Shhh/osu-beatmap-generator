"""
Dataset pipeline for osu! beatmap generation.

Supports the legacy token-window path and the new continuous-signal path
without changing the default runtime behavior.
"""

from __future__ import annotations

import glob
import io
import json
import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .representation import DEFAULT_FRAME_RATE, encode_osu_content_to_signal
from .tokenizer import (
    BEAT_QUANT,
    PAD,
    Residuals,
    difficulty_to_bin,
    parse_osu_bpm,
    tokenize_beatmap,
)


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 441  # ~20 ms at 22050 Hz
N_MELS = 128
N_FEATURES = N_MELS + 1  # mel + onset channel

WINDOW_SEC = 6.0
STRIDE_SEC = 3.0
PREDICT_SEC = 3.0

WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SR)
STRIDE_SAMPLES = int(STRIDE_SEC * TARGET_SR)
PREDICT_START_SEC = (WINDOW_SEC - PREDICT_SEC) / 2

WINDOW_SAMPLE = "window"
FULL_SONG_SAMPLE = "full_song"

_COLLATE_DIAG_PRINTED = False


def build_mel_transform(device: str = "cpu") -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        normalized=True,
    ).to(device)


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Load audio bytes (MP3/OGG/WAV) into a mono waveform tensor."""
    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    return waveform, TARGET_SR


def compute_onset_strength(waveform: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Compute onset strength aligned to mel frames and normalized to [0, 1]."""
    import librosa

    y = waveform.squeeze().cpu().numpy()
    onset = librosa.onset.onset_strength(y=y, sr=TARGET_SR, hop_length=HOP_LENGTH)
    onset = torch.from_numpy(onset).float()
    if onset.shape[0] < n_frames:
        onset = torch.nn.functional.pad(onset, (0, n_frames - onset.shape[0]))
    else:
        onset = onset[:n_frames]
    onset = onset / (onset.max() + 1e-7)
    return onset.unsqueeze(0)


def build_feature_tensor(
    waveform: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    mel = mel_transform(waveform)  # [1, N_MELS, T]
    onset = compute_onset_strength(waveform, mel.shape[-1])  # [1, T]
    return torch.cat([mel, onset.unsqueeze(0)], dim=1)  # [1, N_FEATURES, T]


def normalize_mel_batch(mel_batch: torch.Tensor) -> torch.Tensor:
    mel_log = torch.log(mel_batch + 1e-7)
    mel_ch = mel_log[:, :, :N_MELS, :]
    onset_ch = mel_log[:, :, N_MELS:, :]
    mel_ch = (mel_ch - mel_ch.mean()) / (mel_ch.std() + 1e-6)
    onset_ch = (onset_ch - onset_ch.mean()) / (onset_ch.std() + 1e-6)
    return torch.cat([mel_ch, onset_ch], dim=2)


def infer_beatmap_status(beatmap: dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of beatmap ranking status from dataset metadata."""
    for key in ("status", "beatmap_status", "approval_status"):
        value = beatmap.get(key)
        if value is not None:
            return str(value).strip().lower()

    for key in ("approved", "ranked"):
        value = beatmap.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip().lower()
        try:
            value = int(value)
        except Exception:
            return str(value).strip().lower()
        mapping = {
            -2: "graveyard",
            -1: "wip",
            0: "pending",
            1: "ranked",
            2: "approved",
            3: "qualified",
            4: "loved",
        }
        return mapping.get(value, str(value))

    return None


def is_ranked_status(status: Optional[str]) -> bool:
    if status is None:
        return False
    normalized = status.strip().lower()
    return normalized == "ranked"


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class AudioAugmenter:
    """Applies time stretch with inverse token mapping."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(
        self,
        waveform: torch.Tensor,
        sr: int,
        token_deltas_ms: Optional[list[float]] = None,
    ) -> tuple[torch.Tensor, list[float]]:
        if not self.enabled or random.random() > 0.5:
            return waveform, token_deltas_ms or []

        stretch_factor = random.uniform(0.9, 1.1)
        effects = [["tempo", str(stretch_factor)]]
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        except Exception:
            augmented = waveform
            stretch_factor = 1.0

        adjusted_deltas = token_deltas_ms or []
        if stretch_factor != 1.0 and adjusted_deltas:
            adjusted_deltas = [delta / stretch_factor for delta in adjusted_deltas]
        return augmented, adjusted_deltas


def _build_sample(
    *,
    mel: torch.Tensor,
    tokens: list[int],
    residuals: list[Residuals],
    difficulty_id: int,
    bpm: float,
    star_rating: float,
    beatmap_status: Optional[str],
    signal: Optional[torch.Tensor],
    sample_type: str,
    frame_rate: float,
    target_start_ms: float = 0.0,
) -> dict[str, Any]:
    return {
        "mel": mel,
        "tokens": tokens,
        "residuals": residuals,
        "difficulty_id": difficulty_id,
        "bpm": bpm,
        "star_rating": float(star_rating),
        "beatmap_status": beatmap_status,
        "signal": signal,
        "sample_type": sample_type,
        "frame_rate": float(frame_rate),
        "target_start_ms": float(target_start_ms),
    }


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------
def extract_windows(
    waveform: torch.Tensor,
    osu_content: str,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    difficulty_id: int,
    bpm: float,
    star_rating: float,
    beatmap_status: Optional[str],
    augmenter: Optional[AudioAugmenter] = None,
    frame_rate: float = DEFAULT_FRAME_RATE,
) -> list[dict[str, Any]]:
    """
    Extract overlapping 6 s windows with legacy token targets and a 3 s signal target.
    """
    num_samples = waveform.shape[-1]
    windows: list[dict[str, Any]] = []
    ms_per_beat = 60000.0 / max(bpm, 1e-6)

    if augmenter is not None:
        waveform, _ = augmenter(waveform, TARGET_SR)

    start = 0
    while start + WINDOW_SAMPLES <= num_samples:
        chunk = waveform[:, start:start + WINDOW_SAMPLES]
        mel = build_feature_tensor(chunk, mel_transform)

        window_start_ms = (start / TARGET_SR) * 1000.0
        predict_start_ms = window_start_ms + PREDICT_START_SEC * 1000.0
        predict_end_ms = predict_start_ms + PREDICT_SEC * 1000.0

        tok_obj = tokenize_beatmap(
            osu_content,
            ms_per_beat=ms_per_beat,
            window_start_ms=predict_start_ms,
            window_end_ms=predict_end_ms,
        )
        signal = encode_osu_content_to_signal(
            osu_content,
            waveform=chunk,
            sample_rate=TARGET_SR,
            frame_rate=frame_rate,
            window_start_ms=predict_start_ms,
            window_end_ms=predict_end_ms,
            duration_ms=PREDICT_SEC * 1000.0,
        )

        windows.append(_build_sample(
            mel=mel,
            tokens=tok_obj.tokens,
            residuals=tok_obj.residuals,
            difficulty_id=difficulty_id,
            bpm=bpm,
            star_rating=star_rating,
            beatmap_status=beatmap_status,
            signal=signal,
            sample_type=WINDOW_SAMPLE,
            frame_rate=frame_rate,
            target_start_ms=predict_start_ms,
        ))

        start += STRIDE_SAMPLES

    return windows


def extract_full_song_sample(
    waveform: torch.Tensor,
    osu_content: str,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    difficulty_id: int,
    bpm: float,
    star_rating: float,
    beatmap_status: Optional[str],
    frame_rate: float = DEFAULT_FRAME_RATE,
) -> dict[str, Any]:
    """Extract a full-song sample for AE / flow stages."""
    ms_per_beat = 60000.0 / max(bpm, 1e-6)
    mel = build_feature_tensor(waveform, mel_transform)
    tok_obj = tokenize_beatmap(osu_content, ms_per_beat=ms_per_beat)
    signal = encode_osu_content_to_signal(
        osu_content,
        waveform=waveform,
        sample_rate=TARGET_SR,
        frame_rate=frame_rate,
    )
    return _build_sample(
        mel=mel,
        tokens=tok_obj.tokens,
        residuals=tok_obj.residuals,
        difficulty_id=difficulty_id,
        bpm=bpm,
        star_rating=star_rating,
        beatmap_status=beatmap_status,
        signal=signal,
        sample_type=FULL_SONG_SAMPLE,
        frame_rate=frame_rate,
        target_start_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad token sequences, signals, and spectrograms."""
    mels = []
    all_tokens = []
    all_time_res = []
    all_x_res = []
    all_y_res = []
    signals = []
    signal_lengths = []

    max_mel_t = max(sample["mel"].shape[-1] for sample in batch)
    max_tok_len = max(len(sample["tokens"]) for sample in batch)
    have_signal = all(sample.get("signal") is not None for sample in batch)
    max_signal_t = max(sample["signal"].shape[-1] for sample in batch) if have_signal else 0

    for sample in batch:
        mel = sample["mel"]
        if mel.shape[-1] < max_mel_t:
            mel = torch.nn.functional.pad(mel, (0, max_mel_t - mel.shape[-1]))
        mels.append(mel)

        pad_len = max_tok_len - len(sample["tokens"])
        all_tokens.append(sample["tokens"] + [PAD] * pad_len)
        all_time_res.append([r.time_offset_ms for r in sample["residuals"]] + [0.0] * pad_len)
        all_x_res.append([r.x_offset_px for r in sample["residuals"]] + [0.0] * pad_len)
        all_y_res.append([r.y_offset_px for r in sample["residuals"]] + [0.0] * pad_len)

        if have_signal:
            signal = sample["signal"]
            signal_lengths.append(signal.shape[-1])
            if signal.shape[-1] < max_signal_t:
                signal = torch.nn.functional.pad(signal, (0, max_signal_t - signal.shape[-1]))
            signals.append(signal)

    mel_normed = normalize_mel_batch(torch.stack(mels, dim=0))

    global _COLLATE_DIAG_PRINTED
    if not _COLLATE_DIAG_PRINTED:
        _COLLATE_DIAG_PRINTED = True
        mel_ch = mel_normed[:, :, :N_MELS, :]
        onset_ch = mel_normed[:, :, N_MELS:, :]
        print(
            f"[collate diag] post-norm shape={mel_normed.shape} | "
            f"mel mean={mel_ch.mean():.3f} std={mel_ch.std():.3f}",
            flush=True,
        )
        print(
            f"[collate diag] onset mean={onset_ch.mean():.3f} std={onset_ch.std():.3f}",
            flush=True,
        )

    signal_tensor = None
    signal_mask = None
    if have_signal:
        signal_tensor = torch.stack(signals, dim=0)
        signal_mask = torch.zeros(signal_tensor.shape[0], signal_tensor.shape[-1], dtype=torch.bool)
        for idx, length in enumerate(signal_lengths):
            signal_mask[idx, :length] = True

    return {
        "mel": mel_normed,
        "tokens": torch.tensor(all_tokens, dtype=torch.long),
        "time_residuals": torch.tensor(all_time_res, dtype=torch.float32).clamp(-BEAT_QUANT / 2, BEAT_QUANT / 2),
        "x_residuals": torch.tensor(all_x_res, dtype=torch.float32).clamp(-8.0, 8.0),
        "y_residuals": torch.tensor(all_y_res, dtype=torch.float32).clamp(-6.0, 6.0),
        "difficulty_id": torch.tensor([sample["difficulty_id"] for sample in batch], dtype=torch.long),
        "bpm": torch.tensor([[sample["bpm"]] for sample in batch], dtype=torch.float32),
        "star_rating": torch.tensor([[sample.get("star_rating", 4.0)] for sample in batch], dtype=torch.float32),
        "signal": signal_tensor,
        "signal_mask": signal_mask,
        "sample_type": [sample.get("sample_type", WINDOW_SAMPLE) for sample in batch],
        "frame_rate": torch.tensor([[sample.get("frame_rate", DEFAULT_FRAME_RATE)] for sample in batch], dtype=torch.float32),
        "target_start_ms": torch.tensor([[sample.get("target_start_ms", 0.0)] for sample in batch], dtype=torch.float32),
        "beatmap_status": [sample.get("beatmap_status") for sample in batch],
        "meta": [sample.get("meta", {}) for sample in batch],
        "key": [sample.get("key") for sample in batch],
        "url": [sample.get("url") for sample in batch],
    }


# ---------------------------------------------------------------------------
# Streaming dataset from HuggingFace
# ---------------------------------------------------------------------------
class OsuStreamingDataset(IterableDataset):
    """
    Streams grouped beatmaps and emits window or full-song samples.
    """

    def __init__(
        self,
        split: str = "train",
        mel_transform=None,
        augmenter: Optional[AudioAugmenter] = None,
        curriculum_filter: Optional[str] = None,
        sample_type: str = WINDOW_SAMPLE,
        include_legacy_windows: bool = True,
        quality_filter: str = "all",
        signal_required: bool = False,
        frame_rate: float = DEFAULT_FRAME_RATE,
    ):
        self.split = split
        self.mel_transform = mel_transform
        self.augmenter = augmenter
        self.curriculum_filter = curriculum_filter
        self.sample_type = sample_type
        self.include_legacy_windows = include_legacy_windows
        self.quality_filter = quality_filter
        self.signal_required = signal_required
        self.frame_rate = frame_rate

    def _load_stream(self):
        from datasets import Audio, load_dataset

        ds = load_dataset(
            "project-riz/osu-beatmaps",
            "original",
            split="train",
            streaming=True,
        )
        try:
            ds = ds.cast_column("mp3", Audio(decode=False))
        except Exception:
            pass
        return ds

    def _passes_curriculum_filter(self, osu_content: str) -> bool:
        if self.curriculum_filter == "circles_only":
            if "2,0," in osu_content or ",8,0," in osu_content:
                return False
        return True

    def _passes_quality_filter(self, beatmap_status: Optional[str]) -> bool:
        if self.quality_filter == "all":
            return True
        if beatmap_status is None:
            raise RuntimeError(
                "ranked_only requested, but beatmap status metadata is unavailable in the dataset stream."
            )
        return is_ranked_status(beatmap_status)

    def __iter__(self):
        ds = self._load_stream()
        mel_tfm = self.mel_transform or build_mel_transform("cpu")

        for example in ds:
            try:
                json_data = example.get("json")
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)

                audio_bytes = example.get("mp3")
                if audio_bytes is None:
                    continue
                if isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
                    audio_bytes = audio_bytes["bytes"]

                waveform, _ = load_audio_from_bytes(audio_bytes)
                beatmaps = json_data.get("beatmaps", [])

                for beatmap in beatmaps:
                    osu_content = beatmap.get("content", "")
                    if not osu_content or int(beatmap.get("mode", 0)) != 0:
                        continue

                    if not self._passes_curriculum_filter(osu_content):
                        continue

                    beatmap_status = infer_beatmap_status(beatmap)
                    if not self._passes_quality_filter(beatmap_status):
                        continue

                    star_rating = float(beatmap.get("difficultyrating", 4.0))
                    difficulty_id = difficulty_to_bin(star_rating)
                    bpm = parse_osu_bpm(osu_content)

                    if self.sample_type == WINDOW_SAMPLE:
                        for sample in extract_windows(
                            waveform=waveform,
                            osu_content=osu_content,
                            mel_transform=mel_tfm,
                            difficulty_id=difficulty_id,
                            bpm=bpm,
                            star_rating=star_rating,
                            beatmap_status=beatmap_status,
                            augmenter=self.augmenter,
                            frame_rate=self.frame_rate,
                        ):
                            if len(sample["tokens"]) > 3:
                                yield sample
                    elif self.sample_type == FULL_SONG_SAMPLE:
                        sample = extract_full_song_sample(
                            waveform=waveform,
                            osu_content=osu_content,
                            mel_transform=mel_tfm,
                            difficulty_id=difficulty_id,
                            bpm=bpm,
                            star_rating=star_rating,
                            beatmap_status=beatmap_status,
                            frame_rate=self.frame_rate,
                        )
                        yield sample
                    else:
                        raise ValueError(f"Unsupported sample_type: {self.sample_type}")

            except RuntimeError:
                raise
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Cached WebDataset reader
# ---------------------------------------------------------------------------
class OsuCachedDataset(IterableDataset):
    """Load preprocessed shards from WebDataset .tar files."""

    def __init__(
        self,
        shard_pattern: str,
        shuffle: bool = True,
        shards: Optional[list[str]] = None,
        sample_type: str = WINDOW_SAMPLE,
        signal_required: bool = False,
        quality_filter: str = "all",
    ):
        self.shard_pattern = shard_pattern
        self.shuffle = shuffle
        self.shards = shards
        self.sample_type = sample_type
        self.signal_required = signal_required
        self.quality_filter = quality_filter

    def _passes_quality_filter(self, status: Optional[str]) -> bool:
        if self.quality_filter == "all":
            return True
        if status is None:
            raise RuntimeError(
                "ranked_only requested, but cached samples do not contain beatmap status metadata. "
                "Re-run preprocessing with status metadata available."
            )
        return is_ranked_status(status)

    def __iter__(self):
        import webdataset as wds

        shards = list(self.shards) if self.shards is not None else sorted(glob.glob(self.shard_pattern))
        if shards:
            shards = [path for path in shards if os.path.getsize(path) > 0]
        if not shards:
            print("[dataset] No non-empty shards found for pattern/shard list.", flush=True)
            return

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            shards = shards[worker_info.id::worker_info.num_workers]
            if not shards:
                print(f"[dataset] Worker {worker_info.id} has 0 shards after split.", flush=True)
                return

        if self.shuffle:
            random.shuffle(shards)

        dataset = wds.WebDataset(shards, shardshuffle=False, empty_check=False)
        if self.shuffle:
            dataset = dataset.shuffle(1000)

        for sample in dataset:
            try:
                meta = {}
                if "meta.json" in sample:
                    raw_meta = sample["meta.json"]
                    if isinstance(raw_meta, bytes):
                        meta = json.loads(raw_meta.decode("utf-8"))
                    else:
                        meta = json.loads(raw_meta)

                sample_type = meta.get("sample_type", WINDOW_SAMPLE)
                if sample_type != self.sample_type:
                    continue

                beatmap_status = meta.get("beatmap_status")
                if not self._passes_quality_filter(beatmap_status):
                    continue

                mel = torch.load(io.BytesIO(sample["mel.pt"]), weights_only=False)
                tokens_data = torch.load(io.BytesIO(sample["tokens.pt"]), weights_only=False)
                signal = None
                if "signal.pt" in sample:
                    signal = torch.load(io.BytesIO(sample["signal.pt"]), weights_only=False)
                elif self.signal_required:
                    raise RuntimeError(
                        "Signal-aware stage requested, but cached shards are missing signal.pt. "
                        "Re-run preprocessing for the continuous representation."
                    )

                yield {
                    "mel": mel,
                    "tokens": tokens_data["tokens"],
                    "residuals": tokens_data["residuals"],
                    "difficulty_id": int(tokens_data.get("difficulty_id", meta.get("difficulty_id", 2))),
                    "bpm": float(tokens_data.get("bpm", meta.get("bpm", 120.0))),
                    "star_rating": float(meta.get("star_rating", tokens_data.get("star_rating", 4.0))),
                    "beatmap_status": beatmap_status,
                    "signal": signal,
                    "sample_type": sample_type,
                    "frame_rate": float(meta.get("frame_rate", DEFAULT_FRAME_RATE)),
                    "target_start_ms": float(meta.get("target_start_ms", 0.0)),
                    "meta": meta,
                    "key": sample.get("__key__"),
                    "url": sample.get("__url__"),
                }
            except RuntimeError:
                raise
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def build_dataloader(
    dataset: IterableDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **kwargs,
    )
