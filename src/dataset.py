"""
Dataset pipeline for osu! beatmap generation.

Streams from HuggingFace project-riz/osu-beatmaps (original variant),
explodes grouped beatmaps into individual (audio, beatmap) pairs,
converts audio to mel spectrograms and beatmaps to token sequences,
and supports sliding-window chunking with augmentation.
"""

import glob
import io
import json
import math
import os
import random
from typing import Optional

import torch
import torchaudio
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from .tokenizer import (
    tokenize_beatmap, Residuals, PAD, TOTAL_VOCAB,
    difficulty_to_bin, parse_osu_bpm, BEAT_QUANT,
)


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 441     # ~20ms at 22050 Hz
N_MELS = 128
N_FEATURES = N_MELS + 1  # mel + onset strength channel

WINDOW_SEC = 6.0
STRIDE_SEC = 3.0
PREDICT_SEC = 3.0    # predict middle 3s only

WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SR)      # 132300
STRIDE_SAMPLES = int(STRIDE_SEC * TARGET_SR)       # 66150
PREDICT_START_SEC = (WINDOW_SEC - PREDICT_SEC) / 2  # 1.5s


def build_mel_transform(device: str = "cpu") -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        normalized=True,
    ).to(device)


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Load audio bytes (MP3/OGG/WAV) into a waveform tensor."""
    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
    return waveform, TARGET_SR


def compute_onset_strength(waveform: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Compute onset strength aligned to mel frames, normalized to [0, 1]."""
    import librosa
    y = waveform.squeeze().numpy()
    onset = librosa.onset.onset_strength(y=y, sr=TARGET_SR, hop_length=HOP_LENGTH)
    onset = torch.from_numpy(onset).float()
    if onset.shape[0] < n_frames:
        onset = torch.nn.functional.pad(onset, (0, n_frames - onset.shape[0]))
    else:
        onset = onset[:n_frames]
    onset = onset / (onset.max() + 1e-7)
    return onset.unsqueeze(0)  # [1, T]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class AudioAugmenter:
    """Applies pitch shift and time stretch with inverse token mapping."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(
        self, waveform: torch.Tensor, sr: int, token_deltas_ms: Optional[list[float]] = None
    ) -> tuple[torch.Tensor, list[float]]:
        if not self.enabled or random.random() > 0.5:
            return waveform, token_deltas_ms or []

        stretch_factor = random.uniform(0.9, 1.1)
        effects = [["tempo", str(stretch_factor)]]
        try:
            augmented, new_sr = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sr, effects
            )
        except Exception:
            augmented = waveform
            stretch_factor = 1.0

        adjusted_deltas = token_deltas_ms or []
        if stretch_factor != 1.0 and adjusted_deltas:
            adjusted_deltas = [d / stretch_factor for d in adjusted_deltas]

        return augmented, adjusted_deltas


# ---------------------------------------------------------------------------
# Sliding window extraction
# ---------------------------------------------------------------------------
def extract_windows(
    waveform: torch.Tensor,
    osu_content: str,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    difficulty_id: int,
    bpm: float,
    augmenter: Optional[AudioAugmenter] = None,
) -> list[dict]:
    """
    Extract overlapping windows from a full song.

    Returns list of dicts:
        mel: [1, N_FEATURES, time_frames]  (mel + onset strength)
        tokens: list[int]
        residuals: list[Residuals]
        difficulty_id: int  (0..4)
        bpm: float
    """
    num_samples = waveform.shape[-1]
    windows = []
    ms_per_beat = 60000.0 / bpm

    start = 0
    while start + WINDOW_SAMPLES <= num_samples:
        chunk = waveform[:, start : start + WINDOW_SAMPLES]
        mel = mel_transform(chunk)  # [1, N_MELS, T]

        onset = compute_onset_strength(chunk, mel.shape[-1])  # [1, T]
        mel = torch.cat([mel, onset.unsqueeze(0)], dim=1)     # [1, N_FEATURES, T]

        window_start_ms = (start / TARGET_SR) * 1000.0
        predict_start_ms = window_start_ms + PREDICT_START_SEC * 1000.0
        predict_end_ms = predict_start_ms + PREDICT_SEC * 1000.0

        tok_obj = tokenize_beatmap(
            osu_content,
            ms_per_beat=ms_per_beat,
            window_start_ms=predict_start_ms,
            window_end_ms=predict_end_ms,
        )

        windows.append({
            "mel": mel,
            "tokens": tok_obj.tokens,
            "residuals": tok_obj.residuals,
            "difficulty_id": difficulty_id,
            "bpm": bpm,
        })

        start += STRIDE_SAMPLES

    return windows


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict:
    """Pad token sequences and stack mel spectrograms."""
    mels = []
    all_tokens = []
    all_time_res = []
    all_x_res = []
    all_y_res = []

    max_mel_t = max(b["mel"].shape[-1] for b in batch)
    max_tok_len = max(len(b["tokens"]) for b in batch)

    for b in batch:
        mel = b["mel"]
        pad_t = max_mel_t - mel.shape[-1]
        if pad_t > 0:
            mel = torch.nn.functional.pad(mel, (0, pad_t))
        mels.append(mel)

        tokens = b["tokens"]
        residuals = b["residuals"]
        pad_len = max_tok_len - len(tokens)

        all_tokens.append(tokens + [PAD] * pad_len)
        all_time_res.append([r.time_offset_ms for r in residuals] + [0.0] * pad_len)
        all_x_res.append([r.x_offset_px for r in residuals] + [0.0] * pad_len)
        all_y_res.append([r.y_offset_px for r in residuals] + [0.0] * pad_len)

    return {
        "mel": torch.log(torch.stack(mels, dim=0) + 1e-7),
        "tokens": torch.tensor(all_tokens, dtype=torch.long),
        "time_residuals": torch.tensor(all_time_res, dtype=torch.float32).clamp(-BEAT_QUANT / 2, BEAT_QUANT / 2),
        "x_residuals": torch.tensor(all_x_res, dtype=torch.float32).clamp(-8.0, 8.0),
        "y_residuals": torch.tensor(all_y_res, dtype=torch.float32).clamp(-6.0, 6.0),
        "difficulty_id": torch.tensor([b["difficulty_id"] for b in batch], dtype=torch.long),
        "bpm": torch.tensor([[b["bpm"]] for b in batch], dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Streaming dataset from HuggingFace
# ---------------------------------------------------------------------------
class OsuStreamingDataset(IterableDataset):
    """
    Streams from project-riz/osu-beatmaps original variant,
    explodes grouped beatmaps, extracts windowed samples.
    """

    def __init__(
        self,
        split: str = "train",
        mel_transform=None,
        augmenter: Optional[AudioAugmenter] = None,
        max_objects_per_sec: float = 20.0,
        min_objects_per_sec: float = 0.1,
        curriculum_filter: Optional[str] = None,
    ):
        self.split = split
        self.mel_transform = mel_transform
        self.augmenter = augmenter
        self.max_obj_per_sec = max_objects_per_sec
        self.min_obj_per_sec = min_objects_per_sec
        self.curriculum_filter = curriculum_filter

    def _load_stream(self):
        from datasets import load_dataset, Audio
        ds = load_dataset(
            "project-riz/osu-beatmaps", "original",
            split="train", streaming=True,
        )
        try:
            ds = ds.cast_column("mp3", Audio(decode=False))
        except Exception:
            pass
        return ds

    def _passes_filter(self, osu_content: str) -> bool:
        """Apply curriculum and quality filters."""
        if self.curriculum_filter == "circles_only":
            if "2,0," in osu_content or ",8,0," in osu_content:
                has_slider = any(
                    line.strip().split(",")[3]
                    for line in osu_content.split("\n")
                    if "," in line and not line.startswith("[")
                )
                return False
        return True

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

                audio_length = json_data.get("audio_length", 0)
                beatmaps = json_data.get("beatmaps", [])

                waveform, sr = load_audio_from_bytes(audio_bytes)

                for bm in beatmaps:
                    osu_content = bm.get("content", "")
                    if not osu_content:
                        continue

                    mode = bm.get("mode", 0)
                    if mode != 0:
                        continue

                    star_rating = float(bm.get("difficultyrating", 4.0))
                    diff_id = difficulty_to_bin(star_rating)
                    bpm = parse_osu_bpm(osu_content)

                    if not self._passes_filter(osu_content):
                        continue

                    windows = extract_windows(
                        waveform, osu_content,
                        mel_tfm, diff_id, bpm, self.augmenter,
                    )

                    for w in windows:
                        if len(w["tokens"]) > 3:
                            yield w

            except Exception:
                continue


# ---------------------------------------------------------------------------
# WebDataset-based cached dataset
# ---------------------------------------------------------------------------
class OsuCachedDataset(IterableDataset):
    """Load pre-processed shards from WebDataset .tar files."""

    def __init__(
        self,
        shard_pattern: str,
        shuffle: bool = True,
        shards: Optional[list[str]] = None,
    ):
        self.shard_pattern = shard_pattern
        self.shuffle = shuffle
        self.shards = shards

    def __iter__(self):
        import webdataset as wds

        shards = list(self.shards) if self.shards is not None else sorted(
            glob.glob(self.shard_pattern)
        )
        if shards:
            shards = [s for s in shards if os.path.getsize(s) > 0]
        if not shards:
            print("[dataset] No non-empty shards found for pattern/shard list.", flush=True)
            return

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            shards = shards[worker_info.id::worker_info.num_workers]
            if not shards:
                print(
                    f"[dataset] Worker {worker_info.id} has 0 shards after split.",
                    flush=True,
                )
                return

        if self.shuffle:
            random.shuffle(shards)

        dataset = wds.WebDataset(shards, shardshuffle=False, empty_check=False)
        if self.shuffle:
            dataset = dataset.shuffle(1000)

        for sample in dataset:
            try:
                mel = torch.load(io.BytesIO(sample["mel.pt"]), weights_only=False)
                tokens_data = torch.load(io.BytesIO(sample["tokens.pt"]), weights_only=False)
                tokens = tokens_data["tokens"]
                residuals = tokens_data["residuals"]
                difficulty_id = tokens_data.get("difficulty_id", 2)
                bpm = tokens_data.get("bpm", 120.0)
                yield {
                    "mel": mel,
                    "tokens": tokens,
                    "residuals": residuals,
                    "difficulty_id": int(difficulty_id),
                    "bpm": float(bpm),
                    "key": sample.get("__key__"),
                    "url": sample.get("__url__"),
                }
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
