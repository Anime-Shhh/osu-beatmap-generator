"""
Continuous beatmap signal representation utilities.

The signal layout is fixed to [9, T] at roughly 6 ms / frame:
    0 onset pulse
    1 cursor x (normalized 0..1)
    2 cursor y (normalized 0..1)
    3 slider progress
    4 sustain / hold indicator
    5 hitsound scalar
    6 local density hint
    7 combo-progress hint
    8 cursor velocity hint
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from .tokenizer import (
    PLAYFIELD_H,
    PLAYFIELD_W,
    parse_osu_bpm,
    parse_osu_hitobjects,
    parse_osu_metadata,
)


NUM_SIGNAL_CHANNELS = 9
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_FRAME_RATE = 1000.0 / 6.0
DEFAULT_CENTER_X = PLAYFIELD_W / 2.0
DEFAULT_CENTER_Y = PLAYFIELD_H / 2.0

CHANNEL_ONSET = 0
CHANNEL_CURSOR_X = 1
CHANNEL_CURSOR_Y = 2
CHANNEL_SLIDER_PROGRESS = 3
CHANNEL_SUSTAIN = 4
CHANNEL_HITSOUND = 5
CHANNEL_DENSITY = 6
CHANNEL_COMBO = 7
CHANNEL_VELOCITY = 8


def ms_to_frame(time_ms: float, frame_rate: float = DEFAULT_FRAME_RATE) -> int:
    return max(0, int(round(time_ms * frame_rate / 1000.0)))


def frame_to_ms(frame_idx: int, frame_rate: float = DEFAULT_FRAME_RATE) -> float:
    return frame_idx * 1000.0 / frame_rate


def _clamp01(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 0.0, 1.0)


def _normalize_x(x: float) -> float:
    return float(np.clip(x / PLAYFIELD_W, 0.0, 1.0))


def _normalize_y(y: float) -> float:
    return float(np.clip(y / PLAYFIELD_H, 0.0, 1.0))


def _denormalize_x(x: float) -> float:
    return np.clip(x, 0.0, 1.0) * PLAYFIELD_W


def _denormalize_y(y: float) -> float:
    return np.clip(y, 0.0, 1.0) * PLAYFIELD_H


def _estimate_slider_duration_ms(obj: dict, meta: dict, bpm: float) -> float:
    slider_multiplier = float(meta.get("slider_multiplier", 1.4) or 1.4)
    beat_length = 60000.0 / max(bpm, 1e-6)
    length = float(obj.get("length", 100.0) or 100.0)
    slides = max(int(obj.get("slides", 1) or 1), 1)
    span_duration = (length / max(slider_multiplier * 100.0, 1e-6)) * beat_length
    return max(span_duration * slides, beat_length / 4.0)


def _polyline_points_for_slider(obj: dict) -> list[tuple[float, float]]:
    points = [(float(obj["x"]), float(obj["y"]))]
    for point in obj.get("curve_points", []):
        points.append((float(point[0]), float(point[1])))
    if len(points) == 1:
        points.append(points[0])
    return points


def _sample_polyline(points: list[tuple[float, float]], progress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        xs = np.full_like(progress, points[0][0], dtype=np.float32)
        ys = np.full_like(progress, points[0][1], dtype=np.float32)
        return xs, ys

    pts = np.asarray(points, dtype=np.float32)
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1))
    total = float(seg_lengths.sum())
    if total <= 1e-6:
        xs = np.full_like(progress, pts[0, 0], dtype=np.float32)
        ys = np.full_like(progress, pts[0, 1], dtype=np.float32)
        return xs, ys

    seg_cdf = np.cumsum(seg_lengths) / total
    seg_starts = np.concatenate([[0.0], seg_cdf[:-1]])
    xs = np.zeros_like(progress, dtype=np.float32)
    ys = np.zeros_like(progress, dtype=np.float32)

    for idx, p in enumerate(progress):
        seg_idx = int(np.searchsorted(seg_cdf, p, side="right"))
        seg_idx = min(seg_idx, len(seg_lengths) - 1)
        seg_start = seg_starts[seg_idx]
        seg_end = seg_cdf[seg_idx]
        denom = max(seg_end - seg_start, 1e-6)
        local = (p - seg_start) / denom
        point = pts[seg_idx] * (1.0 - local) + pts[seg_idx + 1] * local
        xs[idx] = point[0]
        ys[idx] = point[1]
    return xs, ys


def _spinner_trajectory(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    angle = np.linspace(0.0, math.pi * 2.0, length, endpoint=False)
    radius_x = PLAYFIELD_W * 0.06
    radius_y = PLAYFIELD_H * 0.06
    xs = DEFAULT_CENTER_X + np.cos(angle) * radius_x
    ys = DEFAULT_CENTER_Y + np.sin(angle) * radius_y
    return xs.astype(np.float32), ys.astype(np.float32)


def _add_onset_pulse(signal: np.ndarray, frame_idx: int, amplitude: float = 1.0) -> None:
    for offset in range(-2, 3):
        idx = frame_idx + offset
        if 0 <= idx < signal.shape[1]:
            weight = math.exp(-(offset**2) / 2.0)
            signal[CHANNEL_ONSET, idx] = max(signal[CHANNEL_ONSET, idx], amplitude * weight)


def _build_signal_from_objects(
    objects: list[dict],
    duration_ms: float,
    bpm: float,
    meta: Optional[dict] = None,
    frame_rate: float = DEFAULT_FRAME_RATE,
    window_start_ms: float = 0.0,
) -> torch.Tensor:
    total_frames = max(1, ms_to_frame(duration_ms, frame_rate) + 1)
    signal = np.zeros((NUM_SIGNAL_CHANNELS, total_frames), dtype=np.float32)
    cursor_x = np.full(total_frames, np.nan, dtype=np.float32)
    cursor_y = np.full(total_frames, np.nan, dtype=np.float32)

    bpm = float(bpm or 120.0)
    meta = meta or {}

    prev_frame = 0
    prev_anchor = np.array([_normalize_x(DEFAULT_CENTER_X), _normalize_y(DEFAULT_CENTER_Y)], dtype=np.float32)
    event_frames: list[int] = []
    combo_count = 0

    for obj_idx, obj in enumerate(objects):
        start_ms = float(obj["time"]) - window_start_ms
        start_frame = min(total_frames - 1, ms_to_frame(start_ms, frame_rate))
        event_frames.append(start_frame)

        combo_count = 1 if obj_idx == 0 or obj.get("new_combo") else combo_count + 1
        next_obj_time = float(objects[obj_idx + 1]["time"]) - window_start_ms if obj_idx + 1 < len(objects) else duration_ms
        combo_fill_end = min(total_frames, max(start_frame + 1, ms_to_frame(next_obj_time, frame_rate)))
        signal[CHANNEL_COMBO, start_frame:combo_fill_end] = np.clip(combo_count / 20.0, 0.0, 1.0)

        start_anchor = np.array([
            _normalize_x(float(obj.get("x", DEFAULT_CENTER_X))),
            _normalize_y(float(obj.get("y", DEFAULT_CENTER_Y))),
        ], dtype=np.float32)

        if start_frame > prev_frame:
            interp = np.linspace(0.0, 1.0, start_frame - prev_frame + 1, dtype=np.float32)
            cursor_x[prev_frame:start_frame + 1] = prev_anchor[0] * (1.0 - interp) + start_anchor[0] * interp
            cursor_y[prev_frame:start_frame + 1] = prev_anchor[1] * (1.0 - interp) + start_anchor[1] * interp

        _add_onset_pulse(signal, start_frame)
        signal[CHANNEL_HITSOUND, start_frame] = np.clip(float(obj.get("hit_sound", 0)) / 16.0, 0.0, 1.0)

        end_frame = start_frame
        end_anchor = start_anchor.copy()

        if obj["type"] == "slider":
            duration_obj_ms = _estimate_slider_duration_ms(obj, meta, bpm)
            end_ms = min(duration_ms, start_ms + duration_obj_ms)
            end_frame = min(total_frames - 1, max(start_frame + 1, ms_to_frame(end_ms, frame_rate)))
            frame_count = max(1, end_frame - start_frame + 1)
            signal[CHANNEL_SUSTAIN, start_frame:end_frame + 1] = 1.0
            progress = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
            signal[CHANNEL_SLIDER_PROGRESS, start_frame:end_frame + 1] = progress

            raw_progress = np.linspace(0.0, float(max(int(obj.get("slides", 1) or 1), 1)), frame_count, dtype=np.float32)
            span_progress = raw_progress % 1.0
            reverse_mask = (raw_progress.astype(np.int32) % 2) == 1
            span_progress[reverse_mask] = 1.0 - span_progress[reverse_mask]

            xs, ys = _sample_polyline(_polyline_points_for_slider(obj), span_progress)
            cursor_x[start_frame:end_frame + 1] = _clamp01(xs / PLAYFIELD_W)
            cursor_y[start_frame:end_frame + 1] = _clamp01(ys / PLAYFIELD_H)
            end_anchor = np.array([cursor_x[end_frame], cursor_y[end_frame]], dtype=np.float32)
            signal[CHANNEL_HITSOUND, start_frame:end_frame + 1] = np.clip(float(obj.get("hit_sound", 0)) / 16.0, 0.0, 1.0)

        elif obj["type"] == "spinner":
            end_time = float(obj.get("end_time", obj["time"] + 1000.0)) - window_start_ms
            end_ms = min(duration_ms, max(start_ms, end_time))
            end_frame = min(total_frames - 1, max(start_frame + 1, ms_to_frame(end_ms, frame_rate)))
            frame_count = max(1, end_frame - start_frame + 1)
            signal[CHANNEL_SUSTAIN, start_frame:end_frame + 1] = 1.0
            xs, ys = _spinner_trajectory(frame_count)
            cursor_x[start_frame:end_frame + 1] = _clamp01(xs / PLAYFIELD_W)
            cursor_y[start_frame:end_frame + 1] = _clamp01(ys / PLAYFIELD_H)
            end_anchor = np.array([cursor_x[end_frame], cursor_y[end_frame]], dtype=np.float32)
            signal[CHANNEL_HITSOUND, start_frame:end_frame + 1] = np.clip(float(obj.get("hit_sound", 0)) / 16.0, 0.0, 1.0)

        else:
            cursor_x[start_frame] = start_anchor[0]
            cursor_y[start_frame] = start_anchor[1]

        prev_frame = end_frame
        prev_anchor = end_anchor

    first_valid = np.where(~np.isnan(cursor_x))[0]
    if len(first_valid) == 0:
        cursor_x[:] = _normalize_x(DEFAULT_CENTER_X)
        cursor_y[:] = _normalize_y(DEFAULT_CENTER_Y)
    else:
        first_idx = int(first_valid[0])
        cursor_x[:first_idx] = cursor_x[first_idx]
        cursor_y[:first_idx] = cursor_y[first_idx]
        for idx in range(first_idx + 1, total_frames):
            if np.isnan(cursor_x[idx]):
                cursor_x[idx] = cursor_x[idx - 1]
            if np.isnan(cursor_y[idx]):
                cursor_y[idx] = cursor_y[idx - 1]

    signal[CHANNEL_CURSOR_X] = _clamp01(cursor_x)
    signal[CHANNEL_CURSOR_Y] = _clamp01(cursor_y)

    if event_frames:
        impulse = np.zeros(total_frames, dtype=np.float32)
        impulse[np.clip(np.asarray(event_frames, dtype=np.int64), 0, total_frames - 1)] = 1.0
        density_window = max(1, int(round(frame_rate)))
        density = np.convolve(impulse, np.ones(density_window, dtype=np.float32), mode="same")
        signal[CHANNEL_DENSITY] = np.clip(density / 14.0, 0.0, 1.0)

    dx = np.diff(signal[CHANNEL_CURSOR_X], prepend=signal[CHANNEL_CURSOR_X][0])
    dy = np.diff(signal[CHANNEL_CURSOR_Y], prepend=signal[CHANNEL_CURSOR_Y][0])
    velocity = np.sqrt(dx**2 + dy**2) * frame_rate
    signal[CHANNEL_VELOCITY] = np.clip(velocity / 1.5, 0.0, 1.0)

    return torch.from_numpy(signal.astype(np.float32))


def encode_osu_content_to_signal(
    osu_content: str,
    *,
    waveform: Optional[torch.Tensor] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_rate: float = DEFAULT_FRAME_RATE,
    window_start_ms: float = 0.0,
    window_end_ms: Optional[float] = None,
    duration_ms: Optional[float] = None,
) -> torch.Tensor:
    """Encode raw .osu text into a dense [9, T] signal tensor."""
    all_objects = parse_osu_hitobjects(osu_content)
    bpm = parse_osu_bpm(osu_content)
    meta = parse_osu_metadata(osu_content)

    if duration_ms is None:
        if waveform is not None:
            duration_ms = waveform.shape[-1] * 1000.0 / max(sample_rate, 1)
        elif window_end_ms is not None:
            duration_ms = window_end_ms - window_start_ms
        elif all_objects:
            last_time = max(float(obj.get("end_time", obj["time"])) for obj in all_objects)
            duration_ms = last_time + 1000.0
        else:
            duration_ms = 1000.0

    if window_end_ms is None:
        window_end_ms = window_start_ms + duration_ms

    filtered_objects = []
    for obj in all_objects:
        obj_start = float(obj["time"])
        obj_end = float(obj.get("end_time", obj_start))
        if obj["type"] == "slider":
            obj_end = obj_start + _estimate_slider_duration_ms(obj, parse_osu_metadata(osu_content), bpm)
        if obj_end < window_start_ms or obj_start > window_end_ms:
            continue
        filtered_objects.append(obj)

    filtered_objects.sort(key=lambda item: item["time"])
    window_duration_ms = max(1.0, float(window_end_ms - window_start_ms))
    return _build_signal_from_objects(
        filtered_objects,
        duration_ms=window_duration_ms,
        bpm=bpm,
        meta=meta,
        frame_rate=frame_rate,
        window_start_ms=window_start_ms,
    )


def encode_beatmap_to_signal(
    osu_path: str | Path,
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_rate: float = DEFAULT_FRAME_RATE,
) -> torch.Tensor:
    """Public file-based wrapper for beatmap signal encoding."""
    with open(osu_path, "r", encoding="utf-8") as f:
        osu_content = f.read()
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return encode_osu_content_to_signal(
        osu_content,
        waveform=waveform,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )


def _snap_time_ms(time_ms: float, bpm: float) -> float:
    beat = 60000.0 / max(bpm, 1e-6)
    subdivision = beat / 16.0
    return round(time_ms / subdivision) * subdivision


def _fit_bezier_control_points(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    if len(xs) <= 2:
        if len(xs) == 0:
            return []
        return [(float(xs[-1]), float(ys[-1]))]

    sample_indices = sorted({0, len(xs) // 3, (2 * len(xs)) // 3, len(xs) - 1})
    control_points: list[tuple[float, float]] = []
    last_added: Optional[tuple[int, int]] = None
    for idx in sample_indices[1:]:
        point = (int(round(xs[idx])), int(round(ys[idx])))
        if last_added != point:
            control_points.append((float(point[0]), float(point[1])))
            last_added = point
    return control_points[:4]


def _apply_anti_spam_filter(objects: list[dict], star_rating: float, window_sec: float = 1.0) -> list[dict]:
    if not objects:
        return []
    max_ops = float(np.clip(3.0 + 1.5 * star_rating, 4.0, 14.0))
    window_ms = window_sec * 1000.0
    kept: list[dict] = []
    recent: list[float] = []
    for obj in sorted(objects, key=lambda item: item["time"]):
        recent = [t for t in recent if obj["time"] - t <= window_ms]
        if len(recent) >= max_ops:
            continue
        kept.append(obj)
        recent.append(obj["time"])
    return kept


def decode_signal_to_osu(
    signal: torch.Tensor | np.ndarray,
    bpm: float,
    offset_ms: float,
    *,
    star_rating: float = 4.0,
    frame_rate: float = DEFAULT_FRAME_RATE,
) -> list[dict]:
    """
    Decode a signal back into cleaned hit-object dicts for .osu serialization.
    """
    if isinstance(signal, torch.Tensor):
        arr = signal.detach().cpu().float().numpy()
    else:
        arr = np.asarray(signal, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[0] < NUM_SIGNAL_CHANNELS:
        raise ValueError(f"Expected signal shape [9, T], got {tuple(arr.shape)}")

    onset = arr[CHANNEL_ONSET]
    cursor_x = _denormalize_x(arr[CHANNEL_CURSOR_X])
    cursor_y = _denormalize_y(arr[CHANNEL_CURSOR_Y])
    slider_progress = arr[CHANNEL_SLIDER_PROGRESS]
    sustain = arr[CHANNEL_SUSTAIN]

    threshold = max(0.2, float(onset.mean() + 0.5 * onset.std()))
    peaks: list[int] = []
    for idx in range(1, len(onset) - 1):
        if onset[idx] >= threshold and onset[idx] >= onset[idx - 1] and onset[idx] >= onset[idx + 1]:
            if peaks and idx - peaks[-1] <= 1 and onset[idx] <= onset[peaks[-1]]:
                continue
            if peaks and idx - peaks[-1] <= 1:
                peaks[-1] = idx
            else:
                peaks.append(idx)

    objects: list[dict] = []
    for peak in peaks:
        start_time = _snap_time_ms(offset_ms + frame_to_ms(peak, frame_rate), bpm)
        x = float(np.clip(cursor_x[peak], 0.0, PLAYFIELD_W))
        y = float(np.clip(cursor_y[peak], 0.0, PLAYFIELD_H))

        seg_start = peak
        while seg_start > 0 and sustain[seg_start - 1] > 0.35:
            seg_start -= 1
        seg_end = peak
        while seg_end + 1 < len(sustain) and sustain[seg_end + 1] > 0.35:
            seg_end += 1

        duration_ms = frame_to_ms(max(0, seg_end - peak), frame_rate)
        slider_delta = float(slider_progress[seg_end] - slider_progress[seg_start]) if seg_end > seg_start else 0.0
        spinner_like = duration_ms >= 450.0 and np.std(cursor_x[seg_start:seg_end + 1]) < PLAYFIELD_W * 0.04

        if seg_end > peak + 1 and slider_delta > 0.25:
            xs = np.clip(cursor_x[seg_start:seg_end + 1], 0.0, PLAYFIELD_W)
            ys = np.clip(cursor_y[seg_start:seg_end + 1], 0.0, PLAYFIELD_H)
            end_time = _snap_time_ms(offset_ms + frame_to_ms(seg_end, frame_rate), bpm)
            control_points = _fit_bezier_control_points(xs, ys)
            path = np.stack([xs, ys], axis=1)
            diffs = np.diff(path, axis=0)
            length = float(np.sqrt((diffs**2).sum(axis=1)).sum()) if len(path) > 1 else 100.0
            objects.append({
                "time": start_time,
                "type": "slider",
                "x": x,
                "y": y,
                "curve_type": "B",
                "curve_points": control_points,
                "slides": 1,
                "length": max(length, 50.0),
                "end_time": end_time,
            })
        elif seg_end > peak + 1 and spinner_like:
            end_time = _snap_time_ms(offset_ms + frame_to_ms(seg_end, frame_rate), bpm)
            objects.append({
                "time": start_time,
                "type": "spinner",
                "x": DEFAULT_CENTER_X,
                "y": DEFAULT_CENTER_Y,
                "end_time": max(end_time, start_time + (60000.0 / max(bpm, 1e-6)) / 4.0),
            })
        else:
            objects.append({"time": start_time, "type": "circle", "x": x, "y": y})

    cleaned = []
    last_time = -float("inf")
    min_sep = (60000.0 / max(bpm, 1e-6)) / 16.0 / 2.0
    for obj in sorted(objects, key=lambda item: item["time"]):
        if obj["time"] - last_time < min_sep:
            continue
        cleaned.append(obj)
        last_time = obj["time"]

    return _apply_anti_spam_filter(cleaned, star_rating=star_rating)
