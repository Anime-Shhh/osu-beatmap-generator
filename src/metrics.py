"""
Evaluation metrics for osu! beatmap generation quality.

- Token edit distance
- Timing MAE (ms)
- Hit F1
- Slider IoU
- Signal-based object metrics
- Cursor smoothness
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .representation import decode_signal_to_osu
from .tokenizer import Residuals, detokenize_to_hitobjects


def token_edit_distance(pred_tokens: list[int], true_tokens: list[int]) -> float:
    n, m = len(pred_tokens), len(true_tokens)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if pred_tokens[i - 1] == true_tokens[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, m)


def objects_timing_mae(pred_objs: list[dict], true_objs: list[dict]) -> float:
    if not pred_objs or not true_objs:
        return float("inf")

    pred_times = sorted(obj["time"] for obj in pred_objs)
    true_times = sorted(obj["time"] for obj in true_objs)

    errors = []
    used = set()
    for pred_time in pred_times:
        best_err = float("inf")
        best_idx = -1
        for idx, true_time in enumerate(true_times):
            if idx in used:
                continue
            err = abs(pred_time - true_time)
            if err < best_err:
                best_err = err
                best_idx = idx
        if best_idx >= 0:
            used.add(best_idx)
            errors.append(best_err)
    return float(np.mean(errors)) if errors else float("inf")


def objects_hit_f1(
    pred_objs: list[dict],
    true_objs: list[dict],
    *,
    time_tolerance_ms: float = 50.0,
    pos_tolerance_px: float = 50.0,
) -> dict[str, float]:
    if not pred_objs and not true_objs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_objs or not true_objs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = 0
    matched_true = set()
    for pred in pred_objs:
        best_dist = float("inf")
        best_idx = -1
        for idx, true in enumerate(true_objs):
            if idx in matched_true:
                continue
            time_diff = abs(pred["time"] - true["time"])
            if time_diff > time_tolerance_ms:
                continue
            pos_diff = ((pred["x"] - true["x"]) ** 2 + (pred["y"] - true["y"]) ** 2) ** 0.5
            if pos_diff > pos_tolerance_px:
                continue
            combined = time_diff + pos_diff
            if combined < best_dist:
                best_dist = combined
                best_idx = idx
        if best_idx >= 0:
            tp += 1
            matched_true.add(best_idx)

    precision = tp / len(pred_objs) if pred_objs else 0.0
    recall = tp / len(true_objs) if true_objs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def timing_mae(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    ms_per_beat: float = 500.0,
) -> float:
    pred_objs = detokenize_to_hitobjects(pred_tokens, pred_residuals, ms_per_beat=ms_per_beat)
    true_objs = detokenize_to_hitobjects(true_tokens, true_residuals, ms_per_beat=ms_per_beat)
    return objects_timing_mae(pred_objs, true_objs)


def hit_f1(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    time_tolerance_ms: float = 50.0,
    pos_tolerance_px: float = 50.0,
    ms_per_beat: float = 500.0,
) -> dict[str, float]:
    pred_objs = detokenize_to_hitobjects(pred_tokens, pred_residuals, ms_per_beat=ms_per_beat)
    true_objs = detokenize_to_hitobjects(true_tokens, true_residuals, ms_per_beat=ms_per_beat)
    return objects_hit_f1(
        pred_objs,
        true_objs,
        time_tolerance_ms=time_tolerance_ms,
        pos_tolerance_px=pos_tolerance_px,
    )


def slider_iou(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    buffer_px: float = 10.0,
    ms_per_beat: float = 500.0,
) -> float:
    try:
        from shapely.geometry import LineString
    except ImportError:
        return 0.0

    pred_objs = detokenize_to_hitobjects(pred_tokens, pred_residuals, ms_per_beat=ms_per_beat)
    true_objs = detokenize_to_hitobjects(true_tokens, true_residuals, ms_per_beat=ms_per_beat)

    pred_sliders = [obj for obj in pred_objs if obj["type"] == "slider"]
    true_sliders = [obj for obj in true_objs if obj["type"] == "slider"]
    if not pred_sliders or not true_sliders:
        return 0.0

    def _slider_to_line(obj: dict) -> Optional[LineString]:
        points = [(obj["x"], obj["y"])] + list(obj.get("curve_points", []))
        if len(points) < 2:
            return None
        return LineString(points)

    ious = []
    used = set()
    for pred_slider in pred_sliders:
        pred_line = _slider_to_line(pred_slider)
        if pred_line is None:
            continue
        pred_buf = pred_line.buffer(buffer_px)

        best_iou = 0.0
        best_idx = -1
        for idx, true_slider in enumerate(true_sliders):
            if idx in used or abs(pred_slider["time"] - true_slider["time"]) > 50:
                continue
            true_line = _slider_to_line(true_slider)
            if true_line is None:
                continue
            true_buf = true_line.buffer(buffer_px)
            intersection = pred_buf.intersection(true_buf).area
            union = pred_buf.union(true_buf).area
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            used.add(best_idx)
            ious.append(best_iou)
    return float(np.mean(ious)) if ious else 0.0


def compute_all_metrics(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    ms_per_beat: float = 500.0,
) -> dict[str, float]:
    results = {
        "edit_distance": token_edit_distance(pred_tokens, true_tokens),
        "timing_mae_ms": timing_mae(
            pred_tokens, true_tokens, pred_residuals, true_residuals, ms_per_beat=ms_per_beat
        ),
        "slider_iou": slider_iou(
            pred_tokens, true_tokens, pred_residuals, true_residuals, ms_per_beat=ms_per_beat
        ),
    }
    f1 = hit_f1(
        pred_tokens, true_tokens, pred_residuals, true_residuals, ms_per_beat=ms_per_beat
    )
    results.update({f"hit_{key}": value for key, value in f1.items()})
    return results


def signal_cursor_smoothness(signal: torch.Tensor | np.ndarray) -> float:
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().float().cpu().numpy()
    cursor = np.stack([signal[1], signal[2]], axis=0)
    velocity = np.diff(cursor, axis=1)
    acceleration = np.diff(velocity, axis=1)
    jerk = np.diff(acceleration, axis=1)
    if jerk.size == 0:
        return 0.0
    return float(np.mean(np.sqrt((jerk**2).sum(axis=0))))


def compute_signal_metrics(
    pred_signal: torch.Tensor | np.ndarray,
    true_signal: torch.Tensor | np.ndarray,
    *,
    bpm: float,
    offset_ms: float = 0.0,
    star_rating: float = 4.0,
) -> dict[str, float]:
    pred_objs = decode_signal_to_osu(pred_signal, bpm=bpm, offset_ms=offset_ms, star_rating=star_rating)
    true_objs = decode_signal_to_osu(true_signal, bpm=bpm, offset_ms=offset_ms, star_rating=star_rating)
    f1 = objects_hit_f1(pred_objs, true_objs)
    return {
        "timing_mae_ms": objects_timing_mae(pred_objs, true_objs),
        "hit_precision": f1["precision"],
        "hit_recall": f1["recall"],
        "hit_f1": f1["f1"],
        "cursor_smoothness": signal_cursor_smoothness(pred_signal),
    }
