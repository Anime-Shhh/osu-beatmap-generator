"""
Evaluation metrics for osu! beatmap generation quality.

- Token edit distance
- Timing MAE (ms)
- Hit F1 (within tolerance window)
- Slider IoU (curve overlap)
"""

import numpy as np
from typing import Optional

from .tokenizer import (
    detokenize_to_hitobjects, Residuals,
    TIME_OFFSET, NUM_BEAT_BINS, POS_OFFSET, NUM_POS_BINS,
    bin_to_beat, bin_to_xy,
)


def token_edit_distance(pred_tokens: list[int], true_tokens: list[int]) -> float:
    """Levenshtein edit distance between two token sequences, normalized by max length."""
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


def timing_mae(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    ms_per_beat: float = 500.0,
) -> float:
    """
    Mean absolute error of hit object timing (ms).
    Reconstructs absolute times from token sequences and compares.
    """
    pred_objs = detokenize_to_hitobjects(pred_tokens, pred_residuals, ms_per_beat=ms_per_beat)
    true_objs = detokenize_to_hitobjects(true_tokens, true_residuals, ms_per_beat=ms_per_beat)

    if not pred_objs or not true_objs:
        return float("inf")

    pred_times = sorted([o["time"] for o in pred_objs])
    true_times = sorted([o["time"] for o in true_objs])

    # Match each predicted time to nearest true time (greedy)
    errors = []
    used = set()
    for pt in pred_times:
        best_err = float("inf")
        best_idx = -1
        for i, tt in enumerate(true_times):
            if i in used:
                continue
            err = abs(pt - tt)
            if err < best_err:
                best_err = err
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            errors.append(best_err)

    return np.mean(errors) if errors else float("inf")


def hit_f1(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    time_tolerance_ms: float = 20.0,
    pos_tolerance_px: float = 50.0,
    ms_per_beat: float = 500.0,
) -> dict[str, float]:
    """
    F1 score: a predicted hit is a true positive if it matches a ground truth
    hit within time_tolerance_ms AND pos_tolerance_px.

    Returns dict with precision, recall, f1.
    """
    pred_objs = detokenize_to_hitobjects(pred_tokens, pred_residuals, ms_per_beat=ms_per_beat)
    true_objs = detokenize_to_hitobjects(true_tokens, true_residuals, ms_per_beat=ms_per_beat)

    if not pred_objs and not true_objs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_objs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not true_objs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = 0
    matched_true = set()

    for po in pred_objs:
        best_dist = float("inf")
        best_idx = -1
        for i, to in enumerate(true_objs):
            if i in matched_true:
                continue
            time_diff = abs(po["time"] - to["time"])
            if time_diff > time_tolerance_ms:
                continue
            pos_diff = ((po["x"] - to["x"]) ** 2 + (po["y"] - to["y"]) ** 2) ** 0.5
            if pos_diff > pos_tolerance_px:
                continue
            combined = time_diff + pos_diff
            if combined < best_dist:
                best_dist = combined
                best_idx = i
        if best_idx >= 0:
            tp += 1
            matched_true.add(best_idx)

    precision = tp / len(pred_objs) if pred_objs else 0.0
    recall = tp / len(true_objs) if true_objs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def slider_iou(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    buffer_px: float = 10.0,
    ms_per_beat: float = 500.0,
) -> float:
    """
    Compute IoU of slider curves using buffered line geometry.
    Requires shapely. Returns mean IoU over matched slider pairs.
    """
    try:
        from shapely.geometry import LineString
    except ImportError:
        return 0.0

    pred_objs = detokenize_to_hitobjects(pred_tokens, pred_residuals, ms_per_beat=ms_per_beat)
    true_objs = detokenize_to_hitobjects(true_tokens, true_residuals, ms_per_beat=ms_per_beat)

    pred_sliders = [o for o in pred_objs if o["type"] == "slider"]
    true_sliders = [o for o in true_objs if o["type"] == "slider"]

    if not pred_sliders or not true_sliders:
        return 0.0

    def _slider_to_line(obj: dict) -> Optional[LineString]:
        points = [(obj["x"], obj["y"])]
        for cp in obj.get("curve_points", []):
            points.append(cp)
        if len(points) < 2:
            return None
        return LineString(points)

    ious = []
    used = set()
    for ps in pred_sliders:
        pred_line = _slider_to_line(ps)
        if pred_line is None:
            continue
        pred_buf = pred_line.buffer(buffer_px)

        best_iou = 0.0
        best_idx = -1
        for i, ts in enumerate(true_sliders):
            if i in used:
                continue
            if abs(ps["time"] - ts["time"]) > 50:
                continue
            true_line = _slider_to_line(ts)
            if true_line is None:
                continue
            true_buf = true_line.buffer(buffer_px)
            intersection = pred_buf.intersection(true_buf).area
            union = pred_buf.union(true_buf).area
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0:
            used.add(best_idx)
            ious.append(best_iou)

    return np.mean(ious) if ious else 0.0


def compute_all_metrics(
    pred_tokens: list[int],
    true_tokens: list[int],
    pred_residuals: Optional[list[Residuals]] = None,
    true_residuals: Optional[list[Residuals]] = None,
    ms_per_beat: float = 500.0,
) -> dict[str, float]:
    """Compute all metrics and return as a flat dict."""
    results = {}
    results["edit_distance"] = token_edit_distance(pred_tokens, true_tokens)
    results["timing_mae_ms"] = timing_mae(
        pred_tokens, true_tokens, pred_residuals, true_residuals, ms_per_beat=ms_per_beat,
    )
    f1_results = hit_f1(
        pred_tokens, true_tokens, pred_residuals, true_residuals, ms_per_beat=ms_per_beat,
    )
    results.update({f"hit_{k}": v for k, v in f1_results.items()})
    results["slider_iou"] = slider_iou(
        pred_tokens, true_tokens, pred_residuals, true_residuals, ms_per_beat=ms_per_beat,
    )
    return results
