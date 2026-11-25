"""Onset matching shared by training-time validation and the evaluation harness."""
from __future__ import annotations

import numpy as np


def match_onsets(pred: np.ndarray, ref: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Maximum-cardinality one-to-one ordered matching of sorted 1-D times within ``tol``.

    Greedy two-pointer is optimal for intervals on a line: always match the
    earliest still-matchable pair.
    """
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    pairs, i, j = [], 0, 0
    while i < len(pred) and j < len(ref):
        d = pred[i] - ref[j]
        if abs(d) <= tol:
            pairs.append((i, j))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return pairs


def onset_scores(pred: np.ndarray, ref: np.ndarray, tol: float) -> dict[str, float]:
    m = len(match_onsets(np.sort(pred), np.sort(ref), tol))
    p, r = len(pred), len(ref)
    return {"matched": m, "pred": p, "ref": r,
            "precision": m / p if p else 0.0, "recall": m / r if r else 0.0,
            "f1": 2 * m / (p + r) if p + r else 1.0}


def short_pair_recall(pred: np.ndarray, ref: np.ndarray, ref_beats: np.ndarray, tol: float,
                      max_gap_beats: float = 0.25 + 1e-3) -> tuple[int, int]:
    """Reference consecutive onset pairs with a gap <= 1/4 beat; recovered if both are matched."""
    order = np.argsort(ref)
    ref, ref_beats = np.asarray(ref)[order], np.asarray(ref_beats)[order]
    pairs = match_onsets(np.sort(pred), ref, tol)
    matched = np.zeros(len(ref), bool)
    for _, j in pairs:
        matched[j] = True
    gaps = np.diff(ref_beats)
    short = np.nonzero((gaps > 0) & (gaps <= max_gap_beats))[0]
    return int(np.sum(matched[short] & matched[short + 1])), int(len(short))
