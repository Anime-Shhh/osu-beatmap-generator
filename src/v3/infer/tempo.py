"""Beat activations -> osu! red lines (BPM, offset, meter), plus drift evaluation.

Constant-tempo first: tempo candidates (tempo head, activation autocorrelation,
octave variants) are scored by fitting a grid to activation peaks with iterated
weighted least squares. If residuals are too large, the peaks are segmented
greedily into piecewise-constant sections (rubato / live recordings / tempo
changes), each becoming a red line. BPMs snap to the roundest value whose
accumulated drift stays within tolerance; offsets land on measure starts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.v3.data.audio import FRAME_MS

from src.v3.models.timing import TEMPO_BINS, bin_bpm

PEAK_MIN = 0.25
INLIER_FRAC = 1 / 6
CONST_P90_MS = 8.0
DRIFT_TOL_MS = 2.0


@dataclass
class RedLineFit:
    time_ms: float
    beat_ms: float
    meter: int


def peaks(act: np.ndarray, min_gap_ms: float = 60.0, thr: float = PEAK_MIN) -> tuple[np.ndarray, np.ndarray]:
    a = act
    cand = np.nonzero((a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:]) & (a[1:-1] >= thr))[0] + 1
    gap = int(math.ceil(min_gap_ms / FRAME_MS))
    blocked = np.zeros(len(a) + 2 * gap + 2, bool)
    keep = []
    for i in cand[np.argsort(-a[cand], kind="stable")]:   # non-maximum suppression, strongest first
        if not blocked[i + gap]:
            keep.append(i)
            blocked[i:i + 2 * gap + 1] = True
    keep.sort()
    times, weights = [], []
    for i in keep:
        y0, y1, y2 = a[i - 1], a[i], a[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        times.append((i + off) * FRAME_MS)
        weights.append(y1)
    return np.asarray(times), np.asarray(weights)


def _fit_grid(t: np.ndarray, w: np.ndarray, period: float, iters: int = 4):
    """Weighted LS fit t ~ phase + k*period over inliers; returns (phase, period, k, resid, inlier)."""
    ang = 2 * np.pi * t / period
    phase = (np.angle(np.sum(w * np.exp(1j * ang))) / (2 * np.pi)) * period
    for _ in range(iters):
        k = np.round((t - phase) / period)
        resid = t - (phase + k * period)
        inl = np.abs(resid) < period * INLIER_FRAC
        if inl.sum() < 4:
            break
        A = np.stack([np.ones(inl.sum()), k[inl]], 1)
        W = w[inl]
        sol, *_ = np.linalg.lstsq(A * W[:, None], t[inl] * W, rcond=None)
        phase, period = float(sol[0]), float(sol[1])
    k = np.round((t - phase) / period)
    resid = t - (phase + k * period)
    return phase, period, k, resid, np.abs(resid) < period * INLIER_FRAC


def _score(act, t, w, phase, period, region, prior):
    grid = np.arange(math.ceil((region[0] - phase) / period), math.floor((region[1] - phase) / period) + 1)
    if len(grid) < 4:
        return -1e9
    gt = phase + grid * period
    idx = np.clip(np.rint(gt / FRAME_MS).astype(int), 0, len(act) - 1)
    support = act[idx].mean()
    return float(support + 0.3 * math.log(prior + 1e-6))


def _candidates(act, tempo_probs, region):
    cands = set()
    for b in np.argsort(-tempo_probs)[:3]:
        cands.add(60000.0 / bin_bpm(int(b)))
    a = act[int(region[0] / FRAME_MS): int(region[1] / FRAME_MS)]
    a = a - a.mean()
    if len(a) > 400:
        spec = np.fft.rfft(a, n=2 * len(a))
        ac = np.fft.irfft(spec * np.conj(spec))[: len(a)]
        lo, hi = int(150 / FRAME_MS), int(1500 / FRAME_MS)
        lags = np.argsort(-ac[lo:hi])[:3] + lo
        for lag in lags:
            cands.add(lag * FRAME_MS)
    out = set()
    for p in cands:
        for m in (0.5, 1.0, 2.0, 2 / 3, 1.5):
            q = p * m
            if 150.0 <= q <= 1500.0:
                out.add(round(q, 3))
    return sorted(out)


def _round_bpm(period: float, n_beats: int) -> float:
    bpm = 60000.0 / period
    for rounded in (round(bpm), round(bpm * 2) / 2, round(bpm, 1), round(bpm, 2)):
        p = 60000.0 / rounded
        if abs(p - period) * max(1, n_beats) <= DRIFT_TOL_MS:
            return p
    return period


def _meter(down_act, phase, period, region):
    k = np.arange(math.ceil((region[0] - phase) / period), math.floor((region[1] - phase) / period) + 1)
    if len(k) < 16:
        return 4, 0
    idx = np.clip(np.rint((phase + k * period) / FRAME_MS).astype(int), 0, len(down_act) - 1)
    d = down_act[idx]
    best = (4, 0, -1e9)
    for m in (4, 3):
        for p in range(m):
            on = (k - p) % m == 0
            score = d[on].mean() - d[~on].mean() - (0.1 if m == 3 else 0.0)
            if score > best[2]:
                best = (m, p, score)
    return best[0], best[1]


def fit_red_lines(beat_act, down_act, tempo_probs, region: tuple[float, float]) -> list[RedLineFit]:
    t, w = peaks(beat_act)
    sel = (t >= region[0] - 2000) & (t <= region[1] + 2000)
    t, w = t[sel], w[sel]
    if len(t) < 8:
        return [RedLineFit(region[0], 500.0, 4)]
    best = None
    for p0 in _candidates(beat_act, tempo_probs, region):
        phase, period, k, resid, inl = _fit_grid(t, w, p0)
        if not (150.0 <= period <= 1500.0):
            continue
        prior = tempo_probs[int(np.clip(round(math.log2(max(60000 / period, 40) / 40) / 0.05), 0, TEMPO_BINS - 1))]
        s = _score(beat_act, t, w, phase, period, region, prior)
        if best is None or s > best[0]:
            best = (s, phase, period, resid, inl)
    _, phase, period, resid, inl = best
    p90 = np.percentile(np.abs(resid[inl]), 90) if inl.sum() else 1e9
    coverage = inl.mean() if len(inl) else 0.0
    if p90 <= CONST_P90_MS and coverage >= 0.7:
        segments = [(t[inl], w[inl], phase, period)]
    else:
        segments = _segment(t, w, tempo_probs) or [(t[inl], w[inl], phase, period)]
    lines = []
    for i, (ts, ws, ph, per) in enumerate(segments):
        n_beats = max(1, int((ts[-1] - ts[0]) / per))
        per = _round_bpm(per, n_beats)
        ph = float(np.average(ts - np.round((ts - ph) / per) * per, weights=ws))
        seg_region = (max(region[0], ts[0]), min(region[1], ts[-1])) if len(segments) > 1 else region
        meter, mphase = _meter(down_act, ph, per, seg_region)
        if i == 0:   # the last downbeat at or before the first mapped moment
            d0 = ph + mphase * per
            measure = meter * per
            start = d0 + math.floor((region[0] - d0) / measure) * measure
        else:        # later sections restart on their first beat
            start = ph + math.ceil((ts[0] - ph) / per - 1e-6) * per
        start = round(start) if abs(start - round(start)) <= 1.0 else start
        lines.append(RedLineFit(float(start), float(per), int(meter)))
    return lines


def _local_period(ts: np.ndarray, tempo_probs: np.ndarray) -> float:
    base = float(np.median(np.diff(ts)))
    best, score = base, -1e9
    for m in (0.5, 1.0, 2.0):
        p = base * m
        if not 150.0 <= p <= 1500.0:
            continue
        b = int(np.clip(round(math.log2(max(60000 / p, 40) / 40) / 0.05), 0, TEMPO_BINS - 1))
        s = math.log(tempo_probs[b] + 1e-6) + (0.5 if m == 1.0 else 0.0)
        if s > score:
            best, score = p, s
    return best


def _ls(ts, ws, phase, per):
    k = np.round((ts - phase) / per)
    A = np.stack([np.ones(len(ts)), k], 1)
    sol, *_ = np.linalg.lstsq(A * ws[:, None], ts * ws, rcond=None)
    return float(sol[0]), float(sol[1])


def _segment(t, w, tempo_probs, tol: float = 12.0, max_misses: int = 4):
    """Piecewise-constant tempo sections from strong peaks (tempo changes, rubato, live)."""
    strong = w >= 0.5 * np.percentile(w, 90)
    ts, ws = t[strong], w[strong]
    segs, s = [], 0
    while s < len(ts) - 6:
        per = _local_period(ts[s:s + 12], tempo_probs)
        phase = ts[s]
        idx, misses, first_miss, e = [s], 0, None, s + 1
        while e < len(ts):
            k = round((ts[e] - phase) / per)
            if k >= 1 and abs(ts[e] - (phase + k * per)) <= tol:
                idx.append(e)
                misses, first_miss = 0, None
                if len(idx) % 4 == 0:
                    phase, per = _ls(ts[idx], ws[idx], phase, per)
            else:
                misses += 1
                first_miss = e if first_miss is None else first_miss
                if misses >= max_misses:
                    break
            e += 1
        if len(idx) >= 4:
            phase, per = _ls(ts[idx], ws[idx], phase, per)
            segs.append((ts[idx], ws[idx], phase, per))
        s = first_miss if (misses >= max_misses and first_miss is not None) else e
        if s <= idx[0]:
            s = idx[0] + 1
    merged = []
    for seg in segs:   # merge sections that continue the same grid (e.g. across a break)
        if merged:
            pts, pws, pph, pper = merged[-1]
            if abs(pper - seg[3]) < 0.5 and abs(((seg[2] - pph) / pper) - round((seg[2] - pph) / pper)) * pper <= tol:
                cat_t, cat_w = np.r_[pts, seg[0]], np.r_[pws, seg[1]]
                merged[-1] = (cat_t, cat_w, *_ls(cat_t, cat_w, pph, pper))
                continue
        merged.append(seg)
    return merged


def beat_times(lines, start: float, end: float) -> np.ndarray:
    out = []
    for i, ln in enumerate(lines):
        seg_end = lines[i + 1].time_ms if i + 1 < len(lines) else end
        k0 = math.ceil((start - ln.time_ms) / ln.beat_ms) if i == 0 else 0
        k = k0
        while ln.time_ms + k * ln.beat_ms < seg_end - 1e-6:
            tt = ln.time_ms + k * ln.beat_ms
            if tt >= start - 1e-6:
                out.append(tt)
            k += 1
    return np.asarray(out)


def drift_report(pred_lines, human_lines, region) -> dict[str, float]:
    """Errors of human beats against the predicted grid, with tempo-ratio classification."""
    hb = beat_times(human_lines, *region)
    pb = beat_times(pred_lines, region[0] - 2000, region[1] + 2000)
    if len(hb) < 4 or len(pb) < 4:
        return {"ok": False, "ratio": float("nan"), "max_ms": float("inf"), "median_ms": float("inf")}
    idx = np.clip(np.searchsorted(pb, hb), 1, len(pb) - 1)
    err = np.minimum(np.abs(hb - pb[idx - 1]), np.abs(hb - pb[idx]))
    ratio = np.median(np.diff(hb)) / np.median(np.diff(pb))
    return {"ok": True, "ratio": float(ratio), "max_ms": float(err.max()), "median_ms": float(np.median(err)),
            "p95_ms": float(np.percentile(err, 95))}
