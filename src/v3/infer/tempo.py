"""Beat activations -> osu! red lines (BPM, offset, meter), plus drift evaluation.

Constant-tempo first. Tempo candidates (tempo head, activation autocorrelation, octave variants) are
refined to a precise period by peak phase coherence, fitted to the activation peaks with iterated
weighted least squares, and scored by grid support, the tempo prior, and how much of T's strong
beat-peak weight the grid accounts for. A single grid is kept unless some stretch of the song stops
fitting it (windowed test); only then are the peaks segmented into piecewise-constant sections (tempo
changes, rubato, live recordings), and the split is kept only if it fits clearly better. BPMs snap to
the roundest value whose accumulated drift stays within tolerance; offsets land on measure starts.

Measured on 1,106 held-out songs (2026-11-25), the previous decision rule split 11% of steady songs:
it counted every peak (strong half-beat peaks or slightly jittery peaks failed it), coarse candidates
(tempo bins are ~3.5% apart) lost beat count on long fast songs, and half tempo could win on the tempo
prior alone because it lands on beats too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.v3.data.audio import FRAME_MS

from src.v3.models.timing import TEMPO_BINS, bin_bpm

PEAK_MIN = 0.25
INLIER_FRAC = 1 / 6
DRIFT_TOL_MS = 2.0
REFINE_SPAN, REFINE_STEPS = 0.04, 801   # period refinement: +-4% around each candidate, 0.01% steps
STRONG = 0.5                            # beat peaks at least this high are "strong"
HIT_FRAC = 0.1                          # a strong peak is accounted for within 1/10 beat of a grid beat
EXPLAIN_W = 0.5                         # candidate score weight of the accounted-for strong-peak fraction:
                                        # beat peaks outvote a tempo prior leaning up to ~2.3:1 the other way
WIN_MS = 8000.0                         # constancy test windows
Q_MIN = 0.4                             # one grid unless a window (10th pct) accounts for < 40% strong weight
SPLIT_GAIN = 0.15                       # a split must raise that 10th-percentile score at least this much


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


def _tempo_bin(period: float) -> int:
    return int(np.clip(round(math.log2(max(60000 / period, 40) / 40) / 0.05), 0, TEMPO_BINS - 1))


def refine_period(t: np.ndarray, w: np.ndarray, p0: float) -> float:
    """The period within +-REFINE_SPAN of p0 that maximizes the peaks' phase coherence at the period and
    at half of it (the half-period term keeps the estimate sharp when off-beat peaks cancel the first)."""
    ps = p0 * (1.0 + np.linspace(-REFINE_SPAN, REFINE_SPAN, REFINE_STEPS))
    ww = w / w.sum()
    ang = 2j * np.pi * t[None, :] / ps[:, None]
    c = np.abs(np.exp(ang) @ ww) + np.abs(np.exp(2 * ang) @ ww)
    return float(ps[int(np.argmax(c))])


def explained(t: np.ndarray, w: np.ndarray, phase: float, period: float) -> float:
    """Fraction of strong-peak weight within HIT_FRAC of a beat of the grid."""
    strong = w >= STRONG
    if not strong.any():
        return 0.0
    r = (t[strong] - phase) / period
    return float(np.sum(w[strong] * (np.abs(r - np.round(r)) <= HIT_FRAC)) / np.sum(w[strong]))


def window_scores(t: np.ndarray, w: np.ndarray, beats: np.ndarray, region) -> np.ndarray:
    """Per WIN_MS window of the region: fraction of strong-peak weight within HIT_FRAC beat of ``beats``."""
    keep = (w >= STRONG) & (t >= region[0]) & (t <= region[1])
    ts, ws = t[keep], w[keep]
    if len(beats) < 2 or len(ts) == 0:
        return np.zeros(1)
    i = np.clip(np.searchsorted(beats, ts), 1, len(beats) - 1)
    d = np.minimum(np.abs(ts - beats[i - 1]), np.abs(beats[i] - ts))
    hit = d <= HIT_FRAC * (beats[i] - beats[i - 1])
    out = []
    for a in np.arange(region[0], region[1], WIN_MS):
        m = (ts >= a) & (ts < a + WIN_MS)
        if m.sum() >= 4:
            out.append(np.sum(ws[m] * hit[m]) / np.sum(ws[m]))
    return np.asarray(out) if out else np.zeros(1)


def _fit_quality(lines, t, w, region) -> float:
    """10th percentile over windows: how well the worst stretches of the song fit these red lines."""
    beats = beat_times(lines, region[0] - 2000, region[1] + 2000)
    return float(np.percentile(window_scores(t, w, beats, region), 10))


def fit_red_lines(beat_act, down_act, tempo_probs, region: tuple[float, float]) -> list[RedLineFit]:
    t, w = peaks(beat_act)
    sel = (t >= region[0] - 2000) & (t <= region[1] + 2000)
    t, w = t[sel], w[sel]
    if len(t) < 8:
        return [RedLineFit(region[0], 500.0, 4)]
    best = None
    for p0 in _candidates(beat_act, tempo_probs, region):
        phase, period, _, _, inl = _fit_grid(t, w, refine_period(t, w, p0))
        if not (150.0 <= period <= 1500.0) or inl.sum() < 4:
            continue
        s = (_score(beat_act, t, w, phase, period, region, tempo_probs[_tempo_bin(period)])
             + EXPLAIN_W * explained(t, w, phase, period))
        if best is None or s > best[0]:
            best = (s, phase, period, inl)
    if best is None:
        return [RedLineFit(region[0], 500.0, 4)]
    _, phase, period, inl = best
    const = _lines_from_segments([(t[inl], w[inl], phase, period)], down_act, region)
    q_const = _fit_quality(const, t, w, region)
    if q_const >= Q_MIN:
        return const
    segments = _segment(t, w, tempo_probs)
    if not segments:
        return const
    split = _lines_from_segments(segments, down_act, region)
    return split if _fit_quality(split, t, w, region) >= q_const + SPLIT_GAIN else const


def _lines_from_segments(segments, down_act, region) -> list[RedLineFit]:
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
    """Errors of human beats against the predicted grid, with tempo-ratio classification.

    ``shift_ms`` is the grid's constant offset against the human one (median of predicted - human, > 0 =
    predicted later); ``shape_*`` are the errors after removing it. They are reported separately because
    a single human chart's offset is itself uncertain: against the audio's onsets, human offsets scatter
    with an IQR of ~7 ms across songs, while T's grids scatter by ~2 ms, and 0.93 of the correlation in
    T's constant shifts is with that human scatter (val, 2026-11-25).
    """
    hb = beat_times(human_lines, *region)
    pb = beat_times(pred_lines, region[0] - 2000, region[1] + 2000)
    if len(hb) < 4 or len(pb) < 4:
        return {"ok": False, "ratio": float("nan"), "max_ms": float("inf"), "median_ms": float("inf"),
                "shift_ms": float("nan"), "shape_max_ms": float("inf"), "shape_median_ms": float("inf")}
    idx = np.clip(np.searchsorted(pb, hb), 1, len(pb) - 1)
    lo, hi = pb[idx - 1], pb[idx]
    signed = np.where(np.abs(hb - lo) <= np.abs(hb - hi), lo, hi) - hb
    err = np.abs(signed)
    shift = float(np.median(signed))
    shape = np.abs(signed - shift)
    ratio = np.median(np.diff(hb)) / np.median(np.diff(pb))
    return {"ok": True, "ratio": float(ratio), "max_ms": float(err.max()), "median_ms": float(np.median(err)),
            "p95_ms": float(np.percentile(err, 95)), "shift_ms": shift, "shape_max_ms": float(shape.max()),
            "shape_median_ms": float(np.median(shape))}
