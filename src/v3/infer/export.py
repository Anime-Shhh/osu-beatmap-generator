"""Events + placement -> native .osu text, with exact slider lengths and fitted curves.

Slider paths arrive as 33 points with exact arc length. They are fitted to the
simplest native curve that reproduces them: L (line), P (circular arc), else B
pieces split at corners (red anchors) with least-squares cubic segments; the
written curve is re-rendered with osu! semantics and checked against the target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.v2.slider_path import slider_path

LINE_TOL = 1.0
FIT_TOL = 1.5          # round-trip on 292 human charts: 2.5 px tolerance gave 98.9% of sliders <= 3 px
CORNER_DEG = 45.0


@dataclass
class Placed:
    x: float
    y: float
    path: np.ndarray | None = None     # [33, 2] px for sliders


def _resample(path: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, path[:, 0]), np.interp(t, cum, path[:, 1])], 1)


def _dev(rendered: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(_resample(rendered, len(target)) - target, axis=1)))


def _line_dev(pts):
    a, b = pts[0], pts[-1]
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return float(np.max(np.linalg.norm(pts - a, axis=1)))
    return float(np.max(np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / n))


def _bezier_fit(pts: np.ndarray) -> np.ndarray:
    """Cubic Bezier through pts[0], pts[-1] (chord-length parameters, LS inner controls)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.r_[0.0, np.cumsum(seg)] / max(seg.sum(), 1e-9)
    b1, b2 = 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2
    rhs = pts - np.outer((1 - t) ** 3, pts[0]) - np.outer(t ** 3, pts[-1])
    A = np.stack([b1, b2], 1)
    ctrl, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return np.array([pts[0], ctrl[0], ctrl[1], pts[-1]])


def _bez_eval(c, n=64):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]


def _pieces(pts: np.ndarray, depth: int = 0) -> list[np.ndarray]:
    if len(pts) <= 2 or _line_dev(pts) <= LINE_TOL:
        return [np.array([pts[0], pts[-1]])]
    c = _bezier_fit(pts)
    if _dev(_bez_eval(c), _resample(pts, 33)) <= FIT_TOL or depth >= 4 or len(pts) < 5:
        return [c]
    mid = len(pts) // 2
    return _pieces(pts[: mid + 1], depth + 1) + _pieces(pts[mid:], depth + 1)


def fit_curve(path: np.ndarray, length: float) -> tuple[str, list[tuple[int, int]], float]:
    """Return (curve_type, integer control points after the head, max deviation px)."""
    head = np.rint(path[0])
    candidates = []
    if _line_dev(path) <= LINE_TOL:
        candidates.append(("L", [path[-1]]))
    mid = path[len(path) // 2]
    candidates.append(("P", [mid, path[-1]]))
    ang = np.degrees(np.arctan2(*np.diff(path, axis=0).T[::-1]))
    turn = np.abs((np.diff(ang) + 180) % 360 - 180)
    cuts = [0] + [i + 1 for i in np.nonzero(turn > CORNER_DEG)[0]] + [len(path) - 1]
    ctrl: list[np.ndarray] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1:
            continue
        for piece in _pieces(path[a:b + 1]):
            if not ctrl:
                ctrl.extend(piece)
            else:
                ctrl.append(piece[0])          # duplicated junction point = red anchor (segment break)
                ctrl.extend(piece[1:])
    b_ctrl = ctrl[1:] if ctrl else [path[-1]]
    candidates.append(("B", b_ctrl))
    # Dense polyline fallback (every second point as a red anchor), kept within the 33-point
    # path-evaluation budget: head + 31 controls.
    dense = path[2::2]
    candidates.append(("B", [p for q in dense for p in (q, q)][:-1]))
    best = None
    for kind, pts in candidates:
        controls = [tuple(int(v) for v in np.rint(p)) for p in pts]
        if kind == "B":
            controls = _dedupe_runs(controls)
        try:
            rendered = slider_path(tuple(head), controls, kind, length)
        except ValueError:
            continue
        dev = _dev(rendered, path)
        if best is None or dev < best[2] - 0.25:
            best = (kind, controls, dev)
        if dev <= FIT_TOL:
            return kind, controls, dev
    if best is None:
        raise ValueError("no native curve reproduces the slider path")
    return best


def _dedupe_runs(points):
    """Keep at most two identical consecutive points (osu! red-anchor convention)."""
    out = []
    for p in points:
        if len(out) >= 2 and out[-1] == p and out[-2] == p:
            continue
        out.append(p)
    return out


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.6f}".rstrip("0").rstrip(".")


def write_osu(events, placed: list[Placed], red_lines, *, title: str, artist: str, version: str,
              audio_filename: str, hp: float, cs: float, od: float, ar: float, slider_multiplier: float,
              creator: str = "osu-v3") -> tuple[str, dict]:
    """Return (.osu text, export report)."""
    timing = [(rl.time_ms, f"{_fmt(rl.time_ms)},{rl.beat_ms!r},{rl.meter},2,0,60,1,0") for rl in red_lines]
    objects, report = [], {"slider_fit_px": [], "curve_types": {"L": 0, "P": 0, "B": 0}}
    current_sv = None
    red_times = [rl.time_ms for rl in red_lines]
    for e, p in zip(events, placed):
        t = int(round(e.time_ms))
        combo = 4 if e.new_combo else 0
        x, y = int(round(p.x)), int(round(p.y))
        if e.kind == 0:
            objects.append(f"{x},{y},{t},{1 | combo},0,0:0:0:0:")
        elif e.kind == 2:
            objects.append(f"256,192,{t},{8 | combo},0,{int(round(e.end_ms))},0:0:0:0:")
        else:
            # A red line resets SV to 1: re-emit the green line after any red line.
            last_red = max((rt for rt in red_times if rt <= t + 1e-6), default=None)
            if current_sv is None or abs(current_sv[0] - e.sv) > 1e-9 or (last_red is not None and current_sv[1] < last_red):
                meter = next((rl.meter for rl in reversed(red_lines) if rl.time_ms <= t + 1e-6), 4)
                timing.append((t + 0.1, f"{t},{-100.0 / e.sv!r},{meter},2,0,60,0,0"))
                current_sv = (e.sv, t)
            span_beats = e.span_ticks / 48.0
            length = e.sv * slider_multiplier * 100.0 * span_beats
            kind, controls, dev = fit_curve(p.path, length)
            report["slider_fit_px"].append(dev)
            report["curve_types"][kind] += 1
            curve = kind + "|" + "|".join(f"{cx}:{cy}" for cx, cy in controls)
            objects.append(f"{x},{y},{t},{2 | combo},0,{curve},{e.spans},{length:.6f}".rstrip("0").rstrip("."))
    timing.sort(key=lambda kv: kv[0])
    text = "\n".join([
        "osu file format v14", "", "[General]", f"AudioFilename: {audio_filename}", "AudioLeadIn: 0",
        "PreviewTime: -1", "Countdown: 0", "SampleSet: Soft", "StackLeniency: 0.7", "Mode: 0",
        "LetterboxInBreaks: 0", "WidescreenStoryboard: 0", "", "[Editor]", "DistanceSpacing: 1", "BeatDivisor: 4",
        "GridSize: 16", "TimelineZoom: 1", "", "[Metadata]", f"Title:{title}", f"TitleUnicode:{title}",
        f"Artist:{artist}", f"ArtistUnicode:{artist}", f"Creator:{creator}", f"Version:{version}", "Source:",
        "Tags:osu-v3 generated", "BeatmapID:0", "BeatmapSetID:-1", "", "[Difficulty]", f"HPDrainRate:{_fmt(hp)}",
        f"CircleSize:{_fmt(cs)}", f"OverallDifficulty:{_fmt(od)}", f"ApproachRate:{_fmt(ar)}",
        f"SliderMultiplier:{slider_multiplier!r}", "SliderTickRate:1", "", "[Events]", "//Background and Video events",
        "//Break Periods", "", "[TimingPoints]", *[line for _, line in timing], "", "", "[HitObjects]", *objects, ""])
    return text, report
