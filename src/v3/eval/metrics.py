"""Evaluation metrics on native .osu text (generated vs reference, and vs human bands).

Reference matching saturates near the human-vs-human ceiling, so distributions are
compared *two-sided* against a human reference set: a model is good when its
distance to humans is close to the distance between two disjoint human sets.
"""
from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np

from src.v3.data.charts import convert_chart
from src.v3.data.grid import TICKS, build_grid
from src.v3.data.osu import RedLine
from src.v3.eval.match import match_onsets, onset_scores, short_pair_recall

DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)


def summarize_chart(text: str) -> dict[str, Any] | None:
    meta, cols = convert_chart(text)
    return None if cols is None else summarize_converted(meta, cols)


def summarize_converted(meta: dict, cols: dict) -> dict[str, Any]:
    reds = [RedLine(*r) for r in meta["red_lines"]]
    t = cols["time_ms"]
    grid = build_grid(reds, meta["grid_start_ms"], float(np.max(cols["end_ms"])) + 4000.0)
    beat = np.clip(cols["beat"], 0, len(grid) - 1)
    beat_pos = beat + cols["tick"] / TICKS        # in beats of the active red line (approx. across lines)
    bim = grid.beat_in_measure[beat]
    return {"meta": meta, "time": t, "end": cols["end_ms"], "kind": cols["kind"].astype(int), "tick": cols["tick"],
            "slot": cols["slot"], "beat_pos": beat_pos, "bim": bim, "nc": cols["new_combo"].astype(bool),
            "x": cols["x"], "y": cols["y"], "curve": cols["curve"], "length": cols["pixel_length"],
            "sv": cols["sv"], "spans": cols["spans"], "stars": meta.get("descriptors", {}).get("stars"),
            "headings": cols["headings"], "path_ok": cols["path_ok"].astype(bool)}


# ------------------------------------------------------- per-map vs reference ----

def reference_metrics(gen: dict, ref: dict) -> dict[str, float]:
    out = {}
    for tol in (10, 25, 50):
        s = onset_scores(gen["time"], ref["time"], tol)
        out[f"f1_{tol}"] = s["f1"]
        out[f"matched_{tol}"], out["n_gen"], out["n_ref"] = s["matched"], s["pred"], s["ref"]
    pairs = match_onsets(gen["time"], ref["time"], 25.0)
    typed = sum(1 for i, j in pairs if gen["kind"][i] == ref["kind"][j])
    out["typed_f1_25"] = 2 * typed / max(1, len(gen["time"]) + len(ref["time"]))
    got, total = short_pair_recall(gen["time"], ref["time"], ref["beat_pos"], 25.0)
    out["short_pairs_got"], out["short_pairs_total"] = got, total
    durs = [abs((gen["end"][i] - gen["time"][i]) - (ref["end"][j] - ref["time"][j]))
            for i, j in pairs if gen["kind"][i] == 1 and ref["kind"][j] == 1]
    out["slider_dur_err_median"] = float(np.median(durs)) if durs else float("nan")
    return out


BODY_TOL_PX = 3.0   # = the export round-trip fit tolerance: a body fitted onto the border may overshoot this much


def _body_points(s: dict) -> np.ndarray:
    """Slider-body centerline points [n_sliders, H+1, 2] rebuilt from the stored unit headings."""
    sl = (s["kind"] == 1) & s["path_ok"]
    if not sl.any():
        return np.zeros((0, 2, 2))
    h = s["headings"][sl].astype(np.float64)
    head = np.stack([s["x"][sl], s["y"][sl]], -1)[:, None, :]
    steps = np.cumsum(h * (s["length"][sl] / h.shape[1])[:, None, None], axis=1)
    return np.concatenate([head, head + steps], axis=1)


def body_overshoot_px(s: dict) -> float:
    """Largest distance any slider body runs past the 512x384 playfield (0 if none): a style trait."""
    pts = _body_points(s)
    if not len(pts):
        return 0.0
    o = np.stack([-pts[..., 0], pts[..., 0] - 512, -pts[..., 1], pts[..., 1] - 384], -1)
    return float(max(0.0, o.max()))


def validity(s: dict) -> dict[str, Any]:
    """Game-legal footprint (the same rule placement projects into: src.v3.models.placement): heads inside
    the 512x384 playfield, slider-body centerlines inside osu!'s visible area, no overlapping objects.

    ``path_failures`` (sliders our reconstruction cannot evaluate, e.g. over the evaluation budget) is
    reported, not counted as invalid: it happens on ranked human charts that the game draws fine, and a
    generated map cannot reach the eval with one (candidate selection rejects any failed re-parse).
    """
    from src.v3.models.placement import BODY_HI, BODY_LO, HEAD_HI, HEAD_LO
    live = s["kind"] != 2
    heads_out = int(np.sum(live & ((s["x"] < HEAD_LO[0]) | (s["x"] > HEAD_HI[0]) | (s["y"] < HEAD_LO[1]) | (s["y"] > HEAD_HI[1]))))
    pts, t = _body_points(s), BODY_TOL_PX
    out = (pts[..., 0] < BODY_LO[0] - t) | (pts[..., 0] > BODY_HI[0] + t) | (pts[..., 1] < BODY_LO[1] - t) | (pts[..., 1] > BODY_HI[1] + t)
    bodies_out = int(out.any(1).sum()) if len(pts) else 0
    meta = s["meta"]
    return {"heads_out": heads_out, "bodies_out": bodies_out, "path_failures": meta["path_failures"],
            "overlaps": meta["overlaps"], "off_grid_frac": meta["off_grid_frac"],
            "body_overshoot_px": body_overshoot_px(s),
            "valid": heads_out == 0 and bodies_out == 0 and meta["overlaps"] == 0}


# ------------------------------------------------------------ distributions ----

def map_distributions(s: dict) -> dict[str, Any]:
    t, k = s["time"], s["kind"]
    on = s["slot"] >= 0
    div = []
    for tick in s["tick"][on]:
        div.append(next((d for d in DIVISORS if tick % (TICKS // d) == 0), 0))
    gaps = np.diff(s["beat_pos"])
    q = np.clip(np.rint(gaps * 12), 0, 48).astype(int)          # gaps in 1/12-beat units (cap 4 beats)
    tri = list(zip(q[:-2], q[1:-1], q[2:]))
    live = k != 2
    h = np.stack([s["x"], s["y"]], 1)[live]
    tl = t[live]
    seg = np.diff(h, axis=0)
    dist = np.linalg.norm(seg, axis=1)
    gap_ms = np.maximum(np.diff(tl), 1.0)
    a, b = seg[:-1], seg[1:]
    ok = (dist[:-1] > 8) & (dist[1:] > 8)
    ang = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))[ok]
    gb = np.diff(s["beat_pos"][live])
    short = (gb > 0) & (gb <= 0.25 + 1e-3)
    stacked = dist[short] < 3.0 if short.any() else np.zeros(0, bool)
    sl = k == 1
    sv = s["sv"][sl]
    minutes = max((t[-1] - t[0]) / 60000.0, 1e-3)
    nc = s["nc"] & on
    return {"divisors": div, "trigrams": tri, "spacing_per_ms": dist / gap_ms, "turn": ang,
            "stack": stacked.tolist(), "curves": s["curve"][sl].tolist(), "slider_len": s["length"][sl].tolist(),
            "density_per_min": len(t) / minutes, "sv_changes_per_min": int(np.sum(np.abs(np.diff(sv)) > 1e-6)) / minutes,
            "nc_on_downbeat": float(np.mean((s["bim"][nc] == 0) & (s["tick"][nc] == 0))) if nc.any() else float("nan"),
            "overshoot_px": body_overshoot_px(s)}


def _hist(values, keys):
    c = collections.Counter(values)
    total = max(1, sum(c.values()))
    return np.array([c.get(k, 0) / total for k in keys])


def _js(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _w1(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    qs = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def pooled(dists: list[dict]) -> dict[str, Any]:
    cat = lambda k: [v for d in dists for v in d[k]]
    return {"divisors": cat("divisors"), "trigrams": cat("trigrams"), "spacing": np.array(cat("spacing_per_ms")),
            "turn": np.array(cat("turn")), "stack": np.array(cat("stack"), float), "curves": cat("curves"),
            "slider_len": np.array(cat("slider_len")), "density": np.array([d["density_per_min"] for d in dists]),
            "sv_changes": np.array([d["sv_changes_per_min"] for d in dists]),
            "nc_down": np.array([d["nc_on_downbeat"] for d in dists if not math.isnan(d["nc_on_downbeat"])]),
            "overshoot": np.array([d.get("overshoot_px", math.nan) for d in dists if not math.isnan(d.get("overshoot_px", math.nan))])}


def distribution_distance(a: dict, b: dict) -> dict[str, float]:
    tri_keys = [k for k, _ in collections.Counter(a["trigrams"] + b["trigrams"]).most_common(2000)]
    return {
        "snap_js": _js(_hist(a["divisors"], (0,) + DIVISORS), _hist(b["divisors"], (0,) + DIVISORS)),
        "trigram_js": _js(_hist(a["trigrams"], tri_keys), _hist(b["trigrams"], tri_keys)),
        "spacing_w1": _w1(a["spacing"], b["spacing"]),
        "turn_w1": _w1(a["turn"], b["turn"]),
        "stack_rate_diff": float(abs(np.mean(a["stack"]) - np.mean(b["stack"]))) if len(a["stack"]) and len(b["stack"]) else float("nan"),
        "curve_js": _js(_hist(a["curves"], (1, 2, 3, 4)), _hist(b["curves"], (1, 2, 3, 4))),
        "slider_len_w1": _w1(a["slider_len"], b["slider_len"]),
        "density_w1": _w1(a["density"], b["density"]),
        "sv_changes_w1": _w1(a["sv_changes"], b["sv_changes"]),
        "nc_downbeat_diff": float(abs(np.mean(a["nc_down"]) - np.mean(b["nc_down"]))) if len(a["nc_down"]) and len(b["nc_down"]) else float("nan"),
        "overshoot_w1": _w1(a.get("overshoot", np.zeros(0)), b.get("overshoot", np.zeros(0))),
    }


def summary_stats(p: dict) -> dict[str, float]:
    return {"stack_rate": float(np.mean(p["stack"])) if len(p["stack"]) else float("nan"),
            "density_median": float(np.median(p["density"])) if len(p["density"]) else float("nan"),
            "sv_changes_median": float(np.median(p["sv_changes"])) if len(p["sv_changes"]) else float("nan"),
            "overshoot_rate": float(np.mean(p["overshoot"] > 0)) if len(p.get("overshoot", [])) else float("nan"),
            "curve_mix": {k: float(v) for k, v in zip("LPBC", _hist(p["curves"], (1, 2, 3, 4)))},
            "snap_mix": {str(k): float(v) for k, v in zip((0,) + DIVISORS, _hist(p["divisors"], (0,) + DIVISORS))}}
