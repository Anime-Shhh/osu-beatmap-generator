"""Native chart -> v3 per-object arrays (rhythm + placement targets).

Slider geometry is stored as the *rendered* path (osu! trimming/extension to the
declared pixel length) resampled into ``HEADINGS`` equal arc-length steps of unit
heading vectors. ``path = head + cumsum(heading * pixel_length / HEADINGS)`` is
exact in length by construction; ``path_err`` records the representation error
against the rendered path so the Phase 0 round-trip gate can judge it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.v2.slider_path import slider_path

from .grid import TICKS, GridError, build_grid, locate
from .osu import CIRCLE, SLIDER, SPINNER, Chart, ChartError, parse_chart

HEADINGS = 32  # 99.3% of human sliders within 3 px (16: 98.5%)
STACK_PX = 1.0
EDGE_PX = 1.0
CURVE_CODE = {"L": 1, "P": 2, "B": 3, "C": 4}

SCALAR_COLUMNS: dict[str, Any] = {
    "time_ms": np.float64, "end_ms": np.float64, "kind": np.int8, "new_combo": np.int8,
    "combo_skip": np.int8, "hit_sound": np.int16, "x": np.float32, "y": np.float32,
    "spans": np.int16, "pixel_length": np.float32, "sv": np.float32, "beat_ms": np.float32,
    "beat": np.int32, "tick": np.int16, "slot": np.int16, "resid24": np.float32, "resid48": np.float32,
    "end_beat": np.int32, "end_tick": np.int16, "end_resid48": np.float32, "span_ticks": np.float32,
    "exit_x": np.float32, "exit_y": np.float32, "path_ok": np.int8, "path_err": np.float32,
    "path_len": np.float32, "poly_ratio": np.float32, "curve": np.int8, "n_controls": np.int16,
    "stack_head": np.int8, "stack_exit": np.int8, "edge_touch": np.int8,
}


def headings_from_path(path: np.ndarray, length: float, k: int = HEADINGS) -> tuple[np.ndarray, float, float]:
    """Unit headings of k equal arc-length steps; returns (headings, max_err_px, path_len)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    targets = np.linspace(0.0, total, k + 1)
    pts = np.stack([np.interp(targets, cum, path[:, 0]), np.interp(targets, cum, path[:, 1])], axis=1)
    steps = np.diff(pts, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    headings = np.zeros((k, 2), np.float64)
    last = None
    for i in range(k):
        if norms[i] > 1e-6:
            last = steps[i] / norms[i]
        headings[i] = last if last is not None else (1.0, 0.0)
    # Backfill leading degenerate steps with the first real heading.
    first = next((steps[i] / norms[i] for i in range(k) if norms[i] > 1e-6), np.array([1.0, 0.0]))
    for i in range(k):
        if norms[i] > 1e-6:
            break
        headings[i] = first
    recon = path[0] + np.concatenate(([[0.0, 0.0]], np.cumsum(headings * (length / k), axis=0)))
    # Compare against the rendered path sampled at the same arc lengths of the *declared* length.
    ref_targets = np.linspace(0.0, length, k + 1)
    ref = np.stack([np.interp(ref_targets, cum, path[:, 0]), np.interp(ref_targets, cum, path[:, 1])], axis=1)
    if length > total:  # rendered path shorter than declared (extension suppressed): extrapolate flat
        ref[ref_targets > total] = path[-1]
    err = float(np.max(np.linalg.norm(recon - ref, axis=1)))
    return headings.astype(np.float32), err, total


def _descriptors(text: str) -> dict[str, Any]:
    try:
        import rosu_pp_py as rosu
        attrs = rosu.Difficulty(lazer=False).calculate(rosu.Beatmap(content=text))
        return {"stars": float(attrs.stars), "aim": float(attrs.aim), "speed": float(attrs.speed),
                "slider_factor": float(attrs.slider_factor),
                "speed_note_count": float(attrs.speed_note_count), "max_combo": int(attrs.max_combo),
                "calculator": "rosu-pp-py/4.0.2/stable/no-mods"}
    except Exception as error:  # recorded, never silently replaced
        return {"error": f"{type(error).__name__}: {error}"[:300]}


def convert_chart(text: str, *, descriptors: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Return (meta, columns). ``columns`` is None when the chart is rejected."""
    meta: dict[str, Any] = {"status": "ok", "errors": []}
    try:
        chart = parse_chart(text)
    except ChartError as error:
        return {"status": "rejected", "errors": [f"parse: {error}"]}, None
    meta.update(_chart_meta(chart))
    if chart.mode != 0:
        meta.update(status="rejected", errors=[f"mode {chart.mode} is not osu!standard"])
        return meta, None
    objs = chart.objects
    if not objs:
        meta.update(status="rejected", errors=["no hit objects"])
        return meta, None
    end_times = np.array([o.end_ms if o.kind != CIRCLE else o.time_ms for o in objs])
    grid_start = min(0.0, objs[0].time_ms - 1.0)
    grid_end = float(np.nanmax(end_times)) + 4 * max(r.beat_ms for r in chart.red_lines)
    try:
        grid = build_grid(chart.red_lines, grid_start, grid_end)
    except GridError as error:
        meta.update(status="rejected", errors=[f"grid: {error}"])
        return meta, None
    meta["grid_start_ms"] = grid_start
    n = len(objs)
    cols = {name: np.zeros(n, dtype) for name, dtype in SCALAR_COLUMNS.items()}
    headings = np.zeros((n, HEADINGS, 2), np.float32)
    times = np.array([o.time_ms for o in objs], np.float64)
    loc = locate(grid, times)
    ends = np.where(np.isfinite(end_times), end_times, times)
    end_loc = locate(grid, ends)
    exits = np.zeros((n, 2), np.float64)
    path_failures: list[str] = []
    for i, o in enumerate(objs):
        cols["time_ms"][i] = o.time_ms
        cols["end_ms"][i] = ends[i]
        cols["kind"][i] = o.kind
        cols["new_combo"][i] = o.new_combo
        cols["combo_skip"][i] = o.combo_skip
        cols["hit_sound"][i] = o.hit_sound
        cols["x"][i], cols["y"][i] = o.x, o.y
        cols["spans"][i] = o.spans
        cols["sv"][i] = o.sv
        cols["beat_ms"][i] = o.beat_ms
        exits[i] = (o.x, o.y)
        if o.kind == SLIDER:
            cols["pixel_length"][i] = o.pixel_length
            cols["curve"][i] = CURVE_CODE.get(o.curve_type, 0)
            cols["n_controls"][i] = len(o.control_points)
            cols["span_ticks"][i] = (o.end_ms - o.time_ms) / o.spans / o.beat_ms * TICKS
            poly = np.array([(o.x, o.y), *o.control_points], np.float64)
            poly_len = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))) if len(poly) > 1 else 0.0
            cols["poly_ratio"][i] = poly_len / o.pixel_length if o.pixel_length > 0 else math.nan
            try:
                path = slider_path((o.x, o.y), o.control_points, o.curve_type, o.pixel_length)
                h, err, total = headings_from_path(path, o.pixel_length)
                headings[i] = h
                cols["path_ok"][i] = 1
                cols["path_err"][i] = err
                cols["path_len"][i] = total
                exits[i] = path[-1] if o.spans % 2 == 1 else path[0]
                lo, hi = path.min(axis=0), path.max(axis=0)
                cols["edge_touch"][i] = bool((lo <= EDGE_PX).any() or hi[0] >= 512 - EDGE_PX or hi[1] >= 384 - EDGE_PX)
            except ValueError as error:
                path_failures.append(str(error))
                cols["path_err"][i] = math.nan
        elif o.kind == CIRCLE:
            cols["edge_touch"][i] = bool(min(o.x, o.y) <= EDGE_PX or o.x >= 512 - EDGE_PX or o.y >= 384 - EDGE_PX)
        elif o.kind == SPINNER:
            cols["span_ticks"][i] = (o.end_ms - o.time_ms) / o.beat_ms * TICKS
    cols["exit_x"], cols["exit_y"] = exits[:, 0].astype(np.float32), exits[:, 1].astype(np.float32)
    cols["beat"], cols["tick"], cols["slot"] = loc.beat, loc.tick, loc.slot
    cols["resid24"], cols["resid48"] = loc.resid24_ms, loc.resid48_ms
    cols["end_beat"], cols["end_tick"], cols["end_resid48"] = end_loc.beat, end_loc.tick, end_loc.resid48_ms
    for i in range(1, n):
        if objs[i].kind == SPINNER or objs[i - 1].kind == SPINNER:
            continue
        head = np.array([objs[i].x, objs[i].y])
        if np.linalg.norm(head - np.array([objs[i - 1].x, objs[i - 1].y])) < STACK_PX:
            cols["stack_head"][i] = 1
        elif objs[i - 1].kind == SLIDER and np.linalg.norm(head - exits[i - 1]) < STACK_PX:
            cols["stack_exit"][i] = 1
    cols["headings"] = headings
    held = cols["kind"] != CIRCLE
    meta.update({
        "n_objects": n,
        "n_circles": int(np.sum(cols["kind"] == CIRCLE)),
        "n_sliders": int(np.sum(cols["kind"] == SLIDER)),
        "n_spinners": int(np.sum(cols["kind"] == SPINNER)),
        "first_ms": float(times[0]), "last_end_ms": float(np.nanmax(ends)),
        "n_beats_grid": len(grid),
        "off_grid_frac": float(np.mean(cols["slot"] < 0)),
        "end_off_grid_frac": float(np.mean(np.abs(cols["end_resid48"][held]) > 2.0)) if held.any() else 0.0,
        "overlaps": int(np.sum(times[1:] < ends[:-1] - 2.0)),
        "path_failures": len(path_failures),
        "path_failure_reasons": sorted(set(path_failures))[:5],
        "max_path_err": float(np.nanmax(cols["path_err"])) if (cols["path_ok"] == 1).any() else 0.0,
        "spans_over_8": int(np.sum(cols["spans"] > 8)),
    })
    if descriptors:
        meta["descriptors"] = _descriptors(text)
    return meta, cols


def _chart_meta(chart: Chart) -> dict[str, Any]:
    md = chart.metadata
    return {
        "format_version": chart.version, "mode": chart.mode,
        "creator": md.get("Creator", ""), "diff_name": md.get("Version", ""),
        "artist": md.get("Artist", ""), "title": md.get("Title", ""),
        "cs": chart.cs, "ar": chart.ar, "od": chart.od, "hp": chart.hp,
        "slider_multiplier": chart.slider_multiplier, "tick_rate": chart.tick_rate,
        "stack_leniency": chart.stack_leniency,
        "red_lines": [[r.time_ms, r.beat_ms, r.meter] for r in chart.red_lines],
        "n_green_lines": sum(1 for p in chart.timing_points if not p.red),
        "kiai_ms": _kiai_spans(chart),
        "warnings": chart.warnings[:5],
    }


def _kiai_spans(chart: Chart) -> list[list[float]]:
    spans, start = [], None
    for p in chart.timing_points:
        if p.kiai and start is None:
            start = p.time_ms
        elif not p.kiai and start is not None:
            spans.append([start, p.time_ms])
            start = None
    if start is not None:
        spans.append([start, None])  # open until the end of the map
    return spans


def concat_columns(per_chart: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(SCALAR_COLUMNS) + ["headings"]
    if not per_chart:
        return {k: np.zeros((0,) + ((HEADINGS, 2) if k == "headings" else ()),
                            np.float32 if k == "headings" else SCALAR_COLUMNS[k]) for k in keys}
    return {k: np.concatenate([c[k] for c in per_chart]) for k in keys}
