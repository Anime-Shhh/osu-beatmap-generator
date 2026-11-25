"""Export round-trip gate: human chart -> v3 representation -> .osu -> parse -> compare.

Measures what the representation + exporter lose with no learned model involved:
onset times, kinds, hold end times, and rendered slider paths (osu! semantics).
    python -m src.v3.eval.roundtrip --n 200
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from src.v2.slider_path import slider_path
from src.v3.data.charts import HEADINGS, convert_chart, headings_from_path
from src.v3.data.osu import CIRCLE, SLIDER, SPINNER, RedLine, parse_chart
from src.v3.infer.export import Placed, _resample, write_osu
from src.v3.infer.legalize import Event


def chart_to_export(text: str):
    ch = parse_chart(text)
    events, placed = [], []
    for o in ch.objects:
        e = Event(time_ms=o.time_ms, kind=o.kind, beat=0, tick=0, beat_ms=o.beat_ms, new_combo=o.new_combo)
        p = Placed(o.x, o.y)
        if o.kind == SLIDER:
            path = slider_path((o.x, o.y), o.control_points, o.curve_type, o.pixel_length)
            h, _, _ = headings_from_path(path, o.pixel_length, HEADINGS)
            pts = np.r_[[np.rint([o.x, o.y])], np.rint([o.x, o.y]) + np.cumsum(h * (o.pixel_length / HEADINGS), 0)]
            e.sv, e.spans = o.sv, o.spans
            e.span_ticks = o.pixel_length / (o.sv * ch.slider_multiplier * 100.0) * 48.0
            e.end_ms = o.end_ms
            p.path = pts
        elif o.kind == SPINNER:
            e.end_ms = o.end_ms
        events.append(e)
        placed.append(p)
    return ch, events, placed


def compare(text: str) -> dict:
    ch, events, placed = chart_to_export(text)
    out, report = write_osu(events, placed, ch.red_lines, title="t", artist="a", version="rt", audio_filename="a.mp3",
                            hp=ch.hp, cs=ch.cs, od=ch.od, ar=ch.ar, slider_multiplier=ch.slider_multiplier)
    back = parse_chart(out)
    res = {"objects": len(ch.objects), "count_match": len(back.objects) == len(ch.objects)}
    if not res["count_match"]:
        return res
    t0 = np.array([o.time_ms for o in ch.objects])
    t1 = np.array([o.time_ms for o in back.objects])
    res["time_err_max"] = float(np.max(np.abs(t0 - t1)))
    res["kind_match"] = all(a.kind == b.kind for a, b in zip(ch.objects, back.objects))
    ends = [(a.end_ms, b.end_ms) for a, b in zip(ch.objects, back.objects) if a.kind != CIRCLE]
    res["end_err_max"] = float(max((abs(a - b) for a, b in ends), default=0.0))
    devs = []
    for a, b in zip(ch.objects, back.objects):
        if a.kind != SLIDER:
            continue
        pa = slider_path((a.x, a.y), a.control_points, a.curve_type, a.pixel_length)
        pb = slider_path((b.x, b.y), b.control_points, b.curve_type, b.pixel_length)
        devs.append(float(np.max(np.linalg.norm(_resample(pa, 65) - _resample(pb, 65), axis=1))))
    res["slider_dev"] = devs
    res["curve_types"] = report["curve_types"]
    stars_a = convert_chart(text)[0].get("descriptors", {}).get("stars")
    stars_b = convert_chart(out)[0].get("descriptors", {}).get("stars")
    res["stars"] = [stars_a, stars_b]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="/scratch/asj102/osu_project/datasets/expansion-20261119/human-index/charts.jsonl")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    lines = open(a.index).readlines()
    rows = [json.loads(l) for l in random.Random(0).sample(lines, a.n)]
    devs, stats = [], {"charts": 0, "count_match": 0, "time_exact": 0, "end_le2ms": 0, "kind_match": 0, "errors": 0}
    star_err = []
    for r in rows:
        try:
            res = compare(open(r["native_path"], encoding="utf-8", errors="replace").read())
        except Exception as error:
            stats["errors"] += 1
            continue
        stats["charts"] += 1
        stats["count_match"] += res["count_match"]
        if not res["count_match"]:
            continue
        stats["time_exact"] += res["time_err_max"] <= 1.0
        stats["end_le2ms"] += res["end_err_max"] <= 2.0
        stats["kind_match"] += res["kind_match"]
        devs += res["slider_dev"]
        if None not in res["stars"]:
            star_err.append(abs(res["stars"][0] - res["stars"][1]))
    d = np.array(devs)
    print(json.dumps({**stats, "sliders": len(d), "slider_le3px": float((d <= 3).mean()) if len(d) else None,
                      "slider_dev_p99": float(np.percentile(d, 99)) if len(d) else None,
                      "star_abs_err_mean": float(np.mean(star_err)) if star_err else None,
                      "star_abs_err_p95": float(np.percentile(star_err, 95)) if star_err else None}, indent=1))


if __name__ == "__main__":
    main()
