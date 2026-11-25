"""Phase 0 finalize: splits after dedup, eligibility, vocabularies, eval sets, gates.

Only this step may set ``ready_for_training``. Every decision is recorded with
its reason so a later chat (or person) can audit why a chart is excluded.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.v3.common import io

from .grid import SLOT_TICKS

MAX_OFF_GRID = 0.02
MIN_OBJECTS = 20
PATTERN_COVERAGE = 0.9995
MAX_PATTERNS = 1024
DURATION_CLASSES = 64
SV_CLASSES = 48
MAX_SPANS = 32           # >8 spans occurs in ~11% of charts; >32 is the true outlier
VAL_SELECT = 300
BAND_SAMPLE = 3000


# ------------------------------------------------------------- loading ----

def shard_dirs(root: Path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / "charts" / "shard-*")))


def load_chart_metas(root: Path, allow_partial: bool = False) -> list[dict[str, Any]]:
    dirs = shard_dirs(root)
    if not dirs:
        raise RuntimeError("no chart shards")
    expected = io.read_json(dirs[0] / "done.json")["binding"]["shards"] if (dirs[0] / "done.json").is_file() else None
    allow_partial = allow_partial or os.environ.get("V3_ALLOW_PARTIAL") == "1"  # smoke tests only
    if not allow_partial and (expected is None or len(dirs) != expected):
        raise RuntimeError(f"chart shards incomplete: {len(dirs)} present, expected {expected}")
    metas = []
    for d in dirs:
        if not (d / "done.json").is_file():
            raise RuntimeError(f"chart shard incomplete: {d}")
        for m in io.read_jsonl(d / "charts.jsonl"):
            m["shard"] = d.name
            metas.append(m)
    return metas


def iter_shard_columns(root: Path, columns: Iterable[str]):
    """Yield (shard_name, metas, {col: array}) one shard at a time (bounded memory)."""
    columns = list(columns)
    for d in shard_dirs(root):
        metas = io.read_jsonl(d / "charts.jsonl")
        with np.load(d / "objects.npz") as z:
            cols = {c: z[c] for c in columns}
        yield d.name, metas, cols


def load_columns(root: Path, columns: list[str], human_only: bool = False) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for _, metas, cols in iter_shard_columns(root, columns):
        for m in metas:
            if m["status"] != "ok" or (human_only and m["synthetic"]):
                continue
            a, n = m["obj_offset"], m["obj_count"]
            out[m["chart_sha256"]] = {c: cols[c][a:a + n] for c in columns}
    return out


# -------------------------------------------------------------- helpers ----

def reliable_red_lines(red_lines: list[list[float]]) -> bool:
    if len(red_lines) <= 1:
        return True
    times = [r[0] for r in red_lines]
    return all(b - a >= 5000.0 for a, b in zip(times, times[1:]))


def red_signature(red_lines: list[list[float]]) -> tuple:
    return tuple((round(t, 0), round(b, 3), m) for t, b, m in red_lines)


def beat_patterns(beat: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """24-bit onset masks for every beat that holds at least one on-grid onset."""
    on = (slot >= 0) & (beat >= 0)
    if not on.any():
        return np.zeros(0, np.int64)
    b = beat[on].astype(np.int64)
    bits = (np.int64(1) << slot[on].astype(np.int64))
    order = np.argsort(b, kind="stable")
    b, bits = b[order], bits[order]
    uniq, start = np.unique(b, return_index=True)
    return np.bitwise_or.reduceat(bits, start)


def main_bpm(meta: dict[str, Any]) -> float:
    reds = meta["red_lines"]
    if not reds:
        return 0.0
    end = meta.get("last_end_ms", reds[-1][0])
    spans = [((reds[i + 1][0] if i + 1 < len(reds) else end) - reds[i][0], reds[i][1]) for i in range(len(reds))]
    return 60000.0 / max(spans)[1]


# ------------------------------------------------------------- finalize ----

def run_finalize(root: Path) -> None:
    t0 = time.time()
    out = root / "manifest"
    for stage in ("offsets", "dedup"):
        if not io.is_done(root / stage / "done.json", verify=False):
            raise RuntimeError(f"{stage} has not completed")
    metas = load_chart_metas(root)
    print(f"{len(metas)} chart records", flush=True)

    # Audio availability.
    audio_ok = {}
    for f in glob.glob(str(root / "audio" / "??" / "*.json")):
        r = io.read_json(f)
        audio_ok[r["sha256"]] = bool(r.get("ok"))

    # Offsets.
    off_summary = io.read_json(root / "offsets" / "summary.json")
    corrections = {k: v["correction_ms"] for k, v in off_summary["strata"].items()}
    quarantined_songs = set(off_summary["quarantined_songs"])
    chart_off = {r["chart_sha256"]: r for r in io.read_jsonl(root / "offsets" / "charts.jsonl")}

    # Dedup -> split changes (eval groups that leak into another split are removed from eval).
    leak_groups = set()
    for p in io.read_jsonl(root / "dedup" / "pairs.jsonl"):
        if p["query_split"] != p["match_split"]:
            leak_groups.add(p["query_group"])
    for m in metas:
        m["split_v3"] = m["split"]
        if m["split"] != "train" and m["group"] in leak_groups:
            m["split_v3"] = "quarantine-dup"

    # Per-song red-line consensus for timing labels.
    sigs = collections.defaultdict(collections.Counter)
    for m in metas:
        if m["status"] == "ok" and not m["synthetic"]:
            sigs[m["audio_sha256"]][red_signature(m["red_lines"])] += 1

    # Per-chart maximum slider repeat count (the rhythm vocabulary holds spans 1..MAX_SPANS).
    max_spans: dict[str, int] = {}
    for _, ms, cols in iter_shard_columns(root, ["spans"]):
        for m in ms:
            if m["status"] == "ok":
                a, n = m["obj_offset"], m["obj_count"]
                max_spans[m["chart_sha256"]] = int(cols["spans"][a:a + n].max()) if n else 0

    rows = []
    reasons = collections.Counter()
    for m in metas:
        sha, audio = m["chart_sha256"], m["audio_sha256"]
        why = []
        if m["status"] != "ok":
            why.append("rejected:" + (m["errors"][0].split(":")[0] if m.get("errors") else "?"))
        if not audio_ok.get(audio):
            why.append("audio_missing")
        base = not why
        off = chart_off.get(sha, {})
        rhythm_why = list(why)
        if base:
            if m["synthetic"]:
                rhythm_why.append("synthetic")
            if m["off_grid_frac"] > MAX_OFF_GRID or m["end_off_grid_frac"] > MAX_OFF_GRID:
                rhythm_why.append("off_grid")
            if audio in quarantined_songs:
                rhythm_why.append("offset_quarantine")
            if off.get("chart_outlier"):
                rhythm_why.append("offset_chart_outlier")
            if m["n_objects"] < MIN_OBJECTS:
                rhythm_why.append("too_few_objects")
            if m["overlaps"] > 0:
                rhythm_why.append("overlapping_objects")
            if max_spans.get(sha, 0) > MAX_SPANS:
                rhythm_why.append(f"spans_over_{MAX_SPANS}")
        rhythm_ok = not rhythm_why
        sig_counts = sigs.get(audio)
        consensus = bool(sig_counts) and sig_counts.most_common(1)[0][0] == red_signature(m["red_lines"])
        timing_ok = rhythm_ok and consensus and reliable_red_lines(m["red_lines"])
        placement_ok = base and (m["n_sliders"] == 0 or m["path_failures"] <= 0.02 * m["n_sliders"])
        for r in rhythm_why:
            reasons[r] += 1
        d = m.get("descriptors", {})
        rows.append({
            "chart_sha256": sha, "shard": m["shard"], "obj_offset": m.get("obj_offset"), "obj_count": m.get("obj_count"),
            "split": m["split_v3"], "group": m["group"], "audio_sha256": audio, "synthetic": m["synthetic"],
            "parent_chart_sha256": m.get("parent_chart_sha256"),
            "rhythm_ok": rhythm_ok, "timing_ok": timing_ok, "placement_ok": placement_ok,
            "exclusion": rhythm_why, "offset_correction_ms": corrections.get(off.get("stratum"), 0.0),
            "stratum": off.get("stratum"), "grid_start_ms": m.get("grid_start_ms"), "red_lines": m.get("red_lines"),
            "stars": d.get("stars"), "aim": d.get("aim"), "speed": d.get("speed"),
            "cs": m.get("cs"), "ar": m.get("ar"), "od": m.get("od"), "hp": m.get("hp"),
            "slider_multiplier": m.get("slider_multiplier"), "tick_rate": m.get("tick_rate"),
            "beatmap_set_id": m.get("beatmap_set_id"), "format_version": m.get("format_version"),
            "n_objects": m.get("n_objects"), "n_sliders": m.get("n_sliders"), "main_bpm": main_bpm(m) if base else None,
            "kiai_ms": m.get("kiai_ms"), "creator": m.get("creator"), "diff_name": m.get("diff_name"),
        })
    by_sha = {r["chart_sha256"]: r for r in rows}
    print(f"eligibility built ({time.time() - t0:.0f}s)", flush=True)

    # Vocabularies (train, rhythm-eligible human charts only) + round-trip statistics.
    pattern_counts = collections.Counter()
    multi_red_counts = collections.Counter()
    dur_counts = collections.Counter()
    sv_counts = collections.Counter()
    rt = collections.Counter()
    for _, ms, cols in iter_shard_columns(root, ["beat", "slot", "kind", "span_ticks", "sv", "resid24",
                                                 "end_resid48", "path_ok", "path_err"]):
        for m in ms:
            r = by_sha.get(m["chart_sha256"])
            if r is None or m["status"] != "ok" or m["synthetic"]:
                continue
            a, n = m["obj_offset"], m["obj_count"]
            sl = slice(a, a + n)
            kind = cols["kind"][sl]
            if r["split"] == "val" and r["rhythm_ok"]:
                res = np.abs(cols["resid24"][sl])
                rt["objects"] += n
                rt["objects_le1ms"] += int(np.sum(res <= 1.0))
                rt["objects_on_grid"] += int(np.sum(cols["slot"][sl] >= 0))
                held = kind != 0
                rt["held"] += int(held.sum())
                rt["held_end_le2ms"] += int(np.sum(np.abs(cols["end_resid48"][sl][held]) <= 2.0))
            if r["split"] == "val":
                sliders = (kind == 1)
                ok = sliders & (cols["path_ok"][sl] == 1)
                rt["sliders"] += int(sliders.sum())
                rt["sliders_path_ok"] += int(ok.sum())
                rt["sliders_le3px"] += int(np.sum(cols["path_err"][sl][ok] <= 3.0))
            if r["split"] != "train" or not r["rhythm_ok"]:
                continue
            pats = beat_patterns(cols["beat"][sl], cols["slot"][sl])
            pattern_counts.update(pats.tolist())
            if len(r["red_lines"]) > 1:
                multi_red_counts.update(pats.tolist())
            sliders = kind == 1
            dur_counts.update(np.rint(cols["span_ticks"][sl][sliders]).astype(int).tolist())
            sv_counts.update(np.round(cols["sv"][sl][sliders] * 20).astype(int).tolist())
    total = sum(pattern_counts.values())
    vocab, covered = [], 0
    for mask, c in pattern_counts.most_common(MAX_PATTERNS):
        vocab.append(int(mask))
        covered += c
        if covered / total >= PATTERN_COVERAGE:
            break
    vset = set(vocab)
    multi_total = sum(multi_red_counts.values())
    multi_cov = sum(c for k, c in multi_red_counts.items() if k in vset) / max(1, multi_total)
    dur_total = sum(dur_counts.values())
    durations = [int(k) for k, _ in dur_counts.most_common(DURATION_CLASSES)]
    dur_cov = sum(dur_counts[k] for k in durations) / max(1, dur_total)
    svs = sorted(int(k) for k, _ in sv_counts.most_common(SV_CLASSES))
    sv_cov = sum(sv_counts[k] for k in svs) / max(1, sum(sv_counts.values()))
    vocab_json = {"slot_ticks": list(SLOT_TICKS),
                  "patterns": vocab, "pattern_coverage": covered / total, "pattern_beats": total,
                  "pattern_coverage_multi_red": multi_cov,
                  "span_duration_ticks": durations, "duration_coverage": dur_cov,
                  "sv_classes": [k / 20 for k in svs], "sv_coverage": sv_cov}
    print(f"vocab: {len(vocab)} patterns cover {covered / total:.5f} (multi-red {multi_cov:.5f}); "
          f"durations {dur_cov:.4f}; sv {sv_cov:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # Eval sets.
    rng = random.Random(20261123)
    val_groups = collections.defaultdict(list)
    for r in rows:
        if r["split"] == "val" and r["rhythm_ok"] and r["stars"] is not None:
            val_groups[r["group"]].append(r)
    set_ids = sorted(int(r["beatmap_set_id"] or 0) for g in val_groups.values() for r in g)
    era_cuts = np.percentile(set_ids, [33.3, 66.7]) if set_ids else [0, 0]
    strata = collections.defaultdict(list)
    for g in sorted(val_groups):
        r = rng.choice(sorted(val_groups[g], key=lambda x: x["chart_sha256"]))
        key = (min(int(r["stars"]), 6), int(np.digitize(r["main_bpm"] or 0, [120, 160, 200])),
               int(np.digitize(int(r["beatmap_set_id"] or 0), era_cuts)))
        strata[key].append(r["chart_sha256"])
    for v in strata.values():
        rng.shuffle(v)
    val300, keys = [], sorted(strata)
    while len(val300) < VAL_SELECT and any(strata.values()):
        for k in keys:
            if strata[k] and len(val300) < VAL_SELECT:
                val300.append(strata[k].pop())
    train_groups = collections.defaultdict(list)
    for r in rows:
        if r["split"] == "train" and r["rhythm_ok"]:
            train_groups[r["group"]].append(r["chart_sha256"])
    band = [rng.choice(sorted(v)) for _, v in sorted(train_groups.items())]
    rng.shuffle(band)
    band = band[:BAND_SAMPLE]

    # Gates.
    gates = {
        "objects_on_grid_frac": rt["objects_on_grid"] / max(1, rt["objects"]),
        "objects_le1ms_frac": rt["objects_le1ms"] / max(1, rt["objects"]),
        "held_end_le2ms_frac": rt["held_end_le2ms"] / max(1, rt["held"]),
        "sliders_path_ok_frac": rt["sliders_path_ok"] / max(1, rt["sliders"]),
        "sliders_le3px_frac": rt["sliders_le3px"] / max(1, rt["sliders_path_ok"]),
        "pattern_coverage_multi_red": multi_cov,
    }
    passed = {
        "objects_on_grid": gates["objects_on_grid_frac"] >= 0.995,
        "slider_paths": gates["sliders_le3px_frac"] >= 0.99,
        "pattern_coverage_multi_red": multi_cov >= 0.995,
    }
    counts = collections.Counter()
    for r in rows:
        for flag in ("rhythm_ok", "timing_ok", "placement_ok"):
            if r[flag]:
                counts[f"{flag}:{r['split']}:{'synthetic' if r['synthetic'] else 'human'}"] += 1
    ready = {"timing": passed["objects_on_grid"], "rhythm": passed["objects_on_grid"] and passed["pattern_coverage_multi_red"],
             "placement": passed["slider_paths"]}
    summary = {"ready_for_training": ready, "gates": gates, "gate_passed": passed, "counts": dict(counts),
               "rhythm_exclusions": dict(reasons), "leak_groups_removed_from_eval": len(leak_groups),
               "offset_quarantined_songs": len(quarantined_songs), "offset_corrections_ms": corrections,
               "val300": len(val300), "human_band": len(band), "seconds": time.time() - t0}
    io.write_jsonl_atomic(out / "charts.jsonl", rows)
    io.write_json_atomic(out / "vocab.json", vocab_json)
    io.write_json_atomic(out / "eval_sets.json", {"val300": val300, "human_band_train": band,
                                                   "era_cuts_set_id": [float(x) for x in era_cuts]})
    io.write_json_atomic(out / "summary.json", summary)
    io.mark_done(out / "done.json", [out / f for f in ("charts.jsonl", "vocab.json", "eval_sets.json", "summary.json")],
                 ready_for_training=ready)
    print(json.dumps(summary, indent=1)[:4000], flush=True)
