"""Training caches: per chart-shard, memory-mappable per-field arrays.

Rhythm cache (rhythm-eligible human charts, every split)
  beats:  start_ms f8, beat_ms f4, frac f4, bim i1 (beat in measure), meter i1,
          pattern i2 (0 = EMPTY, 1.. = vocab id), sv i1 (class of the slider
          velocity used by sliders starting in this beat, -1 = none)
  onsets: beat i4, slot i1, kind i1, dur i2 (slider span-duration class, or
          spinner length in beats - 1; -1 for circles), spans i1, nc i1,
          stack i1 (0 none, 1 on previous head, 2 on previous slider exit)
Placement cache (placement-eligible charts incl. teacher variants)
  objects: time_ms f8, kind i1, beat i4, tick i1, bim i1, beat_ms f4, spans i2,
           length f4, dur_ms f4, nc i1, stack i1, edge i1, x f4, y f4,
           headings f2 [N,32,2], path_ok i1, poly_ratio f4
All times include the per-stratum offset correction from the offset audit.
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.v3.common import io

from .audio import FRAME_MS
from .grid import SLOT_TICKS, TICKS, build_grid
from .osu import RedLine

SPINNER_BEATS = 32


def load_manifest(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not io.is_done(root / "manifest" / "done.json", verify=False):
        raise RuntimeError("Phase 0 manifest is not complete")
    return io.read_jsonl(root / "manifest" / "charts.jsonl"), io.read_json(root / "manifest" / "vocab.json")


class PatternMapper:
    def __init__(self, patterns: list[int]):
        self.ids = {m: i + 1 for i, m in enumerate(patterns)}
        self.patterns = np.array(patterns, np.int64)
        self.pop = np.array([bin(m).count("1") for m in patterns])
        self.oov: dict[int, int] = {}

    def __call__(self, mask: int) -> int:
        if mask == 0:
            return 0
        if mask in self.ids:
            return self.ids[mask]
        if mask not in self.oov:
            ham = np.array([bin(int(m) ^ mask).count("1") for m in self.patterns])
            same = self.pop == bin(mask).count("1")
            cost = ham + np.where(same, 0, 100)
            self.oov[mask] = int(np.argmin(cost)) + 1
        return self.oov[mask]


def _grid_for(row: dict[str, Any], audio_frames: int):
    c = row["offset_correction_ms"] or 0.0
    reds = [RedLine(t + c, b, m) for t, b, m in row["red_lines"]]
    end = audio_frames * FRAME_MS
    return build_grid(reds, row["grid_start_ms"] + c, max(end, reds[-1].time_ms + 1.0)), c


def _nearest_index(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - table[None, :]).argmin(1)


def build_shard(root: Path, shard: str, rows: list[dict[str, Any]], vocab: dict[str, Any]) -> dict[str, Any]:
    mapper = PatternMapper(vocab["patterns"])
    durations = np.array(vocab["span_duration_ticks"], np.float64)
    svs = np.array(vocab["sv_classes"], np.float64)
    with np.load(root / "charts" / shard / "objects.npz") as z:
        cols = {k: z[k] for k in z.files}
    frames_of: dict[str, int] = {}
    R = collections.defaultdict(list)
    G = collections.defaultdict(list)
    r_index, g_index = [], []
    n_beats = n_onsets = n_objects = 0
    stats = collections.Counter()
    for row in rows:
        if row["obj_offset"] is None or not (row["rhythm_ok"] or row["placement_ok"]):
            continue
        audio = row["audio_sha256"]
        if audio not in frames_of:
            meta = io.read_json(root / "audio" / audio[:2] / f"{audio}.json")
            frames_of[audio] = int(meta["frames"])
        sl = slice(row["obj_offset"], row["obj_offset"] + row["obj_count"])
        o = {k: v[sl] for k, v in cols.items()}
        try:
            grid, corr = _grid_for(row, frames_of[audio])
        except ValueError as error:
            stats[f"grid_error:{error}"[:60]] += 1
            continue
        # stacking relation to the previous object
        stack = np.where(o["stack_head"] == 1, 1, np.where(o["stack_exit"] == 1, 2, 0)).astype(np.int8)
        common = {"chart_sha256": row["chart_sha256"], "audio_sha256": audio, "split": row["split"],
                  "group": row["group"], "synthetic": row["synthetic"], "parent": row["parent_chart_sha256"],
                  "frames": frames_of[audio], "offset_correction_ms": corr}
        if row["rhythm_ok"]:
            on = (o["slot"] >= 0) & (o["beat"] >= 0) & (o["beat"] < len(grid))
            beat, slot = o["beat"][on].astype(np.int64), o["slot"][on].astype(np.int64)
            masks = np.zeros(len(grid), np.int64)
            np.bitwise_or.at(masks, beat, np.int64(1) << slot)
            pattern = np.array([mapper(int(m)) for m in masks], np.int16)
            stats["oov_beats"] += int(sum(1 for m in masks if m and int(m) not in mapper.ids))
            stats["onset_beats"] += int(np.count_nonzero(masks))
            kind = o["kind"][on]
            dur = np.full(len(kind), -1, np.int16)
            slider = kind == 1
            if slider.any():
                dur[slider] = _nearest_index(o["span_ticks"][on][slider].astype(np.float64), durations)
            spin = kind == 2
            if spin.any():
                beats = np.rint(o["span_ticks"][on][spin] / TICKS).clip(1, SPINNER_BEATS) - 1
                dur[spin] = beats.astype(np.int16)
            sv = np.full(len(grid), -1, np.int8)
            if slider.any():
                sv[beat[slider]] = _nearest_index(o["sv"][on][slider].astype(np.float64), svs).astype(np.int8)
            order = np.lexsort((slot, beat))
            R["start_ms"].append(grid.start_ms); R["beat_ms"].append(grid.beat_ms.astype(np.float32))
            R["frac"].append(grid.frac.astype(np.float32)); R["bim"].append(grid.beat_in_measure.astype(np.int8))
            R["meter"].append(grid.meter.astype(np.int8)); R["pattern"].append(pattern); R["sv"].append(sv)
            R["o_beat"].append(beat[order].astype(np.int32)); R["o_slot"].append(slot[order].astype(np.int8))
            R["o_kind"].append(kind[order].astype(np.int8)); R["o_dur"].append(dur[order])
            R["o_spans"].append(np.clip(o["spans"][on][order], 1, 127).astype(np.int8))
            R["o_nc"].append(o["new_combo"][on][order].astype(np.int8)); R["o_stack"].append(stack[on][order])
            ends = np.where(o["kind"] == 0, o["time_ms"], o["end_ms"])
            r_index.append({**common, "beats": [n_beats, len(grid)], "onsets": [n_onsets, int(on.sum())],
                            "red_signature": json.dumps(row["red_lines"]), "timing_ok": row["timing_ok"],
                            "format_version": row["format_version"],
                            "first_ms": float(o["time_ms"].min() + corr), "last_ms": float(ends.max() + corr),
                            **{k: row[k] for k in ("stars", "aim", "speed", "cs", "ar", "od", "hp",
                                                   "slider_multiplier", "beatmap_set_id", "main_bpm")}})
            n_beats += len(grid)
            n_onsets += int(on.sum())
        if row["placement_ok"]:
            n = row["obj_count"]
            valid = o["beat"] >= 0
            bim = np.where(valid, grid.beat_in_measure[np.clip(o["beat"], 0, len(grid) - 1)], 0)
            G["time_ms"].append(o["time_ms"] + corr); G["kind"].append(o["kind"])
            G["beat"].append(o["beat"]); G["tick"].append(o["tick"].astype(np.int8)); G["bim"].append(bim.astype(np.int8))
            G["beat_ms"].append(o["beat_ms"]); G["spans"].append(o["spans"]); G["length"].append(o["pixel_length"])
            G["dur_ms"].append((o["end_ms"] - o["time_ms"]).astype(np.float32)); G["nc"].append(o["new_combo"])
            G["stack"].append(stack); G["edge"].append(o["edge_touch"]); G["x"].append(o["x"]); G["y"].append(o["y"])
            G["headings"].append(o["headings"].astype(np.float16)); G["path_ok"].append(o["path_ok"])
            G["poly_ratio"].append(o["poly_ratio"])
            g_index.append({**common, "objects": [n_objects, n],
                            **{k: row[k] for k in ("stars", "aim", "speed", "cs", "ar", "od", "hp",
                                                   "slider_multiplier", "beatmap_set_id")}})
            n_objects += n
    out = root / "cache"
    files = []
    for name, data, index in (("rhythm", R, r_index), ("placement", G, g_index)):
        d = out / name / shard
        for key, parts in data.items():
            path = d / f"{key}.npy"
            io.write_npy_atomic(path, np.concatenate(parts) if parts else np.zeros(0))
            files.append(path)
        io.write_jsonl_atomic(d / "index.jsonl", index)
        files.append(d / "index.jsonl")
    return {"files": files, "rhythm_charts": len(r_index), "placement_charts": len(g_index),
            "beats": n_beats, "onsets": n_onsets, "objects": n_objects, "stats": dict(stats)}


def audio_norm(root: Path, rows: list[dict[str, Any]], n: int = 300) -> dict[str, list[float]]:
    rng = np.random.default_rng(0)
    songs = sorted({r["audio_sha256"] for r in rows if r["split"] == "train" and r["rhythm_ok"]})
    pick = rng.choice(songs, size=min(n, len(songs)), replace=False)
    s1 = s2 = None
    count = 0
    for sha in pick:
        f = np.load(root / "audio" / sha[:2] / f"{sha}.npy", mmap_mode="r")[::7].astype(np.float64)
        s1 = f.sum(0) if s1 is None else s1 + f.sum(0)
        s2 = (f ** 2).sum(0) if s2 is None else s2 + (f ** 2).sum(0)
        count += len(f)
    mean = s1 / count
    std = np.sqrt(np.maximum(s2 / count - mean ** 2, 1e-8))
    return {"mean": mean.tolist(), "std": std.tolist(), "songs": len(pick)}


def run_cache(root: Path, shard_index: int) -> None:
    t0 = time.time()
    rows, vocab = load_manifest(root)
    shards = sorted(p.name for p in (root / "charts").glob("shard-*"))
    shard = shards[shard_index]
    marker = root / "cache" / "_done" / f"{shard}.json"
    manifest_sha = io.sha256_file(root / "manifest" / "charts.jsonl")
    code = io.sha256_json({f: io.sha256_file(Path(__file__).parent / f) for f in ("cache.py", "grid.py")})
    binding = {"manifest": manifest_sha, "vocab": io.sha256_file(root / "manifest" / "vocab.json"), "code": code}
    if io.is_done(marker, verify=False, expect={"binding": binding}):
        print(f"cache {shard}: already complete", flush=True)
        return
    # Hash-verify the source chart shard: a stale or truncated input (e.g. written through a node
    # whose filesystem view diverged) must fail loudly here, not become silently wrong training data.
    if not io.is_done(root / "charts" / shard / "done.json", verify=True):
        raise RuntimeError(f"charts/{shard} fails hash verification; rerun that charts task")
    mine = [r for r in rows if r["shard"] == shard]
    result = build_shard(root, shard, mine, vocab)
    if shard_index == 0:
        norm = audio_norm(root, rows)
        io.write_json_atomic(root / "cache" / "audio_norm.json", norm)
        set_ids = sorted(int(r["beatmap_set_id"] or 0) for r in rows if r["split"] == "train" and r["rhythm_ok"])
        cuts = np.percentile(set_ids, np.linspace(0, 100, 9)[1:-1]).tolist()
        sm = collections.defaultdict(list)
        for r in rows:
            if r["split"] == "train" and r["rhythm_ok"] and r["stars"] is not None:
                sm[min(int(r["stars"]), 9)].append(r["slider_multiplier"])
        io.write_json_atomic(root / "cache" / "meta.json", {
            "era_cuts_set_id": cuts, "slider_multiplier_by_star": {k: float(np.median(v)) for k, v in sorted(sm.items())},
            "spinner_beats": SPINNER_BEATS, "slot_ticks": list(SLOT_TICKS)})
    io.mark_done(marker, [], binding=binding, **{k: v for k, v in result.items() if k != "files"},
                 seconds=time.time() - t0)
    print(f"cache {shard}: {json.dumps({k: v for k, v in result.items() if k != 'files'})} {time.time() - t0:.0f}s",
          flush=True)
