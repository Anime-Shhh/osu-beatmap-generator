"""Pre-flight sweep of the training data through the real samplers, before any GPU time is spent.

Every chart (train and val) is built at its first crop, its last crop and one random crop through the
same ``make`` and collate code the trainers use. Placement also builds each chart's teacher variant,
and timing builds each song's first and last crop. Audio content is not read: ``HeaderAudio`` takes the
frame count from the .npy header and returns zeros of the right shape. So every index and geometry path
is checked at one header read per song instead of terabytes of features. That is the class of bug that
crashed the first R launch 500 steps in: a crop timed past its audio, which would recur on every resume.

A chart fails on any exception, on non-finite inputs or targets, or when most of a crop's audio
positions fall outside its audio window (a crop the sampler should never produce).
Exits non-zero on any failure, so GPU jobs chained with --dependency=afterok never start.

    python -m src.v3.train.sweep [--workers 32] [--stages rhythm placement timing] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.v3.common import io
from src.v3.data.audio import FRAME_MS
from src.v3.train import data as D
from src.v3.train.loop import sample_rng

STAGES = ("rhythm", "placement", "timing")
_DS: dict[tuple[str, str], object] = {}   # built before the fork; workers inherit them copy-on-write


class HeaderAudio(D.Audio):
    """Audio with the real window bounds and zero content (frame counts come from .npy headers)."""

    def __init__(self, root: Path):
        super().__init__(root)
        self._shape: dict[str, tuple[int, ...]] = {}

    def window(self, sha, start_ms, end_ms):
        if sha not in self._shape:
            self._shape[sha] = self.frames(sha).shape
        n, c = self._shape[sha]
        a, b = D.window_bounds(n, start_ms, end_ms)
        return np.zeros((b - a, c), np.float32), a * FRAME_MS


def _build(stage: str, split: str, root: Path):
    train = split == "train"
    if stage == "rhythm":
        ds = D.RhythmData(root, split, augment=train)
    elif stage == "placement":
        ds = D.PlacementData(root, split, augment=train)
    else:
        from src.v3.train.stage_timing import TimingData
        ds = TimingData(root, split, seed=0, augment=train)
    ds.audio = HeaderAudio(root)
    return ds


def _finite(item: dict, where: str):
    for k, v in item.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.isfinite(v).all():
            raise ValueError(f"{where}: non-finite {k}")
        if k == "cond" and not all(np.isfinite(float(x)) for x in v.values()):
            raise ValueError(f"{where}: non-finite cond {v}")


def _outside(pos: np.ndarray, n_frames: int) -> tuple[int, int]:
    out = (pos < -1) | (pos > n_frames)
    return int(out.sum()), int(out.size)


def _check_positions(stats, stage, pos, n_frames, where):
    o, t = _outside(pos, n_frames)
    stats[f"{stage}_pos_outside"] += o
    stats[f"{stage}_pos_total"] += t
    if t and o > 0.5 * t:
        raise ValueError(f"{where}: {o}/{t} audio positions outside the {n_frames}-frame window")


def _rhythm(ds, i, stats):
    row = ds.rows[i]
    beats, _ = ds.chart_arrays(row)
    nb = ds.usable_beats(row, beats["start_ms"])
    stats["rhythm_beats_past_audio"] += len(beats["start_ms"]) - nb
    last = nb - min(ds.max_beats, nb)
    items = [ds.make(i, sample_rng(0, i, 10), start=0), ds.make(i, sample_rng(0, i, 11), start=last),
             ds.make(i, sample_rng(0, i, 12))]
    for k, it in enumerate(items):
        where = f"crop {k}"
        if len(it["audio"]) < 2:
            raise ValueError(f"{where}: {len(it['audio'])} audio frames")
        _finite(it, where)
        _check_positions(stats, "rhythm", it["slot_pos"][it["valid"]], len(it["audio"]), where)
    D.collate_rhythm(items)


def _placement(ds, i, stats):
    items = []
    for teacher in (False, True):
        row = ds.teacher[i] if teacher else ds.rows[i]
        if row is None:
            continue
        n_all = row["objects"][1]
        last = n_all - min(ds.max_objects, n_all)
        for k, start in enumerate((0, last)):
            items.append(ds.make(i, sample_rng(0, i, 20 + 2 * teacher + k), teacher=teacher, start=start))
    for k, it in enumerate(items):
        where = f"crop {k}"
        if len(it["kind"]) == 0 or len(it["audio"]) < 2:
            raise ValueError(f"{where}: {len(it['kind'])} objects, {len(it['audio'])} audio frames")
        _finite(it, where)
        _check_positions(stats, "placement", it["apos"], len(it["audio"]), where)
    stats["placement_teacher_rows"] += ds.teacher[i] is not None
    D.collate_placement(items)


def _timing(ds, j, stats):
    from src.v3.train.stage_timing import collate_timing
    song = ds.songs[j]
    lo, hi = ds.start_range(song)
    items = [ds.make(song, sample_rng(0, j, 30 + k), start_ms=s) for k, s in enumerate((lo, hi))]
    for k, it in enumerate(items):
        if len(it["audio"]) < 4:
            raise ValueError(f"crop {k}: {len(it['audio'])} audio frames")
        _finite(it, f"crop {k}")
        stats["timing_valid_frames"] += int(it["valid"].sum())
        stats["timing_frames"] += len(it["valid"])
    collate_timing(items)


def _ident(stage, ds, i):
    if stage == "timing":
        return {"audio": ds.songs[i]["audio"]}
    r = ds.rows[i]
    return {"chart": r["chart_sha256"], "audio": r["audio_sha256"]}


def _work(task):
    stage, split, lo, hi = task
    ds = _DS[stage, split]
    fn = {"rhythm": _rhythm, "placement": _placement, "timing": _timing}[stage]
    stats, fails = collections.Counter(), []
    for i in range(lo, hi):
        try:
            fn(ds, i, stats)
            stats["ok"] += 1
        except Exception as error:
            stats["failed"] += 1
            if len(fails) < 20:
                fails.append({"stage": stage, "split": split, "index": i, **_ident(stage, ds, i),
                              "error": f"{type(error).__name__}: {error}"[:400],
                              "traceback": traceback.format_exc()[-1500:]})
    return stage, split, stats, fails


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--data", default=str(D.DATA_ROOT))
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--limit", type=int, default=0, help="rows per (stage, split); 0 = all")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    root, t0 = Path(a.data), time.time()
    print(f"[sweep] open-file limit {io.raise_open_file_limit()}", flush=True)   # workers inherit it
    tasks = []
    for stage in a.stages:
        for split in ("train", "val"):
            ds = _DS[stage, split] = _build(stage, split, root)
            n = len(ds.songs if stage == "timing" else ds.rows)
            n = min(n, a.limit) if a.limit else n
            tasks += [(stage, split, s, min(n, s + a.chunk)) for s in range(0, n, a.chunk)]
            print(f"[sweep] {stage}/{split}: {n} items", flush=True)
    totals = collections.defaultdict(collections.Counter)
    failures = []
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("fork")) as ex:
        for k, (stage, split, stats, fails) in enumerate(ex.map(_work, tasks, chunksize=1)):
            totals[f"{stage}/{split}"].update(stats)
            failures += fails
            if (k + 1) % 100 == 0:
                print(f"[sweep] {k + 1}/{len(tasks)} chunks, {len(failures)} failures, {time.time() - t0:.0f}s", flush=True)
    report = {"data": str(root), "seconds": round(time.time() - t0, 1),
              "totals": {k: dict(v) for k, v in sorted(totals.items())},
              "failed": sum(v["failed"] for v in totals.values()), "failures": failures[:200]}
    for name, v in report["totals"].items():
        extra = {k: x for k, x in v.items() if k not in ("ok", "failed")}
        print(f"[sweep] {name}: ok {v.get('ok', 0)} failed {v.get('failed', 0)} {extra}", flush=True)
    for f in failures[:20]:
        print(f"[sweep] FAIL {f['stage']}/{f['split']} #{f['index']} {f.get('chart', f['audio'])[:12]}: {f['error']}",
              flush=True)
    if a.out:
        io.write_json_atomic(Path(a.out), report)
    print("SWEEP_OK" if report["failed"] == 0 else f"SWEEP_FAILED {report['failed']}", flush=True)
    sys.exit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
