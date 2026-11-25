"""Phase 0 job entry points (each one sharded, idempotent and resumable).

    python -m src.v3.data.jobs charts --shard K --shards N      # CPU array
    python -m src.v3.data.jobs audio  --shard K --shards N      # GPU array
    python -m src.v3.data.jobs offset                           # GPU, after charts+audio
    python -m src.v3.data.jobs dedup                            # GPU, after audio
    python -m src.v3.data.jobs finalize                         # CPU, after all

A rerun of any command skips work whose done-marker verifies and redoes the rest.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.v3.common import io
from src.v3.common.sharding import array_task, cpu_count, stable_bucket

PROJECT = Path(os.environ.get("OSU_PROJECT_ROOT", "/scratch/asj102/osu_project"))
EXPANSION = PROJECT / "datasets" / "expansion-20261119"
HUMAN_INDEX = EXPANSION / "human-index" / "charts.jsonl"
SYNTH_INDEX = EXPANSION / "geometry-augmentation" / "synthetic-charts.jsonl"
ROOT = Path(os.environ.get("V3_DATA_ROOT", PROJECT / "datasets" / "v3-phase0"))
REPO = Path(__file__).resolve().parents[3]


def code_sha(*rel: str) -> str:
    files = sorted(set(rel))
    return io.sha256_json({f: io.sha256_file(REPO / f) for f in files})


CHART_CODE = ("src/v3/data/osu.py", "src/v3/data/grid.py", "src/v3/data/charts.py", "src/v2/slider_path.py")
AUDIO_CODE = ("src/v3/data/audio.py",)


def load_index(synthetic: bool = True) -> list[dict[str, Any]]:
    rows = []
    for path, synth in ((HUMAN_INDEX, False),) + (((SYNTH_INDEX, True),) if synthetic else ()):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record["synthetic"] = synth
                rows.append(record)
    return rows


def inputs_sha() -> str:
    return io.sha256_json({str(p): [p.stat().st_size, int(p.stat().st_mtime)] for p in (HUMAN_INDEX, SYNTH_INDEX)})


# ---------------------------------------------------------------- charts ----

def _convert_one(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    from src.v3.data.charts import convert_chart
    base = {k: record.get(k) for k in ("chart_sha256", "group", "split", "synthetic", "parent_chart_sha256", "native_path")}
    base["audio_sha256"] = record["audio"]["sha256"]
    base["source_dataset"] = record["source"].get("source_dataset")
    base["beatmap_id"] = record["source"].get("beatmap_id")
    base["beatmap_set_id"] = record["source"].get("beatmap_set_id")
    try:
        with open(record["native_path"], encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        meta, cols = convert_chart(text)
    except Exception as error:  # a crash in one chart must not kill the shard
        meta, cols = {"status": "rejected", "errors": [f"crash: {type(error).__name__}: {error}"[:300]]}, None
    meta.update(base)
    return meta, cols


def run_charts(shard: int, shards: int, limit: int | None = None) -> None:
    from src.v3.data.charts import concat_columns
    out = ROOT / "charts" / f"shard-{shard:03d}"
    binding = {"code": code_sha(*CHART_CODE), "inputs": inputs_sha(), "shards": shards, "limit": limit}
    marker = out / "done.json"
    if io.is_done(marker, expect={"binding": binding}):
        print(f"charts shard {shard}: already complete", flush=True)
        return
    rows = [r for r in load_index() if stable_bucket(r["chart_sha256"], shards) == shard]
    if limit:
        rows = rows[:limit]
    t0 = time.time()
    metas, cols = [], []
    offset = 0
    with mp.get_context("fork").Pool(cpu_count()) as pool:
        for i, (meta, c) in enumerate(pool.imap(_convert_one, rows, chunksize=8)):
            if c is not None:
                meta["obj_offset"], meta["obj_count"] = offset, len(c["time_ms"])
                offset += len(c["time_ms"])
                cols.append(c)
            metas.append(meta)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(rows)} charts, {time.time() - t0:.0f}s", flush=True)
    io.write_npz_atomic(out / "objects.npz", **concat_columns(cols))
    io.write_jsonl_atomic(out / "charts.jsonl", metas)
    ok = sum(m["status"] == "ok" for m in metas)
    io.mark_done(marker, [out / "objects.npz", out / "charts.jsonl"], binding=binding,
                 charts=len(metas), ok=ok, objects=offset, seconds=time.time() - t0)
    print(f"charts shard {shard}: {ok}/{len(metas)} ok, {offset} objects, {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------- audio ----

def unique_audio() -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for record in load_index(synthetic=False):
        audio = record["audio"]
        seen.setdefault(audio["sha256"], audio)
    return [seen[k] for k in sorted(seen)]


def feature_paths(sha: str) -> tuple[Path, Path]:
    base = ROOT / "audio" / sha[:2]
    return base / f"{sha}.npy", base / f"{sha}.json"


def _song_done(sha: str) -> bool:
    npy, meta = feature_paths(sha)
    if not (npy.is_file() and meta.is_file()):
        return False
    try:
        record = io.read_json(meta)
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("ok") and npy.stat().st_size == record.get("npy_bytes")


def _decode_job(audio: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray | None, dict[str, Any]]:
    from src.v3.data import audio as A
    info: dict[str, Any] = {"sha256": audio["sha256"]}
    try:
        data = A.read_locator(audio)
        if io.sha256_bytes(data) != audio["sha256"]:
            raise IOError("audio bytes do not match recorded sha256")
        sniffed = A.sniff(data)
        info.update(container=sniffed.container, lame_tag=sniffed.lame_tag,
                    leading_zero_bytes=sniffed.leading_zero_bytes, nbytes=sniffed.nbytes,
                    locator_kind=audio.get("kind"))
        return audio, A.decode(data), info
    except Exception as error:
        info["error"] = f"{type(error).__name__}: {error}"[:300]
        return audio, None, info


def run_audio(shard: int, shards: int, device: str = "cuda", limit: int | None = None) -> None:
    from src.v3.data import audio as A
    out = ROOT / "audio" / f"_shard-{shard:03d}"
    binding = {"code": code_sha(*AUDIO_CODE), "shards": shards, "limit": limit}
    marker = out / "done.json"
    if io.is_done(marker, expect={"binding": binding}):
        print(f"audio shard {shard}: already complete", flush=True)
        return
    songs = [a for a in unique_audio() if stable_bucket(a["sha256"], shards) == shard]
    if limit:
        songs = songs[:limit]
    todo = [a for a in songs if not _song_done(a["sha256"])]
    print(f"audio shard {shard}: {len(songs)} songs, {len(todo)} to do", flush=True)
    t0 = time.time()
    failures = []
    with cf.ThreadPoolExecutor(max_workers=cpu_count()) as pool:
        futures = [pool.submit(_decode_job, a) for a in todo]
        for i, future in enumerate(cf.as_completed(futures)):
            audio, wav, info = future.result()
            sha = audio["sha256"]
            if wav is not None:
                try:
                    feats = A.features(wav, device)
                    npy, meta = feature_paths(sha)
                    io.write_npy_atomic(npy, feats)
                    info.update(ok=True, frames=int(feats.shape[0]), samples=int(wav.size),
                                seconds=wav.size / A.SR, sr=A.SR, hop=A.HOP, npy_bytes=npy.stat().st_size,
                                code=binding["code"])
                    io.write_json_atomic(meta, info)  # written last: the per-song done marker
                    continue
                except Exception as error:
                    info["error"] = f"features: {type(error).__name__}: {error}"[:300]
            failures.append(info)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(todo)}, {time.time() - t0:.0f}s", flush=True)
    io.write_jsonl_atomic(out / "failures.jsonl", failures)
    ok = sum(_song_done(a["sha256"]) for a in songs)
    if ok + len(failures) < len(songs):
        raise RuntimeError("audio shard incomplete; rerun to resume")
    io.mark_done(marker, [out / "failures.jsonl"], binding=binding, songs=len(songs), ok=ok,
                 failed=len(failures), seconds=time.time() - t0)
    print(f"audio shard {shard}: {ok}/{len(songs)} ok, {len(failures)} failed, {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------- songpass ----

def run_songpass(shard: int, shards: int, device: str = "cuda", limit: int | None = None) -> None:
    from src.v3.data.songpass import fingerprint, onset_envelope
    out = ROOT / "songpass" / f"_shard-{shard:03d}"
    binding = {"code": code_sha("src/v3/data/songpass.py", *AUDIO_CODE), "shards": shards, "limit": limit}
    marker = out / "done.json"
    if io.is_done(marker, expect={"binding": binding}):
        print(f"songpass shard {shard}: already complete", flush=True)
        return
    songs = [a["sha256"] for a in unique_audio() if stable_bucket(a["sha256"], shards) == shard]
    if limit:
        songs = songs[:limit]
    t0, done, missing = time.time(), 0, 0
    for i, sha in enumerate(songs):
        env_path = ROOT / "songpass" / sha[:2] / f"{sha}.env.npy"
        fp_path = ROOT / "songpass" / sha[:2] / f"{sha}.fp.npy"
        if env_path.is_file() and fp_path.is_file():
            done += 1
            continue
        if not _song_done(sha):
            missing += 1
            continue
        feats = np.load(feature_paths(sha)[0], mmap_mode="r")
        env = onset_envelope(feats, device).to("cpu").numpy().astype(np.float16)
        fp = fingerprint(feats, device).to("cpu").numpy().astype(np.int32)
        io.write_npy_atomic(fp_path, fp)
        io.write_npy_atomic(env_path, env)  # written last: per-song done marker
        done += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(songs)} {time.time() - t0:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    io.write_json_atomic(out / "counts.json", {"songs": len(songs), "done": done, "missing_features": missing})
    io.mark_done(marker, [out / "counts.json"], binding=binding)
    print(f"songpass shard {shard}: {done}/{len(songs)} done, {missing} without features, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ main ----

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=["charts", "audio", "songpass", "offset", "dedup", "finalize", "cache"])
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    args = parser.parse_args(argv)
    shard = None
    if args.job == "cache":
        shard = args.shard if args.shard is not None else array_task()
        status = ROOT / "status" / f"cache-{shard:03d}.json"
        io.write_status(status, "running")
        try:
            from src.v3.data.cache import run_cache
            run_cache(ROOT, shard)
            io.write_status(status, "complete")
        except BaseException as error:
            io.write_status(status, "failed", error=f"{type(error).__name__}: {error}"[:500],
                            traceback=traceback.format_exc()[-2000:])
            raise
        return
    if args.job in {"charts", "audio", "songpass"}:
        shard = args.shard if args.shard is not None else array_task()
        status = ROOT / "status" / f"{args.job}-{shard:03d}.json"
    else:
        status = ROOT / "status" / f"{args.job}.json"
    try:
        if shard is not None:
            io.write_status(status, "running")
            if args.job == "charts":
                run_charts(shard, args.shards, args.limit)
            elif args.job == "audio":
                run_audio(shard, args.shards, args.device, args.limit)
            else:
                run_songpass(shard, args.shards, args.device, args.limit)
        elif args.job == "offset":
            io.write_status(status, "running")
            from src.v3.data.offset_audit import run_offset
            run_offset(ROOT, args.device)
        elif args.job == "dedup":
            io.write_status(status, "running")
            from src.v3.data.dedup import run_dedup
            run_dedup(ROOT, args.device)
        else:
            io.write_status(status, "running")
            from src.v3.data.manifest import run_finalize
            run_finalize(ROOT)
        io.write_status(status, "complete")
    except BaseException as error:
        io.write_status(status, "failed", error=f"{type(error).__name__}: {error}"[:500],
                        traceback=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
