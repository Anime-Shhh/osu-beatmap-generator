"""Phase 1 evaluation harness. Generation and scoring are decoupled, so every system is scored identically.

    evals/<set>/<system>/<chart_sha>.osu    one map per eval chart (v3 modes, Dreamer baselines, human ceiling)
    evals/<set>/<system>/<chart_sha>.json   sidecar, written last: {"offset_ms": shift into audio time, ...}
                                            {"error": ...} = the system failed on this chart (scored as a failure)
                                            {"na": ...}    = not applicable (e.g. no second human chart), excluded
    evals/<set>/scores/<system>.json|.jsonl summary with song-grouped bootstrap intervals / per-chart rows
    evals/band/<set>.pkl                    pooled human-band distributions (cached)

Times: every map is moved into *audio time* before matching. Reference charts get their stratum's
offset correction (manifest ``offset_correction_ms``; old-format MP3s are ~15 ms late), generated maps
are already in audio time (offset 0), and a human-ceiling chart gets its own stratum's correction.

Distribution metrics are sample-size biased (JS/W1 shrink as samples grow), so a system is read against
``human_ref``: the same eval set's human references measured against the same band, at the same n.

    python -m src.v3.eval.harness generate --system v3-oracle --rhythm RUN_R --placement RUN_G [--timing RUN_T]
    python -m src.v3.eval.harness ceiling                  # human-vs-human system "human-other"
    python -m src.v3.eval.harness score --system v3-oracle
    python -m src.v3.eval.harness table
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import pickle
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.v3.common import io
from src.v3.common.sharding import array_task, cpu_count, stable_bucket
from src.v3.eval.match import short_pair_recall
from src.v3.eval.metrics import (distribution_distance, map_distributions, pooled, reference_metrics,
                                 summarize_chart, summary_stats, validity)

DEFAULT_SET = "val300"
STAR_BUCKETS = (0, 2, 3, 4, 5, 6, 7, 99)
MACRO = ("f1_10", "f1_25", "f1_50", "typed_f1_25", "valid", "ref_valid", "abs_star_err")
CEILING_MAX_DSTARS = 0.5


# ------------------------------------------------------------------ inputs --

def _root(arg: str | None) -> Path:
    from src.v3.data.jobs import ROOT
    return Path(arg) if arg else ROOT


def eval_rows(root: Path, set_name: str = DEFAULT_SET) -> list[dict[str, Any]]:
    """Manifest rows of an eval set (fixed order) joined with their native .osu paths."""
    from src.v3.data.cache import load_manifest
    from src.v3.data.jobs import load_index
    shas = io.read_json(root / "manifest" / "eval_sets.json")[set_name]
    rows, _ = load_manifest(root)
    by_sha = {r["chart_sha256"]: r for r in rows}
    index = {r["chart_sha256"]: r for r in load_index(synthetic=False)}
    return [dict(by_sha[s], native_path=index[s]["native_path"], audio_locator=index[s]["audio"]) for s in shas]


def system_dir(root: Path, set_name: str, system: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", system):
        raise ValueError(f"bad system name {system!r}")
    return root / "evals" / set_name / system


def _write_text_atomic(path: Path, text: str) -> None:
    with io.atomic_path(path) as tmp:
        Path(tmp).write_text(text, encoding="utf-8")


def _summary_in_audio_time(text: str, offset_ms: float) -> dict | None:
    s = summarize_chart(text)
    if s is not None and offset_ms:
        s["time"] = s["time"] + offset_ms
        s["end"] = s["end"] + offset_ms
    return s


# -------------------------------------------------------------- generation --

def prepare_system(root: Path, set_name: str, system: str, binding: dict) -> tuple[Path, Path]:
    """Create <system> (reranked selection) and <system>-raw (first sample, unfiltered) under one binding."""
    binding = json.loads(json.dumps(binding))   # compare in JSON form (tuples come back from disk as lists)
    dirs = []
    for name in (system, system + "-raw"):
        out = system_dir(root, set_name, name)
        out.mkdir(parents=True, exist_ok=True)
        bfile = out / "binding.json"
        if bfile.exists() and io.read_json(bfile) != {**binding, "view": name}:
            raise RuntimeError(f"{out} holds maps from a different model/config; use a new --system name")
        io.write_json_atomic(bfile, {**binding, "view": name})
        dirs.append(out)
    return dirs[0], dirs[1]


def write_outputs(out: Path, raw_out: Path, sha: str, best: dict | None, raw: dict | None, record: dict) -> None:
    """Maps first, sidecars last (a sidecar marks the chart done). Candidates carry "text" when exported."""
    from src.v3.infer.pipeline import strip_text
    for d, c, label in ((out, best, "no checker-valid candidate"), (raw_out, raw, "first sample failed")):
        side = {**record, "offset_ms": 0.0, "selected": strip_text(c)}
        if c is not None and c.get("text"):
            _write_text_atomic(d / f"{sha}.osu", c["text"])
        else:
            side.setdefault("error", (c or {}).get("error") or label)
        io.write_json_atomic(d / f"{sha}.json", side)


def select_best(candidates: list[dict], stars: float) -> dict | None:
    """The shared reranking rule (v3 and every baseline): checker-valid, closest measured stars."""
    valid = [c for c in candidates if c.get("valid") and c.get("stars") is not None]
    return min(valid, key=lambda c: abs(c["stars"] - stars)) if valid else None

def run_generate(root: Path, set_name: str, system: str, rhythm_run: Path, placement_run: Path,
                 timing_run: Path | None, shard: int, shards: int, n_rhythm: int, n_place: int, seed: int,
                 cfg: float) -> None:
    """GPU job: one map per eval chart with the chart's own conditioning. Resumes per chart."""
    import torch
    from src.v3.data.osu import RedLine
    from src.v3.infer.pipeline import generate_candidates, load_models, strip_text
    from src.v3.train.data import era_bucket
    mode = "oracle-grid" if timing_run is None else "timing-model"
    binding = {"mode": mode, "candidates": [n_rhythm, n_place], "seed": seed, "cfg": cfg,
               "runs": {k: (str(v), io.sha256_file(v / "best.pt")) if v else None
                        for k, v in (("timing", timing_run), ("rhythm", rhythm_run), ("placement", placement_run))}}
    out, raw_out = prepare_system(root, set_name, system, binding)
    status = root / "evals" / set_name / "status" / f"generate-{system}-{shard:03d}.json"
    io.write_status(status, "running", system=system)
    rows = [r for r in eval_rows(root, set_name) if stable_bucket(r["chart_sha256"], shards) == shard]
    M = load_models(rhythm_run, placement_run, timing_run)
    cuts = io.read_json(root / "cache" / "meta.json")["era_cuts_set_id"]
    done = failed = 0
    for i, r in enumerate(rows):
        sha = r["chart_sha256"]
        if (out / f"{sha}.json").exists() and (raw_out / f"{sha}.json").exists():
            done += 1
            continue
        t0 = time.time()
        a = r["audio_sha256"]
        best = raw = None
        try:
            feats = np.load(root / "audio" / a[:2] / f"{a}.npy").astype(np.float32)
            red = None
            if timing_run is None:   # oracle grid: the reference chart's red lines, moved into audio time
                corr = float(r.get("offset_correction_ms") or 0.0)
                red = [RedLine(t + corr, b, int(m)) for t, b, m in r["red_lines"]]
            res = generate_candidates(M, feats, stars=float(r["stars"]), ar=float(r["ar"]), od=float(r["od"]),
                                      cs=float(r["cs"]), hp=float(r["hp"]),
                                      era=era_bucket(r.get("beatmap_set_id"), cuts), red_lines=red,
                                      n_rhythm=n_rhythm, n_place=n_place,
                                      seed=seed * 1_000_003 + int(sha[:8], 16) % (1 << 30), cfg=cfg,
                                      title=sha[:12], artist="eval")
            best, raw = res["best"], (res["candidates"][0] if res["candidates"] else None)
            record = {"cond": res["cond"], "red_lines": [[l.time_ms, l.beat_ms, l.meter] for l in res["lines"]],
                      "candidates": [strip_text(c) for c in res["candidates"]]}
        except Exception as error:   # one chart must not kill the shard; the failure is scored, not hidden
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            record = {"error": f"{type(error).__name__}: {error}"[:500], "traceback": traceback.format_exc()[-3000:]}
        record["seconds"] = time.time() - t0
        write_outputs(out, raw_out, sha, best, raw, record)
        done += 1
        failed += best is None
        print(f"[{system}] {i + 1}/{len(rows)} {sha[:12]} {'ok' if best else 'FAILED ' + str(record.get('error', 'no valid candidate'))[:80]}"
              f" {record['seconds']:.1f}s", flush=True)
    io.write_status(status, "complete", system=system, charts=done, failed=failed)


# ------------------------------------------------------- human ceiling ----

def _mapper(row: dict) -> str:
    """Guest difficulties are named "<mapper>'s ..."; otherwise the set creator mapped it."""
    m = re.match(r"\s*(.+?)['’]s?\s", row.get("diff_name") or "")
    return (m.group(1) if m else row.get("creator") or "").strip().lower()


def run_ceiling(root: Path, set_name: str) -> None:
    """System "human-other": for each eval chart, another human chart on the same audio within 0.5 stars,
    preferring a different mapper. Charts with no such partner are "na" (coverage is reported)."""
    from src.v3.data.cache import load_manifest
    from src.v3.data.jobs import load_index
    out = system_dir(root, set_name, "human-other")
    out.mkdir(parents=True, exist_ok=True)
    rows, _ = load_manifest(root)
    native = {r["chart_sha256"]: r["native_path"] for r in load_index(synthetic=False)}
    by_audio = collections.defaultdict(list)
    for r in rows:
        if not r["synthetic"] and r["chart_sha256"] in native and r.get("stars") is not None:
            by_audio[r["audio_sha256"]].append(r)
    stats = collections.Counter()
    for r in eval_rows(root, set_name):
        sha = r["chart_sha256"]
        cands = [c for c in by_audio[r["audio_sha256"]]
                 if c["chart_sha256"] != sha and abs(c["stars"] - r["stars"]) <= CEILING_MAX_DSTARS]
        if not cands:
            io.write_json_atomic(out / f"{sha}.json", {"na": "no other human chart within 0.5 stars"})
            stats["na"] += 1
            continue
        mine = _mapper(r)
        c = min(cands, key=lambda c: (_mapper(c) == mine, abs(c["stars"] - r["stars"])))
        _write_text_atomic(out / f"{sha}.osu", Path(native[c["chart_sha256"]]).read_text(encoding="utf-8", errors="replace"))
        same = _mapper(c) == mine
        io.write_json_atomic(out / f"{sha}.json", {"offset_ms": float(c.get("offset_correction_ms") or 0.0),
                                                   "source_chart": c["chart_sha256"], "same_mapper": same,
                                                   "dstars": float(c["stars"] - r["stars"])})
        stats["same_mapper" if same else "different_mapper"] += 1
    io.write_json_atomic(out / "binding.json", {"mode": "human-ceiling", "max_dstars": CEILING_MAX_DSTARS})
    print(f"human-other: {dict(stats)}", flush=True)


# ----------------------------------------------------------------- scoring --

def _ref_summary(row: dict) -> dict | None:
    text = Path(row["native_path"]).read_text(encoding="utf-8", errors="replace")
    return _summary_in_audio_time(text, float(row.get("offset_correction_ms") or 0.0))


def _score_one(args) -> dict:
    row, sdir = args
    sha = row["chart_sha256"]
    base = {"chart": sha, "group": row["group"], "stars_ref": row["stars"]}
    side_path = Path(sdir) / f"{sha}.json"
    if not side_path.exists():
        return {**base, "status": "missing"}
    side = io.read_json(side_path)
    if side.get("na"):
        return {**base, "status": "na"}
    ref = _ref_summary(row)
    ref_dist = map_distributions(ref)
    rv = validity(ref)   # the same checks on the human reference: the checker's human baseline
    base.update({"ref_valid": float(rv["valid"]), "ref_heads_out": rv["heads_out"], "ref_bodies_out": rv["bodies_out"],
                 "ref_overlaps": rv["overlaps"], "ref_path_failures": rv["path_failures"]})
    _, ref_short = short_pair_recall(np.zeros(0), ref["time"], ref["beat_pos"], 25.0)   # failures still count
    fail = {**base, "status": "failed", "error": side.get("error", "unparseable map"), "f1_10": 0.0, "f1_25": 0.0,
            "f1_50": 0.0, "typed_f1_25": 0.0, "valid": 0.0, "abs_star_err": float("nan"),
            "matched_25": 0, "n_gen": 0, "n_ref": len(ref["time"]), "short_pairs_got": 0,
            "short_pairs_total": ref_short, "ref_dist": ref_dist}
    osu = Path(sdir) / f"{sha}.osu"
    if "error" in side or not osu.exists():
        return fail
    gen = _summary_in_audio_time(osu.read_text(encoding="utf-8", errors="replace"), float(side.get("offset_ms") or 0.0))
    if gen is None or len(gen["time"]) < 2:
        return fail
    m = reference_metrics(gen, ref)
    v = validity(gen)
    star_err = (gen["stars"] - row["stars"]) if gen["stars"] is not None else float("nan")
    return {**base, "status": "ok", **m, "valid": float(v["valid"]), "heads_out": v["heads_out"],
            "bodies_out": v["bodies_out"], "body_overshoot_px": v["body_overshoot_px"], "path_failures": v["path_failures"], "overlaps": v["overlaps"], "off_grid_frac": v["off_grid_frac"],
            "stars_gen": gen["stars"], "star_err": star_err, "abs_star_err": abs(star_err),
            "selected": side.get("selected"), "dist": map_distributions(gen), "ref_dist": ref_dist}


def _band_one(path: str):
    s = summarize_chart(Path(path).read_text(encoding="utf-8", errors="replace"))
    return None if s is None or len(s["time"]) < 3 else map_distributions(s)


def human_band(root: Path, workers: int) -> dict:
    """Pooled distributions of the train human band (cached; keyed by the band's chart list)."""
    from src.v3.data.jobs import load_index
    shas = io.read_json(root / "manifest" / "eval_sets.json")["human_band_train"]
    key = io.sha256_json(shas)
    cache = root / "evals" / "band" / "human_band_train.pkl"
    if cache.exists():
        with open(cache, "rb") as handle:
            stored = pickle.load(handle)
        if stored["key"] == key:
            return stored["pooled"]
    native = {r["chart_sha256"]: r["native_path"] for r in load_index(synthetic=False)}
    with mp.get_context("fork").Pool(workers) as pool:
        dists = [d for d in pool.map(_band_one, [native[s] for s in shas], chunksize=16) if d is not None]
    band = pooled(dists)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with io.atomic_path(cache) as tmp:
        with open(tmp, "wb") as handle:
            pickle.dump({"key": key, "pooled": band, "charts": len(dists)}, handle)
    return band


def _boot_groups(groups: list[str], reps: int, seed: int = 0):
    """Song-grouped bootstrap: resample whole songs (charts of one song are not independent)."""
    uniq = sorted(set(groups))
    members = collections.defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        pick = rng.integers(len(uniq), size=len(uniq))
        yield np.concatenate([members[uniq[k]] for k in pick])


def _ci(values: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"), float("nan"))


def _pooled_rates(rs: list[dict]) -> dict[str, float]:
    m = sum(r.get("matched_25", 0) for r in rs)
    n = sum(r.get("n_gen", 0) + r.get("n_ref", 0) for r in rs)
    got = sum(r.get("short_pairs_got", 0) for r in rs)
    tot = sum(r.get("short_pairs_total", 0) for r in rs)
    return {"f1_25_pooled": 2 * m / n if n else float("nan"), "short_pair_recall": got / tot if tot else float("nan")}


def aggregate(rows: list[dict], band: dict, reps: int = 1000, dist_reps: int = 200) -> dict:
    scored = [r for r in rows if r["status"] in ("ok", "failed")]
    groups = [r["group"] for r in scored]
    macro = {}
    for k in MACRO:
        vals = np.array([r.get(k, float("nan")) for r in scored], float)
        boots = [float(np.nanmean(vals[idx])) for idx in _boot_groups(groups, reps)] if len(vals) else []
        lo, hi = _ci(boots)
        macro[k] = {"mean": float(np.nanmean(vals)) if len(vals) else float("nan"), "lo": lo, "hi": hi}
    pooled_rates = _pooled_rates(scored)
    boot_rates = [_pooled_rates([scored[i] for i in idx]) for idx in _boot_groups(groups, reps)]
    pooled_ci = {k: {"value": v, **dict(zip(("lo", "hi"), _ci([b[k] for b in boot_rates])))}
                 for k, v in pooled_rates.items()}
    by_bucket = {}
    for lo, hi in zip(STAR_BUCKETS[:-1], STAR_BUCKETS[1:]):
        errs = [r["abs_star_err"] for r in scored if lo <= r["stars_ref"] < hi and math.isfinite(r.get("abs_star_err", math.nan))]
        by_bucket[f"{lo}-{hi if hi < 99 else ''}"] = {"n": len(errs), "mae": float(np.mean(errs)) if errs else float("nan")}
    ok = [r for r in scored if r["status"] == "ok"]
    dist = {}
    if ok:
        gen_p, ref_p = pooled([r["dist"] for r in ok]), pooled([r["ref_dist"] for r in ok])
        sys_d, ref_d = distribution_distance(gen_p, band), distribution_distance(ref_p, band)
        # JS/W1 estimates are biased by sample size and by bootstrap duplication, so percentile intervals on
        # raw distances mislead. The gated quantity is the paired excess over the human references (same
        # resampled songs for both), where that bias largely cancels: 0 = human-level, > 0 = less human.
        boots = collections.defaultdict(list)
        for idx in _boot_groups([r["group"] for r in ok], dist_reps, seed=1):
            sub = [ok[i] for i in idx]
            a = distribution_distance(pooled([r["dist"] for r in sub]), band)
            b = distribution_distance(pooled([r["ref_dist"] for r in sub]), band)
            for k in a:
                boots[k].append(a[k] - b[k])
        dist["system_vs_band"] = sys_d
        dist["human_ref_vs_band"] = ref_d
        dist["excess_over_human"] = {k: {"value": sys_d[k] - ref_d[k], **dict(zip(("lo", "hi"), _ci(boots[k])))}
                                     for k in sys_d}
        dist["system_vs_refs"] = distribution_distance(gen_p, ref_p)
        dist["system_stats"], dist["human_ref_stats"] = summary_stats(gen_p), summary_stats(ref_p)
        disp = [r["selected"].get("median_displacement_px") for r in ok if r.get("selected")]
        dmax = [r["selected"].get("max_displacement_px") for r in ok if r.get("selected")]
        if any(d is not None for d in dmax):
            dist["projection"] = {"median_of_median_px": float(np.median([d for d in disp if d is not None])) if any(d is not None for d in disp) else None,
                                  "p99_of_max_px": float(np.percentile([d for d in dmax if d is not None], 99))}
    counts = collections.Counter(r["status"] for r in rows)
    return {"n": dict(counts), "macro": macro, "pooled": pooled_ci, "star_mae_by_bucket": by_bucket,
            "distribution": dist, "bootstrap": {"unit": "song group", "reps": reps, "dist_reps": dist_reps}}


def run_score(root: Path, set_name: str, system: str, allow_missing: bool = False) -> dict:
    sdir = system_dir(root, set_name, system)
    rows = eval_rows(root, set_name)
    workers = cpu_count()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_score_one, [(r, str(sdir)) for r in rows], chunksize=4)
    missing = sum(r["status"] == "missing" for r in results)
    if missing and not allow_missing:
        raise RuntimeError(f"{system}: {missing}/{len(rows)} charts have no output yet (generation unfinished?)")
    band = human_band(root, workers)
    summary = {"system": system, "set": set_name, "binding": io.read_json(sdir / "binding.json") if (sdir / "binding.json").exists() else None,
               **aggregate(results, band)}
    sc = root / "evals" / set_name / "scores"
    sc.mkdir(parents=True, exist_ok=True)
    io.write_jsonl_atomic(sc / f"{system}.jsonl", [{k: v for k, v in r.items() if k not in ("dist", "ref_dist")} for r in results])
    io.write_json_atomic(sc / f"{system}.json", summary)
    return summary


# ------------------------------------------------------------------- table --

def _fmt(cell) -> str:
    if not isinstance(cell, dict):
        return "—"
    v = cell.get("mean", cell.get("value"))
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    lo, hi = cell.get("lo"), cell.get("hi")
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None and math.isfinite(lo) else f"{v:.3f}"


def run_table(root: Path, set_name: str) -> str:
    sc = root / "evals" / set_name / "scores"
    summaries = [io.read_json(p) for p in sorted(sc.glob("*.json"))]
    lines = [f"# {set_name}: song-grouped bootstrap 95% intervals", "",
             "| system | n ok/failed/na | F1@10 | F1@25 | F1@25 pooled | ≤1/4 pair recall | valid | human valid (same charts) | star MAE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        n = s["n"]
        lines.append(f"| {s['system']} | {n.get('ok', 0)}/{n.get('failed', 0)}/{n.get('na', 0)} | {_fmt(s['macro']['f1_10'])} | "
                     f"{_fmt(s['macro']['f1_25'])} | {_fmt(s['pooled']['f1_25_pooled'])} | {_fmt(s['pooled']['short_pair_recall'])} | "
                     f"{_fmt(s['macro']['valid'])} | {_fmt(s['macro'].get('ref_valid'))} | {_fmt(s['macro']['abs_star_err'])} |")
    keys = ["snap_js", "trigram_js", "spacing_w1", "turn_w1", "stack_rate_diff", "curve_js", "slider_len_w1",
            "density_w1", "sv_changes_w1", "nc_downbeat_diff", "overshoot_w1"]
    raw = lambda d: " | ".join("—" if not math.isfinite(d.get(k, math.nan)) else f"{d[k]:.3f}" for k in keys)
    with_d = [s for s in summaries if "system_vs_band" in s.get("distribution", {})]
    lines += ["", "Distance to the train human band (point estimates; sample-size biased, so read each system "
              "against human_ref = the same eval charts' human references):", "",
              "| system | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    if with_d:
        ref = max(with_d, key=lambda s: s["n"].get("ok", 0))
        lines.append(f"| **human_ref** (n={ref['n'].get('ok', 0)}) | " + raw(ref["distribution"]["human_ref_vs_band"]) + " |")
    for s in with_d:
        lines.append(f"| {s['system']} | " + raw(s["distribution"]["system_vs_band"]) + " |")
    lines += ["", "Paired excess over human references on the same songs (0 = human-level; interval excludes 0 "
              "→ significant), song-grouped bootstrap 95%:", "",
              "| system | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for s in with_d:
        ex = s["distribution"]["excess_over_human"]
        lines.append(f"| {s['system']} | " + " | ".join(_fmt(ex.get(k)) for k in keys) + " |")
    text = "\n".join(lines) + "\n"
    _write_text_atomic(root / "evals" / set_name / "table.md", text)
    return text


# --------------------------------------------------------------------- cli --

def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("command", choices=["generate", "ceiling", "score", "table"])
    p.add_argument("--data", default=None)
    p.add_argument("--set", default=DEFAULT_SET)
    p.add_argument("--system")
    p.add_argument("--rhythm-run")
    p.add_argument("--placement-run")
    p.add_argument("--timing-run", default=None, help="omit for the oracle-grid mode (reference red lines)")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--candidates", default="4x4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--allow-missing", action="store_true")
    a = p.parse_args(argv)
    root = _root(a.data)
    if a.command == "generate":
        nr, ng = (int(x) for x in a.candidates.split("x"))
        run_generate(root, a.set, a.system, Path(a.rhythm_run), Path(a.placement_run),
                     Path(a.timing_run) if a.timing_run else None, array_task(a.shard), a.shards, nr, ng, a.seed, a.cfg)
    elif a.command == "ceiling":
        run_ceiling(root, a.set)
    elif a.command == "score":
        s = run_score(root, a.set, a.system, a.allow_missing)
        print(json.dumps({k: s[k] for k in ("system", "n", "macro", "pooled")}, indent=1))
    else:
        print(run_table(root, a.set))


if __name__ == "__main__":
    main()
