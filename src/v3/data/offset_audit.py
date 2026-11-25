"""Audio/chart offset audit, stratified by how the audio was encoded.

For every human chart, cross-correlate its object onset times against the song's
onset envelope over lags of +-150 ms (1 ms steps, linear interpolation), giving a
best lag and a peak sharpness. Constant decoder/encoder delays differ by stratum
(source x container x LAME tag x .osu format bucket), so corrections are
*per stratum*, relative to the pooled reference: a constant shared by all strata
is a property of the feature definition and is learned, not corrected.

Lag sign: ``lag > 0`` means audio onsets occur later than the chart's object
times; labels are aligned to features by ``t + correction``.
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np

from src.v3.common import io

from .audio import FRAME_MS

LAGS_MS = np.arange(-150.0, 150.5, 1.0)
SHARP_MIN = 4.0          # robust z of the correlation peak counted as confident
# Per-song lag estimates scatter with an IQR of ~5-8 ms (mappers do not always follow
# the loudest onset), while real misalignment (encoder delay, wrong encoding) is >= 25 ms.
SONG_OUTLIER_MS = 30.0
CHART_OUTLIER_MS = 20.0


def version_bucket(v: int) -> str:
    if v < 5:
        return "v<5"
    if v < 14:
        return "v5-13"
    if v == 14:
        return "v14"
    return "v128+"


def chart_lags(env, times_ms: np.ndarray, device: str = "cuda") -> tuple[float, float]:
    import torch
    if len(times_ms) < 8:
        return float("nan"), 0.0
    pos = (torch.as_tensor(times_ms, device=device, dtype=torch.float64)[None, :]
           + torch.as_tensor(LAGS_MS, device=device, dtype=torch.float64)[:, None]) / FRAME_MS
    pos = pos.clamp(0, env.shape[0] - 1.001)
    lo = pos.floor().long()
    frac = (pos - lo).float()
    vals = env[lo] * (1 - frac) + env[lo + 1] * frac
    corr = vals.mean(1).cpu().numpy()
    k = int(np.argmax(corr))
    lag = LAGS_MS[k]
    if 0 < k < len(corr) - 1:  # parabolic refinement
        a, b, c = corr[k - 1], corr[k], corr[k + 1]
        denom = a - 2 * b + c
        if denom < 0:
            lag += 0.5 * (a - c) / denom
    med = np.median(corr)
    mad = np.median(np.abs(corr - med)) * 1.4826 + 1e-6
    return float(lag), float((corr[k] - med) / mad)


def run_offset(root: Path, device: str = "cuda") -> None:
    from .manifest import load_chart_metas, load_columns
    out = root / "offsets"
    t0 = time.time()
    metas = [m for m in load_chart_metas(root) if m["status"] == "ok" and not m["synthetic"]]
    by_song = collections.defaultdict(list)
    for m in metas:
        by_song[m["audio_sha256"]].append(m)
    times = load_columns(root, ["time_ms"], human_only=True)
    rows, songs = [], []
    for n, (sha, charts) in enumerate(sorted(by_song.items())):
        env_path = root / "songpass" / sha[:2] / f"{sha}.env.npy"
        audio_meta = root / "audio" / sha[:2] / f"{sha}.json"
        if not env_path.is_file() or not audio_meta.is_file():
            continue
        import torch
        env = torch.as_tensor(np.load(env_path).astype(np.float32), device=device)
        ainfo = io.read_json(audio_meta)
        lags = []
        for m in charts:
            lag, sharp = chart_lags(env, times[m["chart_sha256"]]["time_ms"], device)
            stratum = f'{m["source_dataset"]}|{ainfo.get("container")}|lame={ainfo.get("lame_tag")}|{version_bucket(m["format_version"])}'
            rows.append({"chart_sha256": m["chart_sha256"], "audio_sha256": sha, "lag_ms": lag,
                         "sharpness": sharp, "stratum": stratum})
            if np.isfinite(lag) and sharp >= SHARP_MIN:
                lags.append((lag, sharp, stratum))
        if lags:
            arr = np.array([l for l, _, _ in lags])
            songs.append({"audio_sha256": sha, "lag_ms": float(np.median(arr)), "n_confident": len(arr),
                          "spread_ms": float(np.ptp(arr)),
                          "stratum": collections.Counter(s for _, _, s in lags).most_common(1)[0][0]})
        if (n + 1) % 2000 == 0:
            print(f"  offsets {n + 1}/{len(by_song)} songs {time.time() - t0:.0f}s", flush=True)
    pooled = float(np.median([s["lag_ms"] for s in songs]))
    strata = {}
    for key in sorted({s["stratum"] for s in songs}):
        vals = np.array([s["lag_ms"] for s in songs if s["stratum"] == key])
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        # Only correct strata with enough evidence and a material, tight shift.
        correct = len(vals) >= 30 and abs(med - pooled) >= 2.0
        strata[key] = {"songs": int(len(vals)), "median_lag_ms": float(med), "iqr_ms": float(q3 - q1),
                       "correction_ms": float(med - pooled) if correct else 0.0}
    corr = {k: v["correction_ms"] for k, v in strata.items()}
    song_lag = {s["audio_sha256"]: s for s in songs}
    quarantined = sorted(s["audio_sha256"] for s in songs
                         if abs(s["lag_ms"] - (pooled + corr[s["stratum"]])) > SONG_OUTLIER_MS)
    for r in rows:
        s = song_lag.get(r["audio_sha256"])
        r["chart_outlier"] = bool(s and np.isfinite(r["lag_ms"]) and r["sharpness"] >= SHARP_MIN
                                  and abs(r["lag_ms"] - s["lag_ms"]) > CHART_OUTLIER_MS)
    io.write_jsonl_atomic(out / "charts.jsonl", rows)
    io.write_jsonl_atomic(out / "songs.jsonl", songs)
    summary = {"pooled_reference_lag_ms": pooled, "strata": strata, "quarantined_songs": quarantined,
               "n_songs_confident": len(songs), "n_charts": len(rows),
               "chart_outliers": int(sum(r["chart_outlier"] for r in rows)),
               "rule": f"song quarantined if |lag - stratum| > {SONG_OUTLIER_MS} ms (confident sharpness >= {SHARP_MIN})",
               "seconds": time.time() - t0}
    io.write_json_atomic(out / "summary.json", summary)
    io.mark_done(out / "done.json", [out / "charts.jsonl", out / "songs.jsonl", out / "summary.json"])
    print(json.dumps({k: v for k, v in summary.items() if k != "quarantined_songs"}, indent=1)[:3000], flush=True)
    print(f"quarantined songs: {len(quarantined)}", flush=True)
