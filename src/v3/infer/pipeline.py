"""Audio file -> playable .osz with the v3 cascade (timing -> rhythm -> placement).

    python -m src.v3.infer.pipeline --audio song.mp3 --out OUT_DIR --stars 4.5 \
        --timing RUN_T --rhythm RUN_R --placement RUN_G [--candidates 4x4]
Every candidate is exported, re-parsed and checked; selection uses only checker-
valid candidates, ranked by |measured stars - requested|. The report keeps all of
them (no silent filtering).
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

from src.v3.common import io
from src.v3.data import audio as A
from src.v3.data.charts import convert_chart
from src.v3.data.grid import SLOT_TICKS, TICKS, build_grid
from src.v3.data.osu import RedLine
from src.v3.infer.export import Placed, write_osu
from src.v3.infer.legalize import events_from_tokens
from src.v3.infer.tempo import RedLineFit, fit_red_lines
from src.v3.models.placement import PlacementModel, sample as place_sample
from src.v3.models.rhythm import ATTRS, RhythmModel
from src.v3.models.timing import TimingModel
from src.v3.train.data import AUDIO_OFFSETS_MS, era_bucket

SLOT_FRAC = np.asarray(SLOT_TICKS, np.float64) / TICKS


def load_stage(run: Path, cls, extra_kwargs=None):
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    binding = ckpt["binding"]
    model = cls(**(extra_kwargs or {}), **binding["stage"]["model"])
    model.load_state_dict(ckpt["ema"])
    return model.eval(), binding


def mapped_region(feats: np.ndarray) -> tuple[float, float]:
    rms = feats[:, 129].astype(np.float32)
    thr = np.median(rms) - 3.0
    on = np.nonzero(rms > thr)[0]
    if len(on) == 0:
        return 0.0, len(feats) * A.FRAME_MS
    return float(on[0] * A.FRAME_MS), float(on[-1] * A.FRAME_MS)


def rhythm_batch(grid, norm_feats, cond, n, device):
    L = len(grid)
    slot_ms = grid.start_ms[:, None] + SLOT_FRAC[None, :] * grid.beat_ms[:, None]
    valid = SLOT_FRAC[None, :] < grid.frac[:, None] - 1e-6
    valid[:, 0] = True
    rep = lambda x: torch.as_tensor(np.asarray(x))[None].expand(n, *np.asarray(x).shape).contiguous().to(device)
    b = {"audio": rep(norm_feats.astype(np.float32)), "slot_pos": rep((slot_ms / A.FRAME_MS).astype(np.float32)),
         "valid": rep(valid), "beat_pos": rep(np.arange(L, dtype=np.float32)),
         "bim": rep(grid.beat_in_measure.astype(np.int64)), "mask": torch.ones(n, L, dtype=torch.bool, device=device)}
    for k in ("stars", "ar", "od", "cs", "hp", "bpm", "sm"):
        b[k] = torch.full((n,), float(cond[k]), device=device)
    b["era"] = torch.full((n,), int(cond["era"]), dtype=torch.long, device=device)
    return b


def placement_batch(events, grid, norm_feats, cond, sm, n, device):
    t = np.array([e.time_ms for e in events])
    kind = np.array([e.kind for e in events])
    bl = np.array([e.beat_ms for e in events])
    end = np.array([e.end_ms for e in events])
    length = np.array([e.sv * sm * 100.0 * e.span_ticks / 48.0 if e.kind == 1 else 0.0 for e in events])
    dur = np.where(kind == 0, 0.0, end - t)
    prev_gap = np.diff(t, prepend=t[0] - 1000.0)
    next_gap = np.r_[t[1:] - end[:-1], 1000.0]
    scal = np.stack([np.log1p(np.maximum(prev_gap, 0) / 100.0), np.log1p(np.maximum(next_gap, 0) / 100.0),
                     np.log2(np.maximum(prev_gap, 1) / bl + 1 / 48), np.log2(np.maximum(next_gap, 1) / bl + 1 / 48),
                     np.log1p(dur / bl), np.where(kind == 1, length / 100.0, 0.0)], -1).astype(np.float32)
    beat = np.array([e.beat for e in events])
    tick = np.array([e.tick for e in events])
    rep = lambda x, dt=None: torch.as_tensor(np.asarray(x, dtype=dt))[None].expand(n, *np.asarray(x).shape).contiguous().to(device)
    b = {"audio": rep(norm_feats, np.float32), "apos": rep((t[:, None] + AUDIO_OFFSETS_MS[None]) / A.FRAME_MS, np.float32),
         "kind": rep(kind, np.int64), "nc": rep([int(e.new_combo) for e in events], np.int64),
         "stack": rep([e.stack for e in events], np.int64), "spans": rep(np.clip([e.spans for e in events], 0, 32), np.int64),
         "tick": rep(np.clip(tick, 0, 47), np.int64), "bim": rep(np.clip(grid.beat_in_measure[beat], 0, 15), np.int64),
         "scal": rep(scal), "beat_pos": rep(beat + tick / 48.0, np.float32), "length": rep(length, np.float32),
         "mask": torch.ones(n, len(events), dtype=torch.bool, device=device)}
    for k in ("stars", "ar", "od", "cs", "hp"):
        b[k] = torch.full((n,), float(cond[k]), device=device)
    b["sm"] = torch.full((n,), float(sm), device=device)
    b["era"] = torch.full((n,), int(cond["era"]), dtype=torch.long, device=device)
    b["source"] = torch.zeros(n, dtype=torch.long, device=device)       # always generate in human mode
    return b


def check_export(text: str) -> dict:
    """Re-parse an exported map and apply the shared legality rule (src.v3.eval.metrics.validity)."""
    from src.v3.eval.metrics import summarize_converted, validity
    meta, cols = convert_chart(text)
    out = {"status": meta["status"], "errors": meta.get("errors", []), "stars": meta.get("descriptors", {}).get("stars"),
           "objects": meta.get("n_objects", 0)}
    if cols is None:
        out["valid"] = False
        return out
    v = validity(summarize_converted(meta, cols))
    # a generated slider we cannot re-evaluate is a candidate failure (unlike a human chart's budget overrun)
    out.update(v, valid=bool(v["valid"] and v["path_failures"] == 0))
    return out


def main_bpm(lines, region: tuple[float, float]) -> float:
    """BPM of the red line covering the most of the mapped region (same definition as training)."""
    lo, hi = region
    spans = []
    for i, l in enumerate(lines):
        a = max(l.time_ms, lo) if i else lo
        b = min(lines[i + 1].time_ms, hi) if i + 1 < len(lines) else hi
        spans.append((max(0.0, b - a), l.beat_ms))
    return 60000.0 / max(spans)[1]


def load_models(rhythm_run: Path, placement_run: Path, timing_run: Path | None = None, device=None) -> dict:
    """Load every stage once (EMA weights from best.pt) plus the data tables they were trained with."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmodel, gb = load_stage(placement_run, PlacementModel)
    root = Path(gb["config"]["extra"]["data"])
    vocab = io.read_json(root / "manifest" / "vocab.json")
    patterns = vocab["patterns"]
    K = max(bin(m).count("1") for m in patterns)
    rmodel, _ = load_stage(rhythm_run, RhythmModel, {"patterns": patterns, "K": K, "n_dur": len(vocab["span_duration_ticks"]),
                                                      "n_sv": len(vocab["sv_classes"])})
    tmodel = load_stage(timing_run, TimingModel)[0].to(device) if timing_run is not None else None
    return {"device": device, "timing": tmodel, "rhythm": rmodel.to(device), "placement": gmodel.to(device),
            "norm": io.read_json(root / "cache" / "audio_norm.json"), "meta": io.read_json(root / "cache" / "meta.json"),
            "vocab": vocab, "runs": {"timing": str(timing_run) if timing_run else None, "rhythm": str(rhythm_run),
                                     "placement": str(placement_run)}}


def generate_candidates(M: dict, feats: np.ndarray, *, stars: float, ar: float, od: float, cs: float, hp: float,
                        era: int = 8, red_lines=None, n_rhythm: int = 4, n_place: int = 4, seed: int = 0,
                        cfg: float = 1.5, title: str = "", artist: str = "", audio_name: str = "audio.mp3") -> dict:
    """Raw (un-normalized) features [T, 130] -> every exported, re-checked candidate plus the selection.

    ``red_lines`` (audio-time RedLine list) replaces the timing model: the oracle-grid evaluation mode.
    """
    device = M["device"]
    torch.manual_seed(seed)
    nf = (feats.astype(np.float32) - np.asarray(M["norm"]["mean"], np.float32)) / np.asarray(M["norm"]["std"], np.float32)
    region = mapped_region(feats)
    if red_lines is None:
        if M["timing"] is None:
            raise ValueError("no timing model loaded and no red_lines given")
        beat, down, tempo = M["timing"].activations(nf, device)
        lines = fit_red_lines(beat, down, tempo, region)
    else:
        lines = [RedLineFit(float(l.time_ms), float(l.beat_ms), int(l.meter)) for l in red_lines]
    grid = build_grid([RedLine(l.time_ms, l.beat_ms, l.meter) for l in lines], 0.0, len(feats) * A.FRAME_MS)
    sm = M["meta"]["slider_multiplier_by_star"].get(str(min(int(stars), 9)), 1.4)
    cond = {"stars": stars, "ar": ar, "od": od, "cs": cs, "hp": hp, "bpm": main_bpm(lines, region), "sm": sm,
            "era": min(era, 8)}
    vocab, rmodel, gmodel = M["vocab"], M["rhythm"], M["placement"]
    patterns = vocab["patterns"]
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        state, logp = rmodel.sample(rhythm_batch(grid, nf, cond, n_rhythm, device), steps=48, cfg=cfg, generator=gen)
    na = {name: rmodel.na_id(name) for name in ATTRS}
    candidates = []
    for r in range(n_rhythm):
        events = events_from_tokens(state["pattern"][r].cpu().numpy(), state["attrs"][r].cpu().numpy(),
                                    logp["sv"][r].float().cpu().numpy(), grid, patterns, vocab["span_duration_ticks"],
                                    vocab["sv_classes"], na)
        if len(events) < 4:
            candidates.append({"rhythm": r, "error": "too few events"})
            continue
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            g = place_sample(gmodel, placement_batch(events, grid, nf, cond, sm, n_place, device), steps=32, generator=gen)
        for p in range(n_place):
            placed = [Placed(float(g["head_px"][p, i, 0]), float(g["head_px"][p, i, 1]),
                             g["path_px"][p, i].float().cpu().numpy() if e.kind == 1 else None)
                      for i, e in enumerate(events)]
            try:
                text, rep = write_osu(events, placed, lines, title=title or "untitled", artist=artist or "Unknown",
                                      version=f"v3 {stars:.1f}* r{r}p{p}", audio_filename=audio_name, hp=hp, cs=cs,
                                      od=od, ar=ar, slider_multiplier=sm)
            except ValueError as error:
                candidates.append({"rhythm": r, "place": p, "error": str(error)})
                continue
            chk = check_export(text)
            fit = rep["slider_fit_px"]
            candidates.append({"rhythm": r, "place": p, "text": text, **chk,
                               "max_displacement_px": float(g["displacement"][p].max()),
                               "median_displacement_px": float(g["displacement"][p].median()),
                               "slider_fit_p99_px": float(np.percentile(fit, 99)) if fit else 0.0,
                               "curve_types": rep["curve_types"]})
    from src.v3.eval.harness import select_best
    best = select_best(candidates, stars)
    return {"lines": lines, "region": region, "cond": cond, "candidates": candidates, "best": best}


def strip_text(c: dict | None) -> dict | None:
    return None if c is None else {k: v for k, v in c.items() if k != "text"}


def generate(audio_path: Path, out_dir: Path, *, stars: float, ar: float, od: float, cs: float, hp: float,
             timing_run: Path, rhythm_run: Path, placement_run: Path, n_rhythm: int = 4, n_place: int = 4,
             seed: int = 0, title: str = "", artist: str = "", era: int = 8, cfg: float = 1.5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    M = load_models(rhythm_run, placement_run, timing_run)
    data = audio_path.read_bytes()
    info = A.sniff(data)
    feats = A.features(A.decode(data), str(M["device"])).astype(np.float32)
    audio_name = "audio" + {"mp3": ".mp3", "ogg": ".ogg", "opus": ".opus", "wav": ".wav", "flac": ".flac"}.get(info.container, ".mp3")
    title = title or audio_path.stem
    res = generate_candidates(M, feats, stars=stars, ar=ar, od=od, cs=cs, hp=hp, era=era, n_rhythm=n_rhythm,
                              n_place=n_place, seed=seed, cfg=cfg, title=title, artist=artist, audio_name=audio_name)
    best = res["best"]
    report = {"audio": str(audio_path), "container": info.container, "lame_tag": info.lame_tag,
              "requested": {"stars": stars, "ar": ar, "od": od, "cs": cs, "hp": hp}, "cond": res["cond"],
              "red_lines": [[l.time_ms, l.beat_ms, l.meter] for l in res["lines"]], "region_ms": res["region"],
              "candidates": [strip_text(c) for c in res["candidates"]], "selected": strip_text(best),
              "within_0.2_stars": bool(best and abs(best["stars"] - stars) <= 0.2), "models": M["runs"]}
    if best is not None:
        osu_name = f"{(artist or 'Unknown')} - {title} (osu-v3) [{best['stars']:.2f}].osu".replace("/", "_")
        (out_dir / osu_name).write_text(best["text"], encoding="utf-8")
        with zipfile.ZipFile(out_dir / f"{title}-v3.osz", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(osu_name, best["text"])
            z.writestr(audio_name, data)
    io.write_json_atomic(out_dir / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stars", type=float, required=True)
    p.add_argument("--ar", type=float, default=None)
    p.add_argument("--od", type=float, default=None)
    p.add_argument("--cs", type=float, default=4.0)
    p.add_argument("--hp", type=float, default=None)
    p.add_argument("--timing", required=True)
    p.add_argument("--rhythm", required=True)
    p.add_argument("--placement", required=True)
    p.add_argument("--candidates", default="4x4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default="")
    p.add_argument("--artist", default="")
    p.add_argument("--cfg", type=float, default=1.5)
    a = p.parse_args()
    nr, ng = (int(x) for x in a.candidates.split("x"))
    s = a.stars
    rep = generate(Path(a.audio), Path(a.out), stars=s, ar=a.ar if a.ar is not None else min(9.5, s + 3.5),
                   od=a.od if a.od is not None else min(9.0, s + 3.0), cs=a.cs,
                   hp=a.hp if a.hp is not None else min(7.0, s + 2.0), timing_run=Path(a.timing),
                   rhythm_run=Path(a.rhythm), placement_run=Path(a.placement), n_rhythm=nr, n_place=ng, seed=a.seed,
                   title=a.title, artist=a.artist, cfg=a.cfg)
    print(json.dumps({k: v for k, v in rep.items() if k != "candidates"}, indent=1))


if __name__ == "__main__":
    main()
