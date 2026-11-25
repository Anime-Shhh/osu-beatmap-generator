"""osu-dreamer baselines on the v3 eval sets, generated under the same rules as v3.

Variants (all upstream inference artifacts, loaded with upstream ``load_inference``):
    dreamer-v8       published v8 weights (pretraining data unknown: possible overlap with any eval song)
    dreamer-ft       v8 fine-tuned 2000 steps on the pilot manifest (633 songs; 0 overlap with val300)
    dreamer-scratch  upstream architecture trained from scratch on the same pilot manifest

Fairness, so differences come from the models and not the harness:
  * audio: the same ffmpeg binary and flags as the v3 features (and therefore the offset audit and the
    references' audio time), resampled to Dreamer's rate, then Dreamer's own ``make_spec``. MP3 decoders
    disagree by ~25 ms on encoder padding; letting Dreamer decode separately would bias F1@10 against it.
  * conditioning: the reference chart's (stars, AR, OD, CS, HP), the request v3 gets.
  * candidates/selection: the same budget (16) and the same rule (``harness.select_best``), plus the same
    unfiltered first-sample view (``<system>-raw``).

    python -m src.v3.eval.dreamer --system dreamer-v8 --model PATH [--shard i --shards 8]
"""
from __future__ import annotations

import argparse
import contextlib
import io as pyio
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from src.v3.common import io
from src.v3.common.sharding import array_task, stable_bucket
from src.v3.data import audio as A
from src.v3.eval.harness import DEFAULT_SET, _root, eval_rows, prepare_system, select_best, write_outputs

PROJECT = Path("/scratch/asj102/osu_project")
MODELS = {
    "dreamer-v8": PROJECT / "imported/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt",
    "dreamer-ft": PROJECT / "runs/dreamer-63885268/final.pt",
    "dreamer-scratch": PROJECT / "runs/dreamer-scratch-20261118/scratch-inference.pt",
}


def dreamer_wave(data: bytes, sr: int) -> np.ndarray:
    """Mono float32 at ``sr`` with exactly the v3 decode command (only the output rate differs)."""
    cmd = [A.FFMPEG, "-nostdin", "-hide_banner", "-v", "error", "-i", "pipe:0",
           "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def run_dreamer(root: Path, set_name: str, system: str, model_path: Path, shard: int, shards: int,
                n: int = 16, chunk: int = 4, steps: int = 8, seed: int = 0) -> None:
    import torch
    from osu_dreamer.data.beatmap.decode import Metadata, decode_beatmap
    from osu_dreamer.data.load_audio import SR, make_spec
    from osu_dreamer.models.inference.artifact import load_inference
    from src.v3.infer.pipeline import check_export, strip_text
    binding = {"mode": "dreamer", "model": [str(model_path), io.sha256_file(model_path)], "candidates": n,
               "steps": steps, "seed": seed, "audio": f"v3 ffmpeg decode @ {SR} Hz + upstream make_spec"}
    out, raw_out = prepare_system(root, set_name, system, binding)
    status = root / "evals" / set_name / "status" / f"generate-{system}-{shard:03d}.json"
    io.write_status(status, "running", system=system)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_inference(model_path).to(device).eval()
    rows = [r for r in eval_rows(root, set_name) if stable_bucket(r["chart_sha256"], shards) == shard]
    done = failed = 0
    for i, r in enumerate(rows):
        sha = r["chart_sha256"]
        if (out / f"{sha}.json").exists() and (raw_out / f"{sha}.json").exists():
            done += 1
            continue
        t0 = time.time()
        stars = float(r["stars"])
        candidates, best, raw = [], None, None
        try:
            spec = make_spec(dreamer_wave(A.read_locator(r["audio_locator"]), SR))
            audio = torch.tensor(spec, device=device).float()
            request = [stars, float(r["ar"]), float(r["od"]), float(r["cs"]), float(r["hp"])]
            torch.manual_seed(seed * 1_000_003 + int(sha[:8], 16) % (1 << 30))
            for start in range(0, n, chunk):
                k = min(chunk, n - start)
                labels = torch.tensor([request] * k, device=device)
                with torch.no_grad():
                    signals, out_labels = model.sample(audio, labels, num_steps=steps)
                for j in range(k):
                    idx = start + j
                    try:
                        with contextlib.redirect_stdout(pyio.StringIO()), contextlib.redirect_stderr(pyio.StringIO()):
                            text = decode_beatmap(Metadata("audio.mp3", sha[:12], "eval", f"{system} {idx}"),
                                                  out_labels[j].float().cpu().numpy(), signals[j].float().cpu().numpy())
                        candidates.append({"sample": idx, "text": text, **check_export(text)})
                    except Exception as error:   # upstream decoder failures are candidate failures
                        candidates.append({"sample": idx, "error": f"decode: {type(error).__name__}: {error}"[:300]})
            best, raw = select_best(candidates, stars), candidates[0]
            record = {"request": request, "candidates": [strip_text(c) for c in candidates]}
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            record = {"error": f"{type(error).__name__}: {error}"[:500], "traceback": traceback.format_exc()[-3000:]}
        record["seconds"] = time.time() - t0
        write_outputs(out, raw_out, sha, best, raw, record)
        done += 1
        failed += best is None
        print(f"[{system}] {i + 1}/{len(rows)} {sha[:12]} "
              f"{'ok' if best else 'FAILED ' + str(record.get('error', 'no valid candidate'))[:80]} "
              f"valid {sum(bool(c.get('valid')) for c in candidates)}/{len(candidates)} {record['seconds']:.1f}s", flush=True)
    io.write_status(status, "complete", system=system, charts=done, failed=failed)


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--system", required=True, choices=sorted(MODELS))
    p.add_argument("--model", default=None, help="override the variant's artifact path")
    p.add_argument("--data", default=None)
    p.add_argument("--set", default=DEFAULT_SET)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--steps", type=int, default=8, help="upstream default")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    run_dreamer(_root(a.data), a.set, a.system, Path(a.model) if a.model else MODELS[a.system],
                array_task(a.shard), a.shards, n=a.candidates, steps=a.steps, seed=a.seed)


if __name__ == "__main__":
    main()
