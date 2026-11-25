"""Entry point for every v3 training stage (single GPU or torchrun DDP).

    torchrun --nproc_per_node=4 -m src.v3.train.run rhythm --run-dir RUN_DIR --size base --max-steps 60000
Resubmitting the same command resumes from RUN_DIR/last.pt (exact resume).

torchrun parses *every* "--opt" in its argv, including the script's, with argparse prefix matching:
an option here that is a prefix of a torchrun option (e.g. "--run" vs "--run-path") aborts the launch,
and a unique prefix would be silently swallowed. tests/v3/test_train_loop.py guards this.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.v3.train.loop import Trainer, TrainConfig


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("stage", choices=["rhythm", "placement", "timing"])
    p.add_argument("--run-dir", dest="run", required=True)
    p.add_argument("--data", default=os.environ.get("V3_DATA_ROOT", "/scratch/asj102/osu_project/datasets/v3-phase0"))
    p.add_argument("--size", default="base")
    p.add_argument("--max-steps", type=int, default=0, help="0 = derive from --budget-gpu-hours")
    p.add_argument("--budget-gpu-hours", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher-p", type=float, default=0.3, help="placement only: teacher-variant swap probability")
    p.add_argument("--val-charts", type=int, default=32)
    p.add_argument("--no-bf16", action="store_true")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    root = Path(a.data)
    ready = json.load(open(root / "manifest" / "summary.json"))["ready_for_training"]
    if not ready.get(a.stage, False) and os.environ.get("V3_ALLOW_PARTIAL") != "1":
        raise SystemExit(f"Phase 0 gates did not mark '{a.stage}' ready_for_training: {ready}")
    if a.stage == "rhythm":
        from src.v3.train.stage_rhythm import RhythmStage
        stage, mode = RhythmStage(root, a.size, a.seed, a.val_charts), "max"
    elif a.stage == "placement":
        from src.v3.train.stage_placement import PlacementStage
        stage, mode = PlacementStage(root, a.size, a.seed, a.val_charts, teacher_p=a.teacher_p), "min"
    else:
        from src.v3.train.stage_timing import TimingStage
        stage, mode = TimingStage(root, a.size, a.seed, a.val_charts), "max"
    cfg = TrainConfig(run_dir=a.run, max_steps=a.max_steps, batch_size=a.batch, accum=a.accum, lr=a.lr,
                      warmup=a.warmup, val_every=a.val_every, log_every=a.log_every, num_workers=a.workers,
                      seed=a.seed, select_mode=mode, bf16=not a.no_bf16,
                      extra={"stage": a.stage, "data": str(root), "budget_gpu_hours": a.budget_gpu_hours})
    Trainer(stage, cfg).fit()


if __name__ == "__main__":
    main(sys.argv[1:])
