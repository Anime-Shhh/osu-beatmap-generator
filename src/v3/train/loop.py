"""Crash-safe, exactly resumable training loop shared by every v3 stage.

Guarantees
* Full-state checkpoint (model, EMA, optimizer, step, every rank's RNG) every
  ``ckpt_minutes`` and at every validation, written atomically; ``last.pt`` and
  ``prev.pt`` are kept, plus ``best.pt`` (EMA weights) chosen by the stage metric.
* Data order is a pure function of (seed, global sample index): resuming at
  step s replays exactly the batches an uninterrupted run would have seen.
* Graceful stop: a ``STOP`` file in the run dir (touched by the sbatch trap on
  Slurm's pre-timeout USR1, works through torchrun), or SIGUSR1/SIGTERM in-process,
  saves a checkpoint and, under Slurm, requeues the job so it resumes by itself.
* A changed config/data/code binding refuses to resume (use a new run dir).
* EMA updates once per *optimizer* step; DDP runs without find_unused_parameters
  so a dead branch errors instead of silently never training (Dreamer's bug).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
import torch.distributed as dist

from src.v3.common import io


@dataclass
class TrainConfig:
    run_dir: str
    max_steps: int
    batch_size: int                 # per GPU, per micro-step
    accum: int = 1
    lr: float = 3e-4
    warmup: int = 1000
    decay_frac: float = 0.2         # warmup-stable-decay: final fraction decays linearly
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    ckpt_minutes: float = 20.0
    val_every: int = 2000
    log_every: int = 50
    num_workers: int = 6
    seed: int = 0
    bf16: bool = True
    select_mode: str = "max"        # stage metric "select": higher (max) or lower (min) is better
    extra: dict[str, Any] = field(default_factory=dict)


class Stage(Protocol):
    def build_model(self) -> torch.nn.Module: ...
    def dataset(self) -> torch.utils.data.Dataset: ...
    def collate(self, items: list) -> dict: ...
    def loss(self, model: torch.nn.Module, batch: dict, step: int) -> tuple[torch.Tensor, dict[str, float]]: ...
    def validate(self, model: torch.nn.Module, step: int) -> dict[str, float]: ...
    def binding(self) -> dict[str, Any]: ...


class _BatchSampler(torch.utils.data.Sampler):
    """Infinite, rank-disjoint, index-addressed batches starting at a micro-step."""

    def __init__(self, start_micro: int, rank: int, world: int, batch: int):
        self.start, self.rank, self.world, self.batch = start_micro, rank, world, batch

    def __iter__(self):
        m = self.start
        while True:
            base = (m * self.world + self.rank) * self.batch
            yield list(range(base, base + self.batch))
            m += 1


def _dist_info():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return world, rank, local


def train_autocast(device_type: str, enabled: bool):
    """bf16 autocast with the weight-cast cache disabled.

    Autocast caches each weight's bf16 copy for the life of the region and does not key that cache on
    grad mode. A no_grad pass inside the region (placement self-conditioning) therefore caches copies
    with no autograd link, and the grad-enabled pass that follows reuses them: those weights get no
    gradient (torch 2.10, verified; it killed the first G/G0 launch at step 0). Re-casting costs one
    extra copy of the weights per forward, negligible next to the step.
    """
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    decay_start = int(cfg.max_steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return cfg.lr
    frac = (step - decay_start) / max(1, cfg.max_steps - decay_start)
    return cfg.lr * (1 - frac * (1 - cfg.min_lr_ratio))


def _param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "embed" in name or "null" in name else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    def __init__(self, stage: Stage, cfg: TrainConfig):
        self.stage, self.cfg = stage, cfg
        self.run = Path(cfg.run_dir)
        self.world, self.rank, self.local = _dist_info()
        if self.world > 1 and not dist.is_initialized():
            from datetime import timedelta
            # nccl on GPUs; gloo lets the identical DDP code path be smoke-tested on CPU nodes.
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    timeout=timedelta(hours=2))  # rank-0 validation may take a while
        self.device = torch.device(f"cuda:{self.local}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.stop_requested = False
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._on_signal)

    # ---------------------------------------------------------------- utils
    def _on_signal(self, signum, frame):
        self.stop_requested = True

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _barrier(self):
        if self.world > 1:
            dist.barrier()

    def _any(self, flag: bool) -> bool:
        if self.world == 1:
            return flag
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(t)
        return bool(t.item() > 0)

    def _log(self, record: dict):
        if self.is_main:
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            with open(self.run / "metrics.jsonl", "a") as handle:
                handle.write(line + "\n")

    def _status(self, state: str, **fields):
        if self.is_main:
            io.write_status(self.run / "status.json", state, **fields)

    def _rng_state(self):
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}

    def _set_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], self.device)

    def _gather_rng(self):
        state = self._rng_state()
        if self.world == 1:
            return [state]
        out = [None] * self.world
        dist.all_gather_object(out, state)
        return out

    # ----------------------------------------------------------- checkpoint
    def save(self, step: int, best_metric, reason: str):
        rngs = self._gather_rng()
        if self.is_main:
            state = {"step": step, "model": self.raw.state_dict(), "ema": self.ema.state_dict(),
                     "opt": self.opt.state_dict(), "rng": rngs, "binding": self.binding,
                     "best_metric": best_metric, "saved_unix": time.time(), "reason": reason}
            last, prev = self.run / "last.pt", self.run / "prev.pt"
            with io.atomic_path(self.run / "next.pt") as tmp:
                torch.save(state, tmp)
            if last.exists():
                os.replace(last, prev)
            os.replace(self.run / "next.pt", last)
            self._status("running" if reason != "final" else "complete", step=step,
                         last_checkpoint=time.strftime("%Y-%m-%d %H:%M:%S"), best=best_metric, reason=reason)
        self._barrier()
        self.last_ckpt = time.time()

    def _try_resume(self):
        for name in ("last.pt", "prev.pt"):
            path = self.run / name
            if not path.is_file():
                continue
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as error:  # truncated by a hard kill: fall back to prev.pt
                print(f"[resume] {name} unreadable ({error}); trying the previous checkpoint", flush=True)
                continue
            if state["binding"] != self.binding:
                raise RuntimeError(f"{path} was written by a different config/data/code binding; "
                                   f"use a new run directory instead of resuming")
            self.raw.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])
            rngs = state["rng"]
            if len(rngs) == self.world:
                self._set_rng(rngs[self.rank])
            else:
                print("[resume] world size changed; RNG streams restart (data order is still exact)", flush=True)
            print(f"[resume] from {name} at step {state['step']}", flush=True)
            return state["step"], state.get("best_metric")
        return 0, None

    # ------------------------------------------------------------------ run
    def fit(self) -> None:
        cfg = self.cfg
        io.raise_open_file_limit()   # before loader workers fork: placement maps ~1,088 files per worker
        self.run.mkdir(parents=True, exist_ok=True)
        if self.is_main and (self.run / "STOP").exists():
            (self.run / "STOP").unlink()
        self._barrier()
        trajectory = {k: v for k, v in asdict(cfg).items()
                      if k not in {"run_dir", "num_workers", "ckpt_minutes", "val_every", "log_every", "select_mode"}}
        self.binding = {"config": json.loads(json.dumps(trajectory)), "stage": self.stage.binding()}
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        np.random.seed(cfg.seed + 1000 * self.rank)
        random.seed(cfg.seed + 1000 * self.rank)
        self.raw = self.stage.build_model().to(self.device)
        self.ema = copy.deepcopy(self.raw).eval().requires_grad_(False)
        model = self.raw
        if self.world > 1:
            # broadcast_buffers=False: every buffer is a constant table (no BatchNorm), and per-forward
            # buffer broadcasts are collectives that deadlock if ranks ever run different numbers of
            # forwards. Auxiliary no-grad passes must still go through self.raw, never the DDP wrapper.
            model = torch.nn.parallel.DistributedDataParallel(
                self.raw, device_ids=[self.local] if self.device.type == "cuda" else None,
                find_unused_parameters=False, broadcast_buffers=False)
        self.opt = torch.optim.AdamW(_param_groups(self.raw, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas,
                                     fused=self.device.type == "cuda")
        step, best = self._try_resume()
        budget = float(cfg.extra.get("budget_gpu_hours", 0) or 0)
        probe_steps = min(200, cfg.warmup)
        if cfg.max_steps <= 0:
            if budget <= 0:
                raise ValueError("max_steps <= 0 requires extra.budget_gpu_hours")
            plan = self.run / "plan.json"
            if plan.exists():
                cfg.max_steps = int(io.read_json(plan)["max_steps"])
            else:
                cfg.max_steps = 10 ** 9   # resolved after the throughput probe (inside warmup)
        if self.is_main:
            io.write_json_atomic(self.run / "binding.json", self.binding)
            n = sum(p.numel() for p in self.raw.parameters())
            print(f"[train] params {n / 1e6:.2f}M  world {self.world}  start step {step}", flush=True)
        if step >= cfg.max_steps:
            self._status("complete", step=step, best=best)
            return
        self._status("running", step=step, best=best)
        loader = torch.utils.data.DataLoader(
            self.stage.dataset(), batch_sampler=_BatchSampler(step * cfg.accum, self.rank, self.world, cfg.batch_size),
            collate_fn=self.stage.collate, num_workers=cfg.num_workers, pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None,
            # A private generator: creating the iterator must not consume the global RNG,
            # or a resumed run's dropout/noise stream would be offset from the original.
            generator=torch.Generator().manual_seed(cfg.seed))
        batches = iter(loader)
        self.last_ckpt = time.time()
        probe_t0 = probe_s0 = None
        if cfg.max_steps == 10 ** 9 and step >= probe_steps:
            probe_steps = step + 60          # stopped after the probe point without a plan: re-probe

        t_log, acc = time.time(), {}
        model.train()
        while step < cfg.max_steps:
            for g in self.opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            logs_sum: dict[str, float] = {}
            for micro in range(cfg.accum):
                batch = _to_device(next(batches), self.device)
                sync = micro == cfg.accum - 1 or self.world == 1
                ctx = model.no_sync() if (self.world > 1 and not sync) else _null()
                with ctx, train_autocast(self.device.type, cfg.bf16):
                    loss, logs = self.stage.loss(model, batch, step)
                    if not torch.isfinite(loss):
                        self._status("failed", step=step, error="non-finite loss")
                        raise FloatingPointError(f"non-finite loss at step {step}: {logs}")
                    (loss / cfg.accum).backward()
                for k, v in logs.items():
                    logs_sum[k] = logs_sum.get(k, 0.0) + float(v) / cfg.accum
            if step < 50:   # DDP (find_unused_parameters=False) aborts on gradless params; name them, any world size
                gradless = [n for n, p in self.raw.named_parameters() if p.requires_grad and p.grad is None]
                if gradless:
                    self._status("failed", step=step, error=f"parameters without gradient: {gradless[:8]}")
                    raise RuntimeError(f"parameters received no gradient at step {step}: {gradless}")
            gnorm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                d = min(cfg.ema_decay, (1 + step) / (10 + step))
                for e, p in zip(self.ema.parameters(), self.raw.parameters()):
                    e.lerp_(p.detach(), 1 - d)
                for e, b in zip(self.ema.buffers(), self.raw.buffers()):
                    e.copy_(b)
            step += 1
            if cfg.max_steps == 10 ** 9 and budget > 0:
                if step >= min(50, probe_steps - 10) and probe_t0 is None:   # skip compile/warm caches
                    probe_t0, probe_s0 = time.time(), step
                elif step == probe_steps and probe_t0 is not None:
                    sec = torch.tensor([(time.time() - probe_t0) / max(1, probe_steps - probe_s0)], device=self.device)
                    if self.world > 1:
                        dist.all_reduce(sec, op=dist.ReduceOp.MAX)
                    cfg.max_steps = max(probe_steps + 1, int(budget * 3600 / (float(sec) * self.world)))
                    if self.is_main:
                        io.write_json_atomic(self.run / "plan.json", {
                            "max_steps": cfg.max_steps, "sec_per_step": float(sec), "world": self.world,
                            "budget_gpu_hours": budget, "probe_steps": probe_steps})
                        print(f"[train] throughput {float(sec):.3f}s/step on {self.world} GPU(s) -> "
                              f"max_steps {cfg.max_steps} for {budget} GPU-h", flush=True)
            for k, v in logs_sum.items():
                acc[k] = acc.get(k, 0.0) + v
            acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(gnorm)
            if step % cfg.log_every == 0:
                dt = time.time() - t_log
                self._log({"step": step, "lr": lr_at(step - 1, cfg), "sec_per_step": dt / cfg.log_every,
                           **{k: v / cfg.log_every for k, v in acc.items()}})
                acc, t_log = {}, time.time()
            stop = self._any(self.stop_requested or (self.run / "STOP").exists())
            if step % cfg.val_every == 0 or step == cfg.max_steps:
                best = self._validate(step, best)
                self.save(step, best, "final" if step == cfg.max_steps else "validation")
            elif stop or (time.time() - self.last_ckpt) / 60 >= cfg.ckpt_minutes:
                self.save(step, best, "stop" if stop else "periodic")
            if stop and step < cfg.max_steps:
                self._status("incomplete", step=step, best=best, reason="stop requested; resumes from last.pt")
                self._requeue()
                return
        self._status("complete", step=step, best=best)

    def _validate(self, step: int, best):
        metrics = {}
        if self.is_main:
            self.ema.eval()
            with torch.no_grad(), train_autocast(self.device.type, self.cfg.bf16):
                metrics = self.stage.validate(self.ema, step)
            self._log({"step": step, "val": metrics})
            value = metrics.get("select")
            if value is not None and math.isfinite(value):
                better = best is None or (value > best if self.cfg.select_mode == "max" else value < best)
                if better:
                    best = float(value)
                    with io.atomic_path(self.run / "best.pt") as tmp:
                        torch.save({"step": step, "ema": self.ema.state_dict(), "metrics": metrics,
                                    "binding": self.binding}, tmp)
                    io.write_json_atomic(self.run / "best.json", {"step": step, "metrics": metrics})
        if self.world > 1:
            box = [best]
            dist.broadcast_object_list(box, src=0)
            best = box[0]
        return best

    def _requeue(self):
        job = os.environ.get("SLURM_JOB_ID")
        if self.is_main and job and os.environ.get("V3_AUTO_REQUEUE", "1") == "1":
            print(f"[train] requeueing Slurm job {job}", flush=True)
            subprocess.run(["scontrol", "requeue", job], check=False)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch


def sample_rng(seed: int, index: int, salt: int = 0) -> np.random.Generator:
    """Deterministic per-sample RNG: the data order is a pure function of the index."""
    return np.random.default_rng([seed, index, salt])
