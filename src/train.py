"""
Training script for OsuMapper.

Supports:
- Curriculum learning (circles -> +sliders -> full)
- Mixed precision (AMP)
- Gradient accumulation
- Scheduled teacher forcing decay
- Checkpoint management (keep top-3)
- wandb / TensorBoard logging
- GPU-adaptive batch size

Usage:
    python -m src.train \
        --data_dir /dev/shm/asj102/osu_train \
        --model_dir /common/users/asj102/osu_project/models \
        --log_dir /common/users/asj102/osu_project/logs \
        --epochs 100
"""

import argparse
import glob
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .model import OsuMapper, get_adaptive_batch_size
from .dataset import (
    OsuCachedDataset, OsuStreamingDataset, build_dataloader,
    build_mel_transform, collate_fn, AudioAugmenter,
)
from .tokenizer import PAD, TOTAL_VOCAB, Residuals
from .metrics import compute_all_metrics
from .splits import (
    parse_split_ratios,
    build_split_lists,
    write_split_manifest,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train OsuMapper")
    p.add_argument("--data_dir", type=str,
                    default="/freespace/local/asj102/osu_cache/shards")
    p.add_argument("--model_dir", type=str,
                    default="/common/users/asj102/osu_project/models")
    p.add_argument("--log_dir", type=str,
                    default="/common/users/asj102/osu_project/logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=0,
                    help="0 = auto-detect from GPU")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--no_persistent_workers", action="store_true")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", action="store_true")
    p.add_argument("--log_every", type=int, default=20,
                    help="Log every N steps (more logging = smaller N)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="osu-mapper")
    p.add_argument("--dry_run", action="store_true",
                    help="Run one batch per epoch, no saving")
    p.add_argument("--curriculum_circles_until", type=int, default=20)
    p.add_argument("--curriculum_sliders_until", type=int, default=40)
    p.add_argument("--teacher_forcing_start", type=float, default=1.0)
    p.add_argument("--teacher_forcing_end", type=float, default=0.5)
    p.add_argument("--teacher_forcing_decay_epochs", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_epochs", type=int, default=25)
    p.add_argument("--early_stop_min_delta", type=float, default=0.01)
    p.add_argument("--split_ratios", type=str, default="0.8,0.1,0.1")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--split_file", type=str, default="")
    p.add_argument("--split_shuffle", action="store_true")
    p.add_argument("--resume", type=str, default="",
                    help="Path to checkpoint to resume from")
    p.add_argument("--stream", action="store_true",
                    help="Stream from HuggingFace instead of cached shards")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 on Ampere+ GPUs")
    p.add_argument("--compile", action="store_true", help="Use torch.compile for speed")

    # Model architecture
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--decoder_layers", type=int, default=6)
    p.add_argument("--decoder_heads", type=int, default=8)
    p.add_argument("--decoder_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    p.add_argument("--w_discrete", type=float, default=0.8)
    p.add_argument("--w_residual", type=float, default=0.2)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CheckpointManager:
    """Keep only top-k checkpoints by validation loss."""

    def __init__(self, model_dir: str, top_k: int = 3):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, model: nn.Module, optimizer, scheduler, epoch: int,
             val_loss: float, scaler=None) -> Path:
        path = self.model_dir / f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        torch.save(state, path)

        self.checkpoints.append((val_loss, path))
        self.checkpoints.sort(key=lambda x: x[0])

        while len(self.checkpoints) > self.top_k:
            _, old_path = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        best = self.model_dir / "best.pt"
        if self.checkpoints and self.checkpoints[0][1] == path:
            torch.save(state, best)

        return path


def get_teacher_forcing_ratio(epoch: int, args) -> float:
    if epoch >= args.teacher_forcing_decay_epochs:
        return args.teacher_forcing_end
    progress = epoch / args.teacher_forcing_decay_epochs
    return args.teacher_forcing_start - progress * (
        args.teacher_forcing_start - args.teacher_forcing_end
    )


def build_dataset(args, epoch: int, split_name: str, split_shards: dict[str, list[str]] | None):
    """Build dataset for train/val/test split."""
    if split_shards and not args.stream:
        shards = split_shards.get(split_name, [])
        if not shards:
            raise ValueError(f"No shards assigned to split '{split_name}'")
        shard_pattern = os.path.join(args.data_dir, "shard-*.tar")
        return OsuCachedDataset(
            shard_pattern,
            shuffle=(split_name == "train"),
            shards=shards,
        )

    curriculum = None
    augmenter = None
    if split_name == "train":
        if epoch < args.curriculum_circles_until:
            curriculum = "circles_only"
        augmenter = AudioAugmenter(enabled=(epoch > 5))

    mel_transform = build_mel_transform("cpu")
    return OsuStreamingDataset(
        split="train",
        mel_transform=mel_transform,
        augmenter=augmenter,
        curriculum_filter=curriculum,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scaler: GradScaler,
    epoch: int,
    args,
    device: torch.device,
    logger=None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    tf_ratio = get_teacher_forcing_ratio(epoch, args)

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    smooth_l1_loss_fn = nn.SmoothL1Loss()

    end = time.time()
    first_batch_time = None

    for step, batch in enumerate(dataloader):
        data_time = time.time() - end
        if first_batch_time is None:
            first_batch_time = data_time

        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        time_res_gt = batch["time_residuals"].to(device)
        x_res_gt = batch["x_residuals"].to(device)
        y_res_gt = batch["y_residuals"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        bpm = batch["bpm"].to(device)

        tgt_input = tokens[:, :-1]
        tgt_output = tokens[:, 1:]
        padding_mask = tgt_input == PAD

        compute_start = time.time()
        with autocast(dtype=torch.bfloat16):
            outputs = model(mel, tgt_input, difficulty_id, bpm, padding_mask)

            logits = outputs["logits"]
            discrete_loss = ce_loss_fn(
                logits.reshape(-1, TOTAL_VOCAB),
                tgt_output.reshape(-1),
            )

            mask = (tgt_output != PAD).unsqueeze(-1).float()
            time_res_loss = smooth_l1_loss_fn(
                outputs["time_residuals"][:, :tgt_output.shape[1]] * mask,
                time_res_gt[:, 1:tgt_output.shape[1] + 1].unsqueeze(-1) * mask,
            )
            x_res_loss = smooth_l1_loss_fn(
                outputs["x_residuals"][:, :tgt_output.shape[1]] * mask,
                x_res_gt[:, 1:tgt_output.shape[1] + 1].unsqueeze(-1) * mask,
            )
            y_res_loss = smooth_l1_loss_fn(
                outputs["y_residuals"][:, :tgt_output.shape[1]] * mask,
                y_res_gt[:, 1:tgt_output.shape[1] + 1].unsqueeze(-1) * mask,
            )
            residual_loss = (time_res_loss + x_res_loss + y_res_loss) / 3.0

            loss = (
                args.w_discrete * discrete_loss
                + args.w_residual * residual_loss
            )
            loss = loss / args.accum_steps

        def _tensor_stats(name: str, tensor: torch.Tensor):
            t = tensor.detach()
            finite = torch.isfinite(t).all().item()
            if t.numel() == 0:
                print(f"{name}: finite={finite} empty", flush=True)
                return
            t_min = float(t.min().item())
            t_max = float(t.max().item())
            print(f"{name}: finite={finite} min={t_min:.6g} max={t_max:.6g}", flush=True)

        def _isfinite(tensor: torch.Tensor) -> bool:
            return bool(torch.isfinite(tensor.detach()).all().item())

        time_res_gt_used = time_res_gt[:, 1:tgt_output.shape[1] + 1]
        x_res_gt_used = x_res_gt[:, 1:tgt_output.shape[1] + 1]
        y_res_gt_used = y_res_gt[:, 1:tgt_output.shape[1] + 1]

        finite_losses = all(
            _isfinite(t)
            for t in (loss, discrete_loss, residual_loss)
        )
        finite_targets = all(
            _isfinite(t)
            for t in (mel, tgt_output, time_res_gt_used, x_res_gt_used, y_res_gt_used)
        )
        finite_outputs = all(
            _isfinite(t)
            for t in (
                logits,
                outputs["time_residuals"],
                outputs["x_residuals"],
                outputs["y_residuals"],
            )
        )

        if not (finite_losses and finite_targets and finite_outputs):
            print("Non-finite detected", flush=True)
            _tensor_stats("mel", mel)
            if tgt_output.numel() > 0:
                print(f"token min: {tgt_output.min().item()}", flush=True)
                print(f"token max: {tgt_output.max().item()}", flush=True)
            _tensor_stats("time_res_gt", time_res_gt_used)
            _tensor_stats("x_res_gt", x_res_gt_used)
            _tensor_stats("y_res_gt", y_res_gt_used)
            _tensor_stats("logits", logits)
            _tensor_stats("time_residuals", outputs["time_residuals"])
            _tensor_stats("x_residuals", outputs["x_residuals"])
            _tensor_stats("y_residuals", outputs["y_residuals"])
            print(f"discrete_loss finite: {torch.isfinite(discrete_loss).item()}", flush=True)
            print(f"residual_loss finite: {torch.isfinite(residual_loss).item()}", flush=True)
            print(f"total_loss finite: {torch.isfinite(loss).item()}", flush=True)
            keys = batch.get("key")
            if keys is not None:
                print(f"sample keys: {keys[:4] if isinstance(keys, list) else keys}", flush=True)
            urls = batch.get("url")
            if urls is not None:
                print(f"sample urls: {urls[:2] if isinstance(urls, list) else urls}", flush=True)
            print("Skipping bad batch", flush=True)
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        compute_time = time.time() - compute_start
        end = time.time()

        total_loss += loss.item() * args.accum_steps
        num_batches += 1

        if logger and step % max(args.log_every, 1) == 0:
            logger({
                "train/loss": loss.item() * args.accum_steps,
                "train/discrete_loss": discrete_loss.item(),
                "train/residual_loss": residual_loss.item(),
                "train/tf_ratio": tf_ratio,
                "train/step": step,
                "timing/data_time_s": data_time,
                "timing/compute_time_s": compute_time,
            })

        if step < 5 or step % max(args.log_every, 1) == 0:
            batch_size = mel.shape[0]
            step_time = data_time + compute_time
            samples_per_s = batch_size / max(step_time, 1e-6)
            msg = (
                f"  step={step:05d} "
                f"loss={loss.item() * args.accum_steps:.4f} "
                f"data={data_time:.3f}s compute={compute_time:.3f}s "
                f"step={step_time:.3f}s samples/s={samples_per_s:.1f}"
            )
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                max_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                msg += f" | gpu_mem={mem_gb:.2f}GB max={max_mem_gb:.2f}GB"
            print(msg, flush=True)

        if args.dry_run and step >= 2:
            break

    if first_batch_time is not None:
        print(f"  first_batch_data_time={first_batch_time:.3f}s", flush=True)

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    args,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_edit_dist = 0.0
    total_timing_mae = 0.0
    total_hit_f1 = 0.0
    num_batches = 0
    num_metrics = 0

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break

        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        time_res_gt = batch["time_residuals"].to(device)
        x_res_gt = batch["x_residuals"].to(device)
        y_res_gt = batch["y_residuals"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        bpm = batch["bpm"].to(device)

        tgt_input = tokens[:, :-1]
        tgt_output = tokens[:, 1:]
        padding_mask = tgt_input == PAD

        with autocast(dtype=torch.bfloat16):
            outputs = model(mel, tgt_input, difficulty_id, bpm, padding_mask)
            logits = outputs["logits"]
            discrete_loss = ce_loss_fn(
                logits.reshape(-1, TOTAL_VOCAB),
                tgt_output.reshape(-1),
            )

        if step == 0:
            mel_no_onset = mel.clone()
            mel_no_onset[:, :, -1, :] = 0.0
            with autocast(dtype=torch.bfloat16):
                outputs_no_onset = model(
                    mel_no_onset, tgt_input, difficulty_id, bpm, padding_mask
                )
            diff = (
                outputs["logits"].float() - outputs_no_onset["logits"].float()
            ).abs().mean()
            print(
                f"[val diag] Onset zero-out logit diff: {diff:.4f} "
                f"(if <0.01 → model may be ignoring onset)",
                flush=True,
            )

        total_loss += discrete_loss.item()
        num_batches += 1

        # Compute metrics on a subset of the batch
        pred_tokens_batch = logits.argmax(dim=-1)
        for b in range(min(2, tokens.shape[0])):
            pred_t = pred_tokens_batch[b].cpu().tolist()
            true_t = tgt_output[b].cpu().tolist()
            true_t = [t for t in true_t if t != PAD]
            pred_t = pred_t[:len(true_t)]

            sample_ms_per_beat = 60000.0 / bpm[b].item()
            metrics = compute_all_metrics(pred_t, true_t, ms_per_beat=sample_ms_per_beat)
            total_edit_dist += metrics["edit_distance"]
            mae = metrics["timing_mae_ms"]
            if mae != float("inf"):
                total_timing_mae += mae
            total_hit_f1 += metrics["hit_f1"]
            num_metrics += 1

            # #region agent log
            if step == 0 and b == 0:
                import json as _json
                from .tokenizer import (
                    TIME_OFFSET as _TO, NUM_BEAT_BINS as _NB,
                    POS_OFFSET as _PO, NUM_POS_BINS as _NP,
                    _NUM_SPECIAL as _NS,
                    detokenize_to_hitobjects as _detok,
                )
                _LOG = "/common/home/asj102/personalCS/osu_beatmap_generator/.cursor/debug-559ead.log"
                _pred_time = [t for t in pred_t if _TO <= t < _TO + _NB]
                _true_time = [t for t in true_t if _TO <= t < _TO + _NB]
                _pred_pos = [t for t in pred_t if _PO <= t < _PO + _NP]
                _true_pos = [t for t in true_t if _PO <= t < _PO + _NP]
                _pred_type = [t for t in pred_t if 4 <= t <= 13]
                _true_type = [t for t in true_t if 4 <= t <= 13]
                _pred_objs = _detok(pred_t, ms_per_beat=sample_ms_per_beat)
                _true_objs = _detok(true_t, ms_per_beat=sample_ms_per_beat)
                _pred_times_ms = sorted([o["time"] for o in _pred_objs])
                _true_times_ms = sorted([o["time"] for o in _true_objs])
                _pred_time_bins_raw = [t - _TO for t in _pred_time]
                _true_time_bins_raw = [t - _TO for t in _true_time]
                _time_tok_acc = sum(1 for p, t in zip(pred_t, true_t) if p == t and _TO <= t < _TO + _NB) / max(len(_true_time), 1)
                _all_acc = sum(1 for p, t in zip(pred_t, true_t) if p == t) / max(len(true_t), 1)
                def _cat(t):
                    if _TO <= t < _TO + _NB: return "time"
                    if _PO <= t < _PO + _NP: return "pos"
                    if 4 <= t <= 13: return "type"
                    return "other"
                _cat_match = {"time_at_time": 0, "pos_at_time": 0, "type_at_time": 0, "other_at_time": 0,
                              "time_at_pos": 0, "pos_at_pos": 0, "type_at_pos": 0, "other_at_pos": 0,
                              "time_at_type": 0, "pos_at_type": 0, "type_at_type": 0, "other_at_type": 0}
                for _p, _t in zip(pred_t, true_t):
                    _pc, _tc = _cat(_p), _cat(_t)
                    if _tc == "time": _cat_match[f"{_pc}_at_time"] += 1
                    elif _tc == "pos": _cat_match[f"{_pc}_at_pos"] += 1
                    elif _tc == "type": _cat_match[f"{_pc}_at_type"] += 1
                _time_bin_err = [abs(p - t) for p, t in zip(_pred_time_bins_raw, _true_time_bins_raw)] if _pred_time_bins_raw and _true_time_bins_raw else []
                _mean_bin_err = sum(_time_bin_err) / len(_time_bin_err) if _time_bin_err else -1
                with open(_LOG, "a") as _f:
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H1", "message": "token_breakdown", "data": {"seq_len": len(true_t), "pred_time_count": len(_pred_time), "true_time_count": len(_true_time), "pred_pos_count": len(_pred_pos), "true_pos_count": len(_true_pos), "pred_type_count": len(_pred_type), "true_type_count": len(_true_type), "pred_obj_count": len(_pred_objs), "true_obj_count": len(_true_objs)}, "timestamp": int(time.time()*1000)}) + "\n")
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H2", "message": "time_bin_dist", "data": {"pred_bins": _pred_time_bins_raw[:30], "true_bins": _true_time_bins_raw[:30], "pred_bin_set": sorted(set(_pred_time_bins_raw)), "true_bin_set": sorted(set(_true_time_bins_raw)), "mean_bin_error": round(_mean_bin_err, 2)}, "timestamp": int(time.time()*1000)}) + "\n")
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H3", "message": "bpm_and_timing", "data": {"bpm": float(bpm[b].item()), "ms_per_beat": round(sample_ms_per_beat, 2), "pred_times_ms": [round(t, 1) for t in _pred_times_ms[:15]], "true_times_ms": [round(t, 1) for t in _true_times_ms[:15]], "timing_mae": round(mae, 2) if mae != float("inf") else -1}, "timestamp": int(time.time()*1000)}) + "\n")
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H4", "message": "accuracy", "data": {"time_tok_accuracy": round(_time_tok_acc, 4), "overall_tok_accuracy": round(_all_acc, 4)}, "timestamp": int(time.time()*1000)}) + "\n")
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H5", "message": "first_30_tokens", "data": {"pred_first30": pred_t[:30], "true_first30": true_t[:30]}, "timestamp": int(time.time()*1000)}) + "\n")
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H1H2", "message": "category_confusion", "data": _cat_match, "timestamp": int(time.time()*1000)}) + "\n")
                    _f.write(_json.dumps({"sessionId": "559ead", "hypothesisId": "H2", "message": "bin_level_errors", "data": {"per_pos_bin_err": _time_bin_err[:20], "mean_bin_error": round(_mean_bin_err, 2)}, "timestamp": int(time.time()*1000)}) + "\n")
            # #endregion

    if num_batches == 0:
        print(
            "[val] No batches produced. Check split shards, num_workers, and empty shard filtering.",
            flush=True,
        )

    n = max(num_batches, 1)
    nm = max(num_metrics, 1)
    return {
        "val/loss": total_loss / n,
        "val/edit_distance": total_edit_dist / nm,
        "val/timing_mae_ms": total_timing_mae / nm,
        "val/hit_f1": total_hit_f1 / nm,
    }


def prepare_cached_splits(args) -> tuple[dict[str, list[str]] | None, str | None]:
    """Create deterministic train/val/test splits and write manifest."""
    shard_pattern = os.path.join(args.data_dir, "shard-*.tar")
    shards = sorted(glob.glob(shard_pattern))
    original_count = len(shards)
    shards = [s for s in shards if os.path.isfile(s) and os.path.getsize(s) > 0]
    if original_count and len(shards) != original_count:
        print(f"Filtered {original_count - len(shards)} empty shards before split.")
    if not shards or args.stream:
        return None, None

    ratios = parse_split_ratios(args.split_ratios)
    split_lists = build_split_lists(
        shard_paths=shards,
        ratios=ratios,
        split_seed=args.split_seed,
        split_shuffle=args.split_shuffle,
    )

    split_file = args.split_file or os.path.join(args.log_dir, "splits.json")
    manifest = write_split_manifest(
        split_file=split_file,
        data_dir=args.data_dir,
        split_lists=split_lists,
        ratios=ratios,
        split_seed=args.split_seed,
        split_shuffle=args.split_shuffle,
    )

    split_shards = {
        "train": split_lists["train"],
        "val": split_lists["val"],
        "test": split_lists["test"],
    }
    train_set = set(split_shards["train"])
    val_set = set(split_shards["val"])
    test_set = set(split_shards["test"])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Split overlap detected across train/val/test shards")
    print(
        f"Split manifest: {split_file} | train={len(split_shards['train'])} "
        f"val={len(split_shards['val'])} test={len(split_shards['test'])}"
    )
    print(
        "Split config: ratios={ratios} seed={seed} shuffle={shuffle}".format(
            ratios=manifest["split_ratios"],
            seed=manifest["split_seed"],
            shuffle=manifest["split_shuffle"],
        )
    )
    return split_shards, split_file


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            print("TF32 enabled.")

    if args.batch_size == 0:
        args.batch_size = get_adaptive_batch_size()
    print(f"Batch size: {args.batch_size} (accum {args.accum_steps} = effective {args.batch_size * args.accum_steps})")
    if args.no_persistent_workers:
        args.persistent_workers = False
    if args.no_pin_memory:
        args.pin_memory = False
    if not args.no_persistent_workers and not args.persistent_workers:
        args.persistent_workers = True
    if not args.no_pin_memory and not args.pin_memory:
        args.pin_memory = True
    print(
        "Loader: workers={w} prefetch={p} persistent={pw} pin_memory={pm}".format(
            w=args.num_workers,
            p=args.prefetch_factor,
            pw=args.persistent_workers if args.num_workers > 0 else False,
            pm=args.pin_memory,
        )
    )

    shard_pattern = os.path.join(args.data_dir, "shard-*.tar")
    shards = sorted(glob.glob(shard_pattern))
    shard_bytes = 0
    if shards:
        for s in shards:
            try:
                shard_bytes += os.path.getsize(s)
            except OSError:
                continue
    shm_free = None
    try:
        shm_free = shutil.disk_usage("/dev/shm").free
    except OSError:
        shm_free = None
    print(f"Data dir: {args.data_dir}")
    print(f"Shard pattern: {shard_pattern}")
    print(f"Shard count: {len(shards)}")
    if shard_bytes:
        print(f"Shard size: {shard_bytes / (1024 ** 3):.2f} GB")
    if shm_free is not None:
        print(f"/dev/shm free: {shm_free / (1024 ** 3):.2f} GB")
        if shard_bytes and shm_free < shard_bytes:
            print("WARNING: shard data larger than free /dev/shm; expect slow I/O.")

    split_shards, split_file = prepare_cached_splits(args)

    # Model
    model = OsuMapper(
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff=args.decoder_ff,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable:,} trainable")

    start_epoch = 0
    best_val_loss = float("inf")
    resume_ckpt = None

    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.4f}")

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"torch.compile failed, continuing without it: {exc}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(args.epochs // 5, 1), T_mult=2
    )
    scaler = GradScaler()

    if resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    ckpt_mgr = CheckpointManager(args.model_dir, top_k=3)

    # Logging
    logger_fn = None
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            logger_fn = wandb.log
        except ImportError:
            print("wandb not available, falling back to TensorBoard")

    if logger_fn is None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.log_dir)
            _step = [0]
            def _tb_log(d):
                for k, v in d.items():
                    tb_writer.add_scalar(k, v, _step[0])
                _step[0] += 1
            logger_fn = _tb_log
        except ImportError:
            logger_fn = lambda d: None

    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Build train split dataset
        dataset = build_dataset(args, epoch, "train", split_shards)
        dataloader = build_dataloader(
            dataset,
            args.batch_size,
            args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
        )

        train_loss = train_one_epoch(
            model, dataloader, optimizer, scaler, epoch, args, device, logger_fn
        )
        scheduler.step()

        # Validation on held-out split
        val_dataset = build_dataset(args, epoch, "val", split_shards)
        val_loader = build_dataloader(
            val_dataset,
            args.batch_size,
            args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
        )
        val_metrics = validate(model, val_loader, args, device)

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(f"  train_loss={train_loss:.4f} | val_loss={val_metrics['val/loss']:.4f}")
        print(f"  edit_dist={val_metrics['val/edit_distance']:.4f} | "
              f"timing_mae={val_metrics['val/timing_mae_ms']:.1f}ms | "
              f"hit_f1={val_metrics['val/hit_f1']:.4f}")
        print(f"  lr={lr:.6f} | time={elapsed:.1f}s")

        logger_payload = {"train/epoch_loss": train_loss, "epoch": epoch, **val_metrics}
        if split_shards:
            logger_payload["data/train_shards"] = len(split_shards["train"])
            logger_payload["data/val_shards"] = len(split_shards["val"])
            logger_payload["data/test_shards"] = len(split_shards["test"])
        logger_fn(logger_payload)

        # Checkpointing
        val_loss = val_metrics["val/loss"]
        if not args.dry_run:
            ckpt_mgr.save(model, optimizer, scheduler, epoch, val_loss, scaler)

        improved = val_loss < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_loss
            no_improve = 0
            print(f"  *** New best val_loss: {best_val_loss:.4f}")
        else:
            no_improve += 1
            if (
                args.early_stop_patience > 0
                and (epoch + 1) >= args.early_stop_min_epochs
                and no_improve >= args.early_stop_patience
            ):
                print(
                    "  Early stopping after {patience} epochs without "
                    "val improvement > {delta:.4f}".format(
                        patience=args.early_stop_patience,
                        delta=args.early_stop_min_delta,
                    )
                )
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.model_dir}")
    if split_file:
        print(f"Split file saved to: {split_file}")


if __name__ == "__main__":
    main()
