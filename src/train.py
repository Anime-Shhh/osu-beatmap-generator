"""
Training entrypoint for legacy token training, AE training, and flow matching.

Legacy behavior remains the default. New stages are opt-in.
"""

from __future__ import annotations

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
from torch.cuda.amp import GradScaler, autocast

from .checkpoints import export_model_state_dict, load_model_state_dict, normalize_model_state_dict
from .dataset import (
    FULL_SONG_SAMPLE,
    WINDOW_SAMPLE,
    AudioAugmenter,
    N_MELS,
    OsuCachedDataset,
    OsuStreamingDataset,
    build_dataloader,
    build_mel_transform,
)
from .flow_model import LatentFlowMatcher, integrate_flow, sample_flow_inputs
from .latent_ae import (
    SignalAutoencoder,
    SignalCritic,
    compute_autoencoder_losses,
    hinge_discriminator_loss,
)
from .metrics import compute_all_metrics, compute_signal_metrics
from .model import OsuMapper, get_adaptive_batch_size
from .splits import build_split_lists, parse_split_ratios, write_split_manifest
from .tokenizer import PAD, TOTAL_VOCAB


def parse_args():
    p = argparse.ArgumentParser(description="Train OsuMapper")
    p.add_argument("--data_dir", type=str, default="/freespace/local/asj102/osu_cache/shards")
    p.add_argument("--model_dir", type=str, default="/common/users/asj102/osu_project/models")
    p.add_argument("--log_dir", type=str, default="/common/users/asj102/osu_project/logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=0, help="0 = auto-detect from GPU")
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
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="osu-mapper")
    p.add_argument("--dry_run", action="store_true")
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
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--stage", type=str, choices=["legacy", "ae", "flow"], default="legacy")
    p.add_argument("--use_flow_matching", action="store_true", help="Alias for --stage flow")
    p.add_argument("--ae_checkpoint", type=str, default="")
    p.add_argument("--signal_required", action="store_true")
    p.add_argument("--quality_filter", type=str, choices=["all", "ranked_only"], default="all")
    p.add_argument("--difficulty_dropout", type=float, default=0.1)
    p.add_argument("--full_song_context", action="store_true")
    p.add_argument("--difficulty", type=float, default=-1.0, help="Optional conditioning override")

    # Legacy architecture
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--decoder_layers", type=int, default=6)
    p.add_argument("--decoder_heads", type=int, default=8)
    p.add_argument("--decoder_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)

    # Signal / latent architecture
    p.add_argument("--latent_dim", type=int, default=96)
    p.add_argument("--ae_hidden_dim", type=int, default=256)
    p.add_argument("--flow_hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument(
        "--flow_encoder_unfreeze_last_n",
        type=int,
        default=0,
        help="Number of AST encoder layers to unfreeze for flow conditioning. "
        "Default 0 keeps AST frozen to avoid OOM on long full-song samples.",
    )
    p.add_argument("--ode_steps", type=int, default=32)
    p.add_argument("--val_ode_steps", type=int, default=12)

    # Loss weights
    p.add_argument("--w_discrete", type=float, default=0.8)
    p.add_argument("--w_residual", type=float, default=0.2)
    p.add_argument("--w_token_aux", type=float, default=0.0)
    p.add_argument("--w_reconstruction_aux", type=float, default=0.1)
    p.add_argument("--w_flow", type=float, default=1.0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_args(args):
    if args.use_flow_matching:
        args.stage = "flow"
    if args.stage in {"ae", "flow"}:
        args.signal_required = True
        args.full_song_context = True


class CheckpointManager:
    """Keep only top-k checkpoints by validation loss."""

    def __init__(self, model_dir: str, top_k: int = 3):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, state: dict, val_loss: float) -> Path:
        epoch = int(state.get("epoch", 0))
        path = self.model_dir / f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
        torch.save(state, path)

        self.checkpoints.append((val_loss, path))
        self.checkpoints.sort(key=lambda item: item[0])
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
    progress = epoch / max(args.teacher_forcing_decay_epochs, 1)
    return args.teacher_forcing_start - progress * (args.teacher_forcing_start - args.teacher_forcing_end)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def build_logger(args):
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
            step = [0]

            def _tb_log(values):
                for key, value in values.items():
                    tb_writer.add_scalar(key, value, step[0])
                step[0] += 1

            logger_fn = _tb_log
        except ImportError:
            logger_fn = lambda _: None
    return logger_fn


def get_sample_type(args) -> str:
    return FULL_SONG_SAMPLE if args.full_song_context else WINDOW_SAMPLE


def build_dataset(args, epoch: int, split_name: str, split_shards: dict[str, list[str]] | None):
    sample_type = get_sample_type(args)
    signal_required = args.signal_required or args.stage != "legacy"
    if split_shards and not args.stream:
        shards = split_shards.get(split_name, [])
        if not shards:
            raise ValueError(f"No shards assigned to split '{split_name}'")
        shard_pattern = os.path.join(args.data_dir, "shard-*.tar")
        return OsuCachedDataset(
            shard_pattern=shard_pattern,
            shuffle=(split_name == "train"),
            shards=shards,
            sample_type=sample_type,
            signal_required=signal_required,
            quality_filter=args.quality_filter,
        )

    curriculum = None
    augmenter = None
    if split_name == "train" and args.stage == "legacy":
        if epoch < args.curriculum_circles_until:
            curriculum = "circles_only"
        augmenter = AudioAugmenter(enabled=(epoch > 5))

    mel_transform = build_mel_transform("cpu")
    return OsuStreamingDataset(
        split="train",
        mel_transform=mel_transform,
        augmenter=augmenter,
        curriculum_filter=curriculum,
        sample_type=sample_type,
        quality_filter=args.quality_filter,
        signal_required=signal_required,
    )


def apply_difficulty_override(difficulty_id, star_rating, args):
    if args.difficulty > 0:
        star_rating = torch.full_like(star_rating, float(args.difficulty))
        bins = []
        for value in star_rating.squeeze(-1).detach().cpu().tolist():
            if value < 2.0:
                bins.append(0)
            elif value < 3.0:
                bins.append(1)
            elif value < 4.0:
                bins.append(2)
            elif value < 5.3:
                bins.append(3)
            else:
                bins.append(4)
        difficulty_id = torch.tensor(bins, device=difficulty_id.device, dtype=difficulty_id.dtype)
    return difficulty_id, star_rating


def compute_param_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def prepare_cached_splits(args) -> tuple[dict[str, list[str]] | None, str | None]:
    shard_pattern = os.path.join(args.data_dir, "shard-*.tar")
    shards = sorted(glob.glob(shard_pattern))
    original_count = len(shards)
    shards = [shard for shard in shards if os.path.isfile(shard) and os.path.getsize(shard) > 0]
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
    print(
        f"Split manifest: {split_file} | train={len(split_lists['train'])} "
        f"val={len(split_lists['val'])} test={len(split_lists['test'])}"
    )
    return {
        "train": split_lists["train"],
        "val": split_lists["val"],
        "test": split_lists["test"],
    }, split_file


def train_one_epoch_legacy(model, dataloader, optimizer, scaler, epoch, args, device, logger=None):
    model.train()
    total_loss = 0.0
    total_discrete = 0.0
    total_residual = 0.0
    num_batches = 0
    optimizer.zero_grad()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    smooth_l1 = nn.SmoothL1Loss()
    tf_ratio = get_teacher_forcing_ratio(epoch, args)

    for step, batch in enumerate(dataloader):
        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        time_res_gt = batch["time_residuals"].to(device)
        x_res_gt = batch["x_residuals"].to(device)
        y_res_gt = batch["y_residuals"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        star_rating = batch["star_rating"].to(device)
        bpm = batch["bpm"].to(device)
        difficulty_id, star_rating = apply_difficulty_override(difficulty_id, star_rating, args)

        tgt_input = tokens[:, :-1]
        tgt_output = tokens[:, 1:]
        padding_mask = tgt_input == PAD

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = model(mel, tgt_input, difficulty_id, bpm, padding_mask)
            logits = outputs["logits"]
            discrete_loss = ce_loss_fn(logits.reshape(-1, TOTAL_VOCAB), tgt_output.reshape(-1))
            mask = (tgt_output != PAD).unsqueeze(-1).float()
            time_loss = smooth_l1(
                outputs["time_residuals"][:, :tgt_output.shape[1]] * mask,
                time_res_gt[:, 1:tgt_output.shape[1] + 1].unsqueeze(-1) * mask,
            )
            x_loss = smooth_l1(
                outputs["x_residuals"][:, :tgt_output.shape[1]] * mask,
                x_res_gt[:, 1:tgt_output.shape[1] + 1].unsqueeze(-1) * mask,
            )
            y_loss = smooth_l1(
                outputs["y_residuals"][:, :tgt_output.shape[1]] * mask,
                y_res_gt[:, 1:tgt_output.shape[1] + 1].unsqueeze(-1) * mask,
            )
            residual_loss = (time_loss + x_loss + y_loss) / 3.0
            loss = (args.w_discrete * discrete_loss + args.w_residual * residual_loss) / max(args.accum_steps, 1)

        scaler.scale(loss).backward()
        if (step + 1) % max(args.accum_steps, 1) == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * max(args.accum_steps, 1)
        total_discrete += discrete_loss.item()
        total_residual += residual_loss.item()
        num_batches += 1

        if logger and step % max(args.log_every, 1) == 0:
            logger({
                "train/loss": loss.item() * max(args.accum_steps, 1),
                "train/discrete_loss": discrete_loss.item(),
                "train/residual_loss": residual_loss.item(),
                "train/tf_ratio": tf_ratio,
            })
        if args.dry_run and step >= 2:
            break

    return {
        "train/loss": total_loss / max(num_batches, 1),
        "train/discrete_loss": total_discrete / max(num_batches, 1),
        "train/residual_loss": total_residual / max(num_batches, 1),
    }


@torch.no_grad()
def validate_legacy(model, dataloader, args, device, max_batches: int = 50):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0.0
    total_edit = 0.0
    total_mae = 0.0
    total_hit_f1 = 0.0
    total_onset_influence = 0.0
    num_batches = 0
    num_metrics = 0

    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break

        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        star_rating = batch["star_rating"].to(device)
        bpm = batch["bpm"].to(device)
        difficulty_id, star_rating = apply_difficulty_override(difficulty_id, star_rating, args)

        tgt_input = tokens[:, :-1]
        tgt_output = tokens[:, 1:]
        padding_mask = tgt_input == PAD

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = model(mel, tgt_input, difficulty_id, bpm, padding_mask)
            logits = outputs["logits"]
            loss = ce_loss_fn(logits.reshape(-1, TOTAL_VOCAB), tgt_output.reshape(-1))
        total_loss += loss.item()

        mel_no_onset = mel.clone()
        mel_no_onset[:, :, N_MELS:, :] = 0.0
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs_no_onset = model(mel_no_onset, tgt_input, difficulty_id, bpm, padding_mask)
        total_onset_influence += float((outputs["logits"].float() - outputs_no_onset["logits"].float()).abs().mean().item())

        pred_tokens_batch = logits.argmax(dim=-1)
        for batch_idx in range(min(2, tokens.shape[0])):
            pred_tokens = pred_tokens_batch[batch_idx].cpu().tolist()
            true_tokens = tgt_output[batch_idx].cpu().tolist()
            true_tokens = [tok for tok in true_tokens if tok != PAD]
            pred_tokens = pred_tokens[:len(true_tokens)]
            sample_ms_per_beat = 60000.0 / bpm[batch_idx].item()
            metrics = compute_all_metrics(pred_tokens, true_tokens, ms_per_beat=sample_ms_per_beat)
            total_edit += metrics["edit_distance"]
            if metrics["timing_mae_ms"] != float("inf"):
                total_mae += metrics["timing_mae_ms"]
            total_hit_f1 += metrics["hit_f1"]
            num_metrics += 1

        num_batches += 1

    return {
        "val/loss": total_loss / max(num_batches, 1),
        "val/edit_distance": total_edit / max(num_metrics, 1),
        "val/timing_mae_ms": total_mae / max(num_metrics, 1),
        "val/hit_f1": total_hit_f1 / max(num_metrics, 1),
        "val/cursor_smoothness": 0.0,
        "val/onset_influence": total_onset_influence / max(num_batches, 1),
    }


def train_one_epoch_ae(ae, critic, dataloader, ae_opt, critic_opt, scaler, args, device, logger=None):
    ae.train()
    critic.train()
    set_requires_grad(critic, True)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    totals = {
        "train/loss": 0.0,
        "train/critic_loss": 0.0,
        "train/l1_loss": 0.0,
        "train/l2_loss": 0.0,
        "train/multi_scale_loss": 0.0,
        "train/adv_loss": 0.0,
        "train/fm_loss": 0.0,
        "train/token_aux_loss": 0.0,
    }
    num_batches = 0
    ae_opt.zero_grad()
    critic_opt.zero_grad()

    for step, batch in enumerate(dataloader):
        signal = batch["signal"]
        if signal is None:
            raise RuntimeError("AE stage requires signal tensors in the dataset.")
        signal = signal.to(device)
        tokens = batch["tokens"].to(device)
        token_len = tokens.shape[1] if args.w_token_aux > 0 else 0

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = ae(signal, token_len=token_len)
            reconstruction = outputs["reconstruction"]
            real_logits = critic(signal)
            fake_logits = critic(reconstruction.detach())
            critic_loss = hinge_discriminator_loss(real_logits, fake_logits) / max(args.accum_steps, 1)
        scaler.scale(critic_loss).backward()
        if (step + 1) % max(args.accum_steps, 1) == 0:
            scaler.unscale_(critic_opt)
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            scaler.step(critic_opt)
            scaler.update()
            critic_opt.zero_grad()

        set_requires_grad(critic, False)
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = ae(signal, token_len=token_len)
            reconstruction = outputs["reconstruction"]
            fake_logits, fake_features = critic(reconstruction, return_features=True)
            with torch.no_grad():
                _, real_features = critic(signal, return_features=True)
            ae_losses = compute_autoencoder_losses(
                reconstruction=reconstruction,
                target=signal,
                fake_logits=fake_logits,
                real_features=real_features,
                fake_features=fake_features,
            )
            token_aux = torch.zeros((), device=device)
            if args.w_token_aux > 0 and "token_logits" in outputs:
                token_aux = ce_loss_fn(
                    outputs["token_logits"].reshape(-1, TOTAL_VOCAB),
                    tokens.reshape(-1),
                )
            loss = (ae_losses.total + args.w_token_aux * token_aux) / max(args.accum_steps, 1)
        scaler.scale(loss).backward()
        if (step + 1) % max(args.accum_steps, 1) == 0:
            scaler.unscale_(ae_opt)
            nn.utils.clip_grad_norm_(ae.parameters(), args.max_grad_norm)
            scaler.step(ae_opt)
            scaler.update()
            ae_opt.zero_grad()
        set_requires_grad(critic, True)

        totals["train/loss"] += loss.item() * max(args.accum_steps, 1)
        totals["train/critic_loss"] += critic_loss.item() * max(args.accum_steps, 1)
        totals["train/l1_loss"] += ae_losses.l1.item()
        totals["train/l2_loss"] += ae_losses.l2.item()
        totals["train/multi_scale_loss"] += ae_losses.multi_scale.item()
        totals["train/adv_loss"] += ae_losses.adversarial.item()
        totals["train/fm_loss"] += ae_losses.feature_matching.item()
        totals["train/token_aux_loss"] += float(token_aux.item()) if isinstance(token_aux, torch.Tensor) else 0.0
        num_batches += 1

        if logger and step % max(args.log_every, 1) == 0:
            logger({
                "train/loss": loss.item() * max(args.accum_steps, 1),
                "train/critic_loss": critic_loss.item() * max(args.accum_steps, 1),
                "train/l1_loss": ae_losses.l1.item(),
                "train/multi_scale_loss": ae_losses.multi_scale.item(),
            })
        if args.dry_run and step >= 2:
            break

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_ae(ae, dataloader, args, device, max_batches: int = 20):
    ae.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_hit_f1 = 0.0
    total_smoothness = 0.0
    total_onset_influence = 0.0
    num_batches = 0
    num_metrics = 0

    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break
        signal = batch["signal"]
        if signal is None:
            raise RuntimeError("AE validation requires signal tensors.")
        signal = signal.to(device)
        bpm = batch["bpm"].to(device)
        star_rating = batch["star_rating"].to(device)
        offset_ms = batch["target_start_ms"].to(device)

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = ae(signal)
            reconstruction = outputs["reconstruction"]
            loss = F.l1_loss(reconstruction, signal) + 0.5 * F.mse_loss(reconstruction, signal)
        total_loss += loss.item()

        signal_no_onset = signal.clone()
        signal_no_onset[:, 0:1, :] = 0.0
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            reconstruction_no_onset = ae(signal_no_onset)["reconstruction"]
        total_onset_influence += float((reconstruction.float() - reconstruction_no_onset.float()).abs().mean().item())

        for idx in range(signal.shape[0]):
            metrics = compute_signal_metrics(
                reconstruction[idx],
                signal[idx],
                bpm=float(bpm[idx].item()),
                offset_ms=float(offset_ms[idx].item()),
                star_rating=float(star_rating[idx].item()),
            )
            if metrics["timing_mae_ms"] != float("inf"):
                total_mae += metrics["timing_mae_ms"]
            total_hit_f1 += metrics["hit_f1"]
            total_smoothness += metrics["cursor_smoothness"]
            num_metrics += 1

        num_batches += 1

    return {
        "val/loss": total_loss / max(num_batches, 1),
        "val/edit_distance": 0.0,
        "val/timing_mae_ms": total_mae / max(num_metrics, 1),
        "val/hit_f1": total_hit_f1 / max(num_metrics, 1),
        "val/cursor_smoothness": total_smoothness / max(num_metrics, 1),
        "val/onset_influence": total_onset_influence / max(num_batches, 1),
    }


def train_one_epoch_flow(ae, flow_model, dataloader, optimizer, scaler, args, device, logger=None):
    ae.eval()
    flow_model.train()
    set_requires_grad(ae, False)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    totals = {
        "train/loss": 0.0,
        "train/flow_loss": 0.0,
        "train/reconstruction_aux_loss": 0.0,
        "train/token_aux_loss": 0.0,
    }
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        signal = batch["signal"]
        if signal is None:
            raise RuntimeError("Flow stage requires signal tensors in the dataset.")
        signal = signal.to(device)
        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        star_rating = batch["star_rating"].to(device)
        bpm = batch["bpm"].to(device)
        difficulty_id, star_rating = apply_difficulty_override(difficulty_id, star_rating, args)

        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                latent = ae.encode(signal)

        _, t, zt, velocity_target = sample_flow_inputs(latent)
        drop_difficulty = random.random() < max(0.0, min(args.difficulty_dropout, 1.0))

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = flow_model(
                z_t=zt,
                t=t,
                mel=mel,
                difficulty_id=difficulty_id,
                difficulty_value=star_rating,
                bpm=bpm,
                drop_difficulty=drop_difficulty,
            )
            flow_loss = F.mse_loss(outputs["velocity"], velocity_target)
            loss = args.w_flow * flow_loss

            reconstruction_aux = torch.zeros((), device=device)
            token_aux = torch.zeros((), device=device)
            predicted_clean = zt + (1.0 - t.view(-1, 1, 1)) * outputs["velocity"]
            if args.w_reconstruction_aux > 0:
                with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                    recon = ae.decode(predicted_clean, output_len=signal.shape[-1])
                reconstruction_aux = F.l1_loss(recon, signal)
                loss = loss + args.w_reconstruction_aux * reconstruction_aux
            if args.w_token_aux > 0:
                token_logits = ae.token_head(predicted_clean, tokens.shape[1])
                token_aux = ce_loss_fn(token_logits.reshape(-1, TOTAL_VOCAB), tokens.reshape(-1))
                loss = loss + args.w_token_aux * token_aux
            loss = loss / max(args.accum_steps, 1)

        scaler.scale(loss).backward()
        if (step + 1) % max(args.accum_steps, 1) == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(flow_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        totals["train/loss"] += loss.item() * max(args.accum_steps, 1)
        totals["train/flow_loss"] += flow_loss.item()
        totals["train/reconstruction_aux_loss"] += reconstruction_aux.item()
        totals["train/token_aux_loss"] += token_aux.item()
        num_batches += 1

        if logger and step % max(args.log_every, 1) == 0:
            logger({
                "train/loss": loss.item() * max(args.accum_steps, 1),
                "train/flow_loss": flow_loss.item(),
                "train/reconstruction_aux_loss": reconstruction_aux.item(),
            })
        if args.dry_run and step >= 2:
            break

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_flow(ae, flow_model, dataloader, args, device, max_batches: int = 10):
    ae.eval()
    flow_model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_hit_f1 = 0.0
    total_smoothness = 0.0
    total_onset_influence = 0.0
    num_batches = 0
    num_metrics = 0

    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break

        signal = batch["signal"]
        if signal is None:
            raise RuntimeError("Flow validation requires signal tensors.")
        signal = signal.to(device)
        mel = batch["mel"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        star_rating = batch["star_rating"].to(device)
        bpm = batch["bpm"].to(device)
        offset_ms = batch["target_start_ms"].to(device)
        difficulty_id, star_rating = apply_difficulty_override(difficulty_id, star_rating, args)

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            latent = ae.encode(signal)
        _, t, zt, velocity_target = sample_flow_inputs(latent)
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = flow_model(
                z_t=zt,
                t=t,
                mel=mel,
                difficulty_id=difficulty_id,
                difficulty_value=star_rating,
                bpm=bpm,
                drop_difficulty=False,
            )
            flow_loss = F.mse_loss(outputs["velocity"], velocity_target)
        total_loss += flow_loss.item()

        mel_no_onset = mel.clone()
        mel_no_onset[:, :, N_MELS:, :] = 0.0
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs_no_onset = flow_model(
                z_t=zt,
                t=t,
                mel=mel_no_onset,
                difficulty_id=difficulty_id,
                difficulty_value=star_rating,
                bpm=bpm,
                drop_difficulty=False,
            )
        total_onset_influence += float((outputs["velocity"].float() - outputs_no_onset["velocity"].float()).abs().mean().item())

        generated_latent = integrate_flow(
            flow_model=flow_model,
            latent_shape=tuple(latent.shape),
            mel=mel,
            difficulty_id=difficulty_id,
            difficulty_value=star_rating,
            bpm=bpm,
            steps=max(args.val_ode_steps, 1),
            guidance_scale=1.0,
            device=device,
        )
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            reconstruction = ae.decode(generated_latent, output_len=signal.shape[-1])
        for idx in range(signal.shape[0]):
            metrics = compute_signal_metrics(
                reconstruction[idx],
                signal[idx],
                bpm=float(bpm[idx].item()),
                offset_ms=float(offset_ms[idx].item()),
                star_rating=float(star_rating[idx].item()),
            )
            if metrics["timing_mae_ms"] != float("inf"):
                total_mae += metrics["timing_mae_ms"]
            total_hit_f1 += metrics["hit_f1"]
            total_smoothness += metrics["cursor_smoothness"]
            num_metrics += 1
        num_batches += 1

    return {
        "val/loss": total_loss / max(num_batches, 1),
        "val/edit_distance": 0.0,
        "val/timing_mae_ms": total_mae / max(num_metrics, 1),
        "val/hit_f1": total_hit_f1 / max(num_metrics, 1),
        "val/cursor_smoothness": total_smoothness / max(num_metrics, 1),
        "val/onset_influence": total_onset_influence / max(num_batches, 1),
    }


def build_legacy_model(args, device):
    model = OsuMapper(
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff=args.decoder_ff,
        dropout=args.dropout,
    ).to(device)
    return model


def build_ae_model(args, device):
    ae = SignalAutoencoder(
        hidden_dim=args.ae_hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    critic = SignalCritic().to(device)
    return ae, critic


def build_flow_model(args, device):
    return LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.flow_hidden_dim,
        cond_dim=args.cond_dim,
        encoder_unfreeze_last_n=args.flow_encoder_unfreeze_last_n,
    ).to(device)


def load_checkpoint_weights(module: nn.Module, state_dict: dict, strict: bool = True):
    load_model_state_dict(module, state_dict, strict=strict)


def main():
    args = parse_args()
    resolve_runtime_args(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.no_persistent_workers:
        args.persistent_workers = False
    if args.no_pin_memory:
        args.pin_memory = False
    if not args.no_persistent_workers and not args.persistent_workers:
        args.persistent_workers = True
    if not args.no_pin_memory and not args.pin_memory:
        args.pin_memory = True

    print(f"Stage: {args.stage}")
    print(f"Batch size: {args.batch_size} (accum {args.accum_steps})")
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
    shard_bytes = sum(os.path.getsize(path) for path in shards if os.path.exists(path))
    print(f"Data dir: {args.data_dir}")
    print(f"Shard count: {len(shards)}")
    if shard_bytes:
        print(f"Shard size: {shard_bytes / (1024 ** 3):.2f} GB")
    try:
        shm_free = shutil.disk_usage("/dev/shm").free
        print(f"/dev/shm free: {shm_free / (1024 ** 3):.2f} GB")
    except OSError:
        pass

    split_shards, split_file = prepare_cached_splits(args)
    logger_fn = build_logger(args)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ckpt_mgr = CheckpointManager(args.model_dir, top_k=3)

    start_epoch = 0
    best_val_loss = float("inf")
    no_improve = 0

    legacy_model = None
    ae = None
    critic = None
    flow_model = None
    optimizer = None
    critic_opt = None

    if args.stage == "legacy":
        legacy_model = build_legacy_model(args, device)
        total, trainable = compute_param_counts(legacy_model)
        print(f"Parameters: {total:,} total, {trainable:,} trainable")
        optimizer = torch.optim.AdamW(legacy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs // 5, 1), T_mult=2
        )
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            load_model_state_dict(legacy_model, ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed legacy model from epoch {start_epoch}")
        if args.compile:
            try:
                legacy_model = torch.compile(legacy_model)
                print("torch.compile enabled.")
            except Exception as exc:
                print(f"torch.compile failed, continuing without it: {exc}")
    elif args.stage == "ae":
        ae, critic = build_ae_model(args, device)
        total, trainable = compute_param_counts(ae)
        print(f"AE parameters: {total:,} total, {trainable:,} trainable")
        optimizer = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs // 5, 1), T_mult=2
        )
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            load_checkpoint_weights(ae, ckpt["model_state_dict"])
            if ckpt.get("critic_state_dict"):
                load_checkpoint_weights(critic, ckpt["critic_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("critic_optimizer_state_dict"):
                critic_opt.load_state_dict(ckpt["critic_optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed AE model from epoch {start_epoch}")
    else:
        ae, _ = build_ae_model(args, device)
        if not args.ae_checkpoint and not args.resume:
            raise ValueError("Flow stage requires --ae_checkpoint (or a resumable flow checkpoint).")
        if args.ae_checkpoint:
            ae_ckpt = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
            ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)
            load_checkpoint_weights(ae, ae_state)
            print(f"Loaded AE checkpoint: {args.ae_checkpoint}")
        set_requires_grad(ae, False)
        ae.eval()

        flow_model = build_flow_model(args, device)
        total, trainable = compute_param_counts(flow_model)
        print(f"Flow parameters: {total:,} total, {trainable:,} trainable")
        print(
            "Flow AST unfreeze_last_n: {n} ({mode})".format(
                n=args.flow_encoder_unfreeze_last_n,
                mode="frozen AST frontend" if args.flow_encoder_unfreeze_last_n == 0 else "partially trainable AST frontend",
            )
        )
        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs // 5, 1), T_mult=2
        )
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            load_checkpoint_weights(flow_model, ckpt["model_state_dict"])
            if ckpt.get("ae_state_dict"):
                load_checkpoint_weights(ae, ckpt["ae_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed flow model from epoch {start_epoch}")
        if args.compile:
            print("torch.compile is left disabled for flow stage to avoid graph breaks in chunked AST conditioning.")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_dataset = build_dataset(args, epoch, "train", split_shards)
        train_loader = build_dataloader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
        )

        val_dataset = build_dataset(args, epoch, "val", split_shards)
        val_loader = build_dataloader(
            val_dataset,
            args.batch_size,
            args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
        )

        if args.stage == "legacy":
            train_metrics = train_one_epoch_legacy(legacy_model, train_loader, optimizer, scaler, epoch, args, device, logger_fn)
            scheduler.step()
            val_metrics = validate_legacy(legacy_model, val_loader, args, device)
            checkpoint_state = {
                "stage": args.stage,
                "epoch": epoch,
                "model_state_dict": export_model_state_dict(legacy_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val/loss"],
                "scaler_state_dict": scaler.state_dict(),
            }
        elif args.stage == "ae":
            train_metrics = train_one_epoch_ae(ae, critic, train_loader, optimizer, critic_opt, scaler, args, device, logger_fn)
            scheduler.step()
            val_metrics = validate_ae(ae, val_loader, args, device)
            checkpoint_state = {
                "stage": args.stage,
                "epoch": epoch,
                "model_state_dict": export_model_state_dict(ae),
                "critic_state_dict": export_model_state_dict(critic),
                "optimizer_state_dict": optimizer.state_dict(),
                "critic_optimizer_state_dict": critic_opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val/loss"],
                "scaler_state_dict": scaler.state_dict(),
            }
        else:
            train_metrics = train_one_epoch_flow(ae, flow_model, train_loader, optimizer, scaler, args, device, logger_fn)
            scheduler.step()
            val_metrics = validate_flow(ae, flow_model, val_loader, args, device)
            checkpoint_state = {
                "stage": args.stage,
                "epoch": epoch,
                "model_state_dict": export_model_state_dict(flow_model),
                "ae_state_dict": export_model_state_dict(ae),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val/loss"],
                "scaler_state_dict": scaler.state_dict(),
                "ae_checkpoint": args.ae_checkpoint,
            }

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(f"  train_loss={train_metrics['train/loss']:.4f} | val_loss={val_metrics['val/loss']:.4f}")
        print(
            f"  timing_mae={val_metrics['val/timing_mae_ms']:.1f}ms | "
            f"hit_f1={val_metrics['val/hit_f1']:.4f} | "
            f"cursor_smoothness={val_metrics['val/cursor_smoothness']:.6f}"
        )
        print(f"  onset_influence={val_metrics['val/onset_influence']:.6f} | lr={lr:.6f} | time={elapsed:.1f}s")

        logger_payload = {"epoch": epoch, **train_metrics, **val_metrics}
        if split_shards:
            logger_payload["data/train_shards"] = len(split_shards["train"])
            logger_payload["data/val_shards"] = len(split_shards["val"])
            logger_payload["data/test_shards"] = len(split_shards["test"])
        logger_fn(logger_payload)

        if not args.dry_run:
            ckpt_mgr.save(checkpoint_state, val_metrics["val/loss"])

        improved = val_metrics["val/loss"] < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_metrics["val/loss"]
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
                    "  Early stopping after {patience} epochs without val improvement > {delta:.4f}".format(
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
