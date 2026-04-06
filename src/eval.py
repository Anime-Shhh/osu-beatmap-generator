"""
Standalone evaluation for legacy, AE, and flow checkpoints.
"""

from __future__ import annotations

import argparse
import os
from statistics import mean, pstdev

import torch

from .checkpoints import load_model_state_dict
from .dataset import FULL_SONG_SAMPLE, WINDOW_SAMPLE, OsuCachedDataset, build_dataloader
from .flow_model import LatentFlowMatcher, integrate_flow, sample_flow_inputs
from .latent_ae import SignalAutoencoder
from .metrics import compute_all_metrics, compute_signal_metrics
from .model import OsuMapper
from .splits import load_split_manifest, resolve_split_shards
from .tokenizer import PAD, Residuals, TOTAL_VOCAB


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on val/test split")
    p.add_argument("--data_dir", type=str, default="/common/users/asj102/osu_project/data/shards")
    p.add_argument("--split_file", type=str, required=True)
    p.add_argument("--split_name", type=str, choices=["val", "test"], default="test")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ae_checkpoint", type=str, default="")
    p.add_argument("--stage", type=str, choices=["auto", "legacy", "ae", "flow"], default="auto")
    p.add_argument("--quality_filter", type=str, choices=["all", "ranked_only"], default="all")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--ode_steps", type=int, default=16)

    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--decoder_layers", type=int, default=6)
    p.add_argument("--decoder_heads", type=int, default=8)
    p.add_argument("--decoder_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=96)
    p.add_argument("--ae_hidden_dim", type=int, default=256)
    p.add_argument("--flow_hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=256)
    return p.parse_args()


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def _resolve_stage(args, checkpoint: dict) -> str:
    if args.stage != "auto":
        return args.stage
    return checkpoint.get("stage", "legacy")


def _load_state(module, state_dict):
    load_model_state_dict(module, state_dict, strict=True)


@torch.no_grad()
def run_eval_legacy(model, dataloader, device: torch.device, max_batches: int = 0):
    import torch.nn as nn
    from torch.cuda.amp import autocast

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    loss_values = []
    edit_values = []
    timing_values = []
    hit_f1_values = []
    onset_influence_values = []

    for step, batch in enumerate(dataloader):
        if max_batches > 0 and step >= max_batches:
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

        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = model(mel, tgt_input, difficulty_id, bpm, padding_mask)
            logits = outputs["logits"]
            loss = ce_loss_fn(logits.reshape(-1, TOTAL_VOCAB), tgt_output.reshape(-1))
        loss_values.append(float(loss.item()))

        mel_no_onset = mel.clone()
        mel_no_onset[:, :, -1:, :] = 0.0
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs_no_onset = model(mel_no_onset, tgt_input, difficulty_id, bpm, padding_mask)
        onset_influence_values.append(
            float((outputs["logits"].float() - outputs_no_onset["logits"].float()).abs().mean().item())
        )

        pred_tokens_batch = logits.argmax(dim=-1)
        for batch_idx in range(tokens.shape[0]):
            true_tokens = tgt_output[batch_idx].cpu().tolist()
            valid_len = len([token for token in true_tokens if token != PAD])
            if valid_len == 0:
                continue
            true_tokens = true_tokens[:valid_len]
            pred_tokens = pred_tokens_batch[batch_idx].cpu().tolist()[:valid_len]

            pred_residuals = []
            true_residuals = []
            for idx in range(valid_len):
                pred_residuals.append(
                    Residuals(
                        time_offset_ms=float(outputs["time_residuals"][batch_idx, idx, 0].item()),
                        x_offset_px=float(outputs["x_residuals"][batch_idx, idx, 0].item()),
                        y_offset_px=float(outputs["y_residuals"][batch_idx, idx, 0].item()),
                    )
                )
                true_residuals.append(
                    Residuals(
                        time_offset_ms=float(time_res_gt[batch_idx, idx + 1].item()),
                        x_offset_px=float(x_res_gt[batch_idx, idx + 1].item()),
                        y_offset_px=float(y_res_gt[batch_idx, idx + 1].item()),
                    )
                )
            ms_per_beat = 60000.0 / bpm[batch_idx].item()
            metrics = compute_all_metrics(
                pred_tokens,
                true_tokens,
                pred_residuals,
                true_residuals,
                ms_per_beat=ms_per_beat,
            )
            edit_values.append(float(metrics["edit_distance"]))
            if metrics["timing_mae_ms"] != float("inf"):
                timing_values.append(float(metrics["timing_mae_ms"]))
            hit_f1_values.append(float(metrics["hit_f1"]))

    return {
        "loss": loss_values,
        "edit_distance": edit_values,
        "timing_mae_ms": timing_values,
        "hit_f1": hit_f1_values,
        "cursor_smoothness": [],
        "onset_influence": onset_influence_values,
    }


@torch.no_grad()
def run_eval_ae(ae, dataloader, device: torch.device, max_batches: int = 0):
    loss_values = []
    timing_values = []
    hit_f1_values = []
    cursor_smoothness_values = []
    onset_influence_values = []

    for step, batch in enumerate(dataloader):
        if max_batches > 0 and step >= max_batches:
            break
        signal = batch["signal"]
        if signal is None:
            raise RuntimeError("AE evaluation requires signal tensors.")
        signal = signal.to(device)
        bpm = batch["bpm"].to(device)
        star_rating = batch["star_rating"].to(device)
        offset_ms = batch["target_start_ms"].to(device)

        outputs = ae(signal)
        reconstruction = outputs["reconstruction"]
        loss = torch.nn.functional.l1_loss(reconstruction, signal) + 0.5 * torch.nn.functional.mse_loss(reconstruction, signal)
        loss_values.append(float(loss.item()))

        signal_no_onset = signal.clone()
        signal_no_onset[:, 0:1, :] = 0.0
        reconstruction_no_onset = ae(signal_no_onset)["reconstruction"]
        onset_influence_values.append(float((reconstruction.float() - reconstruction_no_onset.float()).abs().mean().item()))

        for idx in range(signal.shape[0]):
            metrics = compute_signal_metrics(
                reconstruction[idx],
                signal[idx],
                bpm=float(bpm[idx].item()),
                offset_ms=float(offset_ms[idx].item()),
                star_rating=float(star_rating[idx].item()),
            )
            if metrics["timing_mae_ms"] != float("inf"):
                timing_values.append(float(metrics["timing_mae_ms"]))
            hit_f1_values.append(float(metrics["hit_f1"]))
            cursor_smoothness_values.append(float(metrics["cursor_smoothness"]))

    return {
        "loss": loss_values,
        "edit_distance": [],
        "timing_mae_ms": timing_values,
        "hit_f1": hit_f1_values,
        "cursor_smoothness": cursor_smoothness_values,
        "onset_influence": onset_influence_values,
    }


@torch.no_grad()
def run_eval_flow(ae, flow_model, dataloader, device: torch.device, max_batches: int = 0, ode_steps: int = 16):
    loss_values = []
    timing_values = []
    hit_f1_values = []
    cursor_smoothness_values = []
    onset_influence_values = []

    for step, batch in enumerate(dataloader):
        if max_batches > 0 and step >= max_batches:
            break
        signal = batch["signal"]
        if signal is None:
            raise RuntimeError("Flow evaluation requires signal tensors.")
        signal = signal.to(device)
        mel = batch["mel"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        star_rating = batch["star_rating"].to(device)
        bpm = batch["bpm"].to(device)
        offset_ms = batch["target_start_ms"].to(device)

        latent = ae.encode(signal)
        _, t, zt, velocity_target = sample_flow_inputs(latent)
        outputs = flow_model(
            z_t=zt,
            t=t,
            mel=mel,
            difficulty_id=difficulty_id,
            difficulty_value=star_rating,
            bpm=bpm,
            drop_difficulty=False,
        )
        loss_values.append(float(torch.nn.functional.mse_loss(outputs["velocity"], velocity_target).item()))

        mel_no_onset = mel.clone()
        mel_no_onset[:, :, -1:, :] = 0.0
        outputs_no_onset = flow_model(
            z_t=zt,
            t=t,
            mel=mel_no_onset,
            difficulty_id=difficulty_id,
            difficulty_value=star_rating,
            bpm=bpm,
            drop_difficulty=False,
        )
        onset_influence_values.append(float((outputs["velocity"].float() - outputs_no_onset["velocity"].float()).abs().mean().item()))

        generated_latent = integrate_flow(
            flow_model=flow_model,
            latent_shape=tuple(latent.shape),
            mel=mel,
            difficulty_id=difficulty_id,
            difficulty_value=star_rating,
            bpm=bpm,
            steps=max(ode_steps, 1),
            guidance_scale=1.0,
            device=device,
        )
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
                timing_values.append(float(metrics["timing_mae_ms"]))
            hit_f1_values.append(float(metrics["hit_f1"]))
            cursor_smoothness_values.append(float(metrics["cursor_smoothness"]))

    return {
        "loss": loss_values,
        "edit_distance": [],
        "timing_mae_ms": timing_values,
        "hit_f1": hit_f1_values,
        "cursor_smoothness": cursor_smoothness_values,
        "onset_influence": onset_influence_values,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Split file: {args.split_file}")
    print(f"Split name: {args.split_name}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.split_file):
        raise FileNotFoundError(f"split file not found: {args.split_file}")

    manifest = load_split_manifest(args.split_file)
    split_shards = resolve_split_shards(args.data_dir, manifest)
    shard_list = [path for path in split_shards.get(args.split_name, []) if os.path.exists(path)]
    if not shard_list:
        raise RuntimeError(f"no shards found for split '{args.split_name}' under {args.data_dir}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stage = _resolve_stage(args, checkpoint)
    sample_type = WINDOW_SAMPLE if stage == "legacy" else FULL_SONG_SAMPLE

    dataset = OsuCachedDataset(
        shard_pattern=os.path.join(args.data_dir, "shard-*.tar"),
        shuffle=False,
        shards=shard_list,
        sample_type=sample_type,
        signal_required=(stage != "legacy"),
        quality_filter=args.quality_filter,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers or args.num_workers > 0,
        pin_memory=args.pin_memory or torch.cuda.is_available(),
    )

    print(f"Resolved stage: {stage}")
    print(f"Shards in split '{args.split_name}': {len(shard_list)}")

    if stage == "legacy":
        model = OsuMapper(
            d_model=args.d_model,
            decoder_layers=args.decoder_layers,
            decoder_heads=args.decoder_heads,
            decoder_ff=args.decoder_ff,
            dropout=args.dropout,
        ).to(device)
        load_model_state_dict(model, checkpoint["model_state_dict"])
        model.eval()
        values = run_eval_legacy(model, dataloader, device, max_batches=args.max_batches)
    elif stage == "ae":
        ae = SignalAutoencoder(hidden_dim=args.ae_hidden_dim, latent_dim=args.latent_dim).to(device)
        _load_state(ae, checkpoint["model_state_dict"])
        ae.eval()
        values = run_eval_ae(ae, dataloader, device, max_batches=args.max_batches)
    else:
        ae = SignalAutoencoder(hidden_dim=args.ae_hidden_dim, latent_dim=args.latent_dim).to(device)
        ae_state = checkpoint.get("ae_state_dict")
        if ae_state is None:
            if not args.ae_checkpoint:
                raise ValueError("Flow evaluation requires --ae_checkpoint when the flow checkpoint does not embed AE weights.")
            ae_checkpoint = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
            ae_state = ae_checkpoint.get("model_state_dict", ae_checkpoint)
        _load_state(ae, ae_state)
        ae.eval()

        flow_model = LatentFlowMatcher(
            latent_dim=args.latent_dim,
            hidden_dim=args.flow_hidden_dim,
            cond_dim=args.cond_dim,
        ).to(device)
        _load_state(flow_model, checkpoint["model_state_dict"])
        flow_model.eval()
        values = run_eval_flow(
            ae,
            flow_model,
            dataloader,
            device,
            max_batches=args.max_batches,
            ode_steps=args.ode_steps,
        )

    summary = {
        key: (mean(values[key]) if values[key] else float("nan"), _safe_std(values[key]))
        for key in ("loss", "edit_distance", "timing_mae_ms", "hit_f1", "cursor_smoothness", "onset_influence")
    }

    print("=== Evaluation Summary ===")
    for key in ("loss", "edit_distance", "timing_mae_ms", "hit_f1", "cursor_smoothness", "onset_influence"):
        metric_mean, metric_std = summary[key]
        print(f"{key}: mean={metric_mean:.6f} std={metric_std:.6f}")


if __name__ == "__main__":
    main()
