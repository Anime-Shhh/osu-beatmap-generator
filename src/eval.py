"""
Evaluation script for held-out shard splits.

Usage:
    python -m src.eval \
        --data_dir /common/users/asj102/osu_project/data/shards \
        --split_file /common/users/asj102/osu_project/logs/splits.json \
        --split_name test \
        --checkpoint /common/users/asj102/osu_project/models/best.pt
"""

import argparse
import os
from statistics import mean, pstdev

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from .dataset import OsuCachedDataset, build_dataloader
from .metrics import compute_all_metrics
from .model import OsuMapper
from .splits import load_split_manifest, resolve_split_shards
from .tokenizer import PAD, Residuals, TOTAL_VOCAB


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on val/test split")
    p.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/asj102/osu_project/data/shards",
    )
    p.add_argument("--split_file", type=str, required=True)
    p.add_argument("--split_name", type=str, choices=["val", "test"], default="test")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--max_batches", type=int, default=0, help="0 means full split")

    # Must match trained checkpoint architecture.
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--decoder_layers", type=int, default=6)
    p.add_argument("--decoder_heads", type=int, default=8)
    p.add_argument("--decoder_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


@torch.no_grad()
def run_eval(model, dataloader, device: torch.device, max_batches: int = 0):
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    loss_values = []
    edit_values = []
    timing_values = []
    hit_f1_values = []
    slider_iou_values = []

    for step, batch in enumerate(dataloader):
        if max_batches > 0 and step >= max_batches:
            break

        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        time_res_gt = batch["time_residuals"].to(device)
        x_res_gt = batch["x_residuals"].to(device)
        y_res_gt = batch["y_residuals"].to(device)

        tgt_input = tokens[:, :-1]
        tgt_output = tokens[:, 1:]
        padding_mask = tgt_input == PAD

        with autocast(dtype=torch.float16):
            outputs = model(mel, tgt_input, padding_mask)
            logits = outputs["logits"]
            loss = ce_loss_fn(
                logits.reshape(-1, TOTAL_VOCAB),
                tgt_output.reshape(-1),
            )
        loss_values.append(float(loss.item()))

        pred_tokens_batch = logits.argmax(dim=-1)
        for b in range(tokens.shape[0]):
            true_t = tgt_output[b].cpu().tolist()
            valid_len = len([t for t in true_t if t != PAD])
            if valid_len == 0:
                continue
            true_t = true_t[:valid_len]
            pred_t = pred_tokens_batch[b].cpu().tolist()[:valid_len]

            pred_res = []
            true_res = []
            for idx in range(valid_len):
                pred_res.append(
                    Residuals(
                        time_offset_ms=float(outputs["time_residuals"][b, idx, 0].item()),
                        x_offset_px=float(outputs["x_residuals"][b, idx, 0].item()),
                        y_offset_px=float(outputs["y_residuals"][b, idx, 0].item()),
                    )
                )
                true_res.append(
                    Residuals(
                        time_offset_ms=float(time_res_gt[b, idx + 1].item()),
                        x_offset_px=float(x_res_gt[b, idx + 1].item()),
                        y_offset_px=float(y_res_gt[b, idx + 1].item()),
                    )
                )

            metrics = compute_all_metrics(pred_t, true_t, pred_res, true_res)
            edit_values.append(float(metrics["edit_distance"]))
            if metrics["timing_mae_ms"] != float("inf"):
                timing_values.append(float(metrics["timing_mae_ms"]))
            hit_f1_values.append(float(metrics["hit_f1"]))
            slider_iou_values.append(float(metrics["slider_iou"]))

    return {
        "loss": loss_values,
        "edit_distance": edit_values,
        "timing_mae_ms": timing_values,
        "hit_f1": hit_f1_values,
        "slider_iou": slider_iou_values,
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
    shard_list = split_shards.get(args.split_name, [])
    shard_list = [s for s in shard_list if os.path.exists(s)]
    if not shard_list:
        raise RuntimeError(f"no shards found for split '{args.split_name}' under {args.data_dir}")

    print(
        f"Shards in split '{args.split_name}': {len(shard_list)} "
        f"(manifest total={manifest.get('shards_total', 'n/a')})"
    )

    model = OsuMapper(
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff=args.decoder_ff,
        dropout=args.dropout,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = OsuCachedDataset(
        shard_pattern=os.path.join(args.data_dir, "shard-*.tar"),
        shuffle=False,
        shards=shard_list,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers or args.num_workers > 0,
        pin_memory=args.pin_memory or torch.cuda.is_available(),
    )

    values = run_eval(model, dataloader, device, max_batches=args.max_batches)

    summary = {
        "loss": (mean(values["loss"]) if values["loss"] else float("nan"), _safe_std(values["loss"])),
        "edit_distance": (
            mean(values["edit_distance"]) if values["edit_distance"] else float("nan"),
            _safe_std(values["edit_distance"]),
        ),
        "timing_mae_ms": (
            mean(values["timing_mae_ms"]) if values["timing_mae_ms"] else float("nan"),
            _safe_std(values["timing_mae_ms"]),
        ),
        "hit_f1": (mean(values["hit_f1"]) if values["hit_f1"] else float("nan"), _safe_std(values["hit_f1"])),
        "slider_iou": (
            mean(values["slider_iou"]) if values["slider_iou"] else float("nan"),
            _safe_std(values["slider_iou"]),
        ),
    }

    print("=== Evaluation Summary ===")
    for key in ("loss", "edit_distance", "timing_mae_ms", "hit_f1", "slider_iou"):
        metric_mean, metric_std = summary[key]
        print(f"{key}: mean={metric_mean:.6f} std={metric_std:.6f}")


if __name__ == "__main__":
    main()
