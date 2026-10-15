"""Deliberately opt-in command line training entry point for architecture v2.

Running this module is the only path in the repository that starts v2 model
optimization. It requires ``--confirm-training`` and is never imported by
preprocessing or inference. This guard is intentional: it prevents a shell
mistake on a login node or a machine with an unrelated active GPU run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import SparseSongDataset, collate_sparse_examples
from .corpus import audit_training_corpus
from .geometry import ActiveEventGeometryDiffuser, GeometryDiffusionConfig
from .grammar import MAX_ANCHORS
from .sparse_model import SparseEventPlanner, SparsePlannerConfig, count_trainable_parameters
from .training import (
    CheckpointProvenance,
    TrainingConfig,
    TrainingStage,
    checkpoint_payload,
    load_checkpoint,
    initialize_planner,
    make_optimizer,
    run_epoch,
    save_checkpoint,
    split_manifest_sha256,
)


def _mapping_json(path: str | None, label: str) -> Mapping[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label} JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tempo-posterior osu! v2 (explicit opt-in)")
    parser.add_argument("--records-dir", required=True, help="directory of safe v2 .pt song records")
    parser.add_argument("--split-manifest", required=True, help="immutable source-component split JSON")
    parser.add_argument("--feature-pipeline-key", required=True, help="shared FeatureCacheManifest.pipeline_key")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--stage", choices=[stage.value for stage in TrainingStage], default="planner")
    parser.add_argument("--initialize-planner", help="transfer legacy planner weights into local-audio architecture; fresh optimizer/run")
    parser.add_argument("--alignment-loss-weight", type=float, default=0.2)
    parser.add_argument("--resume", help="compatible v2 checkpoint to resume")
    parser.add_argument("--planner-config", help="optional JSON overrides for SparsePlannerConfig")
    parser.add_argument("--geometry-config", help="optional JSON overrides for GeometryDiffusionConfig")
    parser.add_argument("--epochs", type=int, default=200, help="total target epochs (default: 200)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience in epochs (0 to disable)")
    parser.add_argument("--min-delta", type=float, default=0.001, help="minimum validation loss improvement")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--profile-dropout", type=float, default=0.15)
    parser.add_argument("--boundary-context-dropout", type=float, default=0.15)
    parser.add_argument("--symbolic-context-dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1337, help="base seed for model and per-epoch data order")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--device", default="cuda", help="explicit device, e.g. cuda or cuda:0")
    parser.add_argument("--max-steps", type=int, help="optional bounded steps per epoch for a smoke run")
    parser.add_argument("--allow-diagnostic-data", action="store_true", help="mark small/unverified runs as unsuitable for generation")
    parser.add_argument("--bars-per-song", type=int, default=16, help="balanced bar samples per audio component per epoch")
    parser.add_argument("--spacing-loss-weight", type=float, default=0.1)
    parser.add_argument("--turn-loss-weight", type=float, default=0.05)
    parser.add_argument("--slider-shape-loss-weight", type=float, default=0.1)
    parser.add_argument("--confirm-training", action="store_true", help="required acknowledgement before optimization")
    return parser.parse_args(argv)


def _is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def _reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """All-reduce training or validation metrics across all distributed ranks."""
    if not dist.is_initialized():
        return metrics
    reduced = {}
    for name, value in metrics.items():
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if name == "steps":
            reduced[name] = float(tensor.item())
        else:
            reduced[name] = float(tensor.item() / dist.get_world_size())
    return reduced


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_training:
        raise SystemExit(
            "Refusing to start training without --confirm-training. "
            "This command can allocate an accelerator and create checkpoints."
        )
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(args.seed)
    if args.initialize_planner and (args.resume or args.stage != "planner"):
        raise ValueError("--initialize-planner is only for a fresh planner run, without --resume")
    output_dir = Path(args.checkpoint_dir)
    if any(output_dir.glob("*.pt")) and not args.resume:
        raise ValueError("Checkpoint directory already contains a run; use --resume or a new directory")
    if args.patience < 0 or args.min_delta < 0 or args.bars_per_song < 1:
        raise ValueError("patience/min-delta must be non-negative and bars-per-song positive")

    is_dist = _is_distributed()
    if is_dist:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl")
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu only for diagnostics")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    split_manifest = _mapping_json(args.split_manifest, "split manifest")
    data_audit = audit_training_corpus(args.records_dir, split_manifest,
                                     allow_diagnostic_data=args.allow_diagnostic_data)
    provenance = CheckpointProvenance(
        feature_pipeline_key=args.feature_pipeline_key,
        split_manifest_sha256=split_manifest_sha256(split_manifest),
    )
    stage = TrainingStage(args.stage)
    if stage is TrainingStage.GEOMETRY and not args.resume:
        raise ValueError(
            "geometry stage requires --resume from a compatible trained planner checkpoint; "
            "training geometry against random symbolic context is invalid"
        )
    planner_config = SparsePlannerConfig(**dict(_mapping_json(args.planner_config, "planner config")))
    geometry_payload = dict(_mapping_json(args.geometry_config, "geometry config"))
    geometry_payload.setdefault("context_dim", planner_config.model_dim)
    geometry_config = GeometryDiffusionConfig(**geometry_payload)
    if geometry_config.context_dim != planner_config.model_dim:
        raise ValueError(
            "geometry_config.context_dim must equal planner_config.model_dim; "
            "geometry conditions on planner decoder hidden states"
        )
    if geometry_config.event_feature_dim != 4:
        raise ValueError(
            "geometry_config.event_feature_dim must be 4 for the documented "
            "previous/next-gap, duration, and slider-length record contract"
        )
    if geometry_config.max_anchors != MAX_ANCHORS:
        raise ValueError(f"geometry max_anchors must match the {MAX_ANCHORS}-control record vocabulary")
    planner = SparseEventPlanner(planner_config).to(device)
    geometry = ActiveEventGeometryDiffuser(geometry_config).to(device)
    config = TrainingConfig(
        stage=stage,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation=args.gradient_accumulation,
        profile_dropout=args.profile_dropout,
        boundary_context_dropout=args.boundary_context_dropout,
        symbolic_context_dropout=args.symbolic_context_dropout,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        spacing_loss_weight=args.spacing_loss_weight,
        turn_loss_weight=args.turn_loss_weight,
        slider_shape_loss_weight=args.slider_shape_loss_weight,
        alignment_loss_weight=args.alignment_loss_weight,
    )
    initialization = initialize_planner(args.initialize_planner, planner, expected_provenance=provenance) if args.initialize_planner else None
    optimizer = make_optimizer(planner, geometry, config)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.mixed_precision == "fp16"))
    start_epoch = 0
    global_step = 0
    restored = {}
    if args.resume:
        restored = load_checkpoint(
            args.resume,
            planner=planner,
            geometry=geometry,
            optimizer=optimizer,
            scaler=scaler,
            expected_provenance=provenance,
            restore_rng_state=True,
            optimizer_stage=stage,
        )
        saved_stage = restored.get("_saved_training_stage")
        if saved_stage != stage.value and any(output_dir.glob("*.pt")):
            raise ValueError("Stage handoff requires a new checkpoint directory; existing checkpoints would be overwritten")
        if saved_stage == stage.value:
            start_epoch = int(restored["epoch"]) + 1
            global_step = int(restored["global_step"])
        elif stage is TrainingStage.GEOMETRY and saved_stage == TrainingStage.PLANNER.value:
            start_epoch = 0
            global_step = 0
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
        else:
            raise ValueError(
                f"cannot resume {stage.value} stage from checkpoint stage {saved_stage!r}; "
                "only same-stage resume or planner-to-geometry handoff is supported"
            )

    if start_epoch >= args.epochs:
        if rank == 0:
            print(
                f"Start epoch {start_epoch} is already >= total target epochs {args.epochs}. "
                "Training is already complete.",
                flush=True,
            )
        if is_dist and dist.is_initialized():
            dist.destroy_process_group()
        return 0

    train_dataset = SparseSongDataset(
        args.records_dir,
        split="train",
        expected_pipeline_key=args.feature_pipeline_key,
        clock_mode="mixed",
        sampling_mode="song",
        alignment_frames=planner_config.local_audio_tokens,
        bars_per_song=args.bars_per_song,
    )
    val_dataset = SparseSongDataset(
        args.records_dir,
        split="val",
        expected_pipeline_key=args.feature_pipeline_key,
        clock_mode="mixed",
        sampling_mode="bar",
        alignment_frames=planner_config.local_audio_tokens,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if is_dist
        else None
    )

    train_order_generator = torch.Generator()
    val_order_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_sparse_examples,
        generator=train_order_generator if train_sampler is None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                   shuffle=False, drop_last=False) if is_dist else None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_sparse_examples,
        generator=val_order_generator,
    )

    # Wrap model with DistributedDataParallel if distributed
    if is_dist:
        if stage is TrainingStage.PLANNER:
            train_planner: Any = DDP(
                planner,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
            train_geometry: Any = geometry
        else:
            train_planner = planner
            train_geometry = DDP(
                geometry,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
    else:
        train_planner = planner
        train_geometry = geometry

    if rank == 0:
        print(
            json.dumps(
                {
                    "planner_trainable_parameters": count_trainable_parameters(planner),
                    "geometry_trainable_parameters": count_trainable_parameters(geometry),
                    "stage": stage.value,
                    "device": str(device),
                    "world_size": world_size,
                    "start_epoch": start_epoch,
                    "total_epochs": args.epochs,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "data_audit": data_audit,
                    "train_examples_per_epoch": len(train_dataset),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir = Path(args.checkpoint_dir)
    same_stage = restored.get("_saved_training_stage") == stage.value
    previous_stopping = restored.get("early_stopping", {}) if same_stage else {}
    best_val_loss = float(previous_stopping.get("best_val_loss", float("inf")))
    epochs_without_improvement = int(previous_stopping.get("epochs_without_improvement", 0))

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.monotonic()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_dataset.set_epoch(epoch)
            val_dataset.set_epoch(0)

            train_order_generator.manual_seed(args.seed + epoch)
            val_order_generator.manual_seed(args.seed + 1_000_003)

            raw_train_metrics = run_epoch(
                planner=train_planner,
                geometry=train_geometry,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                config=config,
                scaler=scaler,
                max_steps=args.max_steps,
            )
            train_metrics = _reduce_metrics(raw_train_metrics, device)
            train_seconds = time.monotonic() - epoch_started
            global_step += int(train_metrics["steps"])

            # Fix validation diffusion noise without perturbing training RNG.
            rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(args.seed + 1_000_003)
                raw_val_metrics = run_epoch(
                    planner=train_planner, geometry=train_geometry, loader=val_loader,
                    optimizer=None, device=device, config=config, max_steps=args.max_steps,
                )
            val_metrics = _reduce_metrics(raw_val_metrics, device)

            stop_training = False
            if rank == 0:
                raw_p = planner.module if isinstance(planner, DDP) else planner
                raw_g = geometry.module if (geometry is not None and isinstance(geometry, DDP)) else geometry
                payload = checkpoint_payload(
                    planner=raw_p,
                    geometry=raw_g,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    training_config=config,
                    provenance=provenance,
                )
                val_loss = float(val_metrics["loss"])
                is_best = False
                if val_loss < best_val_loss - args.min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    is_best = True
                else:
                    epochs_without_improvement += 1
                    if args.patience > 0 and epochs_without_improvement >= args.patience:
                        stop_training = True

                payload["initialization"] = initialization or restored.get("initialization")
                payload["data_audit"] = data_audit
                payload["early_stopping"] = {"best_val_loss": best_val_loss,
                                              "epochs_without_improvement": epochs_without_improvement}
                latest_ckpt = save_checkpoint(payload, output_dir / "latest.pt", overwrite=True)
                if is_best:
                    save_checkpoint(payload, output_dir / "best.pt", overwrite=True)
                if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
                    save_checkpoint(payload, output_dir / f"{stage.value}-epoch-{epoch:04d}.pt", overwrite=True)

                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_seconds": train_seconds,
                            "epoch_seconds": time.monotonic() - epoch_started,
                            "train": train_metrics,
                            "val": val_metrics,
                            "is_best": is_best,
                            "best_val_loss": best_val_loss,
                            "epochs_without_improvement": epochs_without_improvement,
                            "checkpoint": str(latest_ckpt),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if is_dist:
                stop_tensor = torch.tensor(1 if stop_training else 0, dtype=torch.int32, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item() == 1)

            if stop_training:
                if rank == 0:
                    print(f"Early stopping triggered at epoch {epoch} (patience={args.patience})", flush=True)
                break
    finally:
        if is_dist and dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI is intentionally not invoked in unit tests.
    raise SystemExit(main())
