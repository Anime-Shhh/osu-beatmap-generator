"""Deliberately opt-in command line training entry point for architecture v2.

Running this module is the only path in the repository that starts v2 model
optimization. It requires ``--confirm-training`` and is never imported by
preprocessing or inference. This guard is intentional: it prevents a shell
mistake on a login node or a machine with an unrelated active GPU run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from .dataset import SparseSongDataset, collate_sparse_examples
from .geometry import ActiveEventGeometryDiffuser, GeometryDiffusionConfig
from .sparse_model import SparseEventPlanner, SparsePlannerConfig, count_trainable_parameters
from .training import (
    CheckpointProvenance,
    TrainingConfig,
    TrainingStage,
    checkpoint_payload,
    load_checkpoint,
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
    parser.add_argument("--resume", help="compatible v2 checkpoint to resume")
    parser.add_argument("--planner-config", help="optional JSON overrides for SparsePlannerConfig")
    parser.add_argument("--geometry-config", help="optional JSON overrides for GeometryDiffusionConfig")
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--confirm-training", action="store_true", help="required acknowledgement before optimization")
    return parser.parse_args(argv)


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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu only for diagnostics")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    split_manifest = _mapping_json(args.split_manifest, "split manifest")
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
    )
    optimizer = make_optimizer(planner, geometry, config)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.mixed_precision == "fp16"))
    start_epoch = 0
    global_step = 0
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
        if saved_stage == stage.value:
            start_epoch = int(restored["epoch"]) + 1
            global_step = int(restored["global_step"])
        elif stage is TrainingStage.GEOMETRY and saved_stage == TrainingStage.PLANNER.value:
            # This is the intentional stage handoff: retain planner weights
            # but initialize a fresh geometry optimizer, scaler, epoch, and
            # RNG progression instead of trying to load Adam moments belonging
            # to the planner parameter set.
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

    train_dataset = SparseSongDataset(
        args.records_dir,
        split="train",
        expected_pipeline_key=args.feature_pipeline_key,
        clock_mode="mixed",
        sampling_mode="song",
    )
    val_dataset = SparseSongDataset(
        args.records_dir,
        split="val",
        expected_pipeline_key=args.feature_pipeline_key,
        # Records normally contain oracle plus jittered/estimated variants.
        # ``mixed`` still permits a corpus-audit-only oracle record to run a
        # bounded smoke check instead of failing at DataLoader construction.
        clock_mode="mixed",
        sampling_mode="bar",
    )
    train_order_generator = torch.Generator()
    val_order_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        # Workers must be recreated each epoch so their dataset copies receive
        # ``set_epoch`` and rotate bar/clock selections correctly. Persistent
        # workers would retain epoch zero and silently undermine sampling.
        persistent_workers=False,
        collate_fn=collate_sparse_examples,
        generator=train_order_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_sparse_examples,
        generator=val_order_generator,
    )
    print(
        json.dumps(
            {
                "planner_trainable_parameters": count_trainable_parameters(planner),
                "geometry_trainable_parameters": count_trainable_parameters(geometry),
                "stage": stage.value,
                "device": str(device),
                "start_epoch": start_epoch,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    output_dir = Path(args.checkpoint_dir)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        # Explicit per-epoch sampler seeds make a resumed run use the same
        # song-component order as an uninterrupted run at that epoch.
        train_order_generator.manual_seed(args.seed + epoch)
        val_order_generator.manual_seed(args.seed + 1_000_003 + epoch)
        train_metrics = run_epoch(
            planner=planner,
            geometry=geometry,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            scaler=scaler,
            max_steps=args.max_steps,
        )
        global_step += int(train_metrics["steps"])
        val_metrics = run_epoch(
            planner=planner,
            geometry=geometry,
            loader=val_loader,
            optimizer=None,
            device=device,
            config=config,
            max_steps=args.max_steps,
        )
        payload = checkpoint_payload(
            planner=planner,
            geometry=geometry,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            training_config=config,
            provenance=provenance,
        )
        checkpoint = save_checkpoint(payload, output_dir / f"{stage.value}-epoch-{epoch:04d}.pt")
        print(
            json.dumps(
                {"epoch": epoch, "train": train_metrics, "val": val_metrics, "checkpoint": str(checkpoint)},
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI is intentionally not invoked in unit tests.
    raise SystemExit(main())
