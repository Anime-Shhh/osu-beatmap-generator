"""Explicit, resumable training primitives for architecture v2.

Nothing in this module starts a run at import time.  The public functions are
used only by the separately invoked ``python -m src.v2.train`` command, which
itself requires an explicit confirmation flag.  This keeps accidental imports
on a login node or an active workstation from consuming a GPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from .dataset import V2_SONG_RECORD_SCHEMA, move_batch_to_device
from .events import IGNORE_INDEX
from .geometry import ActiveEventGeometryDiffuser, geometry_diffusion_loss
from .grammar import EventVocabulary
from .preprocess import ARCHITECTURE_ID
from .sparse_model import SparseEventPlanner, SparsePlannerConfig


CHECKPOINT_SCHEMA = "osu-beatmap-generator/v2-checkpoint/2"


class TrainingStage(str, Enum):
    PLANNER = "planner"
    GEOMETRY = "geometry"


@dataclass(frozen=True)
class TrainingConfig:
    """Small operational controls; model shape lives in its own config."""

    stage: TrainingStage = TrainingStage.PLANNER
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.05
    gradient_accumulation: int = 1
    max_gradient_norm: float = 1.0
    profile_dropout: float = 0.15
    boundary_context_dropout: float = 0.15
    symbolic_context_dropout: float = 0.10
    mixed_precision: str = "bf16"  # bf16 | fp16 | none
    seed: int = 1337
    spacing_loss_weight: float = 0.1
    turn_loss_weight: float = 0.05
    slider_shape_loss_weight: float = 0.1
    geometry_flip_augmentation: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_accumulation < 1 or self.max_gradient_norm <= 0.0:
            raise ValueError("gradient accumulation and max gradient norm must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("spacing_loss_weight", "turn_loss_weight", "slider_shape_loss_weight"):
            if not 0 <= getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("profile_dropout", "boundary_context_dropout", "symbolic_context_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        if self.mixed_precision not in {"bf16", "fp16", "none"}:
            raise ValueError("mixed_precision must be bf16, fp16, or none")


@dataclass(frozen=True)
class CheckpointProvenance:
    """Compatibility boundary that prevents a misleading resume."""

    feature_pipeline_key: str
    split_manifest_sha256: str
    architecture_id: str = ARCHITECTURE_ID
    record_schema: str = V2_SONG_RECORD_SCHEMA
    vocabulary_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("feature_pipeline_key", "split_manifest_sha256", "architecture_id", "record_schema"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.vocabulary_sha256 and len(self.vocabulary_sha256) != 64:
            raise ValueError("vocabulary_sha256 must be a SHA-256 digest when supplied")

    def to_dict(self) -> dict[str, str]:
        return {
            "feature_pipeline_key": self.feature_pipeline_key,
            "split_manifest_sha256": self.split_manifest_sha256,
            "architecture_id": self.architecture_id,
            "record_schema": self.record_schema,
            "vocabulary_sha256": self.vocabulary_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointProvenance":
        return cls(
            feature_pipeline_key=str(payload["feature_pipeline_key"]),
            split_manifest_sha256=str(payload["split_manifest_sha256"]),
            architecture_id=str(payload.get("architecture_id", ARCHITECTURE_ID)),
            record_schema=str(payload.get("record_schema", V2_SONG_RECORD_SCHEMA)),
            vocabulary_sha256=str(payload.get("vocabulary_sha256", "")),
        )


def vocabulary_sha256(vocabulary: EventVocabulary) -> str:
    """Hash exact token IDs, not merely a tokenizer class name."""

    names = [vocabulary.name_for(index) for index in range(len(vocabulary))]
    payload = json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(b"osu-v2-sparse-vocabulary\0" + payload).hexdigest()


def split_manifest_sha256(manifest: Mapping[str, Any] | str | bytes) -> str:
    """Hash an immutable source-component split manifest canonically."""

    if isinstance(manifest, Mapping):
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif isinstance(manifest, str):
        raw = manifest.encode("utf-8")
    else:
        raw = bytes(manifest)
    return hashlib.sha256(b"osu-v2-source-split\0" + raw).hexdigest()


def make_optimizer(
    planner: SparseEventPlanner,
    geometry: ActiveEventGeometryDiffuser | None,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Construct the stage-appropriate optimizer only when a run asks for it."""

    if config.stage is TrainingStage.PLANNER:
        parameters = list(planner.parameters())
    else:
        if geometry is None:
            raise ValueError("geometry stage requires an ActiveEventGeometryDiffuser")
        parameters = list(geometry.parameters())
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))


def _autocast_context(device: torch.device, precision: str):
    if precision == "none" or device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    # CPU float16 autocast has little support and is not a supported v2 run.
    if device.type == "cpu" and dtype is torch.float16:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def planner_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy over exactly non-padding next-token labels."""

    if logits.dim() != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [B,L,V] and labels [B,L]")
    if not labels.ne(IGNORE_INDEX).any():
        raise ValueError("planner batch has no supervised tokens")
    return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=IGNORE_INDEX)


def gather_event_context(
    decoder_hidden: torch.Tensor,
    event_token_indices: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather one causal symbolic context vector for every active event."""

    if decoder_hidden.dim() != 3 or event_token_indices.dim() != 2 or event_mask.shape != event_token_indices.shape:
        raise ValueError("decoder_hidden=[B,L,D], event indices/mask=[B,N] required")
    if decoder_hidden.shape[0] != event_token_indices.shape[0]:
        raise ValueError("event context batch dimensions must agree")
    if event_token_indices.numel() and (
        (event_token_indices < 0).any() or (event_token_indices >= decoder_hidden.shape[1]).any()
    ):
        raise ValueError("event token indices are outside decoder hidden sequence")
    indices = event_token_indices.unsqueeze(-1).expand(-1, -1, decoder_hidden.shape[-1])
    context = decoder_hidden.gather(1, indices)
    return context * event_mask.unsqueeze(-1).to(context.dtype)


def _drop_symbolic_context(context: torch.Tensor, event_mask: torch.Tensor, probability: float) -> torch.Tensor:
    """Expose geometry to imperfect planner context without fabricating labels."""

    if probability <= 0.0:
        return context
    drop = torch.rand(event_mask.shape, device=context.device) < probability
    keep = event_mask & ~drop
    return context * keep.unsqueeze(-1).to(context.dtype)


def forward_planner(
    planner: SparseEventPlanner,
    batch: Mapping[str, Any],
    *,
    profile_dropout: float = 0.0,
    boundary_context_dropout: float = 0.0,
) -> dict[str, torch.Tensor]:
    """One teacher-forced planner pass using real feature timestamps."""

    drop_profile: bool | torch.Tensor = False
    if profile_dropout:
        drop_profile = torch.rand(batch["profile"].shape[0], device=batch["profile"].device) < profile_dropout
    drop_boundary: bool | torch.Tensor = False
    if boundary_context_dropout:
        # Keep condition sampling per-example, not batch-global. Reuse no RNG
        # state from DataLoader workers so resume behavior remains local to the
        # training process.
        drop_boundary = (
            torch.rand(batch["profile"].shape[0], device=batch["profile"].device)
            < boundary_context_dropout
        )
    return planner(
        rhythm_features=batch["rhythm_features"],
        rhythm_times_ms=batch["rhythm_times_ms"],
        semantic_features=batch["semantic_features"],
        semantic_times_ms=batch["semantic_times_ms"],
        bar_mid_times_ms=batch["bar_mid_times_ms"],
        profile=batch["profile"],
        clock_features=batch["clock_features"],
        decoder_tokens=batch["decoder_tokens"],
        target_bar_index=batch["target_bar_index"],
        bar_mask=batch["bar_mask"],
        rhythm_mask=batch["rhythm_mask"],
        semantic_mask=batch["semantic_mask"],
        token_mask=batch["decoder_mask"],
        drop_profile=drop_profile,
        previous_bar_tokens=batch["previous_bar_tokens"],
        previous_bar_mask=batch["previous_bar_mask"],
        drop_boundary=drop_boundary,
    )


def loss_for_batch(
    planner: SparseEventPlanner,
    geometry: ActiveEventGeometryDiffuser | None,
    batch: Mapping[str, Any],
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Compute the stage loss; gradients are owned by the caller."""

    if config.stage is TrainingStage.PLANNER:
        planner_output = forward_planner(
            planner,
            batch,
            profile_dropout=config.profile_dropout,
            boundary_context_dropout=config.boundary_context_dropout,
        )
        token_loss = planner_token_loss(planner_output["logits"], batch["labels"])
        return {"loss": token_loss, "token_loss": token_loss}
    if geometry is None:
        raise ValueError("geometry stage requires a geometry diffuser")
    # Geometry learns from planner-shaped symbolic context but never updates
    # the planner in this stage. Keeping this pass in no_grad avoids retaining
    # a second full decoder graph and makes the stage safe even when callers
    # invoke ``loss_for_batch`` outside ``run_epoch``.
    with torch.no_grad():
        planner_output = forward_planner(
            planner,
            batch,
            profile_dropout=config.profile_dropout,
            boundary_context_dropout=config.boundary_context_dropout,
        )
        token_loss = planner_token_loss(planner_output["logits"], batch["labels"])
    context = gather_event_context(
        planner_output["decoder_hidden"], batch["event_token_indices"], batch["event_mask"]
    )
    context = _drop_symbolic_context(context, batch["event_mask"], config.symbolic_context_dropout)
    if batch["event_mask"].shape[1] == 0:
        # Every DDP rank must enter the diffuser forward/backward, even when
        # its entire batch contains empty bars. A masked dummy event keeps the
        # graph and reducer hooks intact while contributing exactly zero loss.
        batch = dict(batch)
        size = context.shape[0]
        context = context.new_zeros((size, 1, context.shape[-1]))
        for name in ("event_mask", "event_kinds", "anchor_counts", "event_positions"):
            batch[name] = batch[name].new_zeros((size, 1))
        batch["geometry"] = batch["geometry"].new_zeros((size, 1, batch["geometry"].shape[2], 2))
        batch["event_features"] = batch["event_features"].new_zeros((size, 1, 4))
    clean_geometry = batch["geometry"]
    if geometry.training and config.geometry_flip_augmentation:
        # Reflect whole patterns, never individual objects; preserve spacing,
        # slider topology, handedness diversity, and the rectangular playfield.
        flips = torch.rand((clean_geometry.shape[0], 1, 1, 2), device=clean_geometry.device) < 0.5
        clean_geometry = torch.where(flips, 1.0 - clean_geometry, clean_geometry)
    components = geometry_diffusion_loss(
        geometry,
        context,
        batch["event_kinds"],
        batch["anchor_counts"],
        clean_geometry,
        batch["event_mask"],
        event_features=batch["event_features"],
        event_positions=batch["event_positions"],
        spacing_weight=config.spacing_loss_weight,
        turn_weight=config.turn_loss_weight,
        slider_shape_weight=config.slider_shape_loss_weight,
        return_components=True,
    )
    return {"loss": components["geometry_loss"], "token_loss": token_loss.detach(), **components}


def run_epoch(
    *,
    planner: SparseEventPlanner,
    geometry: ActiveEventGeometryDiffuser | None,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device | str,
    config: TrainingConfig,
    scaler: torch.amp.GradScaler | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one explicitly requested train or evaluation epoch.

    ``optimizer=None`` is evaluation. Callers decide whether to put modules in
    train/eval mode and whether this routine is ever invoked; no surrounding
    code calls it automatically.
    """

    target_device = torch.device(device)
    training = optimizer is not None
    if not training:
        # Condition dropout must not corrupt validation or early stopping.
        from dataclasses import replace
        config = replace(config, profile_dropout=0.0, boundary_context_dropout=0.0,
                         symbolic_context_dropout=0.0)
    if config.stage is TrainingStage.GEOMETRY:
        # Geometry consumes symbolic contexts from an evaluation-mode planner.
        # ``loss_for_batch`` uses no_grad for that pass, avoiding both planner
        # gradients and a permanent requires_grad mutation that would surprise
        # callers reusing a model object for a later planner stage.
        planner.eval()
        if geometry is None:
            raise ValueError("geometry stage requires geometry model")
        geometry.train(training)
    else:
        planner.train(training)
    if not training:
        planner.eval()
        if geometry is not None:
            geometry.eval()

    totals: dict[str, float] = {}
    steps = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for source_batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        batch = move_batch_to_device(source_batch, target_device)
        context = _autocast_context(target_device, config.mixed_precision)
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, context:
            losses = loss_for_batch(planner, geometry, batch, config)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite training/validation loss; refusing to update weights or save a misleading checkpoint")
            scaled_loss = losses["loss"] / config.gradient_accumulation
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if (steps + 1) % config.gradient_accumulation == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    geometry.parameters() if config.stage is TrainingStage.GEOMETRY and geometry is not None else planner.parameters(),
                    config.max_gradient_norm,
                )
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().float().cpu())
        steps += 1
    if training and steps and steps % config.gradient_accumulation:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        # The last group can be shorter; undo division by the full group size.
        correction = config.gradient_accumulation / (steps % config.gradient_accumulation)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(
            geometry.parameters() if config.stage is TrainingStage.GEOMETRY and geometry is not None else planner.parameters(),
            config.max_gradient_norm,
        )
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if not steps:
        raise ValueError("epoch loader yielded no batches")
    return {name: value / steps for name, value in totals.items()} | {"steps": float(steps)}


def checkpoint_payload(
    *,
    planner: SparseEventPlanner,
    geometry: ActiveEventGeometryDiffuser | None,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    training_config: TrainingConfig,
    provenance: CheckpointProvenance,
) -> dict[str, Any]:
    """Build a fully versioned checkpoint payload without writing it yet."""

    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")
    vocabulary_digest = vocabulary_sha256(planner.vocabulary)
    if provenance.vocabulary_sha256 and provenance.vocabulary_sha256 != vocabulary_digest:
        raise ValueError("checkpoint provenance vocabulary does not match planner vocabulary")
    finalized_provenance = CheckpointProvenance(
        **{**provenance.to_dict(), "vocabulary_sha256": vocabulary_digest}
    )
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "provenance": finalized_provenance.to_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "training_config": {**asdict(training_config), "stage": training_config.stage.value},
        "planner_config": asdict(planner.config),
        "geometry_config": None if geometry is None else asdict(geometry.config),
        "planner": planner.state_dict(),
        "geometry": None if geometry is None else geometry.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        # Preserve torch RNG only. Data-order generators are reseeded from the
        # recorded epoch in the CLI, while this state covers model dropout,
        # diffusion noise, and condition dropout after a resume.
        "rng_state": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        },
    }


def save_checkpoint(payload: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write a checkpoint after its caller has explicitly trained."""

    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("refusing to write an unknown checkpoint schema")
    destination = Path(path)
    if destination.suffix not in {".pt", ".pth"}:
        raise ValueError("checkpoint path must end in .pt or .pth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        if not overwrite and destination.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    planner: SparseEventPlanner,
    geometry: ActiveEventGeometryDiffuser | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    expected_provenance: CheckpointProvenance | None = None,
    strict: bool = True,
    restore_rng_state: bool = False,
    optimizer_stage: TrainingStage | None = None,
) -> dict[str, Any]:
    """Safely load a compatible checkpoint; reject cache/tokenizer drift."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"unable to load safe v2 checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported v2 checkpoint schema")
    actual = CheckpointProvenance.from_dict(_mapping(payload.get("provenance"), "provenance"))
    current_vocabulary = vocabulary_sha256(planner.vocabulary)
    if actual.vocabulary_sha256 != current_vocabulary:
        raise ValueError("checkpoint vocabulary differs from current planner vocabulary")
    if expected_provenance is not None:
        for name in ("feature_pipeline_key", "split_manifest_sha256", "architecture_id", "record_schema"):
            if getattr(actual, name) != getattr(expected_provenance, name):
                raise ValueError(f"checkpoint {name} differs from requested run")
    training_payload = payload.get("training_config")
    saved_stage: str | None = None
    if isinstance(training_payload, Mapping):
        raw_stage = training_payload.get("stage")
        if raw_stage is not None:
            saved_stage = str(raw_stage)
    optimizer_compatible = optimizer_stage is None or saved_stage == optimizer_stage.value
    if geometry is not None and saved_stage == TrainingStage.GEOMETRY.value:
        saved_space = payload.get("geometry_config", {}).get("coordinate_space", "legacy-logit")
        if geometry.config.coordinate_space != saved_space:
            raise ValueError("checkpoint geometry coordinate space differs; do not resume trained geometry with a new representation")
    planner.load_state_dict(_mapping(payload.get("planner"), "planner"), strict=strict)
    if geometry is not None and payload.get("geometry") is not None:
        geometry.load_state_dict(_mapping(payload["geometry"], "geometry"), strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None and optimizer_compatible:
        optimizer.load_state_dict(_mapping(payload["optimizer"], "optimizer"))
    if scaler is not None and payload.get("scaler") is not None and optimizer_compatible:
        scaler.load_state_dict(_mapping(payload["scaler"], "scaler"))
    if restore_rng_state and optimizer_compatible:
        raw_rng = payload.get("rng_state")
        if not isinstance(raw_rng, Mapping):
            raise ValueError("checkpoint is missing resumable torch RNG state")
        cpu_state = raw_rng.get("torch_cpu")
        if not isinstance(cpu_state, torch.Tensor):
            raise ValueError("checkpoint torch CPU RNG state is invalid")
        torch.set_rng_state(cpu_state.cpu())
        cuda_states = raw_rng.get("torch_cuda", ())
        if torch.cuda.is_available():
            if not isinstance(cuda_states, (tuple, list)) or not all(
                isinstance(state, torch.Tensor) for state in cuda_states
            ):
                raise ValueError("checkpoint CUDA RNG state is invalid")
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG state does not match available device count")
            torch.cuda.set_rng_state_all(list(cuda_states))
    result = dict(payload)
    result["_optimizer_state_loaded"] = optimizer_compatible
    result["_saved_training_stage"] = saved_stage
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} is not a mapping")
    return value


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CheckpointProvenance",
    "TrainingConfig",
    "TrainingStage",
    "checkpoint_payload",
    "forward_planner",
    "gather_event_context",
    "load_checkpoint",
    "loss_for_batch",
    "make_optimizer",
    "planner_token_loss",
    "run_epoch",
    "save_checkpoint",
    "split_manifest_sha256",
    "vocabulary_sha256",
]
