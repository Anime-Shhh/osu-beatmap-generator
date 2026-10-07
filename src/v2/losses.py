"""Mask-aware objectives for the architecture-v2 beat-grid model.

The old residual objective supervised every non-PAD token, including token
categories that have no coordinate or timing residual.  These losses supervise
each field only where it is semantically defined and exclude padded song tails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .events import EventKind, IGNORE_INDEX
from .model import add_coordinate_noise


@dataclass
class PlannerLosses:
    total: torch.Tensor
    occupancy: torch.Tensor
    kind: torch.Tensor
    timing: torch.Tensor
    base_geometry: torch.Tensor
    slider_length: torch.Tensor
    slider_repeats: torch.Tensor
    spinner_duration: torch.Tensor


def collate_beat_grid_targets(targets: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of per-song grid targets and return an explicit slot mask.

    This is deliberately a small, dependency-free collator.  The caller may
    add metadata separately; every tensor below is safe to pass to the v2 loss
    functions without padded values influencing gradients.
    """

    if not targets:
        raise ValueError("cannot collate an empty target list")
    required = {
        "occupancy",
        "kind",
        "time_residual_beats",
        "coordinates",
        "end_beat_delta",
        "slider_length",
        "slider_repeats",
    }
    if any(not required.issubset(target) for target in targets):
        raise ValueError(f"each target must contain: {sorted(required)}")

    lengths = [int(target["occupancy"].shape[0]) for target in targets]
    lanes = int(targets[0]["occupancy"].shape[1])
    if any(target["occupancy"].shape[1] != lanes for target in targets):
        raise ValueError("all targets in a batch must use the same lane count")
    maximum = max(lengths)
    batch_size = len(targets)

    result = {
        "slot_mask": torch.zeros((batch_size, maximum), dtype=torch.bool),
        "occupancy": torch.zeros((batch_size, maximum, lanes), dtype=torch.float32),
        "kind": torch.full((batch_size, maximum, lanes), IGNORE_INDEX, dtype=torch.long),
        "time_residual_beats": torch.zeros((batch_size, maximum, lanes), dtype=torch.float32),
        "coordinates": torch.zeros((batch_size, maximum, lanes, 2), dtype=torch.float32),
        "end_beat_delta": torch.zeros((batch_size, maximum, lanes), dtype=torch.float32),
        "slider_length": torch.zeros((batch_size, maximum, lanes), dtype=torch.float32),
        "slider_repeats": torch.zeros((batch_size, maximum, lanes), dtype=torch.long),
    }
    for index, (target, length) in enumerate(zip(targets, lengths)):
        result["slot_mask"][index, :length] = True
        for key in required:
            result[key][index, :length] = target[key]
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_planner_losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    subdivisions_per_beat: int,
    occupancy_weight: float = 1.0,
    kind_weight: float = 1.0,
    timing_weight: float = 2.0,
    geometry_weight: float = 0.5,
    slider_weight: float = 0.25,
) -> PlannerLosses:
    """Compute class-balanced, semantic-field-aware planner losses."""

    required_outputs = {
        "occupancy_logits",
        "kind_logits",
        "time_residual_beats",
        "base_coordinates",
        "spinner_duration_beats",
        "slider_length",
        "slider_repeat_logits",
    }
    required_targets = {
        "slot_mask",
        "occupancy",
        "kind",
        "time_residual_beats",
        "coordinates",
        "end_beat_delta",
        "slider_length",
        "slider_repeats",
    }
    if not required_outputs.issubset(outputs):
        raise ValueError(f"outputs missing keys: {sorted(required_outputs - set(outputs))}")
    if not required_targets.issubset(targets):
        raise ValueError(f"targets missing keys: {sorted(required_targets - set(targets))}")

    logits = outputs["occupancy_logits"]
    occupancy = targets["occupancy"].to(logits.dtype)
    slot_mask = targets["slot_mask"].bool()
    valid = slot_mask.unsqueeze(-1)
    if logits.shape != occupancy.shape or logits.shape[:2] != slot_mask.shape:
        raise ValueError("occupancy logits/targets/slot_mask have incompatible shapes")

    positives = (occupancy * valid).sum()
    negatives = ((1.0 - occupancy) * valid).sum()
    # Cap the dynamic weight so a nearly empty song cannot dominate a batch.
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 32.0)
    occupancy_per_item = F.binary_cross_entropy_with_logits(
        logits,
        occupancy,
        pos_weight=positive_weight,
        reduction="none",
    )
    occupancy_loss = _masked_mean(occupancy_per_item, valid)

    active = valid & (occupancy > 0.5)
    kind_logits = outputs["kind_logits"]
    kind_target = targets["kind"].long()
    if kind_logits.shape[:-1] != kind_target.shape:
        raise ValueError("kind logits and targets have incompatible shapes")
    if active.any():
        kind_loss = F.cross_entropy(kind_logits[active], kind_target[active])
    else:
        kind_loss = _zero_like(kind_logits)

    residual_bound = 0.5 / float(subdivisions_per_beat)
    timing_error = F.smooth_l1_loss(
        outputs["time_residual_beats"] / residual_bound,
        targets["time_residual_beats"].to(logits.dtype) / residual_bound,
        reduction="none",
    )
    timing_loss = _masked_mean(timing_error, active)

    geometry_error = F.smooth_l1_loss(
        outputs["base_coordinates"],
        targets["coordinates"].to(logits.dtype),
        reduction="none",
    ).mean(dim=-1)
    geometry_loss = _masked_mean(geometry_error, active)

    slider_mask = active & (kind_target == int(EventKind.SLIDER))
    spinner_mask = active & (kind_target == int(EventKind.SPINNER))
    if slider_mask.any():
        slider_length_loss = _masked_mean(
            F.smooth_l1_loss(
                outputs["slider_length"], targets["slider_length"].to(logits.dtype), reduction="none"
            ),
            slider_mask,
        )
        repeat_target = (targets["slider_repeats"].long() - 1).clamp_min(0)
        slider_repeats_loss = F.cross_entropy(
            outputs["slider_repeat_logits"][slider_mask], repeat_target[slider_mask]
        )
    else:
        slider_length_loss = _zero_like(logits)
        slider_repeats_loss = _zero_like(logits)

    if spinner_mask.any():
        spinner_duration_loss = _masked_mean(
            F.smooth_l1_loss(
                outputs["spinner_duration_beats"],
                targets["end_beat_delta"].to(logits.dtype),
                reduction="none",
            ),
            spinner_mask,
        )
    else:
        spinner_duration_loss = _zero_like(logits)

    total = (
        occupancy_weight * occupancy_loss
        + kind_weight * kind_loss
        + timing_weight * timing_loss
        + geometry_weight * geometry_loss
        + slider_weight * (slider_length_loss + slider_repeats_loss + spinner_duration_loss)
    )
    return PlannerLosses(
        total=total,
        occupancy=occupancy_loss,
        kind=kind_loss,
        timing=timing_loss,
        base_geometry=geometry_loss,
        slider_length=slider_length_loss,
        slider_repeats=slider_repeats_loss,
        spinner_duration=spinner_duration_loss,
    )


def coordinate_diffusion_loss(
    refiner,
    context: torch.Tensor,
    clean_coordinates: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    noise_level: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard epsilon-prediction loss, scored only at actual event slots."""

    batch = clean_coordinates.shape[0]
    if noise_level is None:
        noise_level = torch.rand(batch, device=clean_coordinates.device, dtype=clean_coordinates.dtype)
    noisy, target_noise = add_coordinate_noise(clean_coordinates, noise_level)
    predicted_noise = refiner(context, noisy, noise_level)
    error = F.mse_loss(predicted_noise, target_noise, reduction="none").mean(dim=-1)
    return _masked_mean(error, active_mask.bool())
