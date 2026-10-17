"""CPU-only validation, optional difficulty calculation, and candidate ranking.

This module intentionally separates *hard* output gates from a quality score.
An unavailable star calculator or a high learned score must never turn an
invalid map into a packageable result.  Conversely, a valid draft is never
described as automatically rankable: final mapper review remains required.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from .generation import CandidateManifest, GenerationRequest, GenerationResult


@dataclass(frozen=True)
class StrictAuditPolicy:
    """Explicit deterministic rules used before a generated draft is packaged.

    The rules are deliberately narrower than osu!'s human ranking criteria.
    They reject invalid files and clear timeline conflicts but do not pretend to
    judge artistic quality.  Slider overlaps are warnings because nuanced,
    legal mapper intent exists; spinner conflicts are errors because a player
    cannot service another hit object during an active spinner.
    """

    same_tick_tolerance_ms: float = 0.5
    snap_tolerance_beats: float = 1.0 / 96.0
    max_snap_error_beats: float = 0.25
    spinner_overlap_tolerance_ms: float = 0.5
    slider_overlap_tolerance_ms: float = 0.5
    allowed_snap_divisors: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96)
    max_star_error: float | None = None
    minimum_start_gap_ms: float = 0.0
    check_slider_paths: bool = True
    audio_duration_ms: float | None = None
    active_audio_intervals_ms: tuple[tuple[float, float], ...] | None = None
    minimum_audio_coverage: float = 0.85
    maximum_unmapped_edge_ms: float = 5000.0

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_audio_coverage <= 1 or not math.isfinite(self.maximum_unmapped_edge_ms) or self.maximum_unmapped_edge_ms < 0:
            raise ValueError("invalid audio coverage tolerances")
        if self.audio_duration_ms is not None and (not math.isfinite(self.audio_duration_ms) or self.audio_duration_ms <= 0):
            raise ValueError("audio duration must be finite and positive")
        if self.active_audio_intervals_ms is not None:
            previous = 0.0
            for start, end in self.active_audio_intervals_ms:
                if not (math.isfinite(start) and math.isfinite(end) and previous <= start < end and self.audio_duration_ms is not None and end <= self.audio_duration_ms):
                    raise ValueError("active audio intervals must be sorted, disjoint and inside audio")
                previous = end
        if self.same_tick_tolerance_ms < 0.0:
            raise ValueError("same_tick_tolerance_ms must be non-negative")
        if self.minimum_start_gap_ms < 0 or not math.isfinite(self.minimum_start_gap_ms):
            raise ValueError("minimum_start_gap_ms must be finite and non-negative")
        if self.max_star_error is not None and (not math.isfinite(self.max_star_error) or self.max_star_error < 0):
            raise ValueError("max_star_error must be finite and non-negative")
        if self.snap_tolerance_beats < 0.0 or self.max_snap_error_beats <= 0.0:
            raise ValueError("snap tolerances must be positive")
        if not self.allowed_snap_divisors or any(value <= 0 for value in self.allowed_snap_divisors):
            raise ValueError("allowed_snap_divisors must contain positive values")


@dataclass(frozen=True)
class DifficultyCalculation:
    """Result from an optional, pinned external difficulty calculator."""

    available: bool
    stars: Optional[float] = None
    calculator_id: str = "unavailable"
    attributes: Mapping[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.stars is not None and (not math.isfinite(float(self.stars)) or float(self.stars) < 0.0):
            raise ValueError("difficulty stars must be finite and non-negative")
        if self.available and self.stars is None:
            raise ValueError("an available difficulty calculation must include stars")
        for key, value in self.attributes.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"difficulty attribute {key!r} is non-finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "stars": self.stars,
            "calculator_id": self.calculator_id,
            "attributes": {str(key): float(value) for key, value in self.attributes.items()},
            "error": self.error,
        }


class DifficultyCalculator(Protocol):
    """Minimal adapter protocol so tests and deployments can supply a backend."""

    def calculate(self, osu_content: str) -> DifficultyCalculation:
        """Calculate star rating for a compiled ``.osu`` document."""


class RosuDifficultyCalculator:
    """Pinned, in-process no-mod stable difficulty; version is part of the audit."""

    def __init__(self) -> None:
        from importlib.metadata import version
        import rosu_pp_py

        installed = version("rosu-pp-py")
        if installed != "4.0.2":
            raise ValueError(f"Expected rosu-pp-py==4.0.2, found {installed}; rebuild labels before upgrading")
        self.backend = rosu_pp_py
        self.calculator_id = f"rosu-pp-py/{installed}/stable/no-mods"

    def calculate(self, osu_content: str) -> DifficultyCalculation:
        beatmap = self.backend.Beatmap(content=osu_content)
        result = self.backend.Difficulty(lazer=False).calculate(beatmap)
        return DifficultyCalculation(
            available=True, stars=result.stars, calculator_id=self.calculator_id,
            attributes={name: float(getattr(result, name)) for name in ("aim", "speed")
                        if getattr(result, name, None) is not None},
        )


@dataclass(frozen=True)
class UnavailableDifficultyCalculator:
    """Safe default when no official-compatible calculator is configured."""

    calculator_id: str = "unavailable"
    reason: str = "no pinned difficulty calculator was configured"

    def calculate(self, osu_content: str) -> DifficultyCalculation:
        del osu_content
        return DifficultyCalculation(
            available=False,
            calculator_id=self.calculator_id,
            error=self.reason,
        )


@dataclass(frozen=True)
class CallbackDifficultyCalculator:
    """Adapter for an in-process, explicitly pinned calculator callable."""

    callback: Callable[[str], DifficultyCalculation]

    def calculate(self, osu_content: str) -> DifficultyCalculation:
        result = self.callback(osu_content)
        if not isinstance(result, DifficultyCalculation):
            raise TypeError("difficulty callback must return DifficultyCalculation")
        return result


@dataclass(frozen=True)
class CommandDifficultyCalculator:
    """Optional JSON-command adapter for a pinned external calculator.

    ``command`` is an argument vector, never a shell string, and must contain
    exactly one ``{input}`` element.  The invoked program receives a temporary
    ``.osu`` file and must print one JSON object with a finite ``stars`` (or
    ``star_rating``) field.  No command is discovered, downloaded, or executed
    automatically; deployment has to opt in with an audited pinned command.
    """

    command: tuple[str, ...]
    calculator_id: str
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.command or not self.command[0]:
            raise ValueError("difficulty calculator command must be non-empty")
        if self.command.count("{input}") != 1:
            raise ValueError("difficulty calculator command must contain exactly one '{input}' argument")
        if self.timeout_seconds <= 0:
            raise ValueError("difficulty calculator timeout must be positive")
        if not self.calculator_id.strip():
            raise ValueError("calculator_id must be non-empty")

    @property
    def is_available(self) -> bool:
        executable = self.command[0]
        return bool(shutil.which(executable) or (Path(executable).is_file() and os.access(executable, os.X_OK)))

    def calculate(self, osu_content: str) -> DifficultyCalculation:
        if not self.is_available:
            return DifficultyCalculation(
                available=False,
                calculator_id=self.calculator_id,
                error=f"calculator executable is unavailable: {self.command[0]}",
            )
        try:
            with tempfile.TemporaryDirectory(prefix="osu-v2-difficulty-") as directory:
                input_path = Path(directory) / "candidate.osu"
                input_path.write_text(osu_content, encoding="utf-8")
                command = [str(input_path) if item == "{input}" else item for item in self.command]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return DifficultyCalculation(
                available=False,
                calculator_id=self.calculator_id,
                error=f"calculator execution failed: {exc}",
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or f"calculator exited {completed.returncode}"
            return DifficultyCalculation(available=False, calculator_id=self.calculator_id, error=error)
        try:
            payload = json.loads(completed.stdout)
            if not isinstance(payload, Mapping):
                raise ValueError("calculator output must be a JSON object")
            stars_value = payload.get("stars", payload.get("star_rating"))
            if stars_value is None:
                raise ValueError("calculator output is missing stars")
            attributes = payload.get("attributes", {})
            if not isinstance(attributes, Mapping):
                raise ValueError("calculator attributes must be a JSON object")
            return DifficultyCalculation(
                available=True,
                stars=float(stars_value),
                calculator_id=str(payload.get("calculator_id", self.calculator_id)),
                attributes={str(key): float(value) for key, value in attributes.items()},
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return DifficultyCalculation(
                available=False,
                calculator_id=self.calculator_id,
                error=f"invalid calculator JSON: {exc}",
            )


@dataclass(frozen=True)
class Candidate:
    """A sampled map and the non-authoritative model score used for ties."""

    candidate_id: str
    map_spec: Any
    seed: Optional[int] = None
    model_quality_score: Optional[float] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or any(character.isspace() for character in self.candidate_id):
            raise ValueError("candidate_id must be a non-empty whitespace-free identifier")
        if self.model_quality_score is not None:
            score = float(self.model_quality_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("model_quality_score must be finite and within [0, 1]")


@dataclass(frozen=True)
class CandidateAudit:
    """Hard-gate result and transparent deterministic ranking inputs."""

    candidate: Candidate
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: Mapping[str, float]
    difficulty: DifficultyCalculation
    score: Optional[float]
    event_fingerprint: str

    def to_manifest(self, *, selected: bool = False) -> CandidateManifest:
        return CandidateManifest(
            candidate_id=self.candidate.candidate_id,
            seed=self.candidate.seed,
            valid=self.valid,
            selected=selected,
            score=self.score,
            errors=self.errors,
            warnings=self.warnings,
            metrics=dict(self.metrics),
            difficulty=self.difficulty.to_dict(),
            provenance=dict(self.candidate.provenance),
        )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _messages(report: Any, name: str) -> tuple[str, ...]:
    values = _field(report, name, ())
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _map_components(map_spec: Any) -> tuple[tuple[Any, ...], Any]:
    events = _field(map_spec, "events")
    timing_map = _field(map_spec, "timing_map")
    if events is None or timing_map is None:
        raise ValueError("candidate map_spec must expose events and timing_map")
    return tuple(events), timing_map


def _event_times(events: Sequence[Any], timing_map: Any) -> list[tuple[float, float, Any]]:
    values: list[tuple[float, float, Any]] = []
    for event in events:
        beat = float(_field(event, "beat_position"))
        start = float(timing_map.beat_to_time(beat))
        end_beat = _field(event, "end_beat")
        end = start if end_beat is None else float(timing_map.beat_to_time(float(end_beat)))
        values.append((start, end, event))
    return sorted(values, key=lambda value: (value[0], value[1]))


def _kind_name(event: Any) -> str:
    kind = _field(event, "kind")
    return str(getattr(kind, "name", kind)).lower()


def _nearest_snap_error(beat: float, divisors: Iterable[int]) -> float:
    return min(abs(beat - round(beat * divisor) / divisor) for divisor in divisors)


def _canonical_control_points(event: Any) -> tuple[tuple[float | str, float | str], ...]:
    """Return a stable fingerprint fragment without crashing on invalid input."""

    points: list[tuple[float | str, float | str]] = []
    for point in (_field(event, "control_points", ()) or ()):
        try:
            points.append((round(float(point[0]), 2), round(float(point[1]), 2)))
        except (IndexError, TypeError, ValueError):
            points.append(("invalid", "invalid"))
    return tuple(points)


def _generic_strict_audit(map_spec: Any, policy: StrictAuditPolicy) -> tuple[list[str], list[str], dict[str, float], str]:
    """Audit protocol-compatible MapSpecs even when an external parser is absent."""

    events, timing_map = _map_components(map_spec)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        timed = _event_times(events, timing_map)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return [f"unable to convert event timing: {exc}"], warnings, {}, "invalid-timing"
    if not timed:
        errors.append("map contains no hit objects")
        return errors, warnings, {"object_count": 0.0}, "empty"

    snap_errors: list[float] = []
    gap_values: list[float] = []
    object_counts: dict[str, int] = {}
    canonical_events: list[tuple[Any, ...]] = []
    previous_start: Optional[float] = None
    active_spinners: list[tuple[float, int]] = []
    active_sliders: list[tuple[float, int]] = []
    max_jump_speed = 0.0
    previous_position: Optional[tuple[float, float, float]] = None

    for index, (start_ms, end_ms, event) in enumerate(timed):
        kind = _kind_name(event)
        object_counts[kind] = object_counts.get(kind, 0) + 1
        beat = float(_field(event, "beat_position"))
        x, y = float(_field(event, "x")), float(_field(event, "y"))
        snap_error = _nearest_snap_error(beat, policy.allowed_snap_divisors)
        snap_errors.append(snap_error)
        if not (math.isfinite(start_ms) and math.isfinite(end_ms) and math.isfinite(x) and math.isfinite(y)):
            errors.append(f"event {index}: non-finite timeline or coordinate")
            continue
        if kind not in {"circle", "slider", "spinner"}:
            errors.append(f"event {index}: unsupported object kind {kind!r}")
        if not (0.0 <= x <= 512.0 and 0.0 <= y <= 384.0):
            errors.append(f"event {index}: coordinate is outside the osu! playfield")
        if start_ms < 0.0:
            errors.append(f"event {index}: negative timestamp {start_ms:.3f} ms")
        if end_ms < start_ms:
            errors.append(f"event {index}: interval ends before it starts")
        if snap_error > policy.max_snap_error_beats:
            errors.append(f"event {index}: timing is too far from allowed snaps ({snap_error:.6f} beats)")
        elif snap_error > policy.snap_tolerance_beats:
            warnings.append(f"event {index}: off-grid residual {snap_error:.6f} beats")

        while active_spinners and active_spinners[0][0] <= start_ms + policy.spinner_overlap_tolerance_ms:
            active_spinners.pop(0)
        while active_sliders and active_sliders[0][0] <= start_ms + policy.slider_overlap_tolerance_ms:
            active_sliders.pop(0)
        if active_spinners:
            errors.append(f"event {index}: begins while spinner event {active_spinners[0][1]} is active")
        if active_sliders:
            warnings.append(f"event {index}: begins while slider event {active_sliders[0][1]} is active")

        if previous_start is not None:
            gap = start_ms - previous_start
            gap_values.append(gap)
            if gap <= policy.same_tick_tolerance_ms:
                errors.append(f"event {index}: same-tick or non-increasing start time")
            elif gap < policy.minimum_start_gap_ms:
                errors.append(f"event {index}: start gap {gap:.3f} ms is below requested minimum")
        previous_start = start_ms

        if previous_position is not None:
            prior_x, prior_y, prior_time = previous_position
            delta_ms = start_ms - prior_time
            if delta_ms > 0.0:
                max_jump_speed = max(max_jump_speed, math.hypot(x - prior_x, y - prior_y) / (delta_ms / 1000.0))
        if kind != "spinner":
            previous_position = (x, y, start_ms)

        if kind == "spinner" and end_ms > start_ms:
            active_spinners.append((end_ms, index))
            active_spinners.sort()
        elif kind == "slider" and end_ms > start_ms:
            active_sliders.append((end_ms, index))
            active_sliders.sort()

        if kind in {"spinner", "slider"} and _field(event, "end_beat") is None:
            errors.append(f"event {index}: {kind} is missing an end beat")
        if kind == "slider":
            curve_type = str(_field(event, "curve_type", "B")).upper()
            control_points = tuple(_field(event, "control_points", ()) or ())
            repeats = int(_field(event, "slider_repeats", 0) or 0)
            length = float(_field(event, "slider_length", 0.0) or 0.0)
            if curve_type not in {"B", "L", "P", "C"}:
                errors.append(f"event {index}: unsupported slider curve type {curve_type!r}")
            if repeats < 1:
                errors.append(f"event {index}: slider repeats must be positive")
            if not math.isfinite(length) or length <= 0.0:
                errors.append(f"event {index}: slider length must be positive")
            if not control_points:
                errors.append(f"event {index}: slider has no endpoint/control point")
            for point_index, point in enumerate(control_points):
                try:
                    point_x, point_y = float(point[0]), float(point[1])
                except (IndexError, TypeError, ValueError):
                    errors.append(f"event {index}: malformed slider point {point_index}")
                    continue
                if not (math.isfinite(point_x) and math.isfinite(point_y)):
                    errors.append(f"event {index}: slider point {point_index} is non-finite")
            if policy.check_slider_paths:
                from .slider_path import slider_path, circle_radius
                try:
                    path = slider_path((x, y), control_points, curve_type, length)
                    cs = _field(_field(map_spec, "difficulty", {}), "circle_size", 4.0)
                    radius = circle_radius(cs)
                    if (path.min(axis=0) < radius).any() or (path.max(axis=0) > (512-radius, 384-radius)).any():
                        errors.append(f"event {index}: rendered slider body leaves the playfield")
                except (ValueError, TypeError, IndexError) as exc:
                    errors.append(f"event {index}: slider path evaluation failed: {exc}")

        canonical_events.append(
            (
                round(start_ms, 3),
                round(end_ms, 3),
                kind,
                round(x, 2),
                round(y, 2),
                bool(_field(event, "new_combo", False)),
                int(_field(event, "hit_sound", 0) or 0),
                str(_field(event, "curve_type", "")).upper(),
                _canonical_control_points(event),
                int(_field(event, "slider_repeats", 0) or 0),
                round(float(_field(event, "slider_length", 0.0) or 0.0), 3),
            )
        )

    duration_ms = max(item[0] for item in timed) - min(item[0] for item in timed)
    metrics: dict[str, float] = {
        "object_count": float(len(timed)),
        "duration_ms": max(0.0, duration_ms),
        "objects_per_minute": len(timed) * 60000.0 / max(duration_ms, 1.0),
        "circle_count": float(object_counts.get("circle", 0)),
        "slider_count": float(object_counts.get("slider", 0)),
        "spinner_count": float(object_counts.get("spinner", 0)),
        "max_snap_error_beats": max(snap_errors, default=0.0),
        "median_start_gap_ms": float(median(gap_values)) if gap_values else 0.0,
        "max_jump_speed_px_per_second": max_jump_speed,
    }
    from .coverage import audit_coverage
    coverage_errors, coverage_metrics = audit_coverage(timed, policy)
    errors.extend(coverage_errors)
    metrics.update(coverage_metrics)
    fingerprint = hashlib.sha256(
        json.dumps(canonical_events, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    return errors, warnings, metrics, fingerprint


def _mapspec_validation(map_spec: Any) -> tuple[list[str], list[str]]:
    """Use the canonical MapSpec validator when installed, without requiring it."""

    try:
        from .mapspec import MapSpec, validate_map_spec  # Imported lazily for independent modules.
    except ImportError:
        return [], []
    # Evaluation also intentionally accepts a small protocol-shaped object for
    # tests and external inference bridges.  Do not feed those objects into a
    # future stricter MapSpec validator and mistake a type mismatch for a chart
    # error.
    if not isinstance(map_spec, MapSpec):
        return [], []
    try:
        report = validate_map_spec(map_spec)
    except (AttributeError, TypeError, ValueError) as exc:
        return [f"MapSpec validation failed: {exc}"], []
    valid = _field(report, "valid", None)
    errors = list(_messages(report, "errors"))
    warnings = list(_messages(report, "warnings"))
    if valid is False and not errors:
        errors.append("MapSpec validator rejected candidate")
    return errors, warnings


def _compile_candidate(map_spec: Any, compiler: Optional[Callable[[Any], str]] = None) -> str:
    if compiler is not None:
        return compiler(map_spec)
    try:
        from .mapspec import compile_osu
    except ImportError as exc:
        raise RuntimeError("MapSpec compiler is unavailable") from exc
    return compile_osu(map_spec, validate=True)


def _score_candidate(
    *,
    valid: bool,
    warning_count: int,
    target_star: Optional[float],
    difficulty: DifficultyCalculation,
    model_quality_score: Optional[float],
) -> Optional[float]:
    """Deterministic guardrail score, intentionally not a human-quality claim."""

    if not valid:
        return None
    score = 1.0 - min(0.20, 0.01 * warning_count)
    if target_star is not None and difficulty.available and difficulty.stars is not None:
        tolerance = max(0.5, target_star * 0.15)
        # Preserve ordering even when every candidate is far from the target.
        score *= math.exp(-abs(difficulty.stars - target_star) / tolerance)
    if model_quality_score is not None:
        score += 0.10 * model_quality_score
    return max(0.0, min(1.0, score))


def audit_candidate(
    candidate: Candidate,
    *,
    target_star: Optional[float] = None,
    calculator: Optional[DifficultyCalculator] = None,
    compiler: Optional[Callable[[Any], str]] = None,
    policy: StrictAuditPolicy = StrictAuditPolicy(),
) -> CandidateAudit:
    """Run deterministic gates and optional realized-difficulty calculation."""

    generic_errors, generic_warnings, metrics, fingerprint = _generic_strict_audit(candidate.map_spec, policy)
    mapspec_errors, mapspec_warnings = _mapspec_validation(candidate.map_spec)
    errors = list(dict.fromkeys((*generic_errors, *mapspec_errors)))
    warnings = list(dict.fromkeys((*generic_warnings, *mapspec_warnings)))
    valid = not errors
    calculator = calculator or UnavailableDifficultyCalculator()
    if not valid:
        difficulty = DifficultyCalculation(
            available=False,
            calculator_id="not_evaluated",
            error="candidate failed deterministic validation",
        )
    else:
        try:
            osu_content = _compile_candidate(candidate.map_spec, compiler)
            difficulty = calculator.calculate(osu_content)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            difficulty = DifficultyCalculation(
                available=False,
                calculator_id="unavailable",
                error=f"difficulty calculation unavailable: {exc}",
            )
        if not difficulty.available:
            warnings.append("realized difficulty was not calculated")
        if policy.max_star_error is not None and target_star is not None:
            if not difficulty.available:
                errors.append("target-star validation requires an available difficulty calculator")
            elif abs(difficulty.stars - target_star) > policy.max_star_error:
                errors.append(
                    f"realized difficulty {difficulty.stars:.3f} differs from target "
                    f"{target_star:.3f} by more than {policy.max_star_error:.3f} stars"
                )
        valid = not errors
    score = _score_candidate(
        valid=valid,
        warning_count=len(warnings),
        target_star=target_star,
        difficulty=difficulty,
        model_quality_score=candidate.model_quality_score,
    )
    return CandidateAudit(
        candidate=candidate,
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
        difficulty=difficulty,
        score=score,
        event_fingerprint=fingerprint,
    )


def rerank_candidates(
    candidates: Sequence[Candidate],
    *,
    target_star: Optional[float] = None,
    calculator: Optional[DifficultyCalculator] = None,
    compiler: Optional[Callable[[Any], str]] = None,
    policy: StrictAuditPolicy = StrictAuditPolicy(),
) -> tuple[CandidateAudit, ...]:
    """Audit candidates, penalize exact duplicates, then rank valid drafts first.

    This is a safe baseline selector until a separately trained quality model is
    available.  It only detects exact canonical event duplicates; a future
    learned nearest-training-map gate should be added independently rather than
    overstating this as plagiarism detection.
    """

    if not candidates:
        raise ValueError("at least one candidate is required for reranking")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    audits = [
        audit_candidate(
            candidate,
            target_star=target_star,
            calculator=calculator,
            compiler=compiler,
            policy=policy,
        )
        for candidate in candidates
    ]
    def rank_key(audit: CandidateAudit) -> tuple[bool, float, str]:
        # ``0.0`` is a meaningful valid score, not a missing score.
        score = audit.score if audit.score is not None else -1.0
        return (not audit.valid, -score, audit.candidate.candidate_id)

    # Establish an initial stable ordering before deciding which duplicate wins.
    audits.sort(key=rank_key)
    seen_fingerprints: set[str] = set()
    deduplicated: list[CandidateAudit] = []
    for audit in audits:
        if audit.valid and audit.event_fingerprint in seen_fingerprints:
            warnings = (*audit.warnings, "exact duplicate of a higher-ranked candidate")
            score = max(0.0, (audit.score or 0.0) - 0.25)
            audit = replace(audit, warnings=warnings, score=score)
        elif audit.valid:
            seen_fingerprints.add(audit.event_fingerprint)
        deduplicated.append(audit)
    return tuple(sorted(deduplicated, key=rank_key))


def build_generation_result(
    request: GenerationRequest,
    audits: Sequence[CandidateAudit],
    *,
    architecture_id: str,
    model_revision: str,
) -> GenerationResult:
    """Build a manifest-ready result; only the top valid audit is selected."""

    if not audits:
        raise ValueError("at least one candidate audit is required")
    selected = next((audit for audit in audits if audit.valid), None)
    candidate_manifests = tuple(
        audit.to_manifest(selected=selected is not None and audit.candidate.candidate_id == selected.candidate.candidate_id)
        for audit in audits
    )
    result_warnings: list[str] = [
        "Generated maps are playable drafts for human mapper review; they are not automatically ranked maps."
    ]
    if selected is None:
        result_warnings.append("No candidate passed strict validation; no .osu artifact may be packaged.")
    elif not selected.difficulty.available:
        result_warnings.append("Realized difficulty was not calculated because no pinned calculator was available.")
    return GenerationResult(
        request=request,
        architecture_id=architecture_id,
        model_revision=model_revision,
        candidates=candidate_manifests,
        selected_candidate_id=selected.candidate.candidate_id if selected is not None else None,
        warnings=tuple(result_warnings),
    )
