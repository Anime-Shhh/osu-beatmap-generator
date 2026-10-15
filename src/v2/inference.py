"""Audio-feature-to-draft orchestration for architecture v2.

The runtime accepts an audio-derived :class:`FeatureBundle`, not source `.osu`
timing/settings.  Audio decoding, MERT extraction, and beat/downbeat estimation
are explicit adapters outside this module; they must be installed/pinned by the
user rather than downloaded or guessed during an inference import.  Given a
trained checkpoint and those inputs, this module generates candidates, applies
deterministic safety projection, audits them, and exposes only a valid selected
draft for packaging.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence

import torch

from .audio import (
    CanonicalAudioManifest,
    CanonicalAudioSpec,
    FeatureCacheManifest,
    FeatureStreamManifest,
    pcm_sha256,
)
from .clock import (
    AudioClockResolver,
    ClockEvidence,
    ClockResolver,
    MeterSegment,
    TimingHypothesis,
    TimingMap,
    TimingSegment,
)
from .data import SourceIdentity
from .evaluation import Candidate, DifficultyCalculator, StrictAuditPolicy, build_generation_result, rerank_candidates
from .events import BeatEvent, EventKind
from .generation import (
    DEFAULT_AR,
    DEFAULT_CS,
    DEFAULT_HP,
    DEFAULT_OD,
    DEFAULT_SLIDER_MULTIPLIER,
    DEFAULT_SLIDER_TICK_RATE,
    DEFAULT_TARGET_STAR,
    GenerationRequest,
    GenerationResult,
)
from .geometry import (
    ActiveEventGeometryDiffuser,
    GeometryDiffusionConfig,
    sample_geometry_windows,
)
from .grammar import BarEventCodec, EventVocabulary, PlannerEvent
from .mapspec import DifficultySettings, GeneralSettings, MapMetadata, MapSpec
from .packaging import GeneratedPackage, write_generation_package
from .preprocess import ARCHITECTURE_ID, BarDefinition, build_bar_definitions, clock_features_for_bars, compact_profile
from .sparse_model import SparseEventPlanner, SparsePlannerConfig
from .timing import TimingPoint, timing_map_from_points
from .training import CHECKPOINT_SCHEMA, CheckpointProvenance, vocabulary_sha256


class FeatureProvider(Protocol):
    """Boundary for pinned extractors used by a real audio-only deployment."""

    def extract(self, audio_path: str | Path) -> "FeatureBundle":
        """Decode/canonicalize audio and return timestamped feature streams."""


@dataclass(frozen=True)
class FeatureBundle:
    """Timestamped input features and beat-tracker evidence for one song."""

    feature_cache: FeatureCacheManifest
    rhythm_features: torch.Tensor  # [frames, 101], e.g. 96 mel + rhythm auxiliaries.
    rhythm_times_ms: torch.Tensor  # [frames]
    semantic_features: torch.Tensor  # [frames, 128], frozen/PCA-compressed music embeddings.
    semantic_times_ms: torch.Tensor  # [frames]
    clock_evidence: ClockEvidence

    def __post_init__(self) -> None:
        for name, features, times in (
            ("rhythm", self.rhythm_features, self.rhythm_times_ms),
            ("semantic", self.semantic_features, self.semantic_times_ms),
        ):
            if not isinstance(features, torch.Tensor) or features.dim() != 2:
                raise ValueError(f"{name}_features must be [frames, dimensions]")
            if not isinstance(times, torch.Tensor) or times.dim() != 1:
                raise ValueError(f"{name}_times_ms must be [frames]")
            if features.shape[0] != times.numel() or features.shape[0] < 1:
                raise ValueError(f"{name} feature/timestamp rows must be non-empty and aligned")
            if not torch.isfinite(features).all() or not torch.isfinite(times).all():
                raise ValueError(f"{name} features and timestamps must be finite")
            if times.numel() > 1 and not torch.all(times[1:] > times[:-1]):
                raise ValueError(f"{name} timestamps must be strictly increasing")
        if not math.isclose(
            self.feature_cache.canonical_audio.duration_ms,
            self.clock_evidence.audio_duration_ms,
            rel_tol=0.0,
            abs_tol=20.0,
        ):
            raise ValueError("clock evidence duration must match canonical audio within 20 ms")

    @property
    def duration_ms(self) -> float:
        return self.feature_cache.canonical_audio.duration_ms


@dataclass(frozen=True)
class LoadedV2Models:
    """Models loaded from one pinned checkpoint with an audit-friendly ID."""

    planner: SparseEventPlanner
    geometry: ActiveEventGeometryDiffuser
    checkpoint_provenance: CheckpointProvenance
    model_revision: str


@dataclass(frozen=True)
class GeneratedDraft:
    """All candidates plus the selected audit/result; package only on request."""

    result: GenerationResult
    candidates: tuple[Candidate, ...]
    audits: tuple[Any, ...]

    @property
    def selected_candidate(self) -> Candidate | None:
        selected_id = self.result.selected_candidate_id
        return next((candidate for candidate in self.candidates if candidate.candidate_id == selected_id), None)

    def package(
        self,
        output_dir: str | Path,
        *,
        osu_filename: str = "ranked-like-draft.osu",
        manifest_filename: str = "ranked-like-draft.manifest.json",
        audio_source_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> GeneratedPackage:
        selected = self.selected_candidate
        if selected is None:
            raise ValueError("no valid candidate was selected; refusing to package a map")
        return write_generation_package(
            self.result,
            selected,
            output_dir,
            osu_filename=osu_filename,
            manifest_filename=manifest_filename,
            audio_source_path=audio_source_path,
            overwrite=overwrite,
        )


def _safe_tensor(value: Any, name: str, *, dim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dim() != dim:
        raise ValueError(f"{name} must be a {dim}D tensor")
    return value


def feature_bundle_from_payload(payload: Mapping[str, Any]) -> FeatureBundle:
    """Load a feature bundle from a caller-provided safe tensor payload.

    The CLI intentionally requires this payload (or a user-supplied provider),
    so it never starts a MERT/Beat This! model download just because someone
    typed an audio filename.
    """

    cache_payload = payload.get("feature_cache")
    evidence_payload = payload.get("clock_evidence")
    if not isinstance(cache_payload, Mapping) or not isinstance(evidence_payload, Mapping):
        raise ValueError("feature payload needs feature_cache and clock_evidence objects")
    return FeatureBundle(
        feature_cache=FeatureCacheManifest.from_dict(cache_payload),
        rhythm_features=_safe_tensor(payload.get("rhythm_features"), "rhythm_features", dim=2).float(),
        rhythm_times_ms=_safe_tensor(payload.get("rhythm_times_ms"), "rhythm_times_ms", dim=1).float(),
        semantic_features=_safe_tensor(payload.get("semantic_features"), "semantic_features", dim=2).float(),
        semantic_times_ms=_safe_tensor(payload.get("semantic_times_ms"), "semantic_times_ms", dim=1).float(),
        clock_evidence=ClockEvidence(
            audio_duration_ms=float(evidence_payload["audio_duration_ms"]),
            beat_times_ms=tuple(float(value) for value in evidence_payload.get("beat_times_ms", ())),
            downbeat_times_ms=tuple(float(value) for value in evidence_payload.get("downbeat_times_ms", ())),
            beat_confidences=tuple(float(value) for value in evidence_payload.get("beat_confidences", ())),
            downbeat_confidences=tuple(float(value) for value in evidence_payload.get("downbeat_confidences", ())),
            source=str(evidence_payload.get("source", "unknown")),
            extractor_revision=str(evidence_payload.get("extractor_revision", "unknown")),
        ),
    )


def load_feature_bundle(path: str | Path) -> FeatureBundle:
    """Safely load a precomputed feature bundle; arbitrary pickle is rejected."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"unable to load safe v2 feature payload: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("v2 feature payload must be a mapping")
    return feature_bundle_from_payload(payload)


def _checkpoint_revision(path: Path, payload: Mapping[str, Any]) -> str:
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()[:16]
    step = payload.get("global_step", "unknown")
    return f"checkpoint-{step}-{digest}"


def load_v2_models(path: str | Path, *, device: torch.device | str = "cpu") -> LoadedV2Models:
    """Load a trained checkpoint without constructing or downloading audio models."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"unable to load safe v2 checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported v2 checkpoint schema")
    training = payload.get("training_config", {})
    if training.get("stage") != "geometry" or int(payload.get("global_step", 0)) <= 0:
        raise ValueError(
            "Generation requires a trained geometry-stage checkpoint. Planner checkpoints "
            "also contain random geometry weights; train geometry on verified real maps first."
        )
    if payload.get("data_audit", {}).get("production_ready") is False:
        raise ValueError("This checkpoint is marked as diagnostic/unverified data, not generation-ready")
    planner_config_payload = payload.get("planner_config")
    geometry_config_payload = payload.get("geometry_config")
    provenance_payload = payload.get("provenance")
    if not isinstance(planner_config_payload, Mapping) or not isinstance(geometry_config_payload, Mapping):
        raise ValueError("checkpoint must contain planner and geometry configs")
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("checkpoint is missing provenance")
    planner_config = SparsePlannerConfig(**dict(planner_config_payload))
    geometry_config = GeometryDiffusionConfig(**{"coordinate_space": "legacy-logit", **dict(geometry_config_payload)})
    if geometry_config.context_dim != planner_config.model_dim:
        raise ValueError("checkpoint geometry context dimension differs from planner decoder dimension")
    if geometry_config.event_feature_dim != 4:
        raise ValueError("checkpoint geometry feature dimension differs from the v2 record contract")
    planner = SparseEventPlanner(planner_config)
    geometry = ActiveEventGeometryDiffuser(geometry_config)
    provenance = CheckpointProvenance.from_dict(provenance_payload)
    if provenance.vocabulary_sha256 != vocabulary_sha256(planner.vocabulary):
        raise ValueError("checkpoint vocabulary differs; rebuild v3 records and retrain both stages")
    planner_state = payload.get("planner")
    geometry_state = payload.get("geometry")
    if not isinstance(planner_state, Mapping) or not isinstance(geometry_state, Mapping):
        raise ValueError("checkpoint does not contain both trained model state dictionaries")
    planner.load_state_dict(planner_state)
    geometry.load_state_dict(geometry_state)
    target = torch.device(device)
    planner.to(target).eval()
    geometry.to(target).eval()
    return LoadedV2Models(planner, geometry, provenance, _checkpoint_revision(source, payload))


def _request_profile(request: GenerationRequest) -> torch.Tensor:
    return compact_profile(
        target_star=float(request.target_star if request.target_star is not None else DEFAULT_TARGET_STAR),
        approach_rate=float(request.ar if request.ar is not None else DEFAULT_AR),
        circle_size=float(request.cs if request.cs is not None else DEFAULT_CS),
        overall_difficulty=float(request.od if request.od is not None else DEFAULT_OD),
        hp_drain_rate=float(request.hp if request.hp is not None else DEFAULT_HP),
        slider_multiplier=float(
            request.slider_multiplier
            if request.slider_multiplier is not None
            else DEFAULT_SLIDER_MULTIPLIER
        ),
        slider_tick_rate=float(
            request.slider_tick_rate
            if request.slider_tick_rate is not None
            else DEFAULT_SLIDER_TICK_RATE
        ),
    )


def timing_points_from_hypothesis(hypothesis: TimingHypothesis) -> tuple[TimingPoint, ...]:
    """Compile a canonical native timing-map subset from an audio clock.

    Emits red timing points at the union of all tempo change boundaries and
    meter change boundaries. Preserves active slider velocity across inserted
    red points by emitting inherited timing points where necessary, matching
    native osu! decoder semantics.
    """

    # Collect union of red timing point timestamps from timing segments and meter segments
    red_times_set = set(segment.start_ms for segment in hypothesis.timing_map.segments)
    for m_seg in hypothesis.meter_segments:
        red_times_set.add(hypothesis.timing_map.beat_to_time(m_seg.start_beat))

    red_times = sorted(red_times_set)
    points: list[TimingPoint] = []

    for t_ms in red_times:
        beat = hypothesis.timing_map.time_to_beat(t_ms)
        m_seg = hypothesis.meter_map.segment_at_beat(beat)
        if m_seg.denominator <= 0 or (m_seg.denominator & (m_seg.denominator - 1)) != 0:
            raise ValueError(f"meter denominator must be a positive power of two: {m_seg.denominator}")
        scale = 4.0 / m_seg.denominator
        quarter_beat_length_ms = hypothesis.timing_map.beat_length_at_time(t_ms)
        beat_length_ms = quarter_beat_length_ms * scale
        meter = m_seg.numerator

        points.append(
            TimingPoint(
                offset_ms=t_ms,
                beat_length_ms=beat_length_ms,
                meter=meter,
                sample_set=0,
                sample_index=0,
                volume=100,
                uninherited=True,
                effects=0,
            )
        )

    # Track all timestamps where an inherited point is needed:
    # 1. Non-red timestamps with explicit SV segments (excluding query sentinel <= -1e11)
    # 2. Red point timestamps where active SV != 1.0 (to prevent native osu! from resetting SV to 1.0)
    green_points_by_time: dict[float, TimingPoint] = {}

    for segment in hypothesis.timing_map.slider_velocity_segments:
        if segment.start_ms <= -1.0e11 or segment.start_ms in red_times_set:
            continue
        green_points_by_time[segment.start_ms] = TimingPoint(
            offset_ms=segment.start_ms,
            beat_length_ms=-100.0 / segment.multiplier,
            meter=4,
            sample_set=0,
            sample_index=0,
            volume=100,
            uninherited=False,
            effects=0,
        )

    for t_ms in red_times:
        active_sv = hypothesis.timing_map.slider_velocity_at_time(t_ms)
        if not math.isclose(active_sv, 1.0, abs_tol=1e-7):
            green_points_by_time[t_ms] = TimingPoint(
                offset_ms=t_ms,
                beat_length_ms=-100.0 / active_sv,
                meter=4,
                sample_set=0,
                sample_index=0,
                volume=100,
                uninherited=False,
                effects=0,
            )

    for p in green_points_by_time.values():
        points.append(p)

    # Sort stably by offset_ms, with uninherited (red) points before inherited (green) points
    return tuple(sorted(points, key=lambda point: (point.offset_ms, 0 if point.uninherited else 1)))


def reproject_events_for_compiled_timing(
    events: Sequence[BeatEvent],
    hypothesis: TimingHypothesis,
    difficulty: DifficultySettings,
    timing_points: Sequence[TimingPoint],
) -> tuple[BeatEvent, ...]:
    """Preserve absolute object intervals when native meter units are compiled.

    ``TimingHypothesis`` uses quarter-note beat coordinates. osu! timing points
    encode the active meter unit, so 6/8 and 2/2 compilation changes the native
    beat axis. Every onset/end is therefore projected through milliseconds, and
    slider pixel length is recomputed under the compiled red/SV state so repeats
    retain the same physical final tail.
    """

    compiled_timing = timing_map_from_points(timing_points)
    projected: list[BeatEvent] = []
    for event in events:
        start_ms = hypothesis.timing_map.beat_to_time(event.beat_position)
        start_beat = compiled_timing.time_to_beat(start_ms)
        if event.kind == EventKind.SLIDER:
            if event.end_beat is not None:
                end_ms = hypothesis.timing_map.beat_to_time(event.end_beat)
            else:
                end_ms = start_ms + hypothesis.timing_map.slider_duration_ms(
                    start_ms,
                    event.slider_length,
                    event.slider_repeats,
                    difficulty.slider_multiplier,
                )
            duration_ms = max(0.0, end_ms - start_ms)
            beat_length = compiled_timing.beat_length_at_time(start_ms)
            velocity = compiled_timing.slider_velocity_at_time(start_ms)
            length = (
                duration_ms
                * difficulty.slider_multiplier
                * 100.0
                * velocity
                / (beat_length * max(1, event.slider_repeats))
            )
            projected.append(
                replace(
                    event,
                    beat_position=start_beat,
                    end_beat=compiled_timing.time_to_beat(end_ms),
                    slider_length=max(1.0e-3, length),
                )
            )
        elif event.end_beat is not None:
            end_ms = hypothesis.timing_map.beat_to_time(event.end_beat)
            projected.append(
                replace(
                    event,
                    beat_position=start_beat,
                    end_beat=compiled_timing.time_to_beat(end_ms),
                )
            )
        else:
            projected.append(replace(event, beat_position=start_beat))
    return tuple(sorted(projected, key=lambda event: (event.beat_position, event.source_object_index)))


def _slider_length_for_end(
    hypothesis: TimingHypothesis,
    *,
    start_beat: float,
    target_end_beat: float,
    repeats: int,
    slider_multiplier: float,
) -> float:
    start_ms = hypothesis.timing_map.beat_to_time(start_beat)
    end_ms = hypothesis.timing_map.beat_to_time(target_end_beat)
    duration_ms = max(0.0, end_ms - start_ms)
    beat_length = hypothesis.timing_map.beat_length_at_time(start_ms)
    velocity = hypothesis.timing_map.slider_velocity_at_time(start_ms)
    return duration_ms * slider_multiplier * 100.0 * velocity / (beat_length * max(1, repeats))


def _geometry_to_events(
    plans: Sequence[PlannerEvent],
    geometry: torch.Tensor,
    hypothesis: TimingHypothesis,
    difficulty: DifficultySettings,
) -> list[BeatEvent]:
    """Attach sampled coordinates/anchors and derive native slider durations."""

    events: list[BeatEvent] = []
    for index, plan in enumerate(plans):
        head = geometry[index, 0]
        x, y = float(head[0] * 512.0), float(head[1] * 384.0)
        if plan.kind == EventKind.SPINNER:
            # Spinner location has no gameplay effect; a stable center avoids
            # letting geometry noise affect a meaningless field.
            x, y = 256.0, 192.0
        if plan.kind == EventKind.SLIDER:
            count = max(1, min(plan.anchor_count, geometry.shape[1] - 1))
            anchors = tuple(
                (float(point[0] * 512.0), float(point[1] * 384.0))
                for point in geometry[index, 1 : count + 1]
            )
            intended_end = plan.beat_position + max(plan.duration_beats, 1.0 / 16.0)
            length = _slider_length_for_end(
                hypothesis,
                start_beat=plan.beat_position,
                target_end_beat=intended_end,
                repeats=max(1, plan.slider_repeats),
                slider_multiplier=difficulty.slider_multiplier,
            )
            curve_type = plan.curve_type.upper()
            # Native perfect arcs require exactly two control points. Projection
            # cannot invent the missing topology, so canonicalize that one
            # malformed choice to a Bezier instead of emitting invalid syntax.
            if curve_type == "P" and len(anchors) != 2:
                curve_type = "B"
            events.append(
                BeatEvent(
                    kind=plan.kind,
                    beat_position=plan.beat_position,
                    x=x,
                    y=y,
                    end_beat=intended_end,
                    curve_type=curve_type,
                    control_points=anchors,
                    slider_repeats=max(1, plan.slider_repeats),
                    slider_length=max(1.0e-3, length),
                    hit_sound=plan.hit_sound,
                    new_combo=plan.new_combo,
                    source_object_index=index,
                )
            )
        elif plan.kind == EventKind.SPINNER:
            events.append(
                BeatEvent(
                    kind=plan.kind,
                    beat_position=plan.beat_position,
                    x=x,
                    y=y,
                    end_beat=plan.beat_position + max(plan.duration_beats, 1.0 / 16.0),
                    hit_sound=plan.hit_sound,
                    new_combo=plan.new_combo,
                    source_object_index=index,
                )
            )
        else:
            events.append(
                BeatEvent(
                    kind=plan.kind,
                    beat_position=plan.beat_position,
                    x=x,
                    y=y,
                    hit_sound=plan.hit_sound,
                    new_combo=plan.new_combo,
                    source_object_index=index,
                )
            )
    return events


def project_event_intervals(
    events: Sequence[BeatEvent],
    hypothesis: TimingHypothesis,
    difficulty: DifficultySettings,
) -> tuple[BeatEvent, ...]:
    """Trim generated long objects before the next start when feasible.

    This is a deliberately visible safety repair, not a substitute for the
    strict validator. If a long object cannot retain even a 1/16-beat span, it
    becomes a circle at the same planned onset rather than blocking all later
    objects. The candidate audit records the remaining constraints.
    """

    ordered = sorted(events, key=lambda event: (event.beat_position, event.source_object_index))
    repaired: list[BeatEvent] = []
    for index, event in enumerate(ordered):
        next_start_ms = (
            hypothesis.timing_map.beat_to_time(ordered[index + 1].beat_position)
            if index + 1 < len(ordered)
            else None
        )
        if event.kind not in {EventKind.SLIDER, EventKind.SPINNER} or next_start_ms is None:
            repaired.append(event)
            continue
        start_ms = hypothesis.timing_map.beat_to_time(event.beat_position)
        end_beat = event.end_beat if event.end_beat is not None else event.beat_position
        end_ms = hypothesis.timing_map.beat_to_time(end_beat)
        if end_ms < next_start_ms - 1.0:
            repaired.append(event)
            continue
        legal_end_ms = next_start_ms - 1.0
        minimum_end_ms = hypothesis.timing_map.beat_to_time(event.beat_position + 1.0 / 16.0)
        if legal_end_ms <= minimum_end_ms:
            repaired.append(
                BeatEvent(
                    kind=EventKind.CIRCLE,
                    beat_position=event.beat_position,
                    x=event.x,
                    y=event.y,
                    hit_sound=event.hit_sound,
                    new_combo=event.new_combo,
                    combo_skip=event.combo_skip,
                    source_object_index=event.source_object_index,
                )
            )
            continue
        new_end_beat = hypothesis.timing_map.time_to_beat(legal_end_ms)
        if event.kind == EventKind.SPINNER:
            repaired.append(
                replace(event, end_beat=new_end_beat)
            )
        else:
            length = _slider_length_for_end(
                hypothesis,
                start_beat=event.beat_position,
                target_end_beat=new_end_beat,
                repeats=event.slider_repeats,
                slider_multiplier=difficulty.slider_multiplier,
            )
            repaired.append(
                replace(event, end_beat=new_end_beat, slider_length=max(1.0e-3, length))
            )
    return tuple(repaired)


def _planned_geometry_features(
    plans: Sequence[PlannerEvent],
    hypothesis: TimingHypothesis,
    difficulty: DifficultySettings,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build explicit flow/path conditions for one complete symbolic rollout."""

    values: list[tuple[float, float, float, float]] = []
    lengths: list[float] = []
    normalizer = math.log1p(32.0)
    for index, plan in enumerate(plans):
        previous_gap = plan.beat_position - plans[index - 1].beat_position if index else 32.0
        next_gap = plans[index + 1].beat_position - plan.beat_position if index + 1 < len(plans) else 32.0
        duration = max(0.0, plan.duration_beats)
        length = 0.0
        if plan.kind == EventKind.SLIDER:
            length = _slider_length_for_end(
                hypothesis,
                start_beat=plan.beat_position,
                target_end_beat=plan.beat_position + max(duration, 1.0 / 16.0),
                repeats=max(1, plan.slider_repeats),
                slider_multiplier=difficulty.slider_multiplier,
            )
        lengths.append(length)
        values.append(
            (
                min(1.0, math.log1p(max(0.0, previous_gap)) / normalizer),
                min(1.0, math.log1p(max(0.0, next_gap)) / normalizer),
                min(1.0, math.log1p(duration) / normalizer),
                min(1.0, math.log1p(max(0.0, length)) / math.log1p(1000.0)),
            )
        )
    return (
        torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0),
        torch.arange(len(plans), dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(lengths, dtype=torch.float32, device=device).unsqueeze(0),
    )


@torch.no_grad()
def _sample_candidate_map(
    *,
    planner: SparseEventPlanner,
    geometry: ActiveEventGeometryDiffuser,
    features: FeatureBundle,
    hypothesis: TimingHypothesis,
    request: GenerationRequest,
    seed: int,
    temperature: float,
    geometry_steps: int,
    metadata: MapMetadata | None = None,
    minimum_start_gap_ms: float = 0.0,
) -> MapSpec:
    device = next(planner.parameters()).device
    if next(geometry.parameters()).device != device:
        raise ValueError("planner and geometry models must be on the same device")
    end_beat = max(0.0, hypothesis.timing_map.time_to_beat(features.duration_ms))
    bars = build_bar_definitions(hypothesis, end_beat=end_beat)
    bar_mid, clock = clock_features_for_bars(hypothesis, bars)
    profile = _request_profile(request).to(device).unsqueeze(0)
    rhythm = features.rhythm_features.float().to(device).unsqueeze(0)
    semantic = features.semantic_features.float().to(device).unsqueeze(0)
    rhythm_times = features.rhythm_times_ms.float().to(device).unsqueeze(0)
    semantic_times = features.semantic_times_ms.float().to(device).unsqueeze(0)
    bar_mid = bar_mid.to(device).unsqueeze(0)
    clock = clock.to(device).unsqueeze(0)
    bar_mask = torch.ones(bar_mid.shape, dtype=torch.bool, device=device)
    memory = planner.encode_bars(
        rhythm,
        rhythm_times,
        semantic,
        semantic_times,
        bar_mid,
        profile,
        clock,
        bar_mask=bar_mask,
        return_local_frames=bool(planner.config.local_audio_tokens),
    )
    local_frames = None
    if planner.config.local_audio_tokens:
        memory, local_frames = memory

    def memory_for_bar(bar):
        if not planner.config.local_audio_tokens:
            return memory, bar_mask
        from .alignment import query_grid
        queries = query_grid(hypothesis.timing_map, bar.start_beat, bar.length_beats,
                             planner.config.local_audio_tokens).to(device).unsqueeze(0)
        augmented, mask, _, _ = planner.local_audio_memory(
            memory, local_frames, rhythm, rhythm_times, queries,
            torch.tensor([bar.bar_index], device=device), bar_mask=bar_mask)
        return augmented, mask

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    codec = BarEventCodec(planner.vocabulary)
    difficulty = DifficultySettings(
        hp_drain_rate=float(request.hp if request.hp is not None else DEFAULT_HP),
        circle_size=float(request.cs if request.cs is not None else DEFAULT_CS),
        overall_difficulty=float(request.od if request.od is not None else DEFAULT_OD),
        approach_rate=float(request.ar if request.ar is not None else DEFAULT_AR),
        slider_multiplier=float(
            request.slider_multiplier
            if request.slider_multiplier is not None
            else DEFAULT_SLIDER_MULTIPLIER
        ),
        slider_tick_rate=float(
            request.slider_tick_rate
            if request.slider_tick_rate is not None
            else DEFAULT_SLIDER_TICK_RATE
        ),
    )
    planned_bars: list[tuple[BarDefinition, list[int], list[PlannerEvent], list[int] | None]] = []
    previous_tokens: list[int] | None = None
    last_event_beat = None
    # Include 1 ms for the worst-case difference between two rounded native
    # timestamps. Use the fastest beat length to cover tempo changes too.
    gap_beats = ((minimum_start_gap_ms + 1.0) / min(s.beat_length_ms for s in hypothesis.timing_map.segments)
                 if minimum_start_gap_ms else 0.0)
    for bar in bars:
        decode_memory, decode_mask = memory_for_bar(bar)
        tokens = planner.generate_bar(
            decode_memory,
            memory_mask=decode_mask,
            target_bar_index=bar.bar_index,
            bar_length_beats=bar.length_beats,
            minimum_start_gap_beats=gap_beats,
            previous_event_relative_beat=None if last_event_beat is None else last_event_beat - bar.start_beat,
            previous_bar_tokens=previous_tokens,
            max_tokens=planner.config.max_decoder_tokens - 1,
            temperature=temperature,
            generator=generator,
        )
        decoded = codec.decode(tokens, bar_start_beat=bar.start_beat, meter=bar.length_beats)
        planned_bars.append((bar, tokens, decoded, previous_tokens))
        if decoded:
            last_event_beat = decoded[-1].beat_position
        previous_tokens = tokens

    plans: list[PlannerEvent] = []
    contexts: list[torch.Tensor] = []
    for bar, tokens, decoded, prior_tokens in planned_bars:
        if not decoded:
            continue
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        prior_tensor = (
            None
            if prior_tokens is None
            else torch.tensor(prior_tokens, dtype=torch.long, device=device).unsqueeze(0)
        )
        decode_memory, decode_mask = memory_for_bar(bar)
        hidden = planner.decode_hidden(
            token_tensor,
            decode_memory,
            target_bar_index=torch.tensor([bar.bar_index], device=device),
            memory_mask=decode_mask,
            previous_bar_tokens=prior_tensor,
            previous_bar_mask=None if prior_tensor is None else torch.ones_like(prior_tensor, dtype=torch.bool),
        )
        event_positions = torch.tensor(
            codec.event_context_positions(tokens), dtype=torch.long, device=device
        )
        contexts.append(hidden[:, event_positions, :])
        plans.extend(decoded)
    if plans:
        context = torch.cat(contexts, dim=1)
        kinds = torch.tensor([[int(item.kind) for item in plans]], dtype=torch.long, device=device)
        anchors = torch.tensor([[item.anchor_count for item in plans]], dtype=torch.long, device=device)
        mask = torch.ones(kinds.shape, dtype=torch.bool, device=device)
        event_features, event_positions, target_lengths = _planned_geometry_features(
            plans, hypothesis, difficulty, device=device
        )
        full_geometry = sample_geometry_windows(
            geometry,
            context,
            kinds,
            anchors,
            event_mask=mask,
            event_features=event_features,
            event_positions=event_positions,
            steps=geometry_steps,
            generator=generator,
        )
        # Native pixelLength trims/extends the evaluated curve. Scaling the
        # control polygon to that length distorts P/B shapes (its length is not
        # the curve's length), and clamping controls destroys legitimate arcs.
        full_geometry = full_geometry[0].cpu()
    else:
        full_geometry = torch.zeros((0, geometry.config.max_anchors + 1, 2))
    events = project_event_intervals(
        _geometry_to_events(plans, full_geometry, hypothesis, difficulty), hypothesis, difficulty
    )
    star = float(request.target_star if request.target_star is not None else DEFAULT_TARGET_STAR)
    if metadata is not None:
        map_metadata = metadata
    else:
        map_metadata = MapMetadata(
            title="Generated Draft",
            artist="Unknown Artist",
            creator="osu-beatmap-generator",
            version=f"v2 {star:.1f}★ Draft",
            tags="generated ranked-like draft human-review-required",
        )
    timing_points = timing_points_from_hypothesis(hypothesis)
    compiled_events = reproject_events_for_compiled_timing(
        events, hypothesis, difficulty, timing_points
    )
    return MapSpec(
        format_version=14,
        general=GeneralSettings(audio_filename=request.audio_filename, sample_set="Soft", mode=0),
        metadata=map_metadata,
        difficulty=difficulty,
        timing_points=timing_points,
        events=compiled_events,
    )


def generate_draft(
    *,
    models: LoadedV2Models,
    features: FeatureBundle,
    clock_resolver: ClockResolver,
    request: GenerationRequest,
    calculator: DifficultyCalculator | None = None,
    audit_policy: StrictAuditPolicy = StrictAuditPolicy(),
    temperature: float = 0.9,
    geometry_steps: int = 32,
    metadata: MapMetadata | None = None,
) -> GeneratedDraft:
    """Generate and audit a finite candidate set from audio-derived inputs.

    The function does not write files. Use :meth:`GeneratedDraft.package` only
    after an audit selected a valid candidate. A rejected result is useful
    feedback about clock/model quality, never a license to emit invalid `.osu`.
    """

    if request.feature_revision and request.feature_revision != features.feature_cache.pipeline_revision:
        raise ValueError("request feature_revision differs from supplied feature cache")
    if request.style_cluster is not None:
        raise ValueError(
            "style_cluster is reserved until train-only structural clustering is implemented; "
            "refusing to silently ignore a requested style control"
        )
    if request.extra_controls:
        raise ValueError("extra_controls are manifest-only and cannot alter a v1 generation run")
    if models.checkpoint_provenance.feature_pipeline_key != features.feature_cache.pipeline_key:
        raise ValueError("checkpoint was trained against a different feature pipeline")
    if features.rhythm_features.shape[1] != models.planner.config.rhythm_dim:
        raise ValueError("rhythm feature dimension differs from checkpoint planner config")
    if features.semantic_features.shape[1] != models.planner.config.semantic_dim:
        raise ValueError("semantic feature dimension differs from checkpoint planner config")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    hypotheses = clock_resolver.resolve(features.clock_evidence, max_hypotheses=request.candidate_count)
    if not hypotheses:
        raise ValueError("clock resolver produced no timing hypotheses")
    hypotheses = tuple(sorted(hypotheses, key=lambda value: (-value.score, -value.confidence, value.hypothesis_id)))
    base_seed = int(request.seed if request.seed is not None else 0)
    candidates: list[Candidate] = []
    for index in range(request.candidate_count):
        hypothesis = hypotheses[index % len(hypotheses)]
        seed = base_seed + index
        map_spec = _sample_candidate_map(
            planner=models.planner,
            geometry=models.geometry,
            features=features,
            hypothesis=hypothesis,
            request=request,
            seed=seed,
            temperature=temperature,
            geometry_steps=geometry_steps,
            metadata=metadata,
            minimum_start_gap_ms=audit_policy.minimum_start_gap_ms,
        )
        candidates.append(
            Candidate(
                candidate_id=f"candidate-{index:02d}-{hypothesis.hypothesis_id}",
                map_spec=map_spec,
                seed=seed,
                provenance={
                    "clock_hypothesis_id": hypothesis.hypothesis_id,
                    "clock_source": hypothesis.source,
                    "clock_confidence": hypothesis.confidence,
                    "feature_cache_key": features.feature_cache.cache_key,
                    "architecture_id": ARCHITECTURE_ID,
                },
            )
        )
    audits = rerank_candidates(
        candidates,
        target_star=request.target_star,
        calculator=calculator,
        policy=audit_policy,
    )
    result = build_generation_result(
        request,
        audits,
        architecture_id=ARCHITECTURE_ID,
        model_revision=models.model_revision,
    )
    return GeneratedDraft(result=result, candidates=tuple(candidates), audits=tuple(audits))


def create_mel_filterbank(
    sample_rate_hz: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Create triangular mel filterbank matrix [n_mels, n_fft // 2 + 1]."""
    if f_max is None:
        f_max = sample_rate_hz / 2.0

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    fft_freqs = torch.linspace(0.0, sample_rate_hz / 2.0, n_fft // 2 + 1)

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        up_mask = (fft_freqs >= left) & (fft_freqs <= center)
        if (center - left) > 0:
            filterbank[i, up_mask] = (fft_freqs[up_mask] - left) / (center - left)

        down_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if (right - center) > 0:
            filterbank[i, down_mask] = (right - fft_freqs[down_mask]) / (right - center)

    return filterbank


def _load_audio_ffmpeg(path: Path, target_sr: int) -> torch.Tensor:
    """Decode with FFmpeg while explicitly applying arithmetic-mean downmix."""
    import subprocess

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    channels = int(probe.stdout.strip())
    if channels < 1 or channels > 64:
        raise ValueError(f"unsupported audio channel count: {channels}")
    coefficient = 1.0 / channels
    mix = "+".join(f"{coefficient:.17g}*c{index}" for index in range(channels))
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-filter:a",
        f"pan=mono|c0={mix}",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-",
    ]
    process = subprocess.run(command, capture_output=True, check=True)
    if len(process.stdout) < 4 or len(process.stdout) % 4 != 0:
        raise ValueError("FFmpeg returned an empty or truncated audio stream")
    return torch.frombuffer(bytearray(process.stdout), dtype=torch.float32).unsqueeze(0).clone()


def _load_audio_wave(path: Path, target_sr: int) -> torch.Tensor:
    import wave

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_data = wav_file.readframes(frame_count)

    if channels < 1 or frame_count < 1:
        raise ValueError("WAV file contains no audio frames")
    if sample_rate != target_sr:
        raise ValueError(
            "wave fallback does not resample; FFmpeg is required when "
            f"source sample rate {sample_rate} differs from target {target_sr}"
        )
    if sample_width == 1:
        samples = (torch.frombuffer(raw_data, dtype=torch.uint8).float() - 128.0) / 128.0
    elif sample_width == 2:
        samples = torch.frombuffer(raw_data, dtype=torch.int16).float() / 32768.0
    elif sample_width == 4:
        # The stdlib wave module exposes integer PCM WAV. IEEE float WAV uses a
        # different format tag and is intentionally left to FFmpeg/torchaudio.
        samples = torch.frombuffer(raw_data, dtype=torch.int32).float() / 2147483648.0
    else:
        raise ValueError(f"unsupported integer PCM WAV sample width: {sample_width}")

    expected_samples = frame_count * channels
    if samples.numel() != expected_samples:
        raise ValueError("WAV frame data is truncated")
    waveform = samples.view(frame_count, channels).mean(dim=1).unsqueeze(0)
    return waveform


def load_audio(
    audio_path: str | Path,
    target_sr: int = 24000,
    *,
    backend: Literal["auto", "ffmpeg", "wave"] = "auto",
) -> tuple[torch.Tensor, int]:
    """Decode raw audio to canonical float32 using arithmetic-mean downmix.

    ``backend`` exists for deterministic diagnostics. Automatic mode prefers
    FFmpeg (including MP3 and all resampling), then native-rate integer PCM
    WAV fallback. The fallback fails closed for non-native sample rates so a
    linear, aliasing resampler cannot masquerade as the canonical pipeline.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    if target_sr <= 0:
        raise ValueError("target_sr must be positive")
    loaders = {
        "ffmpeg": _load_audio_ffmpeg,
        "wave": _load_audio_wave,
    }
    if backend not in {"auto", *loaders}:
        raise ValueError(f"unsupported audio backend: {backend}")

    failures: list[str] = []
    selected_backends = tuple(loaders) if backend == "auto" else (backend,)
    for backend_name in selected_backends:
        try:
            waveform = loaders[backend_name](path, target_sr)
            if waveform.numel() < 1 or not torch.isfinite(waveform).all():
                raise ValueError("decoder returned empty or non-finite samples")
            return waveform.contiguous(), target_sr
        except Exception as exc:
            failures.append(f"{backend_name}: {exc}")

    detail = "; ".join(failures)
    raise RuntimeError(f"unable to decode audio from {path} ({detail})")


MIN_SUPPORTED_BPM = 20.0
MAX_SUPPORTED_BPM = 1000.0
MAX_ABS_OFFSET_MS = 86_400_000.0
MAX_CLOCK_BEATS = 1_000_000


def estimate_audio_clock(
    onset_envelope: torch.Tensor,
    *,
    sample_rate_hz: int = 24000,
    hop_length: int = 240,
    duration_ms: float,
    bpm_override: float | None = None,
    offset_override_ms: float | None = None,
) -> tuple[tuple[TimingHypothesis, ...], ClockEvidence]:
    """Estimate song tempo, phase, and bounded clock evidence.

    Explicit tempo controls support 20-1000 BPM and offsets up to one day in
    either direction. These bounds prevent malformed controls from producing a
    non-progressing or impractically large beat grid.
    """
    if sample_rate_hz <= 0 or hop_length <= 0:
        raise ValueError("sample_rate_hz and hop_length must be positive")
    if not math.isfinite(duration_ms) or duration_ms <= 0.0:
        raise ValueError("duration_ms must be finite and positive")
    if bpm_override is not None:
        bpm_override = float(bpm_override)
        if not math.isfinite(bpm_override) or not MIN_SUPPORTED_BPM <= bpm_override <= MAX_SUPPORTED_BPM:
            raise ValueError(
                f"bpm_override must be finite and within [{MIN_SUPPORTED_BPM:g}, {MAX_SUPPORTED_BPM:g}]"
            )
    if offset_override_ms is not None:
        offset_override_ms = float(offset_override_ms)
        if not math.isfinite(offset_override_ms) or abs(offset_override_ms) > MAX_ABS_OFFSET_MS:
            raise ValueError(
                f"offset_override_ms must be finite and within +/-{MAX_ABS_OFFSET_MS:g} ms"
            )
    frame_rate_hz = sample_rate_hz / hop_length
    onset = onset_envelope.squeeze()
    if onset.dim() != 1 or onset.numel() < 1:
        raise ValueError("onset_envelope must contain at least one frame")
    onset = torch.nan_to_num(onset.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_frames = onset.shape[0]

    # 1. Determine BPM
    if bpm_override is not None:
        est_bpm = bpm_override
        beat_length_ms = 60000.0 / est_bpm
        confidence = 1.0
        lag_frames = beat_length_ms * frame_rate_hz / 1000.0
    else:
        min_bpm = 60.0
        max_bpm = 240.0
        min_lag = max(2, int(frame_rate_hz * 60.0 / max_bpm))
        max_lag = min(n_frames - 1, int(frame_rate_hz * 60.0 / min_bpm))

        if max_lag > min_lag and n_frames > max_lag:
            centered = onset - onset.mean()
            autocorr: list[float] = []
            for lag in range(min_lag, max_lag + 1):
                prod = float((centered[:-lag] * centered[lag:]).sum().item())
                bpm_at_lag = frame_rate_hz * 60.0 / lag
                prior = math.exp(-0.5 * ((bpm_at_lag - 140.0) / 40.0) ** 2)
                autocorr.append(prod * (0.7 + 0.3 * prior))

            best_idx = int(torch.tensor(autocorr).argmax().item())
            best_lag = min_lag + best_idx

            if 0 < best_idx < len(autocorr) - 1:
                alpha = autocorr[best_idx - 1]
                beta = autocorr[best_idx]
                gamma = autocorr[best_idx + 1]
                denom = 2.0 * (2.0 * beta - alpha - gamma)
                delta = (gamma - alpha) / denom if abs(denom) > 1e-7 else 0.0
                refined_lag = best_lag + delta
            else:
                refined_lag = float(best_lag)

            beat_length_ms = max(200.0, min(1000.0, (refined_lag / frame_rate_hz) * 1000.0))
            est_bpm = 60000.0 / beat_length_ms
            lag_frames = refined_lag
            confidence = 0.85
        else:
            est_bpm = 120.0
            beat_length_ms = 500.0
            lag_frames = 50.0
            confidence = 0.5

    # 2. Determine Beat Phase (Offset)
    if offset_override_ms is not None:
        offset_ms = float(offset_override_ms)
    else:
        step = max(1, int(round(lag_frames)))
        scores: list[float] = []
        for phase in range(min(step, n_frames)):
            grid = torch.arange(phase, n_frames, step, dtype=torch.long)
            score = float(onset[grid].sum().item())
            scores.append(score)
        best_phase = int(torch.tensor(scores).argmax().item()) if scores else 0
        offset_ms = (best_phase / frame_rate_hz) * 1000.0

    beat_times: list[float] = []
    downbeat_times: list[float] = []

    first_beat_idx = max(0, math.ceil(-offset_ms / beat_length_ms))
    last_beat_idx = math.floor((duration_ms - offset_ms) / beat_length_ms)
    beat_count = max(0, last_beat_idx - first_beat_idx + 1)
    if beat_count > MAX_CLOCK_BEATS:
        raise ValueError(
            f"clock controls would create {beat_count} beats; maximum is {MAX_CLOCK_BEATS}"
        )
    for beat_idx in range(first_beat_idx, last_beat_idx + 1):
        current_time = offset_ms + beat_idx * beat_length_ms
        beat_times.append(round(current_time, 2))
        if beat_idx % 4 == 0:
            downbeat_times.append(round(current_time, 2))

    clock_evidence = ClockEvidence(
        audio_duration_ms=duration_ms,
        beat_times_ms=tuple(beat_times),
        downbeat_times_ms=tuple(downbeat_times),
        beat_confidences=tuple(confidence for _ in beat_times),
        downbeat_confidences=tuple(confidence for _ in downbeat_times),
        source="audio_clock_estimator",
        extractor_revision="v2-onset-autocorr-v1",
    )

    primary_timing = TimingMap(
        (TimingSegment(start_ms=round(offset_ms, 2), beat_length_ms=round(beat_length_ms, 3), start_beat=0.0),)
    )
    primary_hyp = TimingHypothesis(
        hypothesis_id=f"est-{est_bpm:.1f}bpm-{offset_ms:.1f}ms",
        timing_map=primary_timing,
        meter_segments=(MeterSegment(0.0, 4, 4),),
        confidence=confidence,
        score=1.0,
        source="audio_clock_estimator",
    )

    hypotheses: list[TimingHypothesis] = [primary_hyp]
    half_bpm = est_bpm / 2.0
    if half_bpm >= 50.0:
        hypotheses.append(
            TimingHypothesis(
                hypothesis_id=f"half-{half_bpm:.1f}bpm-{offset_ms:.1f}ms",
                timing_map=TimingMap(
                    (TimingSegment(start_ms=round(offset_ms, 2), beat_length_ms=round(beat_length_ms * 2, 3), start_beat=0.0),)
                ),
                meter_segments=(MeterSegment(0.0, 4, 4),),
                confidence=confidence * 0.7,
                score=0.7,
                source="audio_clock_estimator",
            )
        )
    double_bpm = est_bpm * 2.0
    if double_bpm <= 300.0:
        hypotheses.append(
            TimingHypothesis(
                hypothesis_id=f"double-{double_bpm:.1f}bpm-{offset_ms:.1f}ms",
                timing_map=TimingMap(
                    (TimingSegment(start_ms=round(offset_ms, 2), beat_length_ms=round(beat_length_ms / 2, 3), start_beat=0.0),)
                ),
                meter_segments=(MeterSegment(0.0, 4, 4),),
                confidence=confidence * 0.7,
                score=0.7,
                source="audio_clock_estimator",
            )
        )

    return tuple(hypotheses), clock_evidence


def extract_audio_features(
    waveform: torch.Tensor,
    sample_rate_hz: int = 24000,
    *,
    models: LoadedV2Models | None = None,
    rhythm_dim: int = 101,
    semantic_dim: int = 128,
    bpm: float | None = None,
    offset_ms: float | None = None,
    pipeline_revision: str | None = None,
    audio_sha256: str | None = None,
) -> tuple[FeatureBundle, tuple[TimingHypothesis, ...]]:
    """Extract log-mel, onset strength, semantic features, and clock hypotheses from decoded audio."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2 or waveform.shape[0] != 1:
        raise ValueError("waveform must be 1D or 2D [1, samples]")

    if models is not None:
        rhythm_dim = models.planner.config.rhythm_dim
        semantic_dim = models.planner.config.semantic_dim

    if pipeline_revision is None:
        pipeline_revision = "v2-multirate-audio-3"

    waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    n_samples = waveform.shape[-1]
    if n_samples < 1:
        raise ValueError("waveform must contain at least one sample")

    duration_ms = n_samples * 1000.0 / sample_rate_hz

    hop_length = int(round(sample_rate_hz * 0.010))
    n_fft = 1024
    win_length = min(n_fft, max(hop_length, 1024))
    # Reflection-centered STFT requires input longer than its n_fft/2 pad.
    analysis_waveform = waveform
    minimum_analysis_samples = n_fft // 2 + 1
    if n_samples < minimum_analysis_samples:
        analysis_waveform = torch.nn.functional.pad(
            waveform, (0, minimum_analysis_samples - n_samples)
        )

    window = torch.hann_window(win_length)
    stft = torch.stft(
        analysis_waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs()
    n_frames = magnitude.shape[-1]

    n_mels = max(1, rhythm_dim - 5 if rhythm_dim > 5 else rhythm_dim)
    mel_fb = create_mel_filterbank(sample_rate_hz, n_fft, n_mels)
    mel_spec = torch.matmul(mel_fb, magnitude)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    diff = torch.relu(magnitude[:, 1:] - magnitude[:, :-1])
    onset_raw = torch.cat([torch.zeros(1), diff.mean(dim=0)])
    onset_norm = onset_raw / (onset_raw.max() + 1e-7)

    rms = magnitude.pow(2).mean(dim=0).sqrt()
    rms_norm = rms / (rms.max() + 1e-7)

    flux = torch.cat([torch.zeros(1), torch.norm(diff, dim=0)])
    flux_norm = flux / (flux.max() + 1e-7)

    if rhythm_dim == 101 and n_mels == 96:
        aux = torch.stack([onset_norm, rms_norm, flux_norm, onset_norm * flux_norm, rms_norm * onset_norm], dim=0)
        rhythm_features = torch.cat([log_mel, aux], dim=0).transpose(0, 1).contiguous()
    elif rhythm_dim > n_mels:
        extra_dim = rhythm_dim - n_mels
        aux_candidates = [onset_norm, rms_norm, flux_norm]
        aux_list = [aux_candidates[i % len(aux_candidates)] for i in range(extra_dim)]
        aux = torch.stack(aux_list, dim=0)
        rhythm_features = torch.cat([log_mel, aux], dim=0).transpose(0, 1).contiguous()
    else:
        rhythm_features = log_mel[:rhythm_dim, :].transpose(0, 1).contiguous()

    rhythm_times_ms = torch.arange(n_frames, dtype=torch.float32) * (hop_length * 1000.0 / sample_rate_hz)

    sem_hop_frames = 4
    n_sem_frames = math.ceil(n_frames / sem_hop_frames)

    padded_len = n_sem_frames * sem_hop_frames
    if padded_len > n_frames:
        mel_padded = torch.nn.functional.pad(log_mel, (0, padded_len - n_frames))
    else:
        mel_padded = log_mel[:, :padded_len]

    mel_groups = mel_padded.view(mel_padded.shape[0], n_sem_frames, sem_hop_frames)
    semantic_starts = torch.arange(n_sem_frames, dtype=torch.long) * sem_hop_frames
    valid_counts = torch.clamp(n_frames - semantic_starts, min=1, max=sem_hop_frames)
    mel_pooled = mel_groups.sum(dim=-1) / valid_counts.to(log_mel.dtype).unsqueeze(0)
    if semantic_dim >= n_mels:
        sem_repeat = (semantic_dim + n_mels - 1) // n_mels
        semantic_features = mel_pooled.repeat(sem_repeat, 1)[:semantic_dim, :].transpose(0, 1).contiguous()
    else:
        semantic_features = mel_pooled[:semantic_dim, :].transpose(0, 1).contiguous()

    frame_ms = hop_length * 1000.0 / sample_rate_hz
    semantic_times_ms = (
        semantic_starts.to(torch.float32) + (valid_counts.to(torch.float32) - 1.0) / 2.0
    ) * frame_ms

    hypotheses, clock_evidence = estimate_audio_clock(
        onset_norm,
        sample_rate_hz=sample_rate_hz,
        hop_length=hop_length,
        duration_ms=duration_ms,
        bpm_override=bpm,
        offset_override_ms=offset_ms,
    )

    # Bulk export avoids Python byte-by-byte iteration over tensor storage.
    raw_pcm_bytes = waveform.to(torch.float32).contiguous().cpu().numpy().tobytes()
    raw_pcm_sha256 = pcm_sha256(raw_pcm_bytes)
    if audio_sha256 is None:
        audio_sha256 = raw_pcm_sha256

    source_identity = SourceIdentity(
        audio_sha256=audio_sha256,
        canonical_audio_sha256=raw_pcm_sha256,
        audio_duration_ms=duration_ms,
    )
    canonical_spec = CanonicalAudioSpec(
        sample_rate_hz=sample_rate_hz,
        decoder_revision="mean-downmix-ffmpeg-resample-or-native-pcm-v3",
    )
    canonical_audio = CanonicalAudioManifest(
        source=source_identity,
        pcm_sha256=raw_pcm_sha256,
        frame_count=n_samples,
        spec=canonical_spec,
    )

    rhythm_hash = hashlib.sha256(rhythm_features.contiguous().cpu().numpy().tobytes()).hexdigest()
    rhythm_t_hash = hashlib.sha256(rhythm_times_ms.contiguous().cpu().numpy().tobytes()).hexdigest()
    semantic_hash = hashlib.sha256(semantic_features.contiguous().cpu().numpy().tobytes()).hexdigest()
    semantic_t_hash = hashlib.sha256(semantic_times_ms.contiguous().cpu().numpy().tobytes()).hexdigest()

    rhythm_stream = FeatureStreamManifest(
        name="rhythm_features",
        extractor="audio-logmel-onset",
        extractor_revision="v2-rhythm-100hz",
        feature_dim=rhythm_dim,
        frame_count=n_frames,
        dtype="float32",
        feature_path="features/rhythm.npy",
        feature_sha256=rhythm_hash,
        timestamps_path="timestamps/rhythm.npy",
        timestamps_sha256=rhythm_t_hash,
        nominal_frame_rate_hz=sample_rate_hz / hop_length,
    )

    semantic_stream = FeatureStreamManifest(
        name="semantic_features",
        extractor="audio-mel-pooled",
        extractor_revision="v2-semantic-25hz-centered-v2",
        feature_dim=semantic_dim,
        frame_count=n_sem_frames,
        dtype="float32",
        feature_path="features/semantic.npy",
        feature_sha256=semantic_hash,
        timestamps_path="timestamps/semantic.npy",
        timestamps_sha256=semantic_t_hash,
        nominal_frame_rate_hz=sample_rate_hz / (hop_length * sem_hop_frames),
    )

    cache_manifest = FeatureCacheManifest(
        canonical_audio=canonical_audio,
        streams=(rhythm_stream, semantic_stream),
        pipeline_revision=pipeline_revision,
    )

    bundle = FeatureBundle(
        feature_cache=cache_manifest,
        rhythm_features=rhythm_features,
        rhythm_times_ms=rhythm_times_ms,
        semantic_features=semantic_features,
        semantic_times_ms=semantic_times_ms,
        clock_evidence=clock_evidence,
    )

    return bundle, hypotheses


class AudioFeatureExtractor:
    """Default pinned audio feature extractor for v2 inference."""

    def __init__(
        self,
        *,
        sample_rate_hz: int = 24000,
        rhythm_dim: int = 101,
        semantic_dim: int = 128,
        models: LoadedV2Models | None = None,
        pipeline_revision: str | None = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.rhythm_dim = rhythm_dim
        self.semantic_dim = semantic_dim
        self.models = models
        self.pipeline_revision = pipeline_revision

    def extract(self, audio_path: str | Path) -> FeatureBundle:
        waveform, sr = load_audio(audio_path, target_sr=self.sample_rate_hz)
        bundle, _ = extract_audio_features(
            waveform,
            sr,
            models=self.models,
            rhythm_dim=self.rhythm_dim,
            semantic_dim=self.semantic_dim,
            pipeline_revision=self.pipeline_revision,
        )
        return bundle


__all__ = [
    "AudioClockResolver",
    "AudioFeatureExtractor",
    "FeatureBundle",
    "FeatureProvider",
    "GeneratedDraft",
    "LoadedV2Models",
    "create_mel_filterbank",
    "estimate_audio_clock",
    "extract_audio_features",
    "feature_bundle_from_payload",
    "generate_draft",
    "load_audio",
    "load_feature_bundle",
    "load_v2_models",
    "project_event_intervals",
    "timing_points_from_hypothesis",
]
