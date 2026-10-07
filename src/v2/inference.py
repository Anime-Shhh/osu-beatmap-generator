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
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch

from .audio import FeatureCacheManifest
from .clock import ClockEvidence, ClockResolver, TimingHypothesis
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
    project_slider_paths,
    sample_geometry_windows,
)
from .grammar import BarEventCodec, EventVocabulary, PlannerEvent
from .mapspec import DifficultySettings, GeneralSettings, MapMetadata, MapSpec
from .packaging import GeneratedPackage, write_generation_package
from .preprocess import ARCHITECTURE_ID, BarDefinition, build_bar_definitions, clock_features_for_bars, compact_profile
from .sparse_model import SparseEventPlanner, SparsePlannerConfig
from .timing import TimingPoint
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
    planner_config_payload = payload.get("planner_config")
    geometry_config_payload = payload.get("geometry_config")
    provenance_payload = payload.get("provenance")
    if not isinstance(planner_config_payload, Mapping) or not isinstance(geometry_config_payload, Mapping):
        raise ValueError("checkpoint must contain planner and geometry configs")
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("checkpoint is missing provenance")
    planner_config = SparsePlannerConfig(**dict(planner_config_payload))
    geometry_config = GeometryDiffusionConfig(**dict(geometry_config_payload))
    if geometry_config.context_dim != planner_config.model_dim:
        raise ValueError("checkpoint geometry context dimension differs from planner decoder dimension")
    if geometry_config.event_feature_dim != 4:
        raise ValueError("checkpoint geometry feature dimension differs from the v2 record contract")
    planner = SparseEventPlanner(planner_config)
    geometry = ActiveEventGeometryDiffuser(geometry_config)
    planner_state = payload.get("planner")
    geometry_state = payload.get("geometry")
    if not isinstance(planner_state, Mapping) or not isinstance(geometry_state, Mapping):
        raise ValueError("checkpoint does not contain both trained model state dictionaries")
    planner.load_state_dict(planner_state)
    geometry.load_state_dict(geometry_state)
    provenance = CheckpointProvenance.from_dict(provenance_payload)
    if provenance.vocabulary_sha256 != vocabulary_sha256(planner.vocabulary):
        raise ValueError("checkpoint sparse vocabulary does not match its model")
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

    Timing/SV values are emitted; Kiai and presentation effects remain absent
    until they have an explicit predictor. This avoids the old system's hidden
    reuse of a reference `.osu` timing section at audio-only inference.
    """

    points: list[TimingPoint] = []
    for segment in hypothesis.timing_map.segments:
        meter = hypothesis.meter_map.segment_at_beat(segment.start_beat).numerator
        points.append(
            TimingPoint(
                offset_ms=segment.start_ms,
                beat_length_ms=segment.beat_length_ms,
                meter=meter,
                sample_set=0,
                sample_index=0,
                volume=100,
                uninherited=True,
                effects=0,
            )
        )
    for segment in hypothesis.timing_map.slider_velocity_segments:
        # The timing utility's negative-time segment is a query sentinel, not
        # an osu! source point. Do not serialize it.
        if segment.start_ms <= -1.0e11:
            continue
        points.append(
            TimingPoint(
                offset_ms=segment.start_ms,
                beat_length_ms=-100.0 / segment.multiplier,
                meter=4,
                sample_set=0,
                sample_index=0,
                volume=100,
                uninherited=False,
                effects=0,
            )
        )
    return tuple(sorted(points, key=lambda point: point.offset_ms))


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
    )
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
    for bar in bars:
        tokens = planner.generate_bar(
            memory,
            memory_mask=bar_mask,
            target_bar_index=bar.bar_index,
            bar_length_beats=bar.length_beats,
            previous_bar_tokens=previous_tokens,
            max_tokens=planner.config.max_decoder_tokens - 1,
            temperature=temperature,
            generator=generator,
        )
        decoded = codec.decode(tokens, bar_start_beat=bar.start_beat, meter=bar.length_beats)
        planned_bars.append((bar, tokens, decoded, previous_tokens))
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
        hidden = planner.decode_hidden(
            token_tensor,
            memory,
            target_bar_index=torch.tensor([bar.bar_index], device=device),
            memory_mask=bar_mask,
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
        full_geometry = project_slider_paths(
            full_geometry, kinds, anchors, target_lengths, event_mask=mask
        )[0].cpu()
    else:
        full_geometry = torch.zeros((0, geometry.config.max_anchors + 1, 2))
    events = project_event_intervals(
        _geometry_to_events(plans, full_geometry, hypothesis, difficulty), hypothesis, difficulty
    )
    star = float(request.target_star if request.target_star is not None else DEFAULT_TARGET_STAR)
    return MapSpec(
        format_version=14,
        general=GeneralSettings(audio_filename=request.audio_filename, sample_set="Soft", mode=0),
        metadata=MapMetadata(
            title="Generated Draft",
            artist="Unknown Artist",
            creator="osu-beatmap-generator",
            version=f"v2 {star:.1f}★ Draft",
            tags="generated ranked-like draft human-review-required",
        ),
        difficulty=difficulty,
        timing_points=timing_points_from_hypothesis(hypothesis),
        events=events,
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


__all__ = [
    "FeatureBundle",
    "FeatureProvider",
    "GeneratedDraft",
    "LoadedV2Models",
    "feature_bundle_from_payload",
    "generate_draft",
    "load_feature_bundle",
    "load_v2_models",
    "project_event_intervals",
    "timing_points_from_hypothesis",
]
