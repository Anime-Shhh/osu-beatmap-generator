"""Explicit construction of leakage-safe v2 song records.

This module does not stream Hugging Face, decode audio, download MERT, or run
a beat tracker on import.  Those expensive/external operations belong to
separately invoked extractor adapters.  Once timestamped feature tensors and a
canonical cache manifest exist, these helpers build the compact, auditable
records consumed by :mod:`src.v2.dataset`.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - preprocessing needs the train env.
    raise RuntimeError("src.v2.preprocess requires PyTorch") from exc

from .audio import FeatureCacheManifest
from .clock import MeterSegment, TimingHypothesis
from .data import SourceSplitAssignment
from .dataset import (
    V2_SONG_RECORD_SCHEMA,
    song_record_index,
    song_record_index_path,
    validate_song_record,
)
from .events import BeatEvent, EventKind, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .grammar import BarEventCodec, EventVocabulary, MAX_ANCHORS
from .mapspec import MapSpec
from .timing import TimingMap


ARCHITECTURE_ID = "tempo-posterior-hierarchical-event-diffusion/v2"


@dataclass(frozen=True)
class BarDefinition:
    """A real meter-aware decoding span expressed in continuous beats."""

    bar_index: int
    start_beat: float
    end_beat: float

    @property
    def length_beats(self) -> float:
        return self.end_beat - self.start_beat


def meter_segments_from_mapspec(map_spec: MapSpec) -> tuple[MeterSegment, ...]:
    """Translate native red timing-point meters to inferred-clock bar segments.

    osu!'s file format stores only the numerator, so generated/native maps use
    a quarter-note denominator. A rare red point that lands mid-bar begins a
    new canonical bar instead of silently letting an old meter continue across
    it; the source record retains the original timing point for audit.
    """

    timing = map_spec.timing_map
    result: list[MeterSegment] = []
    for segment in timing.segments:
        if not result:
            result.append(MeterSegment(0.0, segment.meter, 4))
            continue
        # Red timing points commonly change BPM inside a continuing meter. They
        # are timing-map boundaries, not automatically bar boundaries.
        if segment.meter == result[-1].numerator:
            continue
        previous = result[-1]
        bars_since_previous = (segment.start_beat - previous.start_beat) / previous.bar_length_beats
        rounded = round(bars_since_previous)
        start_bar = None
        if not math.isclose(bars_since_previous, rounded, abs_tol=1e-6):
            # Canonicalize an awkward historical point as the next absolute
            # bar. This makes its behavior explicit and prevents a fixed 4/4
            # segmentation from silently crossing it.
            prior_map = TimingHypothesis(
                hypothesis_id="meter-canonicalization",
                timing_map=timing,
                meter_segments=tuple(result),
                confidence=1.0,
                source="native-map",
                is_oracle=True,
            ).meter_map
            start_bar = prior_map.bar_position(segment.start_beat).bar_index + 1
        result.append(MeterSegment(segment.start_beat, segment.meter, 4, start_bar=start_bar))
    return tuple(result)


def oracle_timing_hypothesis(map_spec: MapSpec) -> TimingHypothesis:
    """Build a marked oracle clock for targets only, never audio-only inference."""

    return TimingHypothesis(
        hypothesis_id="oracle-map-timing",
        timing_map=map_spec.timing_map,
        meter_segments=meter_segments_from_mapspec(map_spec),
        confidence=1.0,
        score=1.0,
        source="source-osu-timing",
        is_oracle=True,
    )


def build_bar_definitions(hypothesis: TimingHypothesis, *, end_beat: float) -> tuple[BarDefinition, ...]:
    """Build bar spans without assuming 4/4 or a constant BPM.

    Meter transitions are honored as boundaries. The sparse grammar supports a
    fractional quarter-note bar length (for example 3/8), provided its start
    offset can be represented by the allowed rational snap vocabulary.
    """

    if not math.isfinite(end_beat) or end_beat < 0.0:
        raise ValueError("end_beat must be finite and non-negative")
    meter_map = hypothesis.meter_map
    meter_starts = tuple(segment.start_beat for segment in meter_map.segments[1:])
    bars: list[BarDefinition] = []
    start = 0.0
    index = 0
    # One bar is retained for an empty or very short song so the planner has a
    # valid target context; there is no implicit fixed-duration window.
    while not bars or start < end_beat - 1e-7:
        meter = meter_map.segment_at_beat(start + 1e-8)
        natural_end = start + meter.bar_length_beats
        upcoming = next((value for value in meter_starts if value > start + 1e-7), None)
        end = min(natural_end, upcoming) if upcoming is not None else natural_end
        if end <= start + 1e-7:
            raise ValueError("meter map did not advance while constructing bars")
        bars.append(BarDefinition(index, start, end))
        index += 1
        start = end
        if index > 4096:
            raise ValueError("song has more than 4096 bars; split it before preprocessing")
    return tuple(bars)


def clock_features_for_bars(
    hypothesis: TimingHypothesis,
    bars: Sequence[BarDefinition],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actual bar-mid timestamps and four clock conditioning features.

    Features are intentionally bounded/simple: normalized log BPM, normalized
    bar length, timing confidence, and local tempo-change magnitude.
    Crucially, this does *not* include source-map Kiai state: the v1 audio-only
    clock adapter has no Kiai predictor, so exposing it during training would
    be an oracle leak. Beat/downbeat posterior detail remains in the rhythmic
    feature stream rather than pretending a heuristic feature vector is an
    audio representation.
    """

    mid_times: list[float] = []
    values: list[tuple[float, float, float, float]] = []
    previous_bpm: float | None = None
    for bar in bars:
        midpoint_beat = (bar.start_beat + bar.end_beat) * 0.5
        time_ms = hypothesis.timing_map.beat_to_time(midpoint_beat)
        beat_length = hypothesis.timing_map.beat_length_at_time(time_ms)
        bpm = 60000.0 / beat_length
        tempo_change = 0.0
        if previous_bpm is not None:
            # Bounded signed log-ratio makes a red-point transition visible
            # without smuggling presentation fields such as Kiai into a
            # timing-only inference contract.
            tempo_change = max(-1.0, min(1.0, math.log(bpm / previous_bpm) / math.log(2.0)))
        mid_times.append(time_ms)
        values.append(
            (
                math.log(max(bpm, 1.0)) / math.log(300.0),
                min(1.0, bar.length_beats / 12.0),
                hypothesis.confidence,
                tempo_change,
            )
        )
        previous_bpm = bpm
    return torch.tensor(mid_times, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)


def compact_profile(
    *,
    target_star: float,
    approach_rate: float,
    circle_size: float,
    overall_difficulty: float,
    hp_drain_rate: float,
    slider_multiplier: float,
    slider_tick_rate: float,
) -> torch.Tensor:
    """Six normalized controls consumed by the planner.

    HP and OD share one compact difficulty-axis component, while slider
    multiplier/tick rate remain explicit: otherwise source slider duration and
    geometry targets would be trained under settings unavailable at inference.
    This is explicit rather than training with an unavailable raw mapper ID.
    """

    if slider_multiplier <= 0.0 or slider_tick_rate <= 0.0:
        raise ValueError("slider multiplier and tick rate must be positive")
    def normalized_slider_setting(value: float) -> float:
        # Log scaling preserves ordinary low settings and high historical
        # values without making a 4.0 clip indistinguishable from 10.0.
        return min(1.0, max(0.0, math.log(value / 0.05) / math.log(10.0 / 0.05)))

    values = (
        target_star / 10.0,
        approach_rate / 10.0,
        circle_size / 10.0,
        (overall_difficulty + hp_drain_rate) / 20.0,
        normalized_slider_setting(slider_multiplier),
        normalized_slider_setting(slider_tick_rate),
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("profile values must be finite")
    return torch.tensor(values, dtype=torch.float32)


def _feature_tensor(value: torch.Tensor, name: str, *, dim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dim() != dim:
        raise ValueError(f"{name} must be a {dim}D torch Tensor")
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if value.shape[0] < 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return value


def _timestamps(value: torch.Tensor, name: str, count: int) -> torch.Tensor:
    value = _feature_tensor(value, name, dim=1)
    if value.numel() != count:
        raise ValueError(f"{name} must align with its feature stream")
    if value.numel() > 1 and not torch.all(value[1:] > value[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    return value


def _selected_events(events: Sequence[BeatEvent], bar: BarDefinition) -> tuple[BeatEvent, ...]:
    return tuple(
        event
        for event in sorted(events, key=lambda item: (item.beat_position, item.source_object_index))
        if bar.start_beat <= event.beat_position < bar.end_beat
    )


def reproject_events_absolute(
    events: Sequence[BeatEvent],
    *,
    source_timing: TimingMap,
    target_timing: TimingMap,
) -> tuple[BeatEvent, ...]:
    """Move event beat coordinates between clocks without moving physical time.

    This is the required primitive for any future non-oracle clock target. The
    current record schema deliberately disables such variants because it has no
    variant-specific bar-target container; retaining only shifted audio queries
    with oracle tokens would teach a systematic timing error.
    """

    projected: list[BeatEvent] = []
    for event in events:
        start_ms = source_timing.beat_to_time(event.beat_position)
        end_beat = None
        if event.end_beat is not None:
            end_ms = source_timing.beat_to_time(event.end_beat)
            end_beat = target_timing.time_to_beat(end_ms)
        projected.append(
            replace(
                event,
                beat_position=target_timing.time_to_beat(start_ms),
                end_beat=end_beat,
            )
        )
    return tuple(sorted(projected, key=lambda event: (event.beat_position, event.source_object_index)))


def _geometry_targets(events: Sequence[BeatEvent]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    geometry = torch.zeros((len(events), MAX_ANCHORS + 1, 2), dtype=torch.float32)
    kinds = torch.empty((len(events),), dtype=torch.long)
    anchor_counts = torch.zeros((len(events),), dtype=torch.long)
    lengths = torch.zeros((len(events),), dtype=torch.float32)
    for index, event in enumerate(events):
        geometry[index, 0] = torch.tensor((event.x / PLAYFIELD_WIDTH, event.y / PLAYFIELD_HEIGHT))
        kinds[index] = int(event.kind)
        if event.kind == EventKind.SLIDER:
            if not 1 <= len(event.control_points) <= MAX_ANCHORS:
                raise ValueError(
                    f"slider at beat {event.beat_position:.3f} has unsupported anchor count "
                    f"{len(event.control_points)} (expected 1..{MAX_ANCHORS})"
                )
            anchor_counts[index] = len(event.control_points)
            for anchor_index, (x, y) in enumerate(event.control_points, start=1):
                geometry[index, anchor_index] = torch.tensor((x / PLAYFIELD_WIDTH, y / PLAYFIELD_HEIGHT))
            lengths[index] = float(event.slider_length)
    if (geometry < 0).any() or (geometry > 1).any():
        raise ValueError("source event geometry lies outside the osu! playfield")
    return geometry, kinds, anchor_counts, lengths


def _geometry_event_features(
    selected: Sequence[BeatEvent],
    all_events: Sequence[BeatEvent],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Features that make geometry aware of local movement context.

    The causal symbolic hidden state knows the current plan, but geometry also
    needs explicit previous/next timing gaps and target path scale to learn
    readable spacing. Features are bounded log values so a rare long break or
    spinner does not dominate the diffusion conditioner.
    """

    ordered = tuple(sorted(all_events, key=lambda item: (item.beat_position, item.source_object_index)))
    indices = {id(event): index for index, event in enumerate(ordered)}
    values: list[tuple[float, float, float, float]] = []
    positions: list[int] = []
    normalizer = math.log1p(32.0)
    for event in selected:
        index = indices[id(event)]
        prior_gap = event.beat_position - ordered[index - 1].beat_position if index else 32.0
        next_gap = (
            ordered[index + 1].beat_position - event.beat_position
            if index + 1 < len(ordered)
            else 32.0
        )
        duration = max(0.0, (event.end_beat or event.beat_position) - event.beat_position)
        values.append(
            (
                min(1.0, math.log1p(max(0.0, prior_gap)) / normalizer),
                min(1.0, math.log1p(max(0.0, next_gap)) / normalizer),
                min(1.0, math.log1p(duration) / normalizer),
                min(1.0, math.log1p(max(0.0, event.slider_length)) / math.log1p(1000.0)),
            )
        )
        positions.append(index)
    return (
        torch.tensor(values, dtype=torch.float32).reshape(len(selected), 4),
        torch.tensor(positions, dtype=torch.long),
    )


def _bar_target(
    events: Sequence[BeatEvent],
    bar: BarDefinition,
    codec: BarEventCodec,
) -> dict[str, Any]:
    selected = _selected_events(events, bar)
    tokens = torch.tensor(
        codec.encode(selected, bar_start_beat=bar.start_beat, meter=bar.length_beats),
        dtype=torch.long,
    )
    # Use the completed-event token, not the initial EVENT marker: causal
    # geometry conditioning must see kind/duration/topology fields.
    event_positions = torch.tensor(codec.event_context_positions(tokens.tolist()), dtype=torch.long)
    geometry, kinds, anchors, lengths = _geometry_targets(selected)
    geometry_features, global_positions = _geometry_event_features(selected, events)
    if event_positions.numel() != len(selected):
        raise AssertionError("grammar encoder did not produce one completed context position per event")
    return {
        "bar_index": bar.bar_index,
        "bar_start_beat": bar.start_beat,
        "bar_length_beats": bar.length_beats,
        "tokens": tokens,
        "geometry": geometry,
        "event_kinds": kinds,
        "anchor_counts": anchors,
        "event_token_indices": event_positions,
        "target_lengths_px": lengths,
        "event_features": geometry_features,
        "event_positions": global_positions,
    }


def build_song_record(
    *,
    map_spec: MapSpec,
    source_assignment: SourceSplitAssignment,
    feature_cache: FeatureCacheManifest,
    rhythm_features: torch.Tensor,
    rhythm_times_ms: torch.Tensor,
    semantic_features: torch.Tensor,
    semantic_times_ms: torch.Tensor,
    target_star: float,
    clock_hypotheses: Sequence[TimingHypothesis] = (),
    architecture_id: str = ARCHITECTURE_ID,
) -> dict[str, Any]:
    """Create one serializable song record from externally extracted features.

    The oracle timing hypothesis is retained strictly as a supervision target.
    Non-oracle clocks are rejected until the record schema carries their own
    reprojected bars and targets. Reusing oracle labels with shifted audio
    queries is an unsafe augmentation, not robustness training.
    """

    assignment_source = source_assignment.source
    assignment_source.validate_training_provenance()
    cache_source = feature_cache.canonical_audio.source
    # A feature cache belongs to an audio asset, while a source assignment can
    # additionally name one difficulty/map/mapper. Compare immutable audio
    # evidence rather than dataclass equality so multiple maps for the same
    # song can reuse one cache without losing their provenance.
    if assignment_source.audio_sha256 != cache_source.audio_sha256:
        raise ValueError("source assignment and feature cache have different encoded audio")
    if (
        assignment_source.canonical_audio_sha256 is not None
        and cache_source.canonical_audio_sha256 is not None
        and assignment_source.canonical_audio_sha256 != cache_source.canonical_audio_sha256
    ):
        raise ValueError("source assignment and feature cache have different canonical audio")
    if (
        assignment_source.beatmap_id is not None
        and map_spec.metadata.beatmap_id > 0
        and assignment_source.beatmap_id != str(map_spec.metadata.beatmap_id)
    ):
        raise ValueError("source assignment beatmap ID does not match parsed map metadata")
    if (
        assignment_source.beatmap_set_id is not None
        and map_spec.metadata.beatmap_set_id > 0
        and assignment_source.beatmap_set_id != str(map_spec.metadata.beatmap_set_id)
    ):
        raise ValueError("source assignment beatmap set ID does not match parsed map metadata")
    if not architecture_id.strip():
        raise ValueError("architecture_id must be non-empty")
    rhythm_features = _feature_tensor(rhythm_features, "rhythm_features", dim=2)
    rhythm_times_ms = _timestamps(rhythm_times_ms, "rhythm_times_ms", rhythm_features.shape[0])
    semantic_features = _feature_tensor(semantic_features, "semantic_features", dim=2)
    semantic_times_ms = _timestamps(semantic_times_ms, "semantic_times_ms", semantic_features.shape[0])
    oracle = oracle_timing_hypothesis(map_spec)
    if clock_hypotheses:
        raise ValueError(
            "non-oracle clock variants are disabled until variant-specific "
            "absolute-time targets are represented in the song-record schema"
        )
    variants = (oracle,)
    if any(hypothesis.meter_segments != oracle.meter_segments for hypothesis in variants):
        raise ValueError("clock hypotheses must share the canonical meter topology")
    if len({hypothesis.hypothesis_id for hypothesis in variants}) != len(variants):
        raise ValueError("clock hypothesis IDs must be unique")

    max_event_beat = max(
        (
            max(event.beat_position, event.end_beat or event.beat_position)
            for event in map_spec.events
        ),
        default=0.0,
    )
    if max_event_beat < -1e-7:
        raise ValueError("source map has all hit objects before its first red timing point")
    duration_end_beat = oracle.timing_map.time_to_beat(feature_cache.canonical_audio.duration_ms)
    # The tiny event epsilon includes an object exactly on a bar boundary
    # without adding a speculative post-audio bar when the song itself ends on
    # that boundary.
    bars = build_bar_definitions(
        oracle,
        end_beat=max(0.0, max_event_beat + 1e-7, duration_end_beat),
    )
    primary_times, primary_clock = clock_features_for_bars(oracle, bars)
    codec = BarEventCodec(EventVocabulary())
    targets = [_bar_target(map_spec.events, bar, codec) for bar in bars]
    clock_variants = []
    for hypothesis in variants:
        times, clock = clock_features_for_bars(hypothesis, bars)
        clock_variants.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "is_oracle": hypothesis.is_oracle,
                "bar_mid_times_ms": times,
                "clock_features": clock,
                "hypothesis": hypothesis.to_dict(),
            }
        )
    record = {
        "schema_version": V2_SONG_RECORD_SCHEMA,
        "metadata": {
            "cache_key": feature_cache.cache_key,
            "feature_pipeline_key": feature_cache.pipeline_key,
            "source_component_id": source_assignment.component_id,
            "split": source_assignment.split,
            "architecture_id": architecture_id,
            "source": source_assignment.source.to_dict(),
            "feature_pipeline_revision": feature_cache.pipeline_revision,
        },
        "rhythm_features": rhythm_features,
        "rhythm_times_ms": rhythm_times_ms,
        "semantic_features": semantic_features,
        "semantic_times_ms": semantic_times_ms,
        "bar_mid_times_ms": primary_times,
        "clock_features": primary_clock,
        "profile": compact_profile(
            target_star=target_star,
            approach_rate=map_spec.difficulty.approach_rate,
            circle_size=map_spec.difficulty.circle_size,
            overall_difficulty=map_spec.difficulty.overall_difficulty,
            hp_drain_rate=map_spec.difficulty.hp_drain_rate,
            slider_multiplier=map_spec.difficulty.slider_multiplier,
            slider_tick_rate=map_spec.difficulty.slider_tick_rate,
        ),
        "bar_targets": targets,
        "clock_variants": clock_variants,
        "oracle_timing_hypothesis": oracle.to_dict(),
    }
    validate_song_record(record)
    return record


def write_song_record(record: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write one validated record and its tensor-free JSON index.

    The sidecar prevents dataset construction from deserializing every full
    feature tensor just to find a split/bar count.  It is checked again against
    the full payload at first sample access, so it is an optimization rather
    than a second source of truth.
    """

    validate_song_record(record)
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("v2 song record paths must end in .pt")
    index_destination = song_record_index_path(destination)
    index_payload = song_record_index(record, validate=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite v2 record: {destination}")
    if index_destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite v2 record index: {index_destination}")
    record_descriptor, record_temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    index_descriptor, index_temporary_name = tempfile.mkstemp(
        prefix=f".{index_destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(record_descriptor)
    os.close(index_descriptor)
    record_temporary = Path(record_temporary_name)
    index_temporary = Path(index_temporary_name)
    record_replaced = False
    index_replaced = False
    try:
        torch.save(dict(record), record_temporary)
        index_temporary.write_text(
            json.dumps(index_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        if not overwrite and destination.exists():
            raise FileExistsError(f"refusing to overwrite v2 record: {destination}")
        if not overwrite and index_destination.exists():
            raise FileExistsError(f"refusing to overwrite v2 record index: {index_destination}")
        os.replace(record_temporary, destination)
        record_replaced = True
        os.replace(index_temporary, index_destination)
        index_replaced = True
    except BaseException:
        record_temporary.unlink(missing_ok=True)
        index_temporary.unlink(missing_ok=True)
        # A record that was replaced just before a filesystem error must not
        # retain an old sidecar: dataset construction will safely fall back to
        # validating the full tensor record instead of trusting stale split or
        # bar-count metadata. The record itself is never deleted here.
        if record_replaced and not index_replaced:
            index_destination.unlink(missing_ok=True)
        raise
    return destination


def ranked_standard_source_filter(record: Mapping[str, Any]) -> bool:
    """Conservative source filter for a quality-first project-riz manifest.

    It accepts the common field spellings without guessing at absent metadata;
    unknown status or mode is excluded rather than quietly entering the primary
    Ranked corpus. Approved/Loved can be prepared into separate tiers.
    """

    mode = str(record.get("mode", record.get("game_mode", ""))).strip().lower()
    status = str(record.get("status", record.get("approval_status", ""))).strip().lower()
    standard = mode in {"osu", "standard", "0"}
    ranked = status == "ranked"
    return standard and ranked


def source_manifest_line(record: Mapping[str, Any]) -> str:
    """Emit stable JSON for an immutable, reviewable source manifest."""

    return json.dumps(dict(record), sort_keys=True, separators=(",", ":"))


__all__ = [
    "ARCHITECTURE_ID",
    "BarDefinition",
    "build_bar_definitions",
    "build_song_record",
    "clock_features_for_bars",
    "compact_profile",
    "meter_segments_from_mapspec",
    "oracle_timing_hypothesis",
    "ranked_standard_source_filter",
    "reproject_events_absolute",
    "source_manifest_line",
    "write_song_record",
]
