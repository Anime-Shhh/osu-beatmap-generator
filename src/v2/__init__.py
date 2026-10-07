"""Architecture-v2 primitives for sparse-event osu!standard generation.

The v2 package is intentionally separate from the legacy tokenizer and latent
flow experiment. It provides structured target handling, group-safe data
splits, and a hierarchical sparse-event model scaffold without changing legacy
commands.
"""

from importlib import import_module

from .audio import (
    CANONICAL_SAMPLE_RATE_HZ,
    CanonicalAudioManifest,
    CanonicalAudioSpec,
    FeatureCacheManifest,
    FeatureStreamManifest,
    TimestampAlignment,
    apply_scalar_alignment,
    build_timestamp_alignment,
    pcm_sha256,
    regular_timestamps_ms,
    validate_feature_stream_timestamps,
    validate_timestamps_ms,
)
from .clock import (
    BarPosition,
    ClockEvidence,
    ClockResolver,
    MeterMap,
    MeterSegment,
    OracleClockResolver,
    StaticClockResolver,
    TimingHypothesis,
    jitter_timing_hypothesis,
    select_timing_hypothesis,
)
from .data import (
    SourceIdentity,
    SourceSplitAssignment,
    assert_group_disjoint,
    assert_source_split_disjoint,
    assign_group_split,
    audio_sha256,
    build_source_split_assignments,
    build_source_split_manifest,
    source_component_groups,
)
from .timing import (
    KiaiSegment,
    SliderVelocitySegment,
    TimingMap,
    TimingPoint,
    TimingSegment,
    parse_timing_map,
    parse_timing_points,
    timing_map_from_points,
)

__all__ = [
    "BeatEvent",
    "BeatGridTargets",
    "BarPosition",
    "CANONICAL_SAMPLE_RATE_HZ",
    "CanonicalAudioManifest",
    "CanonicalAudioSpec",
    "ClockEvidence",
    "ClockResolver",
    "DifficultySettings",
    "EventKind",
    "FeatureCacheManifest",
    "FeatureStreamManifest",
    "GeneralSettings",
    "HitSample",
    "KiaiSegment",
    "MapMetadata",
    "MapSpec",
    "MeterMap",
    "MeterSegment",
    "OracleClockResolver",
    "SourceIdentity",
    "SourceSplitAssignment",
    "StaticClockResolver",
    "TimingMap",
    "TimingHypothesis",
    "TimingPoint",
    "TimingSegment",
    "TimestampAlignment",
    "SliderVelocitySegment",
    "ValidationReport",
    "ValidationPolicy",
    "apply_scalar_alignment",
    "assign_group_split",
    "assert_group_disjoint",
    "assert_source_split_disjoint",
    "audio_sha256",
    "build_source_split_assignments",
    "build_source_split_manifest",
    "build_timestamp_alignment",
    "build_beat_grid_targets",
    "compile_osu",
    "events_from_osu",
    "jitter_timing_hypothesis",
    "mapspec_from_osu",
    "parse_timing_map",
    "parse_timing_points",
    "pcm_sha256",
    "regular_timestamps_ms",
    "select_timing_hypothesis",
    "source_component_groups",
    "timing_map_from_points",
    "validate_feature_stream_timestamps",
    "validate_timestamps_ms",
    "validate_events",
    "validate_map_spec",
]


_EVENT_EXPORTS = {
    "BeatEvent",
    "BeatGridTargets",
    "EventKind",
    "HitSample",
    "ValidationReport",
    "build_beat_grid_targets",
    "events_from_osu",
    "validate_events",
}


_MAPSPEC_EXPORTS = {
    "DifficultySettings",
    "GeneralSettings",
    "MapMetadata",
    "MapSpec",
    "ValidationPolicy",
    "compile_osu",
    "mapspec_from_osu",
    "validate_map_spec",
}


def __getattr__(name: str):
    """Keep metadata/timing tools usable in lightweight preprocessing envs.

    The event-grid and model modules require PyTorch, while audio-hash split
    assignment does not.  Lazy exports prevent a split-audit command from
    failing merely because it runs on a CPU/login node without the training
    environment activated.
    """

    if name in _EVENT_EXPORTS:
        module = import_module(f"{__name__}.events")
        return getattr(module, name)
    if name in _MAPSPEC_EXPORTS:
        module = import_module(f"{__name__}.mapspec")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
