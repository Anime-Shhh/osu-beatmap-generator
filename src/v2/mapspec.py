"""Canonical, compiler-ready osu!standard map representation.

The legacy tokenizer is intentionally not used here. ``MapSpec`` retains the
fields that determine whether a generated draft is playable and editable:
native timing points, settings, complete sliders/spinners, combo state, and
native hitsound fields. It is deliberately a small standard-library layer so
it can run during preprocessing and on login nodes without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping, Optional

from .events import (
    BeatEvent,
    EventKind,
    HitSample,
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
    ValidationReport,
    events_from_osu,
    serialize_hitobjects,
    validate_events,
)
from .timing import TimingMap, TimingPoint, parse_timing_points, timing_map_from_points


SUPPORTED_SAMPLE_SETS = frozenset({"Normal", "Soft", "Drum", "None"})
NATIVE_SAMPLE_SET_IDS = frozenset({0, 1, 2, 3})
DEFAULT_SNAP_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96)
_FORMAT_VERSION_RE = re.compile(r"^osu file format v(\d+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class GeneralSettings:
    """The generated-map subset of osu!'s ``[General]`` section."""

    audio_filename: str
    audio_lead_in: int = 0
    preview_time: int = -1
    countdown: int = 0
    sample_set: str = "Soft"
    stack_leniency: float = 0.7
    mode: int = 0


@dataclass(frozen=True)
class MapMetadata:
    """Human-facing metadata. It never carries mapper-style conditioning."""

    title: str
    artist: str
    creator: str
    version: str
    source: str = ""
    tags: str = ""
    title_unicode: str = ""
    artist_unicode: str = ""
    beatmap_id: int = 0
    beatmap_set_id: int = -1


@dataclass(frozen=True)
class DifficultySettings:
    """Native standard-mode difficulty and slider settings."""

    hp_drain_rate: float = 6.0
    circle_size: float = 4.0
    overall_difficulty: float = 8.0
    approach_rate: float = 9.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0


@dataclass(frozen=True)
class MapSpec:
    """An immutable intermediate representation for one osu!standard draft.

    All temporal fields in events use continuous beat position. Native timing
    points are retained separately rather than encoded as hidden conditioning
    values. ``timing_map`` is derived, never serialized as an independent
    source of truth.
    """

    format_version: int
    general: GeneralSettings
    metadata: MapMetadata
    difficulty: DifficultySettings
    timing_points: tuple[TimingPoint, ...]
    events: tuple[BeatEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timing_points", tuple(self.timing_points))
        object.__setattr__(self, "events", tuple(self.events))

    @property
    def timing_map(self) -> TimingMap:
        return timing_map_from_points(self.timing_points)


@dataclass(frozen=True)
class ValidationPolicy:
    """Strict output gate for ranked-like generated drafts.

    The policy is intentionally configurable so source-corpus audits can use
    the same validator without pretending every historical map follows the
    generator's conservative output rules.
    """

    min_start_gap_ms: float = 1.0
    reject_slider_overlap: bool = True
    reject_spinner_overlap: bool = True
    require_sorted_events: bool = True
    require_beat_snap: bool = True
    snap_denominators: tuple[int, ...] = DEFAULT_SNAP_DENOMINATORS
    snap_tolerance_beats: float = 1.0 / 192.0
    max_slider_control_points: int = 32
    # Optional conservative policy only. Native osu! extends a path when its
    # declared pixelLength exceeds the computed curve, so this is not syntax validity.
    require_slider_path_feasibility: bool = False
    slider_path_tolerance_px: float = 2.0
    slider_end_tolerance_ms: float = 2.0
    allow_custom_samples: bool = False
    require_nonempty_events: bool = True

    def __post_init__(self) -> None:
        if self.min_start_gap_ms < 0.0 or self.slider_end_tolerance_ms < 0.0:
            raise ValueError("timing tolerances must be non-negative")
        if self.snap_tolerance_beats < 0.0:
            raise ValueError("snap_tolerance_beats must be non-negative")
        if self.max_slider_control_points < 1 or self.slider_path_tolerance_px < 0.0:
            raise ValueError("slider path limits must be positive/non-negative")
        if not self.snap_denominators or any(value <= 0 for value in self.snap_denominators):
            raise ValueError("snap_denominators must contain positive values")


def _section_lines(osu_content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    active: Optional[str] = None
    for raw_line in osu_content.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            active = line[1:-1]
            sections.setdefault(active, [])
        elif active is not None:
            sections[active].append(raw_line.rstrip("\r"))
    return sections


def _key_values(lines: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//") or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def _parse_int(values: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, str(default)))
    except ValueError:
        return default


def _parse_float(values: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, str(default)))
    except ValueError:
        return default


def _format_number(value: float, *, digits: int = 6) -> str:
    if not math.isfinite(value):
        raise ValueError("cannot serialize a non-finite number")
    rounded = round(value)
    if abs(value - rounded) < 10 ** -(digits + 1):
        return str(int(rounded))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _format_timing_point(point: TimingPoint) -> str:
    return ",".join(
        (
            _format_number(point.offset_ms),
            _format_number(point.beat_length_ms),
            str(int(point.meter)),
            str(int(point.sample_set)),
            str(int(point.sample_index)),
            str(int(point.volume)),
            "1" if point.uninherited else "0",
            str(int(point.effects)),
        )
    )


def mapspec_from_osu(osu_content: str) -> MapSpec:
    """Parse the generator-relevant standard-mode fields from an ``.osu`` file.

    Parsing is permissive by design; call :func:`validate_map_spec` to audit
    malformed corpus records or to gate generated output. It intentionally
    ignores backgrounds, storyboards, breaks, and custom assets because v1
    generation does not produce them.
    """

    sections = _section_lines(osu_content)
    general_values = _key_values(sections.get("General", ()))
    metadata_values = _key_values(sections.get("Metadata", ()))
    difficulty_values = _key_values(sections.get("Difficulty", ()))

    format_version = 14
    for raw_line in osu_content.splitlines():
        match = _FORMAT_VERSION_RE.match(raw_line.strip())
        if match:
            format_version = int(match.group(1))
            break

    general = GeneralSettings(
        audio_filename=general_values.get("AudioFilename", "audio.mp3"),
        audio_lead_in=_parse_int(general_values, "AudioLeadIn", 0),
        preview_time=_parse_int(general_values, "PreviewTime", -1),
        countdown=_parse_int(general_values, "Countdown", 0),
        sample_set=general_values.get("SampleSet", "Soft"),
        stack_leniency=_parse_float(general_values, "StackLeniency", 0.7),
        mode=_parse_int(general_values, "Mode", 0),
    )
    metadata = MapMetadata(
        title=metadata_values.get("Title", "Unknown Title"),
        artist=metadata_values.get("Artist", "Unknown Artist"),
        creator=metadata_values.get("Creator", "Unknown Creator"),
        version=metadata_values.get("Version", "Generated"),
        source=metadata_values.get("Source", ""),
        tags=metadata_values.get("Tags", ""),
        title_unicode=metadata_values.get("TitleUnicode", ""),
        artist_unicode=metadata_values.get("ArtistUnicode", ""),
        beatmap_id=_parse_int(metadata_values, "BeatmapID", 0),
        beatmap_set_id=_parse_int(metadata_values, "BeatmapSetID", -1),
    )
    difficulty = DifficultySettings(
        hp_drain_rate=_parse_float(difficulty_values, "HPDrainRate", 6.0),
        circle_size=_parse_float(difficulty_values, "CircleSize", 4.0),
        overall_difficulty=_parse_float(difficulty_values, "OverallDifficulty", 8.0),
        approach_rate=_parse_float(difficulty_values, "ApproachRate", 9.0),
        slider_multiplier=_parse_float(difficulty_values, "SliderMultiplier", 1.4),
        slider_tick_rate=_parse_float(difficulty_values, "SliderTickRate", 1.0),
    )
    timing_points = parse_timing_points(osu_content)
    events = tuple(events_from_osu(osu_content, timing_map_from_points(timing_points)))
    return MapSpec(
        format_version=format_version,
        general=general,
        metadata=metadata,
        difficulty=difficulty,
        timing_points=timing_points,
        events=events,
    )


def _field_is_safe(value: str) -> bool:
    return "\n" not in value and "\r" not in value


def _sample_is_valid(sample: HitSample, *, allow_custom_samples: bool) -> bool:
    return (
        sample.normal_set in NATIVE_SAMPLE_SET_IDS
        and sample.addition_set in NATIVE_SAMPLE_SET_IDS
        and sample.custom_index >= 0
        and 0 <= sample.volume <= 100
        and (allow_custom_samples or not sample.filename)
        and _field_is_safe(sample.filename)
    )


def _nearest_snap_residual(beat: float, denominators: Iterable[int]) -> float:
    """Distance to the closest allowed rational beat position."""

    closest = math.inf
    for denominator in denominators:
        if denominator <= 0:
            continue
        snapped = round(beat * denominator) / denominator
        closest = min(closest, abs(beat - snapped))
    return closest


def _event_end_time_ms(
    event: BeatEvent,
    timing_map: TimingMap,
    slider_multiplier: float,
) -> Optional[float]:
    start_time_ms = timing_map.beat_to_time(event.beat_position)
    if event.kind == EventKind.SLIDER:
        return start_time_ms + timing_map.slider_duration_ms(
            start_time_ms,
            event.slider_length,
            event.slider_repeats,
            slider_multiplier,
        )
    if event.kind == EventKind.SPINNER and event.end_beat is not None:
        return timing_map.beat_to_time(event.end_beat)
    return None


def _validate_timing_points(
    timing_points: tuple[TimingPoint, ...],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not timing_points:
        errors.append("map has no timing points")
        return
    has_red = False
    prior_offset = -math.inf
    for index, point in enumerate(timing_points):
        if not (math.isfinite(point.offset_ms) and math.isfinite(point.beat_length_ms)):
            errors.append(f"timing point {index}: offset and beat length must be finite")
        if point.offset_ms < prior_offset:
            errors.append(f"timing point {index}: timing points are not time ordered")
        prior_offset = point.offset_ms
        if point.uninherited:
            has_red = True
            if point.beat_length_ms <= 0.0:
                errors.append(f"timing point {index}: red points require positive beat length")
        elif point.beat_length_ms >= 0.0:
            errors.append(f"timing point {index}: inherited points require negative beat length")
        if not 1 <= point.meter <= 16:
            errors.append(f"timing point {index}: meter must be between 1 and 16")
        if point.sample_set not in NATIVE_SAMPLE_SET_IDS:
            errors.append(f"timing point {index}: invalid sample set")
        if point.sample_index < 0:
            errors.append(f"timing point {index}: sample index must be non-negative")
        if not 0 <= point.volume <= 100:
            errors.append(f"timing point {index}: volume must be between 0 and 100")
        if point.effects < 0:
            errors.append(f"timing point {index}: effects must be non-negative")
        elif point.effects & ~0x9:
            warnings.append(f"timing point {index}: unknown timing effect bits are preserved")
    if not has_red:
        errors.append("map has no valid red timing point")


def _validate_settings(map_spec: MapSpec, errors: list[str]) -> None:
    general = map_spec.general
    if general.mode != 0:
        errors.append("only osu!standard mode (Mode: 0) is supported")
    if not general.audio_filename or not _field_is_safe(general.audio_filename):
        errors.append("AudioFilename must be a non-empty single-line filename")
    elif "/" in general.audio_filename or "\\" in general.audio_filename:
        errors.append("AudioFilename must not contain a path separator")
    if general.sample_set not in SUPPORTED_SAMPLE_SETS:
        errors.append("General SampleSet must be Normal, Soft, Drum, or None")
    if general.countdown not in {0, 1, 2, 3}:
        errors.append("Countdown must be in [0, 3]")
    if not math.isfinite(general.stack_leniency) or not 0.0 <= general.stack_leniency <= 1.0:
        errors.append("StackLeniency must be finite and between 0 and 1")

    if map_spec.format_version < 3:
        errors.append("osu file format version must be at least 3")
    for field_name in ("title", "artist", "creator", "version"):
        value = getattr(map_spec.metadata, field_name)
        if not value or not _field_is_safe(value):
            errors.append(f"Metadata {field_name} must be non-empty and single-line")
    for field_name in ("source", "tags", "title_unicode", "artist_unicode"):
        if not _field_is_safe(getattr(map_spec.metadata, field_name)):
            errors.append(f"Metadata {field_name} must be single-line")

    difficulty = map_spec.difficulty
    for field_name in ("hp_drain_rate", "circle_size", "overall_difficulty", "approach_rate"):
        value = getattr(difficulty, field_name)
        if not math.isfinite(value) or not 0.0 <= value <= 10.0:
            errors.append(f"Difficulty {field_name} must be finite and between 0 and 10")
    if not math.isfinite(difficulty.slider_multiplier) or difficulty.slider_multiplier <= 0.0:
        errors.append("SliderMultiplier must be finite and positive")
    if not math.isfinite(difficulty.slider_tick_rate) or difficulty.slider_tick_rate <= 0.0:
        errors.append("SliderTickRate must be finite and positive")


def _validate_event_fields(
    event: BeatEvent,
    index: int,
    policy: ValidationPolicy,
    errors: list[str],
) -> None:
    if not isinstance(event.kind, EventKind):
        errors.append(f"event {index}: unsupported object kind")
    if event.hit_sound < 0 or event.hit_sound > 15:
        errors.append(f"event {index}: hit sound must use native bit flags [0, 15]")
    if not 0 <= event.combo_skip <= 7:
        errors.append(f"event {index}: combo skip must be in [0, 7]")
    elif event.combo_skip and not event.new_combo:
        errors.append(f"event {index}: combo skip requires new_combo")
    if not _sample_is_valid(event.hit_sample, allow_custom_samples=policy.allow_custom_samples):
        errors.append(f"event {index}: invalid or unsupported custom hit sample")

    if event.kind == EventKind.SLIDER:
        if len(event.control_points) > policy.max_slider_control_points:
            errors.append(
                f"event {index}: slider has more than {policy.max_slider_control_points} control points"
            )
        if event.curve_type.upper() == "P" and len(event.control_points) != 2:
            errors.append(f"event {index}: perfect-curve slider requires exactly two control points")
        if (
            policy.require_slider_path_feasibility
            and event.curve_type.upper() in {"B", "L"}
            and event.control_points
        ):
            points = ((event.x, event.y), *event.control_points)
            polyline_length = sum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(points, points[1:])
            )
            # A Bezier or linear curve's geometric length cannot exceed its
            # control polyline. (Perfect arcs and Catmull paths need a native
            # evaluator, so this conservative bound intentionally does not
            # overclaim validity for them.) Rejecting a requested travel
            # distance beyond this upper bound prevents a sampler from
            # declaring a duration its generated simple path cannot provide.
            if event.slider_length > polyline_length + policy.slider_path_tolerance_px:
                errors.append(
                    f"event {index}: slider length exceeds its control-path feasibility bound"
                )
        edge_count = event.slider_repeats + 1
        if event.slider_edge_sounds and len(event.slider_edge_sounds) != edge_count:
            errors.append(f"event {index}: slider edge-sound count must equal repeats + 1")
        if event.slider_edge_sets and len(event.slider_edge_sets) != edge_count:
            errors.append(f"event {index}: slider edge-set count must equal repeats + 1")
        for edge_index, sound in enumerate(event.slider_edge_sounds):
            if sound < 0 or sound > 15:
                errors.append(f"event {index}: slider edge sound {edge_index} is invalid")
        for edge_index, (normal_set, addition_set) in enumerate(event.slider_edge_sets):
            if normal_set not in NATIVE_SAMPLE_SET_IDS or addition_set not in NATIVE_SAMPLE_SET_IDS:
                errors.append(f"event {index}: slider edge set {edge_index} is invalid")
    elif event.kind == EventKind.SPINNER and (
        abs(event.x - PLAYFIELD_WIDTH / 2.0) > 1e-6
        or abs(event.y - PLAYFIELD_HEIGHT / 2.0) > 1e-6
    ):
        errors.append(f"event {index}: spinners must use the canonical playfield center")


def _validate_intervals(
    events: tuple[BeatEvent, ...],
    timing_map: TimingMap,
    difficulty: DifficultySettings,
    policy: ValidationPolicy,
    errors: list[str],
) -> None:
    active: list[tuple[float, EventKind, int]] = []
    previous_start: Optional[float] = None
    previous_serialized_start: Optional[int] = None
    for index, event in enumerate(events):
        start = timing_map.beat_to_time(event.beat_position)
        if not math.isfinite(start):
            # ``validate_events`` already reports the malformed event; avoid
            # turning an audit into an exception while checking later events.
            continue
        serialized_start = int(round(start))
        if previous_start is not None:
            gap = start - previous_start
            if gap < 0.0:
                errors.append(f"event {index}: event order is not monotonic in time")
            elif gap < policy.min_start_gap_ms:
                errors.append(
                    f"event {index}: start gap {gap:.3f} ms is below {policy.min_start_gap_ms:.3f} ms"
                )
        if previous_serialized_start == serialized_start:
            errors.append(f"event {index}: shares serialized timestamp {serialized_start} ms")
        previous_start = start
        previous_serialized_start = serialized_start

        remaining: list[tuple[float, EventKind, int]] = []
        for active_end, active_kind, active_index in active:
            if active_end > start + 1e-6:
                if active_kind == EventKind.SPINNER and policy.reject_spinner_overlap:
                    errors.append(
                        f"event {index}: starts during spinner {active_index} occupancy"
                    )
                elif active_kind == EventKind.SLIDER and policy.reject_slider_overlap:
                    errors.append(
                        f"event {index}: starts during slider {active_index} occupancy"
                    )
                remaining.append((active_end, active_kind, active_index))
        active = remaining

        try:
            end = _event_end_time_ms(event, timing_map, difficulty.slider_multiplier)
        except ValueError:
            # The base event validator reports invalid slider fields; keeping
            # validation non-throwing makes it useful in corpus audits.
            end = None
        if event.kind == EventKind.SLIDER and end is not None and event.end_beat is not None:
            declared_end = timing_map.beat_to_time(event.end_beat)
            if abs(declared_end - end) > policy.slider_end_tolerance_ms:
                errors.append(
                    f"event {index}: slider end beat disagrees with native length/SV duration"
                )
        if end is not None and end > start + 1e-6:
            active.append((end, event.kind, index))


def validate_map_spec(
    map_spec: MapSpec,
    policy: ValidationPolicy = ValidationPolicy(),
) -> ValidationReport:
    """Run the strict, interval-aware output gate for a generated draft."""

    errors: list[str] = []
    warnings: list[str] = []
    _validate_settings(map_spec, errors)
    _validate_timing_points(map_spec.timing_points, errors, warnings)

    if policy.require_nonempty_events and not map_spec.events:
        errors.append("map has no hit objects")
    if policy.require_sorted_events:
        expected = tuple(
            sorted(
                map_spec.events,
                key=lambda item: (item.beat_position, item.source_object_index),
            )
        )
        if expected != map_spec.events:
            errors.append("events must be sorted by beat position and source order")

    try:
        timing_map = map_spec.timing_map
    except (TypeError, ValueError, OverflowError) as error:
        errors.append(f"cannot construct timing map: {error}")
        return ValidationReport(tuple(errors), tuple(warnings))

    try:
        base_report = validate_events(map_spec.events, timing_map)
    except (TypeError, ValueError, OverflowError) as error:
        errors.append(f"cannot validate event clock conversion: {error}")
        return ValidationReport(tuple(errors), tuple(warnings))
    errors.extend(base_report.errors)
    warnings.extend(base_report.warnings)
    for index, event in enumerate(map_spec.events):
        _validate_event_fields(event, index, policy, errors)
        if policy.require_beat_snap and math.isfinite(event.beat_position):
            residual = _nearest_snap_residual(event.beat_position, policy.snap_denominators)
            if residual > policy.snap_tolerance_beats:
                errors.append(
                    f"event {index}: beat position is {residual:.6f} beats from allowed snap"
                )

    if map_spec.events:
        _validate_intervals(map_spec.events, timing_map, map_spec.difficulty, policy, errors)
    return ValidationReport(tuple(errors), tuple(warnings))


def compile_osu(
    map_spec: MapSpec,
    *,
    validate: bool = True,
    policy: ValidationPolicy = ValidationPolicy(),
) -> str:
    """Compile a canonical ``MapSpec`` into a complete editor-ready `.osu` file.

    The compiler is deliberately deterministic. It emits no custom assets,
    backgrounds, storyboards, or hidden inferred defaults; all generated
    gameplay semantics come from the map spec itself.
    """

    if validate:
        report = validate_map_spec(map_spec, policy)
        if not report.valid:
            joined = "; ".join(report.errors)
            raise ValueError(f"refusing to compile invalid MapSpec: {joined}")

    general = map_spec.general
    metadata = map_spec.metadata
    difficulty = map_spec.difficulty
    timing_map = map_spec.timing_map
    lines = [
        f"osu file format v{int(map_spec.format_version)}",
        "",
        "[General]",
        f"AudioFilename: {general.audio_filename}",
        f"AudioLeadIn: {int(general.audio_lead_in)}",
        f"PreviewTime: {int(general.preview_time)}",
        f"Countdown: {int(general.countdown)}",
        f"SampleSet: {general.sample_set}",
        f"StackLeniency: {_format_number(general.stack_leniency)}",
        f"Mode: {int(general.mode)}",
        "LetterboxInBreaks: 0",
        "WidescreenStoryboard: 0",
        "",
        "[Editor]",
        "DistanceSpacing: 1",
        "BeatDivisor: 4",
        "GridSize: 4",
        "TimelineZoom: 1",
        "",
        "[Metadata]",
        f"Title: {metadata.title}",
        f"TitleUnicode: {metadata.title_unicode or metadata.title}",
        f"Artist: {metadata.artist}",
        f"ArtistUnicode: {metadata.artist_unicode or metadata.artist}",
        f"Creator: {metadata.creator}",
        f"Version: {metadata.version}",
        f"Source: {metadata.source}",
        f"Tags: {metadata.tags}",
        f"BeatmapID: {int(metadata.beatmap_id)}",
        f"BeatmapSetID: {int(metadata.beatmap_set_id)}",
        "",
        "[Difficulty]",
        f"HPDrainRate: {_format_number(difficulty.hp_drain_rate)}",
        f"CircleSize: {_format_number(difficulty.circle_size)}",
        f"OverallDifficulty: {_format_number(difficulty.overall_difficulty)}",
        f"ApproachRate: {_format_number(difficulty.approach_rate)}",
        f"SliderMultiplier: {_format_number(difficulty.slider_multiplier)}",
        f"SliderTickRate: {_format_number(difficulty.slider_tick_rate)}",
        "",
        "[Events]",
        "//Background and Video events",
        "",
        "[TimingPoints]",
    ]
    lines.extend(_format_timing_point(point) for point in map_spec.timing_points)
    lines.extend(("", "[Colours]", "", "[HitObjects]"))
    lines.extend(serialize_hitobjects(map_spec.events, timing_map))
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_SNAP_DENOMINATORS",
    "DifficultySettings",
    "GeneralSettings",
    "MapMetadata",
    "MapSpec",
    "ValidationPolicy",
    "compile_osu",
    "mapspec_from_osu",
    "validate_map_spec",
]
