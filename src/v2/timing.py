"""Lossless-enough tempo-map utilities for v2 targets.

The legacy path reduces a map to its first BPM.  That is not safe for maps
with red timing-point changes: beat-relative targets then acquire systematic
timing errors.  ``TimingMap`` retains every uninherited timing point and maps
between milliseconds and a continuous beat coordinate.

Inherited timing points (slider velocity changes) are deliberately not folded
into beat time. They belong to slider semantics and must be preserved as a
separate target rather than being mistaken for a BPM change. Native timing
records also retain meter, Kiai effects, sample state, and volume so a model
can compile an editor-ready output without fabricating those fields.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Optional


DEFAULT_BEAT_LENGTH_MS = 500.0
DEFAULT_METER = 4
TIMING_EFFECT_KIAI = 1
TIMING_EFFECT_OMIT_FIRST_BAR_LINE = 8


@dataclass(frozen=True)
class TimingPoint:
    """One native ``[TimingPoints]`` record.

    Red (uninherited) points own the beat clock and meter. Inherited points
    own slider velocity. The remaining fields are retained because collapsing
    them loses Kiai, meter, sample, and volume semantics needed to compile an
    editor-ready map.
    """

    offset_ms: float
    beat_length_ms: float
    meter: int = DEFAULT_METER
    sample_set: int = 0
    sample_index: int = 0
    volume: int = 100
    uninherited: bool = True
    effects: int = 0

    @property
    def start_ms(self) -> float:
        return self.offset_ms

    @property
    def is_kiai(self) -> bool:
        return bool(self.effects & TIMING_EFFECT_KIAI)

    @property
    def omit_first_bar_line(self) -> bool:
        return bool(self.effects & TIMING_EFFECT_OMIT_FIRST_BAR_LINE)

    @property
    def slider_velocity_multiplier(self) -> Optional[float]:
        if self.uninherited or self.beat_length_ms >= 0.0:
            return None
        return -100.0 / self.beat_length_ms


@dataclass(frozen=True)
class TimingSegment:
    """One constant-tempo interval in a continuous beat coordinate system."""

    start_ms: float
    beat_length_ms: float
    start_beat: float
    meter: int = DEFAULT_METER

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class SliderVelocitySegment:
    """An inherited timing-point multiplier that affects slider duration."""

    start_ms: float
    multiplier: float


@dataclass(frozen=True)
class MeterSegment:
    """The active numerator from a red timing point onward."""

    start_ms: float
    meter: int


@dataclass(frozen=True)
class KiaiSegment:
    """A transition in Kiai state, derived from native timing effects."""

    start_ms: float
    enabled: bool


class TimingMap:
    """Piecewise-linear conversion between map time and musical beat time."""

    def __init__(
        self,
        segments: Iterable[TimingSegment],
        slider_velocity_segments: Iterable[SliderVelocitySegment] = (),
        *,
        timing_points: Iterable[TimingPoint] = (),
    ):
        ordered = tuple(sorted(segments, key=lambda item: item.start_ms))
        if not ordered:
            ordered = (TimingSegment(0.0, DEFAULT_BEAT_LENGTH_MS, 0.0),)
        if any(
            not (
                math.isfinite(segment.start_ms)
                and math.isfinite(segment.beat_length_ms)
                and math.isfinite(segment.start_beat)
            )
            for segment in ordered
        ):
            raise ValueError("Timing segments must be finite")
        if any(segment.beat_length_ms <= 0 for segment in ordered):
            raise ValueError("Timing segments must have a positive beat length")
        if any(segment.meter < 1 for segment in ordered):
            raise ValueError("Timing segment meters must be positive")
        if any(
            current.start_ms <= previous.start_ms
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError("Timing segments must be strictly increasing in time")
        if any(
            current.start_beat <= previous.start_beat
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError("Timing segments must be strictly increasing in beat position")

        supplied_velocity_segments = tuple(slider_velocity_segments)
        velocity_ordered = tuple(
            sorted(supplied_velocity_segments, key=lambda item: item.start_ms)
        )
        if not velocity_ordered:
            velocity_ordered = (SliderVelocitySegment(0.0, 1.0),)
        if any(
            not (math.isfinite(segment.start_ms) and math.isfinite(segment.multiplier))
            for segment in velocity_ordered
        ):
            raise ValueError("Slider velocity segments must be finite")
        if any(segment.multiplier <= 0 for segment in velocity_ordered):
            raise ValueError("Slider velocity multipliers must be positive")
        if any(
            current.start_ms <= previous.start_ms
            for previous, current in zip(velocity_ordered, velocity_ordered[1:])
        ):
            raise ValueError("Slider velocity segments must be strictly increasing in time")
        source_points = tuple(timing_points)
        if source_points:
            # Python's sort is stable, so the last record at a timestamp keeps
            # osu!'s final-point precedence.
            source_points = tuple(
                sorted(source_points, key=lambda point: point.offset_ms)
            )
        else:
            generated_points = [
                TimingPoint(
                    offset_ms=segment.start_ms,
                    beat_length_ms=segment.beat_length_ms,
                    meter=segment.meter,
                    uninherited=True,
                )
                for segment in ordered
            ]
            # A negative-infinity default SV segment is a query sentinel, not
            # an actual source timing point.
            for segment in velocity_ordered if supplied_velocity_segments else ():
                if segment.start_ms <= -1.0e11:
                    continue
                generated_points.append(
                    TimingPoint(
                        offset_ms=segment.start_ms,
                        beat_length_ms=-100.0 / segment.multiplier,
                        uninherited=False,
                    )
                )
            source_points = tuple(
                sorted(generated_points, key=lambda point: point.offset_ms)
            )

        self.segments = ordered
        self._start_times = tuple(segment.start_ms for segment in ordered)
        self._start_beats = tuple(segment.start_beat for segment in ordered)
        self.slider_velocity_segments = velocity_ordered
        self._slider_velocity_times = tuple(segment.start_ms for segment in velocity_ordered)
        self.timing_points = source_points
        self._timing_point_times = tuple(point.offset_ms for point in source_points)

    def segment_at_time(self, time_ms: float) -> TimingSegment:
        index = max(0, bisect_right(self._start_times, float(time_ms)) - 1)
        return self.segments[index]

    def segment_at_beat(self, beat: float) -> TimingSegment:
        index = max(0, bisect_right(self._start_beats, float(beat)) - 1)
        return self.segments[index]

    def timing_point_at_time(self, time_ms: float) -> Optional[TimingPoint]:
        """Return the final native timing point active at ``time_ms``."""

        if not self.timing_points:
            return None
        index = bisect_right(self._timing_point_times, float(time_ms)) - 1
        return self.timing_points[index] if index >= 0 else None

    def time_to_beat(self, time_ms: float) -> float:
        segment = self.segment_at_time(time_ms)
        return segment.start_beat + (float(time_ms) - segment.start_ms) / segment.beat_length_ms

    def beat_to_time(self, beat: float) -> float:
        segment = self.segment_at_beat(beat)
        return segment.start_ms + (float(beat) - segment.start_beat) * segment.beat_length_ms

    def beat_length_at_time(self, time_ms: float) -> float:
        return self.segment_at_time(time_ms).beat_length_ms

    def meter_at_time(self, time_ms: float) -> int:
        """Return the active red-point time-signature numerator."""

        return self.segment_at_time(time_ms).meter

    def slider_velocity_at_time(self, time_ms: float) -> float:
        index = max(0, bisect_right(self._slider_velocity_times, float(time_ms)) - 1)
        return self.slider_velocity_segments[index].multiplier

    def kiai_at_time(self, time_ms: float) -> bool:
        """Return Kiai state from the final active native timing point."""

        point = self.timing_point_at_time(time_ms)
        return point.is_kiai if point is not None else False

    def meter_segments(self) -> tuple[MeterSegment, ...]:
        return tuple(MeterSegment(segment.start_ms, segment.meter) for segment in self.segments)

    def kiai_segments(self) -> tuple[KiaiSegment, ...]:
        """Return Kiai state transitions rather than redundant point records."""

        transitions: list[KiaiSegment] = []
        previous_state = False
        index = 0
        while index < len(self.timing_points):
            point = self.timing_points[index]
            end = index + 1
            while (
                end < len(self.timing_points)
                and self.timing_points[end].offset_ms == point.offset_ms
            ):
                end += 1
            state = self.timing_points[end - 1].is_kiai
            if state != previous_state:
                transitions.append(KiaiSegment(point.offset_ms, state))
                previous_state = state
            index = end
        return tuple(transitions)

    def slider_duration_ms(
        self,
        start_ms: float,
        pixel_length: float,
        repeats: int,
        slider_multiplier: float,
    ) -> float:
        """Compute native slider duration from its initial timing/SV settings.

        osu! slider duration uses the red timing point and inherited slider
        velocity active at slider start.  Persisting this derivation is far more
        faithful than the legacy model's hard-coded 100-pixel, one-repeat
        reconstruction.
        """

        if pixel_length <= 0 or repeats <= 0 or slider_multiplier <= 0:
            raise ValueError("slider pixel length, repeats, and multiplier must be positive")
        beat_length = self.beat_length_at_time(start_ms)
        velocity = self.slider_velocity_at_time(start_ms)
        span_duration = pixel_length / (slider_multiplier * 100.0 * velocity) * beat_length
        return span_duration * repeats

    def as_dicts(self) -> list[dict[str, float | int]]:
        return [
            {
                "start_ms": segment.start_ms,
                "beat_length_ms": segment.beat_length_ms,
                "start_beat": segment.start_beat,
                "bpm": segment.bpm,
                "meter": segment.meter,
            }
            for segment in self.segments
        ]

    def slider_velocity_dicts(self) -> list[dict[str, float]]:
        return [
            {"start_ms": segment.start_ms, "multiplier": segment.multiplier}
            for segment in self.slider_velocity_segments
        ]

    def timing_point_dicts(self) -> list[dict[str, float | int | bool]]:
        """Structured native timing records for cache manifests and audits."""

        return [
            {
                "offset_ms": point.offset_ms,
                "beat_length_ms": point.beat_length_ms,
                "meter": point.meter,
                "sample_set": point.sample_set,
                "sample_index": point.sample_index,
                "volume": point.volume,
                "uninherited": point.uninherited,
                "effects": point.effects,
                "kiai": point.is_kiai,
            }
            for point in self.timing_points
        ]


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def parse_timing_points(osu_content: str) -> tuple[TimingPoint, ...]:
    """Parse native timing records, including meter, SV, and Kiai effects.

    Malformed lines are skipped. Semantic validation belongs to ``MapSpec`` so
    a corpus audit can report source defects instead of crashing while parsing
    a large dataset.
    """

    in_timing_points = False
    points: list[TimingPoint] = []
    for raw_line in osu_content.splitlines():
        line = raw_line.strip()
        if line == "[TimingPoints]":
            in_timing_points = True
            continue
        if not in_timing_points:
            continue
        if line.startswith("["):
            break
        if not line or line.startswith("//"):
            continue

        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            offset_ms = float(fields[0])
            beat_length_ms = float(fields[1])
        except ValueError:
            continue
        # Older formats omit the seventh column. A positive beat length is the
        # only defensible fallback for red/inherited classification.
        uninherited = (
            _parse_int(fields[6], 1) == 1
            if len(fields) >= 7
            else beat_length_ms > 0.0
        )
        points.append(
            TimingPoint(
                offset_ms=offset_ms,
                beat_length_ms=beat_length_ms,
                meter=_parse_int(fields[2], DEFAULT_METER)
                if len(fields) >= 3
                else DEFAULT_METER,
                sample_set=_parse_int(fields[3], 0) if len(fields) >= 4 else 0,
                sample_index=_parse_int(fields[4], 0) if len(fields) >= 5 else 0,
                volume=_parse_int(fields[5], 100) if len(fields) >= 6 else 100,
                uninherited=uninherited,
                effects=_parse_int(fields[7], 0) if len(fields) >= 8 else 0,
            )
        )
    # Stable ordering retains final-line precedence at duplicate timestamps.
    return tuple(sorted(points, key=lambda point: point.offset_ms))


def _timing_point_records(osu_content: str) -> list[tuple[float, float, bool]]:
    """Backward-compatible compact view used by older v2 callers."""

    return [
        (point.offset_ms, point.beat_length_ms, point.uninherited)
        for point in parse_timing_points(osu_content)
    ]


def _last_point_per_time(points: Iterable[TimingPoint]) -> list[TimingPoint]:
    """osu! applies the final point of one type at any shared timestamp."""

    by_time: dict[float, TimingPoint] = {}
    for point in points:
        by_time[point.offset_ms] = point
    return sorted(by_time.values(), key=lambda point: point.offset_ms)


def timing_map_from_points(points: Iterable[TimingPoint]) -> TimingMap:
    """Build the clock view from canonical native timing points."""

    points = tuple(sorted(points, key=lambda point: point.offset_ms))
    red_points = _last_point_per_time(
        point
        for point in points
        if point.uninherited and point.beat_length_ms > 0.0
    )
    inherited_points = _last_point_per_time(
        point
        for point in points
        if not point.uninherited and point.beat_length_ms < 0.0
    )
    if not red_points:
        return TimingMap(
            (TimingSegment(0.0, DEFAULT_BEAT_LENGTH_MS, 0.0),),
            timing_points=points,
        )

    first = red_points[0]
    segments = [
        TimingSegment(first.offset_ms, first.beat_length_ms, 0.0, meter=first.meter)
    ]
    for point in red_points[1:]:
        previous = segments[-1]
        start_beat = previous.start_beat + (
            (point.offset_ms - previous.start_ms) / previous.beat_length_ms
        )
        segments.append(
            TimingSegment(point.offset_ms, point.beat_length_ms, start_beat, meter=point.meter)
        )
    # A sentinel keeps the default velocity active before the first inherited
    # point without fabricating a source timing point at t=0.
    velocities = [SliderVelocitySegment(-1.0e12, 1.0)]
    for point in inherited_points:
        multiplier = point.slider_velocity_multiplier
        if multiplier is not None and multiplier > 0.0:
            velocities.append(SliderVelocitySegment(point.offset_ms, multiplier))
    return TimingMap(segments, velocities, timing_points=points)


def parse_timing_map(osu_content: str) -> TimingMap:
    """Parse all timing semantics from an ``.osu`` document.

    Beat zero is anchored at the first red timing point. Events before that
    point retain negative beat positions rather than inheriting a fabricated
    first BPM. The complete native timing-point sequence remains available on
    the returned map for source integrity and compilation.
    """

    return timing_map_from_points(parse_timing_points(osu_content))
