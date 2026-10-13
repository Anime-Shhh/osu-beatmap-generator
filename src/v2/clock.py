"""Clock-hypothesis contracts for audio-only beatmap generation.

The map planner must be trained and evaluated with both oracle and imperfect
musical clocks.  This module defines the hand-off from a beat/downbeat system
to the planner without selecting, downloading, or executing a particular beat
tracker.  It is intentionally CPU-only and serializable for cache audits.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .audio import validate_timestamps_ms
from .timing import SliderVelocitySegment, TimingMap, TimingSegment


DEFAULT_METER_NUMERATOR = 4
DEFAULT_METER_DENOMINATOR = 4


@dataclass(frozen=True)
class MeterSegment:
    """One time-signature interval expressed in continuous quarter-note beats.

    ``start_bar`` is optional for a normal, bar-aligned meter change.  It is
    required only when a source chart changes meter mid-bar, where automatic
    bar numbering would otherwise be ambiguous.  osu! timing points expose the
    numerator; the denominator defaults to 4 and stays explicit in v2 targets.
    """

    start_beat: float
    numerator: int = DEFAULT_METER_NUMERATOR
    denominator: int = DEFAULT_METER_DENOMINATOR
    start_bar: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_beat):
            raise ValueError("meter start_beat must be finite")
        if isinstance(self.numerator, bool) or not isinstance(self.numerator, int):
            raise ValueError("meter numerator must be an integer")
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int):
            raise ValueError("meter denominator must be an integer")
        if self.numerator <= 0:
            raise ValueError("meter numerator must be positive")
        if self.denominator <= 0 or self.denominator & (self.denominator - 1):
            raise ValueError("meter denominator must be a positive power of two")
        if self.start_bar is not None and (
            isinstance(self.start_bar, bool) or not isinstance(self.start_bar, int)
        ):
            raise ValueError("meter start_bar must be an integer when supplied")
        if self.start_bar is not None and self.start_bar < 0:
            raise ValueError("meter start_bar must be non-negative when supplied")

    @property
    def bar_length_beats(self) -> float:
        """Bar length in the quarter-note beat coordinate used by TimingMap."""

        return self.numerator * 4.0 / self.denominator

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "start_beat": self.start_beat,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "start_bar": self.start_bar,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeterSegment":
        return cls(
            start_beat=float(payload["start_beat"]),
            numerator=int(payload.get("numerator", DEFAULT_METER_NUMERATOR)),
            denominator=int(payload.get("denominator", DEFAULT_METER_DENOMINATOR)),
            start_bar=(None if payload.get("start_bar") is None else int(payload["start_bar"])),
        )


@dataclass(frozen=True)
class BarPosition:
    """A beat's deterministic bar position under a :class:`MeterMap`."""

    bar_index: int
    beat_in_bar: float
    bar_length_beats: float
    phase: float


class MeterMap:
    """Piecewise meter map that turns continuous beats into bar coordinates."""

    def __init__(self, segments: Iterable[MeterSegment] = ()):
        ordered = tuple(sorted(segments, key=lambda segment: segment.start_beat))
        if not ordered:
            ordered = (MeterSegment(0.0),)
        if any(
            current.start_beat <= previous.start_beat
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError("meter segments must be strictly increasing in beat position")
        if not math.isclose(ordered[0].start_beat, 0.0, abs_tol=1e-7):
            raise ValueError("the first meter segment must begin at beat zero")

        start_bars = [0 if ordered[0].start_bar is None else ordered[0].start_bar]
        if start_bars[0] != 0:
            raise ValueError("the first meter segment must begin at bar zero")
        for previous, current in zip(ordered, ordered[1:]):
            previous_start_bar = start_bars[-1]
            exact_bars = (current.start_beat - previous.start_beat) / previous.bar_length_beats
            rounded_bars = round(exact_bars)
            if current.start_bar is None:
                if not math.isclose(exact_bars, rounded_bars, abs_tol=1e-6):
                    raise ValueError(
                        "a mid-bar meter change requires an explicit start_bar"
                    )
                start_bars.append(previous_start_bar + rounded_bars)
            else:
                if current.start_bar < previous_start_bar:
                    raise ValueError("meter start_bar values must not decrease")
                start_bars.append(current.start_bar)

        self.segments = ordered
        self._start_beats = tuple(segment.start_beat for segment in ordered)
        self._start_bars = tuple(start_bars)

    def segment_index_at_beat(self, beat: float) -> int:
        return max(0, bisect_right(self._start_beats, float(beat)) - 1)

    def segment_at_beat(self, beat: float) -> MeterSegment:
        return self.segments[self.segment_index_at_beat(beat)]

    def bar_position(self, beat: float) -> BarPosition:
        """Return a stable absolute bar index and phase for a continuous beat."""

        index = self.segment_index_at_beat(beat)
        segment = self.segments[index]
        relative = float(beat) - segment.start_beat
        bar_offset = math.floor(relative / segment.bar_length_beats)
        beat_in_bar = relative - bar_offset * segment.bar_length_beats
        phase = beat_in_bar / segment.bar_length_beats
        return BarPosition(
            bar_index=self._start_bars[index] + bar_offset,
            beat_in_bar=beat_in_bar,
            bar_length_beats=segment.bar_length_beats,
            phase=phase,
        )

    def to_dicts(self) -> list[dict[str, int | float | None]]:
        """Serialize both explicit and derived bar starts for auditability."""

        return [
            {
                **segment.to_dict(),
                "resolved_start_bar": start_bar,
            }
            for segment, start_bar in zip(self.segments, self._start_bars)
        ]


@dataclass(frozen=True)
class ClockEvidence:
    """Timestamped output of an external beat/downbeat or onset system.

    It is intentionally model-agnostic: a Beat This!, madmom, or learned
    resolver adapter can convert its posteriors into these candidate timestamps
    without leaking an `.osu` timing map into audio-only inference.
    """

    audio_duration_ms: float
    beat_times_ms: tuple[float, ...] = ()
    downbeat_times_ms: tuple[float, ...] = ()
    beat_confidences: tuple[float, ...] = ()
    downbeat_confidences: tuple[float, ...] = ()
    source: str = "unknown"
    extractor_revision: str = "unknown"

    def __post_init__(self) -> None:
        if not math.isfinite(self.audio_duration_ms) or self.audio_duration_ms <= 0.0:
            raise ValueError("audio_duration_ms must be finite and positive")
        beat_times = validate_timestamps_ms(self.beat_times_ms, require_nonempty=False)
        downbeat_times = validate_timestamps_ms(self.downbeat_times_ms, require_nonempty=False)
        object.__setattr__(self, "beat_times_ms", beat_times)
        object.__setattr__(self, "downbeat_times_ms", downbeat_times)
        self._validate_confidences(self.beat_confidences, len(beat_times), "beat_confidences")
        self._validate_confidences(
            self.downbeat_confidences, len(downbeat_times), "downbeat_confidences"
        )
        object.__setattr__(
            self, "beat_confidences", tuple(float(value) for value in self.beat_confidences)
        )
        object.__setattr__(
            self,
            "downbeat_confidences",
            tuple(float(value) for value in self.downbeat_confidences),
        )
        if not self.source.strip() or not self.extractor_revision.strip():
            raise ValueError("clock evidence source and extractor_revision must be non-empty")

    @staticmethod
    def _validate_confidences(
        confidences: Sequence[float], expected_length: int, field_name: str
    ) -> None:
        if confidences and len(confidences) != expected_length:
            raise ValueError(f"{field_name} must be empty or align with its timestamps")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in confidences):
            raise ValueError(f"{field_name} values must lie in [0, 1]")


def _timing_map_dict(timing_map: TimingMap) -> dict[str, list[dict[str, float]]]:
    return {
        "segments": timing_map.as_dicts(),
        "slider_velocity_segments": timing_map.slider_velocity_dicts(),
    }


def _timing_map_from_dict(payload: Mapping[str, Any]) -> TimingMap:
    raw_segments = payload.get("segments")
    raw_velocities = payload.get("slider_velocity_segments", [])
    if not isinstance(raw_segments, list) or not isinstance(raw_velocities, list):
        raise ValueError("timing map requires segment arrays")
    if not raw_segments or any(not isinstance(segment, Mapping) for segment in raw_segments):
        raise ValueError("timing map requires non-empty object segments")
    if any(not isinstance(segment, Mapping) for segment in raw_velocities):
        raise ValueError("slider velocity segments must be objects")
    return TimingMap(
        (
            TimingSegment(
                start_ms=float(segment["start_ms"]),
                beat_length_ms=float(segment["beat_length_ms"]),
                start_beat=float(segment["start_beat"]),
            )
            for segment in raw_segments
        ),
        (
            SliderVelocitySegment(
                start_ms=float(segment["start_ms"]), multiplier=float(segment["multiplier"])
            )
            for segment in raw_velocities
        ),
    )


@dataclass(frozen=True)
class TimingHypothesis:
    """One scored musical-clock candidate passed to the event planner."""

    hypothesis_id: str
    timing_map: TimingMap
    meter_segments: tuple[MeterSegment, ...] = (MeterSegment(0.0),)
    confidence: float = 0.0
    score: float = 0.0
    source: str = "unknown"
    is_oracle: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.timing_map, TimingMap):
            raise ValueError("timing_map must be a TimingMap")
        if not self.hypothesis_id.strip() or not self.source.strip():
            raise ValueError("hypothesis_id and source must be non-empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("hypothesis confidence must lie in [0, 1]")
        if not math.isfinite(self.score):
            raise ValueError("hypothesis score must be finite")
        normalized_meters = tuple(self.meter_segments) or (MeterSegment(0.0),)
        object.__setattr__(self, "meter_segments", normalized_meters)
        # Constructing validates bar-aligned changes and all segment ordering.
        MeterMap(normalized_meters)

    @property
    def meter_map(self) -> MeterMap:
        return MeterMap(self.meter_segments)

    def beat_phase(self, time_ms: float) -> tuple[float, float]:
        """Return sin/cos of local quarter-note phase for model conditioning."""

        phase = self.timing_map.time_to_beat(time_ms) % 1.0
        angle = 2.0 * math.pi * phase
        return math.sin(angle), math.cos(angle)

    def bar_position(self, time_ms: float) -> BarPosition:
        return self.meter_map.bar_position(self.timing_map.time_to_beat(time_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "timing_map": _timing_map_dict(self.timing_map),
            "meter_segments": [segment.to_dict() for segment in self.meter_segments],
            "confidence": self.confidence,
            "score": self.score,
            "source": self.source,
            "is_oracle": self.is_oracle,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimingHypothesis":
        timing_map = payload.get("timing_map")
        meter_segments = payload.get("meter_segments", [])
        if not isinstance(timing_map, Mapping) or not isinstance(meter_segments, list):
            raise ValueError("timing hypothesis requires timing_map and meter_segments")
        if any(not isinstance(segment, Mapping) for segment in meter_segments):
            raise ValueError("every meter segment must be an object")
        return cls(
            hypothesis_id=str(payload["hypothesis_id"]),
            timing_map=_timing_map_from_dict(timing_map),
            meter_segments=tuple(
                MeterSegment.from_dict(segment)
                for segment in meter_segments
            ),
            confidence=float(payload.get("confidence", 0.0)),
            score=float(payload.get("score", 0.0)),
            source=str(payload.get("source", "unknown")),
            is_oracle=bool(payload.get("is_oracle", False)),
        )


@runtime_checkable
class ClockResolver(Protocol):
    """Adapter boundary for a learned or signal-processing clock resolver."""

    def resolve(
        self, evidence: ClockEvidence, *, max_hypotheses: int = 4
    ) -> tuple[TimingHypothesis, ...]:
        """Return ordered candidates without reading `.osu` metadata at inference."""


class StaticClockResolver:
    """Deterministic resolver useful for oracle, fixture, and ablation paths."""

    def __init__(self, hypotheses: Iterable[TimingHypothesis]):
        candidates = tuple(hypotheses)
        if not candidates:
            raise ValueError("StaticClockResolver requires at least one hypothesis")
        if len({hypothesis.hypothesis_id for hypothesis in candidates}) != len(candidates):
            raise ValueError("StaticClockResolver hypothesis IDs must be unique")
        self._hypotheses = tuple(
            sorted(
                candidates,
                key=lambda hypothesis: (
                    -hypothesis.score,
                    -hypothesis.confidence,
                    hypothesis.hypothesis_id,
                ),
            )
        )

    def resolve(
        self, evidence: ClockEvidence, *, max_hypotheses: int = 4
    ) -> tuple[TimingHypothesis, ...]:
        if max_hypotheses <= 0:
            raise ValueError("max_hypotheses must be positive")
        return self._hypotheses[:max_hypotheses]


class AudioClockResolver:
    """Dynamic clock resolver estimating hypotheses from audio clock evidence or presets."""

    def __init__(self, hypotheses: Iterable[TimingHypothesis] | None = None):
        self._preset_hypotheses = tuple(hypotheses) if hypotheses else ()

    def resolve(
        self, evidence: ClockEvidence, *, max_hypotheses: int = 4
    ) -> tuple[TimingHypothesis, ...]:
        if max_hypotheses <= 0:
            raise ValueError("max_hypotheses must be positive")
        if self._preset_hypotheses:
            return self._preset_hypotheses[:max_hypotheses]

        beats = list(evidence.beat_times_ms)
        if not beats:
            return (
                TimingHypothesis(
                    hypothesis_id="fallback-120bpm",
                    timing_map=TimingMap((TimingSegment(start_ms=0.0, beat_length_ms=500.0, start_beat=0.0),)),
                    meter_segments=(MeterSegment(0.0, 4, 4),),
                    confidence=0.5,
                    score=0.5,
                    source="audio_clock_fallback",
                ),
            )[:max_hypotheses]

        if len(beats) >= 2:
            intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
            intervals.sort()
            median_interval = intervals[len(intervals) // 2]
        else:
            median_interval = 500.0

        offset_ms = float(beats[0])
        primary = TimingHypothesis(
            hypothesis_id=f"evidence-{round(60000.0 / median_interval, 1)}bpm-{round(offset_ms, 1)}ms",
            timing_map=TimingMap(
                (TimingSegment(start_ms=round(offset_ms, 2), beat_length_ms=round(median_interval, 3), start_beat=0.0),)
            ),
            meter_segments=(MeterSegment(0.0, 4, 4),),
            confidence=0.9,
            score=1.0,
            source=evidence.source,
        )
        return (primary,)[:max_hypotheses]


class OracleClockResolver(StaticClockResolver):
    """Explicit training-only resolver; never register it for audio inference."""

    def __init__(self, hypothesis: TimingHypothesis):
        if not hypothesis.is_oracle:
            hypothesis = TimingHypothesis(
                hypothesis_id=hypothesis.hypothesis_id,
                timing_map=hypothesis.timing_map,
                meter_segments=hypothesis.meter_segments,
                confidence=hypothesis.confidence,
                score=hypothesis.score,
                source=hypothesis.source,
                is_oracle=True,
            )
        super().__init__((hypothesis,))


def select_timing_hypothesis(
    hypotheses: Iterable[TimingHypothesis],
) -> TimingHypothesis:
    """Select the deterministic top candidate after a validator/reranker score."""

    candidates = tuple(hypotheses)
    if not candidates:
        raise ValueError("cannot select a timing hypothesis from an empty sequence")
    return min(
        candidates,
        key=lambda hypothesis: (
            -hypothesis.score,
            -hypothesis.confidence,
            hypothesis.hypothesis_id,
        ),
    )


def jitter_timing_hypothesis(
    hypothesis: TimingHypothesis,
    *,
    offset_ms: float,
    time_scale: float,
    confidence: float | None = None,
    source: str = "jittered_oracle",
) -> TimingHypothesis:
    """Create a coherent imperfect clock for robust planner training.

    ``time_scale`` multiplies all beat lengths: values above one make the
    hypothesis slower, values below one faster.  Timing changes and inherited
    SV timestamps are transformed together, while their beat coordinates and
    meter structure stay intact.  This is deliberately an affine perturbation;
    a learned resolver supplies non-affine hypotheses separately.
    """

    if not math.isfinite(offset_ms):
        raise ValueError("offset_ms must be finite")
    if not math.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError("time_scale must be finite and positive")
    anchor_ms = hypothesis.timing_map.segments[0].start_ms

    def transform_time(value: float) -> float:
        return anchor_ms + (value - anchor_ms) * time_scale + offset_ms

    timing_map = TimingMap(
        (
            TimingSegment(
                start_ms=transform_time(segment.start_ms),
                beat_length_ms=segment.beat_length_ms * time_scale,
                start_beat=segment.start_beat,
            )
            for segment in hypothesis.timing_map.segments
        ),
        (
            SliderVelocitySegment(
                start_ms=transform_time(segment.start_ms), multiplier=segment.multiplier
            )
            for segment in hypothesis.timing_map.slider_velocity_segments
        ),
    )
    identity_payload = json.dumps(
        {
            "parent": hypothesis.hypothesis_id,
            "offset_ms": offset_ms,
            "time_scale": time_scale,
            "source": source,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hypothesis_id = "jitter:" + hashlib.sha256(identity_payload).hexdigest()[:20]
    return TimingHypothesis(
        hypothesis_id=hypothesis_id,
        timing_map=timing_map,
        meter_segments=hypothesis.meter_segments,
        confidence=hypothesis.confidence if confidence is None else confidence,
        score=hypothesis.score,
        source=source,
        is_oracle=False,
    )
