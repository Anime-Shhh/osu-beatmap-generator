"""Structured osu!standard events and beat-grid targets for architecture v2.

Unlike the legacy tokenizer and dense signal representation, this module keeps
the fields that determine whether a slider or spinner is playable: all red
timing changes are handled by :class:`TimingMap`; slider repeat count, pixel
length, curve type, and anchors are retained; and output validation is explicit.
It intentionally does not claim to be a complete osu! file parser.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Optional

try:
    import torch
except ModuleNotFoundError:  # Enables parser/split audits outside the train env.
    torch = None  # type: ignore[assignment]

from .timing import TimingMap, parse_timing_map


PLAYFIELD_WIDTH = 512.0
PLAYFIELD_HEIGHT = 384.0
IGNORE_INDEX = -100


class EventKind(IntEnum):
    CIRCLE = 0
    SLIDER = 1
    SPINNER = 2


@dataclass(frozen=True)
class HitSample:
    """Native per-object sample fields, with no dependency on custom assets."""

    normal_set: int = 0
    addition_set: int = 0
    custom_index: int = 0
    volume: int = 0
    filename: str = ""

    def as_osu_field(self) -> str:
        return (
            f"{self.normal_set}:{self.addition_set}:{self.custom_index}:"
            f"{self.volume}:{self.filename}"
        )

    @property
    def is_default(self) -> bool:
        return (
            self.normal_set == 0
            and self.addition_set == 0
            and self.custom_index == 0
            and self.volume == 0
            and not self.filename
        )


@dataclass(frozen=True)
class BeatEvent:
    """A hit object expressed in continuous musical time.

    ``beat_position`` is continuous so a model can predict a coarse grid slot
    and a bounded microtiming residual without forcing all outputs to snap.
    ``end_beat`` is used by spinners and records the derived slider endpoint
    during training.  Slider serialization remains native (repeat count,
    pixel length, and timing/SV), rather than inventing a fixed duration.
    """

    kind: EventKind
    beat_position: float
    x: float
    y: float
    end_beat: Optional[float] = None
    curve_type: str = "B"
    control_points: tuple[tuple[float, float], ...] = ()
    slider_repeats: int = 1
    slider_length: float = 0.0
    hit_sound: int = 0
    new_combo: bool = False
    combo_skip: int = 0
    hit_sample: HitSample = field(default_factory=HitSample)
    slider_edge_sounds: tuple[int, ...] = ()
    slider_edge_sets: tuple[tuple[int, int], ...] = ()
    source_object_index: int = -1


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class BeatGridTargets:
    """Dense, *bounded* view of sparse events for a beat-grid planner.

    Shapes use ``[slots, lanes]`` rather than silently overwriting two objects
    that quantize to the same beat slot.  The planner's lane count is therefore
    a declared modeling choice and overfull slots fail preprocessing loudly.
    """

    slot_origin: int
    subdivisions_per_beat: int
    occupancy: torch.Tensor
    kind: torch.Tensor
    time_residual_beats: torch.Tensor
    coordinates: torch.Tensor
    end_beat_delta: torch.Tensor
    slider_length: torch.Tensor
    slider_repeats: torch.Tensor

    @property
    def num_slots(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def lanes(self) -> int:
        return int(self.occupancy.shape[1])


def _iter_hitobject_lines(osu_content: str) -> Iterable[str]:
    in_hitobjects = False
    for raw_line in osu_content.splitlines():
        line = raw_line.strip()
        if line == "[HitObjects]":
            in_hitobjects = True
            continue
        if not in_hitobjects:
            continue
        if line.startswith("["):
            return
        if line and not line.startswith("//"):
            yield line


def _kind_from_type_bits(type_bits: int) -> Optional[EventKind]:
    if type_bits & 1:
        return EventKind.CIRCLE
    if type_bits & 2:
        return EventKind.SLIDER
    if type_bits & 8:
        return EventKind.SPINNER
    return None


def _parse_hit_sample(value: str) -> HitSample:
    """Parse osu!'s ``normal:addition:index:volume:filename`` field."""

    fields = value.split(":", maxsplit=4)

    def as_int(index: int) -> int:
        if index >= len(fields) or not fields[index].strip():
            return 0
        try:
            return int(fields[index])
        except ValueError:
            return 0

    return HitSample(
        normal_set=as_int(0),
        addition_set=as_int(1),
        custom_index=as_int(2),
        volume=as_int(3),
        filename=fields[4] if len(fields) >= 5 else "",
    )


def _parse_edge_sets(value: str) -> tuple[tuple[int, int], ...]:
    edge_sets: list[tuple[int, int]] = []
    for item in value.split("|"):
        fields = item.split(":", maxsplit=1)
        try:
            normal_set = int(fields[0]) if fields[0] else 0
            addition_set = int(fields[1]) if len(fields) > 1 and fields[1] else 0
        except ValueError:
            normal_set, addition_set = 0, 0
        edge_sets.append((normal_set, addition_set))
    return tuple(edge_sets)


def _parse_edge_sounds(value: str) -> tuple[int, ...]:
    sounds: list[int] = []
    for item in value.split("|"):
        try:
            sounds.append(int(item) if item else 0)
        except ValueError:
            sounds.append(0)
    return tuple(sounds)


def _parse_hitobjects(osu_content: str) -> list[dict]:
    """Parse the target fields required for faithful v2 event targets."""

    objects: list[dict] = []
    for object_index, line in enumerate(_iter_hitobject_lines(osu_content)):
        fields = line.split(",")
        if len(fields) < 5:
            continue
        try:
            x = float(fields[0])
            y = float(fields[1])
            time_ms = float(fields[2])
            type_bits = int(fields[3])
            hit_sound = int(fields[4])
        except ValueError:
            continue

        kind = _kind_from_type_bits(type_bits)
        if kind is None:
            continue
        obj: dict = {
            "kind": kind,
            "x": x,
            "y": y,
            "time_ms": time_ms,
            "hit_sound": hit_sound,
            "new_combo": bool(type_bits & 4),
            "combo_skip": (type_bits >> 4) & 0x7,
            "source_object_index": object_index,
        }

        if kind == EventKind.SLIDER:
            if len(fields) < 8:
                continue
            curve_parts = fields[5].split("|")
            curve_type = curve_parts[0].upper() if curve_parts else "B"
            control_points: list[tuple[float, float]] = []
            for raw_point in curve_parts[1:]:
                try:
                    point_x, point_y = raw_point.split(":", maxsplit=1)
                    control_points.append((float(point_x), float(point_y)))
                except ValueError:
                    continue
            try:
                repeats = max(1, int(fields[6]))
                length = float(fields[7])
            except ValueError:
                continue
            obj.update(
                curve_type=curve_type,
                control_points=tuple(control_points),
                slider_repeats=repeats,
                slider_length=length,
            )
            if len(fields) >= 9 and fields[8]:
                obj["slider_edge_sounds"] = _parse_edge_sounds(fields[8])
            if len(fields) >= 10 and fields[9]:
                obj["slider_edge_sets"] = _parse_edge_sets(fields[9])
            if len(fields) >= 11:
                obj["hit_sample"] = _parse_hit_sample(fields[10])
        elif kind == EventKind.SPINNER:
            if len(fields) < 6:
                continue
            try:
                obj["end_time_ms"] = float(fields[5])
            except ValueError:
                continue

            if len(fields) >= 7:
                obj["hit_sample"] = _parse_hit_sample(fields[6])
        elif len(fields) >= 6:
            obj["hit_sample"] = _parse_hit_sample(fields[5])

        objects.append(obj)
    return objects


def _parse_slider_multiplier(osu_content: str) -> float:
    for raw_line in osu_content.splitlines():
        line = raw_line.strip()
        if line.startswith("SliderMultiplier:"):
            try:
                value = float(line.split(":", maxsplit=1)[1])
                if value > 0:
                    return value
            except ValueError:
                pass
    return 1.4


def events_from_osu(osu_content: str, timing_map: Optional[TimingMap] = None) -> list[BeatEvent]:
    """Convert a standard-mode ``.osu`` document to structured beat events."""

    timing_map = timing_map or parse_timing_map(osu_content)
    slider_multiplier = _parse_slider_multiplier(osu_content)
    events: list[BeatEvent] = []
    for obj in _parse_hitobjects(osu_content):
        time_ms = float(obj["time_ms"])
        end_beat = None
        if obj["kind"] == EventKind.SPINNER:
            end_beat = timing_map.time_to_beat(float(obj["end_time_ms"]))
        elif obj["kind"] == EventKind.SLIDER:
            slider_end_ms = time_ms + timing_map.slider_duration_ms(
                time_ms,
                float(obj["slider_length"]),
                int(obj["slider_repeats"]),
                slider_multiplier,
            )
            end_beat = timing_map.time_to_beat(slider_end_ms)
        events.append(
            BeatEvent(
                kind=obj["kind"],
                beat_position=timing_map.time_to_beat(time_ms),
                x=float(obj["x"]),
                y=float(obj["y"]),
                end_beat=end_beat,
                curve_type=str(obj.get("curve_type", "B")),
                control_points=tuple(obj.get("control_points", ())),
                slider_repeats=int(obj.get("slider_repeats", 1)),
                slider_length=float(obj.get("slider_length", 0.0)),
                hit_sound=int(obj.get("hit_sound", 0)),
                new_combo=bool(obj.get("new_combo", False)),
                combo_skip=int(obj.get("combo_skip", 0)),
                hit_sample=obj.get("hit_sample", HitSample()),
                slider_edge_sounds=tuple(obj.get("slider_edge_sounds", ())),
                slider_edge_sets=tuple(obj.get("slider_edge_sets", ())),
                source_object_index=int(obj["source_object_index"]),
            )
        )
    return sorted(events, key=lambda event: (event.beat_position, event.source_object_index))


def events_to_hitobjects(events: Iterable[BeatEvent], timing_map: TimingMap) -> list[dict]:
    """Convert beat events back to hit-object dictionaries without resnapping."""

    objects: list[dict] = []
    for event in sorted(events, key=lambda item: (item.beat_position, item.source_object_index)):
        item = {
            "time": timing_map.beat_to_time(event.beat_position),
            "type": event.kind.name.lower(),
            "x": event.x,
            "y": event.y,
            "hit_sound": event.hit_sound,
            "new_combo": event.new_combo,
            "combo_skip": event.combo_skip,
            "hit_sample": event.hit_sample,
        }
        if event.kind == EventKind.SLIDER:
            item.update(
                curve_type=event.curve_type,
                curve_points=list(event.control_points),
                slides=event.slider_repeats,
                length=event.slider_length,
                edge_sounds=list(event.slider_edge_sounds),
                edge_sets=list(event.slider_edge_sets),
            )
        elif event.kind == EventKind.SPINNER:
            item["end_time"] = timing_map.beat_to_time(
                event.end_beat if event.end_beat is not None else event.beat_position
            )
        objects.append(item)
    return objects


def serialize_hitobjects(events: Iterable[BeatEvent], timing_map: TimingMap) -> list[str]:
    """Serialize valid v2 events to the ``[HitObjects]`` portion of an .osu file."""

    lines: list[str] = []
    for event in sorted(events, key=lambda item: (item.beat_position, item.source_object_index)):
        x = int(round(event.x))
        y = int(round(event.y))
        time_ms = int(round(timing_map.beat_to_time(event.beat_position)))
        type_bits = {EventKind.CIRCLE: 1, EventKind.SLIDER: 2, EventKind.SPINNER: 8}[event.kind]
        if event.new_combo:
            type_bits |= 4
            type_bits |= (max(0, min(7, int(event.combo_skip))) << 4)
        if event.kind == EventKind.CIRCLE:
            lines.append(
                f"{x},{y},{time_ms},{type_bits},{event.hit_sound},"
                f"{event.hit_sample.as_osu_field()}"
            )
        elif event.kind == EventKind.SLIDER:
            curve = event.curve_type.upper()
            for point_x, point_y in event.control_points:
                curve += f"|{int(round(point_x))}:{int(round(point_y))}"
            line = (
                f"{x},{y},{time_ms},{type_bits},{event.hit_sound},{curve},"
                f"{event.slider_repeats},{event.slider_length:.6f}"
            )
            if event.slider_edge_sounds or event.slider_edge_sets or not event.hit_sample.is_default:
                edge_count = event.slider_repeats + 1
                edge_sounds = event.slider_edge_sounds or (0,) * edge_count
                edge_sets = event.slider_edge_sets or ((0, 0),) * edge_count
                line += "," + "|".join(str(int(sound)) for sound in edge_sounds)
                line += "," + "|".join(
                    f"{int(normal)}:{int(addition)}"
                    for normal, addition in edge_sets
                )
                line += "," + event.hit_sample.as_osu_field()
            lines.append(line)
        else:
            end_beat = event.end_beat if event.end_beat is not None else event.beat_position
            end_time_ms = int(round(timing_map.beat_to_time(end_beat)))
            lines.append(
                f"{x},{y},{time_ms},{type_bits},{event.hit_sound},{end_time_ms},"
                f"{event.hit_sample.as_osu_field()}"
            )
    return lines


def validate_events(
    events: Iterable[BeatEvent],
    timing_map: TimingMap,
    *,
    min_gap_ms: Optional[float] = None,
) -> ValidationReport:
    """Check basic serializability and safety constraints before map output.

    This is intentionally conservative but not a replacement for osu!lazer's
    parser, strain calculator, or mapper review.  In particular, equal-time
    stacked events are legal and are not rejected here.
    """

    errors: list[str] = []
    warnings: list[str] = []
    prior_time: Optional[float] = None
    for index, event in enumerate(sorted(events, key=lambda item: (item.beat_position, item.source_object_index))):
        fields = (event.beat_position, event.x, event.y)
        if not all(math.isfinite(value) for value in fields):
            errors.append(f"event {index}: non-finite beat position or coordinate")
            continue
        if not (0.0 <= event.x <= PLAYFIELD_WIDTH and 0.0 <= event.y <= PLAYFIELD_HEIGHT):
            errors.append(f"event {index}: coordinate is outside the osu! playfield")

        time_ms = timing_map.beat_to_time(event.beat_position)
        if prior_time is not None and time_ms < prior_time:
            errors.append(f"event {index}: timestamps are not monotonic")
        if prior_time is not None and min_gap_ms is not None and 0.0 < time_ms - prior_time < min_gap_ms:
            warnings.append(f"event {index}: interval below requested {min_gap_ms:.1f} ms gap")
        prior_time = time_ms

        if event.kind == EventKind.SLIDER:
            if event.curve_type.upper() not in {"B", "L", "P", "C"}:
                errors.append(f"event {index}: unsupported slider curve type {event.curve_type!r}")
            if event.slider_repeats < 1:
                errors.append(f"event {index}: slider repeats must be positive")
            if not math.isfinite(event.slider_length) or event.slider_length <= 0.0:
                errors.append(f"event {index}: slider length must be positive")
            if not event.control_points:
                errors.append(f"event {index}: slider has no endpoint/control point")
            for point_index, (point_x, point_y) in enumerate(event.control_points):
                if not (math.isfinite(point_x) and math.isfinite(point_y)):
                    errors.append(f"event {index}: slider point {point_index} is non-finite")
                elif not (0.0 <= point_x <= PLAYFIELD_WIDTH and 0.0 <= point_y <= PLAYFIELD_HEIGHT):
                    errors.append(f"event {index}: slider point {point_index} is off playfield")
        elif event.kind == EventKind.SPINNER:
            if event.end_beat is None or not math.isfinite(event.end_beat):
                errors.append(f"event {index}: spinner is missing an end beat")
            elif event.end_beat <= event.beat_position:
                errors.append(f"event {index}: spinner end must follow start")
    return ValidationReport(tuple(errors), tuple(warnings))


def build_beat_grid_targets(
    events: Iterable[BeatEvent],
    *,
    subdivisions_per_beat: int = 16,
    max_events_per_slot: int = 2,
    slot_origin: Optional[int] = None,
    num_slots: Optional[int] = None,
) -> BeatGridTargets:
    """Pack sparse events into multi-lane beat-grid training targets.

    No event is silently clipped.  A preprocessing run must explicitly choose a
    larger lane count or a finer grid if it encounters too many simultaneous
    events, making data loss visible before a training run.
    """

    if torch is None:
        raise RuntimeError(
            "build_beat_grid_targets requires PyTorch; activate the project training environment"
        )
    if subdivisions_per_beat <= 0:
        raise ValueError("subdivisions_per_beat must be positive")
    if max_events_per_slot <= 0:
        raise ValueError("max_events_per_slot must be positive")

    ordered = sorted(events, key=lambda item: (item.beat_position, item.source_object_index))
    quantized_slots = [int(round(event.beat_position * subdivisions_per_beat)) for event in ordered]
    if slot_origin is None:
        slot_origin = min(quantized_slots, default=0)
    if num_slots is None:
        last_slot = max(quantized_slots, default=slot_origin)
        num_slots = max(1, last_slot - slot_origin + 1)
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")

    occupancy = torch.zeros((num_slots, max_events_per_slot), dtype=torch.float32)
    kind = torch.full((num_slots, max_events_per_slot), IGNORE_INDEX, dtype=torch.long)
    time_residual = torch.zeros((num_slots, max_events_per_slot), dtype=torch.float32)
    coordinates = torch.zeros((num_slots, max_events_per_slot, 2), dtype=torch.float32)
    end_beat_delta = torch.zeros((num_slots, max_events_per_slot), dtype=torch.float32)
    slider_length = torch.zeros((num_slots, max_events_per_slot), dtype=torch.float32)
    slider_repeats = torch.zeros((num_slots, max_events_per_slot), dtype=torch.long)

    lanes_used: defaultdict[int, int] = defaultdict(int)
    residual_limit = 0.5 / subdivisions_per_beat + 1e-6
    for event, absolute_slot in zip(ordered, quantized_slots):
        slot = absolute_slot - slot_origin
        if not 0 <= slot < num_slots:
            raise ValueError(
                f"event at beat {event.beat_position:.6f} lies outside requested grid "
                f"[{slot_origin}, {slot_origin + num_slots - 1}]"
            )
        lane = lanes_used[slot]
        if lane >= max_events_per_slot:
            raise ValueError(
                f"slot {absolute_slot} contains more than {max_events_per_slot} events; "
                "increase max_events_per_slot or use a finer beat grid"
            )
        lanes_used[slot] += 1

        residual = event.beat_position - absolute_slot / subdivisions_per_beat
        if abs(residual) > residual_limit:
            raise AssertionError("rounding produced an out-of-range microtiming residual")
        occupancy[slot, lane] = 1.0
        kind[slot, lane] = int(event.kind)
        time_residual[slot, lane] = residual
        coordinates[slot, lane] = torch.tensor((event.x / PLAYFIELD_WIDTH, event.y / PLAYFIELD_HEIGHT))
        if event.end_beat is not None:
            end_beat_delta[slot, lane] = event.end_beat - event.beat_position
        slider_length[slot, lane] = event.slider_length
        slider_repeats[slot, lane] = event.slider_repeats

    return BeatGridTargets(
        slot_origin=slot_origin,
        subdivisions_per_beat=subdivisions_per_beat,
        occupancy=occupancy,
        kind=kind,
        time_residual_beats=time_residual,
        coordinates=coordinates,
        end_beat_delta=end_beat_delta,
        slider_length=slider_length,
        slider_repeats=slider_repeats,
    )


@dataclass(frozen=True)
class EventLanes:
    """Frame-aligned binary event activity lanes on a uniform time grid (e.g. 10ms).

    Lanes:
    - circle_impulses: 1 at circle onset frame
    - slider_onsets: 1 at slider head frame
    - slider_tails: 1 at true physical slider tail frame (after all repeats)
    - slider_spans: 1 throughout [onset, tail] span
    - spinner_spans: 1 throughout [onset, end] span
    """

    frame_times_ms: Any  # [T] torch.Tensor
    circle_impulses: Any  # [T] torch.Tensor
    slider_onsets: Any  # [T] torch.Tensor
    slider_tails: Any  # [T] torch.Tensor
    slider_spans: Any  # [T] torch.Tensor
    spinner_spans: Any  # [T] torch.Tensor

    def as_tensor(self) -> Any:
        """Stack lanes into [T, 5] tensor."""
        if torch is None:
            raise RuntimeError("EventLanes.as_tensor requires PyTorch")
        return torch.stack(
            [
                self.circle_impulses,
                self.slider_onsets,
                self.slider_tails,
                self.slider_spans,
                self.spinner_spans,
            ],
            dim=-1,
        )


def build_event_lanes(
    events: Iterable[BeatEvent],
    timing_map: TimingMap,
    *,
    frame_times_ms: Any,
    slider_multiplier: float = 1.4,
) -> EventLanes:
    """Compute exact frame-aligned binary event lanes from events and timing."""
    if torch is None:
        raise RuntimeError("build_event_lanes requires PyTorch")
    n_frames = frame_times_ms.shape[0]
    circle_impulses = torch.zeros(n_frames, dtype=torch.float32)
    slider_onsets = torch.zeros(n_frames, dtype=torch.float32)
    slider_tails = torch.zeros(n_frames, dtype=torch.float32)
    slider_spans = torch.zeros(n_frames, dtype=torch.float32)
    spinner_spans = torch.zeros(n_frames, dtype=torch.float32)

    if n_frames == 0:
        return EventLanes(
            frame_times_ms=frame_times_ms,
            circle_impulses=circle_impulses,
            slider_onsets=slider_onsets,
            slider_tails=slider_tails,
            slider_spans=slider_spans,
            spinner_spans=spinner_spans,
        )

    # Frame step estimate for indexing
    times = frame_times_ms.cpu()

    def _frame_idx(t_ms: float) -> int:
        idx = int(torch.searchsorted(times, torch.tensor(t_ms), right=False).item())
        if idx > 0 and idx < n_frames:
            # Check if previous frame center is closer
            if abs(times[idx - 1].item() - t_ms) < abs(times[idx].item() - t_ms):
                idx = idx - 1
        return min(max(0, idx), n_frames - 1)

    for event in events:
        start_ms = timing_map.beat_to_time(event.beat_position)
        start_idx = _frame_idx(start_ms)

        if event.kind == EventKind.CIRCLE:
            circle_impulses[start_idx] = 1.0
        elif event.kind == EventKind.SLIDER:
            slider_onsets[start_idx] = 1.0
            if event.slider_length > 0:
                duration_ms = timing_map.slider_duration_ms(
                    start_ms,
                    event.slider_length,
                    max(1, event.slider_repeats),
                    slider_multiplier,
                )
                end_ms = start_ms + duration_ms
            elif event.end_beat is not None:
                end_ms = timing_map.beat_to_time(event.end_beat)
            else:
                end_ms = start_ms

            end_idx = _frame_idx(end_ms)
            slider_tails[end_idx] = 1.0
            span_start = min(start_idx, end_idx)
            span_end = max(start_idx, end_idx)
            slider_spans[span_start : span_end + 1] = 1.0
        elif event.kind == EventKind.SPINNER:
            if event.end_beat is not None:
                end_ms = timing_map.beat_to_time(event.end_beat)
            else:
                end_ms = start_ms
            end_idx = _frame_idx(end_ms)
            span_start = min(start_idx, end_idx)
            span_end = max(start_idx, end_idx)
            spinner_spans[span_start : span_end + 1] = 1.0

    return EventLanes(
        frame_times_ms=frame_times_ms,
        circle_impulses=circle_impulses,
        slider_onsets=slider_onsets,
        slider_tails=slider_tails,
        slider_spans=slider_spans,
        spinner_spans=spinner_spans,
    )
