"""Explicit osu!standard parser for v3.

All times returned here are *audio-relative milliseconds as the game plays them*:
files with format version < 5 receive osu!'s +24 ms legacy offset on every
timing point and hit object. Slider durations follow stable semantics: the red
line active at the slider start supplies the beat length, the last timing point
at or before the start supplies slider velocity (a red line resets SV to 1, a
green line at the same timestamp overrides it), SV is clamped to [0.1, 10].
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

CIRCLE, SLIDER, SPINNER = 0, 1, 2
LEGACY_OFFSET_MS = 24.0


class ChartError(ValueError):
    """A chart that cannot be represented faithfully; recorded, never silently fixed."""


@dataclass(frozen=True)
class RedLine:
    time_ms: float
    beat_ms: float
    meter: int


@dataclass(frozen=True)
class TimingPoint:
    time_ms: float
    red: bool
    beat_ms: float       # red: beat length; green: raw negative value
    meter: int
    kiai: bool
    order: int           # file order, for stable tie-breaking


@dataclass
class HitObject:
    kind: int
    time_ms: float
    x: float
    y: float
    new_combo: bool
    combo_skip: int
    hit_sound: int
    curve_type: str = ""
    control_points: tuple[tuple[float, float], ...] = ()
    spans: int = 1
    pixel_length: float = 0.0
    end_ms: float = math.nan
    sv: float = 1.0
    beat_ms: float = math.nan


@dataclass
class Chart:
    version: int
    general: dict[str, str]
    metadata: dict[str, str]
    difficulty: dict[str, str]
    timing_points: list[TimingPoint]
    red_lines: list[RedLine]
    objects: list[HitObject]
    slider_multiplier: float
    tick_rate: float
    cs: float
    ar: float
    od: float
    hp: float
    stack_leniency: float
    mode: int
    warnings: list[str] = field(default_factory=list)


def _sections(text: str) -> tuple[int, dict[str, list[str]]]:
    version = -1
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("//"):
            continue
        if version < 0 and line.lower().startswith("osu file format v"):
            try:
                version = int(line[len("osu file format v"):].strip())
            except ValueError:
                version = 0
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return version, sections


def _key_values(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _float(values: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(values.get(key, default))
        return value if math.isfinite(value) else default
    except ValueError:
        return default


def parse_timing_points(lines: list[str], offset: float) -> list[TimingPoint]:
    points: list[TimingPoint] = []
    for order, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            time_ms = float(parts[0]) + offset
            beat = float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(time_ms) and math.isfinite(beat)):
            continue
        meter = 4
        if len(parts) > 2:
            try:
                meter = int(parts[2]) or 4
            except ValueError:
                meter = 4
        red = beat > 0
        if len(parts) > 6:
            try:
                red = parts[6].strip() == "1" and beat > 0
            except ValueError:
                pass
        kiai = False
        if len(parts) > 7:
            try:
                kiai = bool(int(parts[7]) & 1)
            except ValueError:
                kiai = False
        points.append(TimingPoint(time_ms, red, beat, max(1, meter), kiai, order))
    points.sort(key=lambda p: (p.time_ms, p.order))
    return points


class TimingLookup:
    """Red-line and slider-velocity lookups with stable tie-breaking."""

    def __init__(self, points: list[TimingPoint]):
        reds = [p for p in points if p.red]
        if not reds:
            raise ChartError("no red (uninherited) timing point")
        self.red_times = [p.time_ms for p in reds]
        self.reds = reds
        # SV events: at equal times a red resets first, then greens apply.
        events = sorted(points, key=lambda p: (p.time_ms, 0 if p.red else 1, p.order))
        self.sv_times = [p.time_ms for p in events]
        self.sv_values = []
        for p in events:
            if p.red:
                self.sv_values.append(1.0)
            else:
                sv = -100.0 / p.beat_ms if p.beat_ms < 0 else 1.0
                self.sv_values.append(min(10.0, max(0.1, sv)))

    def red_at(self, time_ms: float) -> TimingPoint:
        i = bisect.bisect_right(self.red_times, time_ms + 1e-6) - 1
        return self.reds[max(i, 0)]  # osu! extends the first red line backwards

    def sv_at(self, time_ms: float) -> float:
        i = bisect.bisect_right(self.sv_times, time_ms + 1e-6) - 1
        return self.sv_values[i] if i >= 0 else 1.0


def _parse_objects(lines: list[str], offset: float, warnings: list[str]) -> list[HitObject]:
    objects: list[HitObject] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x, y, t = float(parts[0]), float(parts[1]), float(parts[2]) + offset
            bits, sound = int(parts[3]), int(parts[4])
        except ValueError:
            warnings.append("unparseable hit object skipped")
            continue
        combo = bool(bits & 4)
        skip = (bits >> 4) & 7
        if bits & 1:
            objects.append(HitObject(CIRCLE, t, x, y, combo, skip, sound))
        elif bits & 2:
            if len(parts) < 8:
                raise ChartError("slider missing curve/slides/length fields")
            curve = parts[5].split("|")
            controls = []
            for item in curve[1:]:
                try:
                    px, py = item.split(":", 1)
                    controls.append((float(px), float(py)))
                except ValueError:
                    continue
            try:
                spans = int(parts[6])
                length = float(parts[7])
            except ValueError as error:
                raise ChartError("slider slides/length unparseable") from error
            objects.append(HitObject(SLIDER, t, x, y, combo, skip, sound,
                                     curve_type=(curve[0] or "B").upper()[:1],
                                     control_points=tuple(controls), spans=max(1, spans),
                                     pixel_length=length))
        elif bits & 8:
            if len(parts) < 6:
                raise ChartError("spinner missing end time")
            try:
                end = float(parts[5]) + offset
            except ValueError as error:
                raise ChartError("spinner end unparseable") from error
            objects.append(HitObject(SPINNER, t, 256.0, 192.0, combo, skip, sound, end_ms=end))
        # mania hold notes (bit 128) are not standard objects; ignore.
    objects.sort(key=lambda o: o.time_ms)
    return objects


def parse_chart(text: str) -> Chart:
    version, sections = _sections(text)
    offset = LEGACY_OFFSET_MS if 0 <= version < 5 else 0.0
    general = _key_values(sections.get("General", []))
    metadata = _key_values(sections.get("Metadata", []))
    difficulty = _key_values(sections.get("Difficulty", []))
    mode = int(_float(general, "Mode", 0))
    warnings: list[str] = []
    points = parse_timing_points(sections.get("TimingPoints", []), offset)
    lookup = TimingLookup(points)
    sm = _float(difficulty, "SliderMultiplier", 1.4)
    if not 0.3 <= sm <= 4.0:
        warnings.append(f"slider multiplier {sm} outside stable range")
        sm = min(3.6, max(0.4, sm))
    od = _float(difficulty, "OverallDifficulty", 5.0)
    ar = _float(difficulty, "ApproachRate", od)  # old formats: AR defaults to OD
    objects = _parse_objects(sections.get("HitObjects", []), offset, warnings)
    for obj in objects:
        red = lookup.red_at(obj.time_ms)
        obj.beat_ms = red.beat_ms
        if obj.kind == SLIDER:
            obj.sv = lookup.sv_at(obj.time_ms)
            if not (obj.pixel_length > 0 and math.isfinite(obj.pixel_length)):
                raise ChartError("slider pixel length must be positive")
            span = obj.pixel_length / (sm * 100.0 * obj.sv) * red.beat_ms
            obj.end_ms = obj.time_ms + span * obj.spans
    reds = [RedLine(p.time_ms, p.beat_ms, p.meter) for p in points if p.red]
    # Collapse red lines at an identical timestamp: the later one wins (stable behavior).
    dedup: list[RedLine] = []
    for red in reds:
        if dedup and abs(dedup[-1].time_ms - red.time_ms) < 1e-6:
            dedup[-1] = red
        else:
            dedup.append(red)
    return Chart(version=version, general=general, metadata=metadata, difficulty=difficulty,
                 timing_points=points, red_lines=dedup, objects=objects, slider_multiplier=sm,
                 tick_rate=_float(difficulty, "SliderTickRate", 1.0),
                 cs=_float(difficulty, "CircleSize", 4.0), ar=ar, od=od,
                 hp=_float(difficulty, "HPDrainRate", 5.0),
                 stack_leniency=_float(general, "StackLeniency", 0.7), mode=mode, warnings=warnings)
