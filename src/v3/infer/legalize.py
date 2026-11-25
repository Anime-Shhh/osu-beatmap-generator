"""Rhythm tokens -> timed, playable events (no overlapping holds, sectioned SV)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.v3.data.grid import SLOT_TICKS, TICKS, BeatGrid

MIN_GAP_BEATS = 1 / 16
MIN_GAP_MS = 30.0
MIN_SPAN_TICKS = 6          # 1/8 beat: shorter sliders become circles
SV_CHANGE_PENALTY = 3.0     # nats; humans change SV per section, not per slider


@dataclass
class Event:
    time_ms: float
    kind: int                # 0 circle, 1 slider, 2 spinner
    beat: int
    tick: int
    beat_ms: float
    new_combo: bool = False
    stack: int = 0
    spans: int = 1
    span_ticks: float = 0.0
    end_ms: float = 0.0
    sv: float = 1.0


def events_from_tokens(pattern, attrs, sv_logp, grid: BeatGrid, patterns: list[int], durations: list[int],
                       sv_classes: list[float], na: dict[str, int]) -> list[Event]:
    """pattern [L] ids, attrs [L, K, 5] ids (NA allowed), sv_logp [L, n_sv]; L beats of ``grid``."""
    masks = np.array([0] + list(patterns), np.int64)
    n_dur = len(durations)
    events: list[Event] = []
    for j, pid in enumerate(pattern):
        m = int(masks[pid]) if 0 <= pid < len(masks) else 0
        slots = [s for s in range(24) if m >> s & 1]
        for k, slot in enumerate(slots):
            tick = SLOT_TICKS[slot]
            kind, dur, spans, nc, stack = (int(v) for v in attrs[j, k]) if k < attrs.shape[1] else (0, 0, 0, 0, 0)
            kind = kind if kind in (0, 1, 2) else 0
            ev = Event(time_ms=float(grid.start_ms[j] + tick / TICKS * grid.beat_ms[j]), kind=kind, beat=j, tick=tick,
                       beat_ms=float(grid.beat_ms[j]), new_combo=nc == 1, stack=stack if stack in (1, 2) else 0)
            if kind == 1:
                ev.span_ticks = float(durations[dur]) if 0 <= dur < n_dur else 24.0
                ev.spans = spans + 1 if 0 <= spans < 32 else 1
            elif kind == 2:
                ev.span_ticks = float((dur - n_dur + 1) * TICKS) if n_dur <= dur < na["dur"] else 2.0 * TICKS
            events.append(ev)
    events.sort(key=lambda e: e.time_ms)
    _fix_holds(events)
    _assign_sv(events, sv_logp, sv_classes)
    if events:
        events[0].new_combo = True
    for i, e in enumerate(events):          # spinners and the object after them start combos
        if e.kind == 2 or (i and events[i - 1].kind == 2):
            e.new_combo = True
    return events


def _fix_holds(events: list[Event]) -> None:
    for i, e in enumerate(events):
        nxt = events[i + 1].time_ms if i + 1 < len(events) else float("inf")
        limit = nxt - max(MIN_GAP_MS, MIN_GAP_BEATS * e.beat_ms)
        if e.kind == 0:
            e.end_ms = e.time_ms
            continue
        span_ms = e.span_ticks / TICKS * e.beat_ms
        if e.kind == 2:
            e.end_ms = min(e.time_ms + span_ms, limit)
            if e.end_ms - e.time_ms < e.beat_ms:          # a spinner shorter than a beat is not a spinner
                e.kind, e.end_ms = 0, e.time_ms
            continue
        while e.spans > 1 and e.time_ms + span_ms * e.spans > limit:
            e.spans -= 1
        if e.time_ms + span_ms * e.spans > limit:
            fit_ticks = (limit - e.time_ms) / e.beat_ms * TICKS
            fit_ticks = np.floor(fit_ticks / 3) * 3          # stay on the 1/16 grid
            if fit_ticks < MIN_SPAN_TICKS:
                e.kind, e.spans, e.end_ms = 0, 1, e.time_ms
                continue
            e.span_ticks = float(fit_ticks)
            span_ms = e.span_ticks / TICKS * e.beat_ms
        e.end_ms = e.time_ms + span_ms * e.spans
        if e.kind == 1:
            e.stack = e.stack  # unchanged; stacking onto a slider exit is resolved in placement


def _assign_sv(events: list[Event], sv_logp, sv_classes: list[float]) -> None:
    sliders = [e for e in events if e.kind == 1]
    if not sliders:
        return
    emit = np.stack([np.asarray(sv_logp[e.beat], np.float64) for e in sliders])      # [S, C]
    n, c = emit.shape
    score = emit[0].copy()
    back = np.zeros((n, c), np.int64)
    for i in range(1, n):
        stay = score
        switch = score.max() - SV_CHANGE_PENALTY
        back[i] = np.where(stay >= switch, np.arange(c), score.argmax())
        score = np.maximum(stay, switch) + emit[i]
    state = int(score.argmax())
    for i in range(n - 1, -1, -1):
        sliders[i].sv = float(sv_classes[state])
        state = int(back[i, state])
