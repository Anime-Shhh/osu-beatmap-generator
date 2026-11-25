"""Beat grid built from red lines, with partial beats and per-red-line resets.

Unit conventions
* A *beat token* is one beat of the active red line. The beat counter restarts
  at every red line (fixing src/v2/timing.py, which kept counting across them).
* When the next red line arrives mid-beat, the last beat before it is *partial*:
  ``frac < 1`` and only slots inside it are valid.
* Within a beat, positions use 48 ticks. The 24 *slots* used by the rhythm model
  are the union of the 1/16 and 1/12 grids (multiples of 3 or 4 ticks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .osu import RedLine

TICKS = 48
SLOT_TICKS = tuple(t for t in range(TICKS) if t % 3 == 0 or t % 4 == 0)
assert len(SLOT_TICKS) == 24
TICK_TO_SLOT = {t: i for i, t in enumerate(SLOT_TICKS)}
SNAP_TOL_MS = 2.0
PARTIAL_EPS_MS = 2.0
MIN_BEAT_MS, MAX_BEAT_MS = 15.0, 6000.0  # 10 .. 4000 BPM; outside is a broken red line
MAX_BEATS = 30000


class GridError(ValueError):
    pass


@dataclass(frozen=True)
class BeatGrid:
    start_ms: np.ndarray     # float64 [B]
    beat_ms: np.ndarray      # float64 [B]
    frac: np.ndarray         # float64 [B], 1 for full beats
    red_index: np.ndarray    # int32 [B]
    beat_in_measure: np.ndarray  # int16 [B], resets at each red line
    meter: np.ndarray        # int16 [B]

    def __len__(self) -> int:
        return len(self.start_ms)


def build_grid(red_lines: list[RedLine], start_ms: float, end_ms: float) -> BeatGrid:
    if not red_lines:
        raise GridError("no red lines")
    if end_ms <= start_ms:
        raise GridError("empty time range")
    starts, lengths, fracs, reds, phases, meters = [], [], [], [], [], []
    for index, red in enumerate(red_lines):
        if not (MIN_BEAT_MS <= red.beat_ms <= MAX_BEAT_MS):
            raise GridError(f"red line beat length {red.beat_ms} outside [{MIN_BEAT_MS}, {MAX_BEAT_MS}]")
        seg_end = red_lines[index + 1].time_ms if index + 1 < len(red_lines) else end_ms
        if index + 1 == len(red_lines):
            seg_end = max(seg_end, red.time_ms)
        k = 0
        if index == 0 and start_ms < red.time_ms:
            k = -math.ceil((red.time_ms - start_ms) / red.beat_ms - 1e-9)
        while True:
            s = red.time_ms + k * red.beat_ms
            remaining = seg_end - s
            if remaining <= PARTIAL_EPS_MS:
                break
            frac = 1.0 if remaining >= red.beat_ms - PARTIAL_EPS_MS else remaining / red.beat_ms
            starts.append(s)
            lengths.append(red.beat_ms)
            fracs.append(frac)
            reds.append(index)
            phases.append(k % red.meter)
            meters.append(red.meter)
            if len(starts) > MAX_BEATS:
                raise GridError("grid exceeds beat budget")
            if frac < 1.0:
                break
            k += 1
    return BeatGrid(np.asarray(starts, np.float64), np.asarray(lengths, np.float64),
                    np.asarray(fracs, np.float64), np.asarray(reds, np.int32),
                    np.asarray(phases, np.int16), np.asarray(meters, np.int16))


@dataclass(frozen=True)
class Located:
    beat: np.ndarray        # int32 beat token index (-1 if before the grid)
    tick: np.ndarray        # int16 nearest 48-tick in [0, 48)
    slot: np.ndarray        # int16 24-slot index, -1 if off-grid (> SNAP_TOL_MS)
    resid48_ms: np.ndarray  # float32 signed residual to nearest 48-tick
    resid24_ms: np.ndarray  # float32 signed residual to nearest 24-slot tick


_ALL_TICKS = np.arange(TICKS, dtype=np.float64)
_SLOT_TICKS_F = np.asarray(SLOT_TICKS, dtype=np.float64)


def _nearest(grid: BeatGrid, j: int, t: float, ticks: np.ndarray) -> tuple[int, int, float]:
    """Nearest valid candidate among ``ticks`` of beat j, or tick 0 of beat j+1."""
    length, start = grid.beat_ms[j], grid.start_ms[j]
    valid = ticks[(ticks == 0) | (ticks < grid.frac[j] * TICKS - 1e-6)]
    times = start + valid / TICKS * length
    k = int(np.argmin(np.abs(t - times)))
    best = (j, int(valid[k]), float(t - times[k]))
    if j + 1 < len(grid):
        nxt = float(t - grid.start_ms[j + 1])
        if abs(nxt) < abs(best[2]):
            best = (j + 1, 0, nxt)
    return best


def locate(grid: BeatGrid, times_ms: np.ndarray, tol_ms: float = SNAP_TOL_MS) -> Located:
    times = np.atleast_1d(np.asarray(times_ms, np.float64))
    beats = np.searchsorted(grid.start_ms, times, side="right") - 1
    out_beat = np.full(times.shape, -1, np.int32)
    out_tick = np.zeros(times.shape, np.int16)
    out_slot = np.full(times.shape, -1, np.int16)
    r48 = np.full(times.shape, np.nan, np.float32)
    r24 = np.full(times.shape, np.nan, np.float32)
    for i, (t, j) in enumerate(zip(times, beats)):
        if j < 0:
            continue
        b48, tick48, res48 = _nearest(grid, int(j), float(t), _ALL_TICKS)
        b24, tick24, res24 = _nearest(grid, int(j), float(t), _SLOT_TICKS_F)
        out_beat[i] = b48
        out_tick[i] = tick48
        r48[i] = res48
        r24[i] = res24
        if abs(res24) <= tol_ms:  # on the 24-slot grid: tick/beat are the slot's
            out_beat[i], out_tick[i] = b24, tick24
            out_slot[i] = TICK_TO_SLOT[tick24]
    return Located(out_beat, out_tick, out_slot, r48, r24)


def time_of(grid: BeatGrid, beat: np.ndarray, tick: np.ndarray) -> np.ndarray:
    """Inverse of locate for on-grid positions (48-tick resolution)."""
    beat = np.asarray(beat)
    return grid.start_ms[beat] + np.asarray(tick, np.float64) / TICKS * grid.beat_ms[beat]


def slot_time(grid: BeatGrid, beat: np.ndarray, slot: np.ndarray) -> np.ndarray:
    ticks = np.asarray(SLOT_TICKS)[np.asarray(slot)]
    return time_of(grid, beat, ticks)
