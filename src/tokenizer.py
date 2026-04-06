"""
Grid-based tokenizer for osu! beatmap hit objects.

Encodes continuous playfield coordinates (512x384) into a 32x32 discrete grid
with residual offsets for sub-bin precision. Handles circles, sliders (parametric
Bezier encoding), and spinners.
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------
PAD = 0
BOS = 1
EOS = 2
SEP = 3

# Object type tokens (offset at 4)
TYPE_CIRCLE = 4
TYPE_SLIDER_START = 5
TYPE_SLIDER_CONTROL = 6
TYPE_SLIDER_END = 7
TYPE_SPINNER_START = 8
TYPE_SPINNER_END = 9

# Slider curve type tokens
CURVE_BEZIER = 10
CURVE_LINEAR = 11
CURVE_PERFECT = 12
CURVE_CATMULL = 13

_NUM_SPECIAL = 14

# Grid and time parameters
GRID_W = 32
GRID_H = 32
PLAYFIELD_W = 512
PLAYFIELD_H = 384
BIN_W = PLAYFIELD_W / GRID_W   # 16.0 px
BIN_H = PLAYFIELD_H / GRID_H   # 12.0 px

NUM_POS_BINS = GRID_W * GRID_H  # 1024

# Beat-relative timing: bins represent fractions of a beat, not milliseconds.
# 128 bins at 1/16-beat resolution covers 0 to ~8 beats.
NUM_BEAT_BINS = 128
BEAT_QUANT = 1.0 / 16.0        # each bin = 1/16 of a beat
MAX_BEAT_DELTA = (NUM_BEAT_BINS - 1) * BEAT_QUANT  # 7.9375 beats

# Vocabulary layout:
#   [0..13]     special + type + curve tokens
#   [14..1037]  position bins  (32*32 = 1024)
#   [1038..1165] beat delta bins (128)
POS_OFFSET = _NUM_SPECIAL
TIME_OFFSET = POS_OFFSET + NUM_POS_BINS
VOCAB_SIZE = TIME_OFFSET + NUM_BEAT_BINS  # 1166

# Difficulty is now a conditioning input, not a token in the sequence.
# 5 bins: EASY=0, NORMAL=1, HARD=2, INSANE=3, EXPERT=4
NUM_DIFF_BINS = 5
TOTAL_VOCAB = VOCAB_SIZE  # 1166

MAX_SLIDER_CONTROL_POINTS = 8


# ---------------------------------------------------------------------------
# Residual container
# ---------------------------------------------------------------------------
@dataclass
class Residuals:
    """Continuous residuals for sub-bin precision."""
    time_offset_ms: float = 0.0   # beat residual in [-BEAT_QUANT/2, +BEAT_QUANT/2]
    x_offset_px: float = 0.0      # [-BIN_W/2, +BIN_W/2] px
    y_offset_px: float = 0.0      # [-BIN_H/2, +BIN_H/2] px


@dataclass
class TokenizedObject:
    """A single tokenized hit object with discrete tokens and residuals."""
    tokens: list[int] = field(default_factory=list)
    residuals: list[Residuals] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def xy_to_bin(x: float, y: float) -> tuple[int, Residuals]:
    """Convert playfield (x, y) to grid bin index and residual offsets."""
    gx = x / BIN_W
    gy = y / BIN_H
    bx = max(0, min(GRID_W - 1, int(gx)))
    by = max(0, min(GRID_H - 1, int(gy)))
    bin_idx = by * GRID_W + bx
    res = Residuals(
        x_offset_px=x - (bx * BIN_W + BIN_W / 2),
        y_offset_px=y - (by * BIN_H + BIN_H / 2),
    )
    return POS_OFFSET + bin_idx, res


def bin_to_xy(token: int, res: Optional[Residuals] = None) -> tuple[float, float]:
    """Convert grid bin token back to playfield coordinates."""
    idx = token - POS_OFFSET
    bx = idx % GRID_W
    by = idx // GRID_W
    x = bx * BIN_W + BIN_W / 2
    y = by * BIN_H + BIN_H / 2
    if res is not None:
        x += res.x_offset_px
        y += res.y_offset_px
    x = max(0, min(PLAYFIELD_W, x))
    y = max(0, min(PLAYFIELD_H, y))
    return x, y


def beat_to_bin(delta_beats: float) -> tuple[int, float]:
    """Quantize beat delta to bin index and residual (in beats)."""
    clamped = max(0.0, min(MAX_BEAT_DELTA, delta_beats))
    bin_idx = int(round(clamped / BEAT_QUANT))
    bin_idx = max(0, min(NUM_BEAT_BINS - 1, bin_idx))
    residual = delta_beats - bin_idx * BEAT_QUANT
    residual = max(-BEAT_QUANT / 2, min(BEAT_QUANT / 2, residual))
    return TIME_OFFSET + bin_idx, residual


def bin_to_beat(token: int, residual: float = 0.0) -> float:
    """Convert beat bin token back to beat delta."""
    idx = token - TIME_OFFSET
    return idx * BEAT_QUANT + residual


def difficulty_to_bin(star_rating: float) -> int:
    """Map star rating to a difficulty conditioning bin (0-4)."""
    if star_rating < 2.0:
        return 0   # EASY
    elif star_rating < 3.0:
        return 1   # NORMAL
    elif star_rating < 4.0:
        return 2   # HARD
    elif star_rating < 5.3:
        return 3   # INSANE
    else:
        return 4   # EXPERT


# ---------------------------------------------------------------------------
# .osu file parser
# ---------------------------------------------------------------------------
def _parse_hit_type(type_int: int) -> str:
    """Decode the type bitfield from .osu hit object line."""
    if type_int & 1:
        return "circle"
    elif type_int & 2:
        return "slider"
    elif type_int & 8:
        return "spinner"
    return "circle"


_CURVE_MAP = {"B": CURVE_BEZIER, "L": CURVE_LINEAR, "P": CURVE_PERFECT, "C": CURVE_CATMULL}
_CURVE_UNMAP = {v: k for k, v in _CURVE_MAP.items()}


def parse_osu_hitobjects(osu_content: str) -> list[dict]:
    """
    Parse the [HitObjects] section of an .osu file.

    Returns a list of dicts with keys:
        time (ms), type (str), x, y,
        and for sliders: curve_type, curve_points, slides, length
        and for spinners: end_time
    """
    lines = osu_content.split("\n")
    in_hitobjects = False
    objects = []

    for line in lines:
        line = line.strip()
        if line == "[HitObjects]":
            in_hitobjects = True
            continue
        if in_hitobjects:
            if line.startswith("[") or not line:
                if line.startswith("["):
                    break
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                time_ms = float(parts[2])
                type_int = int(parts[3])
                hit_sound = int(parts[4]) if len(parts) > 4 else 0
            except (ValueError, IndexError):
                continue

            hit_type = _parse_hit_type(type_int)
            obj = {
                "time": time_ms,
                "type": hit_type,
                "x": x,
                "y": y,
                "hit_sound": hit_sound,
                "new_combo": bool(type_int & 4),
            }

            if hit_type == "slider" and len(parts) >= 8:
                curve_data = parts[5]
                curve_parts = curve_data.split("|")
                curve_type_char = curve_parts[0] if curve_parts else "B"
                control_points = []
                for cp in curve_parts[1:]:
                    try:
                        cx, cy = cp.split(":")
                        control_points.append((float(cx), float(cy)))
                    except (ValueError, IndexError):
                        continue
                try:
                    slides = int(parts[6])
                    length = float(parts[7])
                except (ValueError, IndexError):
                    slides = 1
                    length = 100.0
                obj["curve_type"] = curve_type_char
                obj["curve_points"] = control_points[:MAX_SLIDER_CONTROL_POINTS]
                obj["slides"] = slides
                obj["length"] = length

            elif hit_type == "spinner" and len(parts) >= 6:
                try:
                    obj["end_time"] = float(parts[5])
                except (ValueError, IndexError):
                    obj["end_time"] = time_ms + 1000

            objects.append(obj)

    return objects


def parse_osu_metadata(osu_content: str) -> dict:
    """Extract useful metadata from .osu file header sections."""
    meta = {}
    for line in osu_content.split("\n"):
        line = line.strip()
        if line.startswith("OverallDifficulty:"):
            meta["od"] = float(line.split(":")[1])
        elif line.startswith("ApproachRate:"):
            meta["ar"] = float(line.split(":")[1])
        elif line.startswith("CircleSize:"):
            meta["cs"] = float(line.split(":")[1])
        elif line.startswith("HPDrainRate:"):
            meta["hp"] = float(line.split(":")[1])
        elif line.startswith("SliderMultiplier:"):
            meta["slider_multiplier"] = float(line.split(":")[1])
        elif line.startswith("Mode:"):
            meta["mode"] = int(line.split(":")[1].strip())
    return meta


def parse_osu_bpm(osu_content: str) -> float:
    """Extract primary BPM from the first uninherited [TimingPoints] entry."""
    in_timing = False
    for line in osu_content.splitlines():
        line = line.strip()
        if line == "[TimingPoints]":
            in_timing = True
            continue
        if not in_timing:
            continue
        if line.startswith("["):
            break
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            beat_length = float(parts[1])
        except ValueError:
            continue
        uninherited = len(parts) < 7 or parts[6].strip() == "1"
        if uninherited and beat_length > 0:
            return 60000.0 / beat_length
    return 120.0


# ---------------------------------------------------------------------------
# Tokenize / Detokenize
# ---------------------------------------------------------------------------
def tokenize_beatmap(
    osu_content: str,
    ms_per_beat: float = 500.0,
    window_start_ms: float = 0.0,
    window_end_ms: float = float("inf"),
) -> TokenizedObject:
    """
    Convert an .osu file (or window of it) into a token sequence with residuals.

    Time deltas are expressed in **beats** (delta_ms / ms_per_beat) so the model
    learns rhythm independently of BPM.

    Sequence: [BOS] <objects...> [EOS]
    Difficulty and BPM are handled as separate conditioning inputs, not tokens.

    Token sequence format per hit object:
        <BEAT_BIN> <TYPE> <POS_BIN> [slider params...]

    For sliders:
        <BEAT_BIN> <SLIDER_START> <POS_BIN> <CURVE_TYPE>
        [<SLIDER_CONTROL> <POS_BIN>] * N
        <SLIDER_END> <POS_BIN(last)>

    For spinners:
        <BEAT_BIN> <SPINNER_START> <POS_BIN>
        <BEAT_BIN(duration)> <SPINNER_END>
    """
    objects = parse_osu_hitobjects(osu_content)
    objects = [o for o in objects if window_start_ms <= o["time"] < window_end_ms]
    objects.sort(key=lambda o: o["time"])

    result = TokenizedObject()
    result.tokens.append(BOS)
    result.residuals.append(Residuals())

    prev_time = window_start_ms

    for obj in objects:
        delta_ms = obj["time"] - prev_time
        delta_beats = delta_ms / ms_per_beat
        time_tok, time_res = beat_to_bin(delta_beats)
        pos_tok, pos_res = xy_to_bin(obj["x"], obj["y"])
        pos_res.time_offset_ms = time_res

        if obj["type"] == "circle":
            result.tokens.extend([time_tok, TYPE_CIRCLE, pos_tok])
            result.residuals.extend([Residuals(time_offset_ms=time_res), Residuals(), pos_res])

        elif obj["type"] == "slider":
            curve_type_tok = _CURVE_MAP.get(obj.get("curve_type", "B"), CURVE_BEZIER)
            result.tokens.extend([time_tok, TYPE_SLIDER_START, pos_tok, curve_type_tok])
            result.residuals.extend([
                Residuals(time_offset_ms=time_res), Residuals(), pos_res, Residuals(),
            ])

            for cp_x, cp_y in obj.get("curve_points", []):
                cp_tok, cp_res = xy_to_bin(cp_x, cp_y)
                result.tokens.extend([TYPE_SLIDER_CONTROL, cp_tok])
                result.residuals.extend([Residuals(), cp_res])

            last_cp = obj.get("curve_points", [])
            if last_cp:
                end_tok, end_res = xy_to_bin(last_cp[-1][0], last_cp[-1][1])
            else:
                end_tok, end_res = pos_tok, Residuals(
                    x_offset_px=pos_res.x_offset_px, y_offset_px=pos_res.y_offset_px
                )
            result.tokens.extend([TYPE_SLIDER_END, end_tok])
            result.residuals.extend([Residuals(), end_res])

        elif obj["type"] == "spinner":
            result.tokens.extend([time_tok, TYPE_SPINNER_START, pos_tok])
            result.residuals.extend([Residuals(time_offset_ms=time_res), Residuals(), pos_res])
            duration_ms = obj.get("end_time", obj["time"] + 1000) - obj["time"]
            duration_beats = duration_ms / ms_per_beat
            dur_tok, dur_res = beat_to_bin(duration_beats)
            result.tokens.extend([dur_tok, TYPE_SPINNER_END])
            result.residuals.extend([Residuals(time_offset_ms=dur_res), Residuals()])

        prev_time = obj["time"]

    result.tokens.append(EOS)
    result.residuals.append(Residuals())
    return result


def detokenize_to_hitobjects(
    tokens: list[int],
    residuals: Optional[list[Residuals]] = None,
    base_time_ms: float = 0.0,
    ms_per_beat: float = 500.0,
) -> list[dict]:
    """
    Convert token sequence back to a list of hit object dicts
    suitable for writing to .osu format.

    Beat-relative deltas are multiplied by ms_per_beat to recover absolute
    millisecond timestamps.
    """
    if residuals is None:
        residuals = [Residuals()] * len(tokens)

    objects = []
    current_time = base_time_ms
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        if tok < _NUM_SPECIAL and tok not in (TYPE_CIRCLE, TYPE_SLIDER_START, TYPE_SPINNER_START):
            i += 1
            continue

        # Beat delta token
        if TIME_OFFSET <= tok < TIME_OFFSET + NUM_BEAT_BINS:
            delta_beats = bin_to_beat(tok, residuals[i].time_offset_ms)
            current_time += delta_beats * ms_per_beat
            i += 1
            if i >= n:
                break

            obj_type_tok = tokens[i]
            i += 1

            if obj_type_tok == TYPE_CIRCLE:
                if i >= n:
                    break
                pos_tok = tokens[i]
                x, y = bin_to_xy(pos_tok, residuals[i])
                objects.append({"time": current_time, "type": "circle", "x": x, "y": y})
                i += 1

            elif obj_type_tok == TYPE_SLIDER_START:
                if i >= n:
                    break
                pos_tok = tokens[i]
                sx, sy = bin_to_xy(pos_tok, residuals[i])
                i += 1

                curve_type = "B"
                if i < n and tokens[i] in _CURVE_UNMAP:
                    curve_type = _CURVE_UNMAP[tokens[i]]
                    i += 1

                control_points = []
                while i < n and tokens[i] == TYPE_SLIDER_CONTROL:
                    i += 1
                    if i < n and POS_OFFSET <= tokens[i] < POS_OFFSET + NUM_POS_BINS:
                        cx, cy = bin_to_xy(tokens[i], residuals[i])
                        control_points.append((cx, cy))
                        i += 1

                if i < n and tokens[i] == TYPE_SLIDER_END:
                    i += 1
                    if i < n and POS_OFFSET <= tokens[i] < POS_OFFSET + NUM_POS_BINS:
                        i += 1

                objects.append({
                    "time": current_time, "type": "slider",
                    "x": sx, "y": sy, "curve_type": curve_type,
                    "curve_points": control_points,
                    "slides": 1, "length": 100.0,
                })

            elif obj_type_tok == TYPE_SPINNER_START:
                if i >= n:
                    break
                pos_tok = tokens[i]
                spx, spy = bin_to_xy(pos_tok, residuals[i])
                i += 1

                end_time = current_time + 1000
                if i < n and TIME_OFFSET <= tokens[i] < TIME_OFFSET + NUM_BEAT_BINS:
                    dur_beats = bin_to_beat(tokens[i], residuals[i].time_offset_ms)
                    end_time = current_time + dur_beats * ms_per_beat
                    i += 1
                    if i < n and tokens[i] == TYPE_SPINNER_END:
                        i += 1

                objects.append({
                    "time": current_time, "type": "spinner",
                    "x": spx, "y": spy, "end_time": end_time,
                })
            else:
                pass
        else:
            i += 1

    return objects


def hitobjects_to_osu_lines(objects: list[dict]) -> list[str]:
    """Convert hit object dicts to .osu [HitObjects] lines."""
    lines = []
    for obj in objects:
        x = int(round(obj["x"]))
        y = int(round(obj["y"]))
        t = int(round(obj["time"]))

        if obj["type"] == "circle":
            # x,y,time,type,hitSound,objectParams...
            lines.append(f"{x},{y},{t},1,0,0:0:0:0:")

        elif obj["type"] == "slider":
            ct = obj.get("curve_type", "B")
            cps = obj.get("curve_points", [])
            curve_str = ct
            for cx, cy in cps:
                curve_str += f"|{int(round(cx))}:{int(round(cy))}"
            slides = obj.get("slides", 1)
            length = obj.get("length", 100.0)
            lines.append(f"{x},{y},{t},2,0,{curve_str},{slides},{length:.1f}")

        elif obj["type"] == "spinner":
            end_t = int(round(obj.get("end_time", t + 1000)))
            lines.append(f"256,192,{t},8,0,{end_t}")

    return lines
