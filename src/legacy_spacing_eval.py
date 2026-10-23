"""Descriptive free-running spacing audits, never an inference/reranking policy."""
import math
import numpy as np
from .tokenizer import parse_osu_hitobjects, parse_osu_metadata

GAP_BINS_MS = (50, 100, 200, 400, 800, float('inf'))


def audit_raw_head_tokens(tokens, residuals):
    """Pre-clamp diagnostics over sampled windows, including discarded onsets."""
    from .tokenizer import TYPE_CIRCLE, TYPE_SLIDER_START, POS_OFFSET, TIME_OFFSET, GRID_W, BIN_W, BIN_H
    heads = outside = invalid = 0
    for i in range(1, len(tokens)):
        if tokens[i-1] not in (TYPE_CIRCLE, TYPE_SLIDER_START):
            continue
        if not POS_OFFSET <= tokens[i] < TIME_OFFSET:
            invalid += 1
            continue
        heads += 1
        cell = tokens[i]-POS_OFFSET
        x = (cell % GRID_W + .5)*BIN_W + residuals[i].x_offset_px
        y = (cell // GRID_W + .5)*BIN_H + residuals[i].y_offset_px
        outside += int(not (0 <= x <= 512 and 0 <= y <= 384))
    return dict(raw_head_tokens=heads, raw_head_bounds_violations=outside,
                invalid_head_position_tokens=invalid)


def distribution(values):
    if not values:
        return {'count': 0, 'mean': None, 'p50': None, 'p95': None, 'p99': None, 'max': None}
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.quantile(a, .5)),
                p95=float(np.quantile(a, .95)), p99=float(np.quantile(a, .99)), max=float(a.max()))


def audit_generated_map(osu_content, *, duration_ms, target_star):
    """Audit the serialized output, including its native slider length extension.

    Head speed is a proxy, not cursor speed or osu!'s aim formula. Head coordinates
    have already passed the unchanged tokenizer's clamp; report that explicitly.
    """
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError('positive finite audio duration required')
    from .v2.slider_path import slider_path, circle_radius  # read-only native path audit
    from .v2.evaluation import RosuDifficultyCalculator  # pinned no-mod calculator
    objects = parse_osu_hitobjects(osu_content)
    pairs = []
    previous = None
    head_violations = slider_invalid = slider_outside = bad_times = 0
    nonpositive_gaps = 0
    radius = circle_radius(parse_osu_metadata(osu_content).get('cs', 4))
    for obj in objects:
        x, y, t = obj['x'], obj['y'], obj['time']
        if not (0 <= t <= duration_ms): bad_times += 1
        if obj['type'] == 'spinner':
            previous = None
            continue
        if not (0 <= x <= 512 and 0 <= y <= 384): head_violations += 1
        if previous is not None:
            gap = t - previous['time']
            distance = math.hypot(x-previous['x'], y-previous['y'])
            if gap <= 0: nonpositive_gaps += 1
            pairs.append(dict(gap_ms=gap, distance_px=distance,
                              head_speed_px_s=distance*1000/gap if gap > 0 else None))
        previous = obj
        if obj['type'] == 'slider':
            try:
                if obj.get('slides', 0) < 1: raise ValueError('invalid repeats')
                path = slider_path((x,y), obj.get('curve_points',()), obj.get('curve_type','B'), obj.get('length',0))
                if (path.min(0)<radius).any() or (path.max(0)>(512-radius,384-radius)).any(): slider_outside += 1
            except (ValueError,TypeError,IndexError):
                slider_invalid += 1
    bins = {}
    lower = 0
    for upper in GAP_BINS_MS:
        subset = [p for p in pairs if lower < p['gap_ms'] <= upper]
        bins[f'({lower},{upper}]'] = distribution([p['distance_px'] for p in subset])
        lower = upper
    calculator = RosuDifficultyCalculator()
    try:
        stars = calculator.calculate(osu_content).to_dict()
    except Exception as exc:
        stars = {'available': False, 'stars': None, 'error': str(exc)}
    return dict(target_star=target_star, duration_ms=duration_ms, objects=len(objects),
        sliders=sum(o['type']=='slider' for o in objects), events_per_minute=len(objects)*60000/duration_ms,
        first_head_ms=min((o['time'] for o in objects),default=None),
        last_head_ms=max((o['time'] for o in objects),default=None),
        head_playfield_violations=head_violations, invalid_sliders=slider_invalid,
        slider_body_violations=slider_outside, starts_outside_audio=bad_times,
        nonpositive_head_gaps=nonpositive_gaps, spacing=distribution([p['distance_px'] for p in pairs]),
        head_speed=distribution([p['head_speed_px_s'] for p in pairs if p['head_speed_px_s'] is not None]),
        distance_by_gap_ms=bins, pairs=pairs, realized_difficulty=stars,
        limitations=['Head-to-head only; excludes spinner transitions, not slider-tail travel.',
                     'Heads are post-clamp; zero violations does not prove raw predictions stayed in bounds.',
                     'Slider paths are numerically approximated and do not include stacking.'])
