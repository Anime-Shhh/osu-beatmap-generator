"""Conservative constant-tempo fitting from detected beats (no chart clock)."""
import math
import statistics


def fit_global_grid(beats_ms):
    beats = sorted(float(v) for v in beats_ms)
    if any(not math.isfinite(v) for v in beats):
        raise ValueError('nonfinite detected beat')
    if len(beats) < 16 or beats[-1]-beats[0] < 8000:
        return {'accepted': False, 'reason': 'insufficient beat span'}
    gaps = [b-a for a, b in zip(beats, beats[1:]) if b > a]
    if not gaps: return {'accepted': False, 'reason': 'no positive intervals'}
    initial = statistics.median(gaps)
    if not 200 <= initial <= 1500:
        return {'accepted': False, 'reason': 'tempo outside diagnostic bounds'}
    center = beats[len(beats)//2]
    times = [v-center for v in beats]
    def coherence(period):
        angles = [v*2*math.pi/period for v in times]
        x, y = sum(map(math.cos, angles)), sum(map(math.sin, angles))
        return math.hypot(x, y), math.atan2(y, x)*period/(2*math.pi)
    # Global phase coherence avoids accumulated rounding drift from 20ms beat
    # tracker frames (a median interval of 440ms need not mean 136.36 BPM).
    low, high = initial*.94, initial*1.06
    candidates = [low+(high-low)*i/400 for i in range(401)]
    period = max(candidates, key=lambda x: coherence(x)[0])
    for _ in range(2):
        radius = (high-low)/400
        low, high = period-radius, period+radius
        candidates = [low+(high-low)*i/40 for i in range(41)]
        period = max(candidates, key=lambda x: coherence(x)[0])
    _, phase = coherence(period)
    origin = center+phase
    for _ in range(3):
        indices = [round((v-origin)/period) for v in beats]
        selected = [(i, v) for i, v in zip(indices, beats) if abs(v-origin-i*period) < .15*period]
        if len(selected) < 16: return {'accepted': False, 'reason': 'insufficient coherent beats'}
        mx = statistics.mean(i for i, _ in selected)
        my = statistics.mean(v for _, v in selected)
        xx = sum((i-mx)**2 for i, _ in selected)
        if not xx: return {'accepted': False, 'reason': 'degenerate grid'}
        period = sum((i-mx)*(v-my) for i, v in selected)/xx
        origin = my-period*mx
    residual = sorted(abs(v-origin-round((v-origin)/period)*period) for v in beats)
    fraction = sum(v < 25 for v in residual)/len(residual)
    return {'accepted': fraction > .9, 'bpm': 60000/period, 'beat_length_ms': period,
            'offset_ms': origin % period, 'median_residual_ms': statistics.median(residual),
            'p90_residual_ms': residual[math.ceil(.9*len(residual))-1],
            'fraction_within_25ms': fraction,
            'note': 'constant-tempo diagnostic; no reference chart used'}
