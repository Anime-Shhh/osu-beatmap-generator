"""Chart-alignment supervision derived from native times, never model inputs.

Targets mean 'this mapper selected an event here at this difficulty', not
'every acoustic onset must be clicked' or an instrument attribution label.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .clock import TimingHypothesis
from .events import EventKind
from .grammar import BarEventCodec, RESIDUAL_TICKS_PER_BEAT


def query_grid(timing, start_beat: float, length_beats: float, count: int) -> torch.Tensor:
    if count < 2 or length_beats <= 0:
        raise ValueError('alignment grid needs at least two samples and positive bar length')
    # Beat-space queries mapped individually to milliseconds handle tempo changes
    # inside a bar; interpolating its endpoints would not.
    return torch.tensor([timing.beat_to_time(start_beat + i * length_beats / count)
                         for i in range(count)], dtype=torch.float32)


def reference_events(record):
    timing = TimingHypothesis.from_dict(record['oracle_timing_hypothesis']).timing_map
    codec = BarEventCodec()
    events = []
    for bar in record['bar_targets']:
        # Existing labels can quantize an onset just inside the bar onto its
        # right boundary. Accept only the codec's documented half-residual-step
        # error here; generation still enforces strict bar membership.
        tolerance = 0.5 / RESIDUAL_TICKS_PER_BEAT + 1e-9
        for event in codec.decode(bar['tokens'].tolist(), bar_start_beat=0.,
                                  meter=bar['bar_length_beats'] + tolerance):
            beat = bar['bar_start_beat'] + event.beat_position
            events.append((timing.beat_to_time(beat),
                           timing.beat_to_time(beat + event.duration_beats), int(event.kind)))
    return events


def chart_targets(times, events, *, tolerance_ms=25.0):
    """Soft onset labels (circle/slider/spinner) plus slider-active occupancy."""
    if tolerance_ms <= 0:
        raise ValueError('tolerance must be positive')
    result = torch.zeros(len(times), 4, dtype=torch.float32)
    # At very slow tempi the grid may be coarse. Ensure every in-bar onset can
    # supervise a neighbor, but retain a 25 ms tolerance on normal grids.
    step = float((times[1:] - times[:-1]).max()) if len(times) > 1 else 0.
    width = max(tolerance_ms, step)
    for start, end, kind in events:
        distance = (times - start).abs()
        pulse = (1. - distance / width).clamp_min(0.)
        result[:, kind] = torch.maximum(result[:, kind], pulse)
        if kind == int(EventKind.SLIDER):
            result[:,3] = torch.maximum(result[:,3], ((times >= start) & (times < end)).float())
    return result


def alignment_loss(logits, targets, mask):
    if logits.shape != targets.shape or mask.shape != logits.shape[:2]:
        raise ValueError('alignment logits/targets/mask shapes differ')
    if not torch.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any():
        raise ValueError('alignment targets must be finite probabilities')
    weight = mask.unsqueeze(-1)
    # Fixed positive weighting avoids batch-dependent objectives and unstable
    # ratios for empty/sparse bars. Tunable loss weight controls overall influence.
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets.float(),
                pos_weight=logits.new_tensor([5.,5.,5.,1.]).float(), reduction='none')
    return (losses * weight).sum() / (weight.sum().clamp_min(1) * logits.shape[-1])
