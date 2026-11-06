"""Change only timing metadata and SV, preserving generated starts and durations."""
import math
import numpy as np
from .metrics import events_of
from src.beatmap_flow.clock import fit_global_grid
from src.v2.mapspec import mapspec_from_osu


def clock_candidate(beats,downbeats):
    fit=fit_global_grid(beats)
    accepted=fit.get('accepted',False) and fit['median_residual_ms']<=10 and fit['p90_residual_ms']<=25
    # Beat fit alone cannot establish meter: require consecutive downbeats four fitted beats apart.
    db=np.asarray(downbeats,dtype=float)
    if len(db)<4:accepted=False
    else:
        bar=np.diff(db)/fit.get('beat_length_ms',1)
        accepted=accepted and bool(np.mean(np.abs(bar-4)<.15)>=.9)
    return {**fit,'accepted':bool(accepted),'meter':4 if accepted else None}


def replace_timing(text,clock):
    if not clock.get('accepted'):raise ValueError('clock needs review')
    old=mapspec_from_osu(text);before=events_of(old)
    period=float(clock['beat_length_ms']);offset=float(clock['offset_ms'])
    if not math.isfinite(period) or period<=0:raise ValueError('invalid beat length')
    sm=old.difficulty.slider_multiplier
    # Rewind the red line before every object, without changing phase.
    first=min((e['start_ms'] for e in before),default=0)
    origin=offset+math.floor((min(0,first)-offset)/period)*period
    lines=[f'{origin:.9f},{period:.12f},4,0,0,50,1,0']
    for e,timed in zip(old.events,before):
        if int(e.kind)!=1:continue
        duration=timed['end_ms']-timed['start_ms']
        sv=e.slider_length*e.slider_repeats*period/(duration*100*sm)
        if not .1<=sv<=10:raise ValueError('unsupported slider velocity')
        lines.append(f"{timed['start_ms']:.9f},{-100/sv:.12f},4,0,0,50,0,0")
    start=text.index('[TimingPoints]');end=text.index('[HitObjects]',start)
    out=text[:start]+'[TimingPoints]\n'+'\n'.join(lines)+'\n\n'+text[end:]
    after=events_of(mapspec_from_osu(out))
    if len(before)!=len(after) or any(abs(a[k]-b[k])>2 for a,b in zip(before,after) for k in ('start_ms','end_ms')):
        raise ValueError('timing replacement altered generated events')
    return out
