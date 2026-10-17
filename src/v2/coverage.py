"""Full-song output coverage, independent of a candidate's star rating.

Energy detects audible regions, not beats or mapper intention. Tolerances allow
short introductions and musical rests; long missing audible passages fail.
"""
from __future__ import annotations
from dataclasses import replace
import math


def active_intervals(features):
    # Known extractor layout: last four starts with relative RMS. Custom feature
    # pipelines fall back to full duration instead of guessing a channel.
    if features.rhythm_features.shape[1] != 101 or features.feature_cache.pipeline_revision != 'v2-multirate-audio-3':
        return ((0., features.duration_ms),)
    times=features.rhythm_times_ms.cpu().tolist()
    rms=features.rhythm_features[:,97].cpu().tolist()
    count = math.ceil(features.duration_ms/1000)
    totals = [0]*count
    active = [0]*count
    for t, value in zip(times, rms):
        second = int(t//1000)
        if 0 <= second < count:
            totals[second] += 1
            active[second] += int(value >= .02)
    intervals=[]
    for second in range(count):
        if totals[second] and active[second]>=max(1, .1*totals[second]):
            intervals.append((float(second*1000),min(features.duration_ms,float((second+1)*1000))))
    return tuple(intervals)


def policy_for_audio(policy, features):
    return replace(policy,audio_duration_ms=features.duration_ms,active_audio_intervals_ms=active_intervals(features))


def audit_coverage(timed, policy):
    duration=policy.audio_duration_ms
    if duration is None:
        return [], {}
    intervals=policy.active_audio_intervals_ms
    if intervals is None:
        intervals=((0.,duration),)
    if not timed:
        return ['no objects cover the audio'], {'audio_duration_ms':duration,'active_audio_coverage':0.}
    first=min(t[0] for t in timed); last=max(t[1] for t in timed)
    errors=[]
    if last>duration+100:
        errors.append('objects extend past the end of the audio')
    total=sum(b-a for a,b in intervals)
    # Each object/hold covers its interval and a small musical-rest allowance.
    ranges=sorted((max(0.,t[0]-3000),min(duration,t[1]+3000)) for t in timed)
    merged=[]
    for a,b in ranges:
        if merged and a<=merged[-1][1]: merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else: merged.append((a,b))
    covered=sum(max(0.,min(b,d)-max(a,c)) for a,b in intervals for c,d in merged)
    fraction=covered/total if total else 1.
    lead=max(0.,first-intervals[0][0]) if intervals else 0.
    tail=max(0.,intervals[-1][1]-last) if intervals else 0.
    if fraction<policy.minimum_audio_coverage:
        errors.append(f'audible-song coverage {fraction:.1%} is below {policy.minimum_audio_coverage:.1%}')
    if lead>policy.maximum_unmapped_edge_ms:
        errors.append(f'unmapped audible intro is {lead/1000:.1f}s')
    if tail>policy.maximum_unmapped_edge_ms:
        errors.append(f'unmapped audible outro is {tail/1000:.1f}s')
    return errors,dict(audio_duration_ms=duration,first_object_ms=first,last_object_end_ms=last,
                       active_audio_coverage=fraction,unmapped_intro_ms=lead,unmapped_outro_ms=tail)
