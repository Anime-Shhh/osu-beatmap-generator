import math
import statistics
import numpy as np
from scripts.compare_native_rhythm import compare, events_of
from scripts.evaluate_rhythm_flow import match_onsets
from scripts.summarize_kaguya_comparison import spatial_pairs
from src.v2.mapspec import mapspec_from_osu, validate_map_spec, ValidationPolicy
from src.v2.slider_path import event_path
from src.v2.evaluation import RosuDifficultyCalculator


def _native_metrics(text, reference, requested=4.5):
    spec=mapspec_from_osu(text);pred=events_of(spec);truth=events_of(reference)
    result=compare(pred,truth,reference.timing_map)
    typed={};errors=[];durations=[]
    for tolerance in (10,25,50):
        pairs=match_onsets(pred,truth,tolerance)
        typed[str(tolerance)]=2*sum(pred[i]['kind']==truth[j]['kind'] for i,j in pairs)/max(1,len(pred)+len(truth))
    for i,j in match_onsets(pred,truth,50):
        if pred[i]['kind']==truth[j]['kind']==1:
            durations.append(abs(pred[i]['end_ms']-pred[i]['start_ms']-truth[j]['end_ms']+truth[j]['start_ms']))
    report=validate_map_spec(spec,ValidationPolicy(require_beat_snap=False,allow_custom_samples=True))
    errors.extend(report.errors)
    for e in spec.events:
        try:path=event_path(e)
        except ValueError as exc:
            errors.append(str(exc));continue
        if np.any(path[:,0]<0) or np.any(path[:,0]>512) or np.any(path[:,1]<0) or np.any(path[:,1]>384):
            errors.append('rendered body outside playfield')
    stars=RosuDifficultyCalculator().calculate(text).stars
    period=reference.timing_map.beat_length_at_time(truth[0]['start_ms']) if truth else 500
    result.update(type_f1=typed,slider_duration_median_ms=statistics.median(durations) if durations else None,
                  slider_duration_errors_ms=durations,slider_duration_matches=len(durations),stars=stars,star_error=abs(stars-requested),
                  invalid=bool(errors),errors=sorted(set(errors)),spatial_pairs=spatial_pairs(pred,period))
    return result


def native_metrics(text, reference, requested=4.5):
    try:return _native_metrics(text,reference,requested)
    except (ValueError,TypeError,OverflowError,IndexError) as exc:
        # Invalid exports remain in the denominator; never silently discard a bad seed.
        result=compare([],events_of(reference),reference.timing_map)
        result.update(type_f1={str(t):0. for t in (10,25,50)},slider_duration_median_ms=None,
                      slider_duration_matches=0,slider_duration_errors_ms=[],stars=None,star_error=12.,
                      invalid=True,errors=[repr(exc)],spatial_pairs={})
        return result


def summarize(rows):
    if not rows:raise ValueError('empty evaluation')
    result={}
    for tolerance in ('10','25','50'):
        n=sum(r['metrics']['onsets'][tolerance]['matches'] for r in rows)
        p=sum(r['metrics']['generated']['objects'] for r in rows)
        t=sum(r['metrics']['reference']['objects'] for r in rows)
        result['f1_'+tolerance]=2*n/max(1,p+t)
        result['type_f1_'+tolerance]=sum(r['metrics']['type_f1'][tolerance] for r in rows)/len(rows)
    denom=sum(r['metrics']['short_circle_pairs']['reference'] for r in rows)
    result['pair_recall']=sum(r['metrics']['short_circle_pairs']['both_as_circles_50ms'] for r in rows)/max(1,denom)
    result['pair_reference_count']=denom
    result['star_error']=statistics.mean(r['metrics']['star_error'] for r in rows)
    result['invalid_rate']=statistics.mean(r['metrics']['invalid'] for r in rows)
    durations=[e for r in rows for e in r['metrics'].get('slider_duration_errors_ms',([r['metrics']['slider_duration_median_ms']] if r['metrics']['slider_duration_median_ms'] is not None else []))]
    result['slider_duration_median_ms']=statistics.median(durations) if durations else None
    result['songs']=len({r['group'] for r in rows})
    return result


def bootstrap(old,new,key='f1_25',samples=2000):
    import random
    oldgroups={};newgroups={}
    for rows,dest in ((old,oldgroups),(new,newgroups)):
        for r in rows:dest.setdefault(r['group'],[]).append(r)
    if oldgroups.keys()!=newgroups.keys():raise ValueError('unmatched sources')
    for k in oldgroups:
        if sorted(r['seed'] for r in oldgroups[k])!=sorted(r['seed'] for r in newgroups[k]):raise ValueError('unmatched seeds')
    keys=sorted(oldgroups);rng=random.Random(1731);values=[]
    for _ in range(samples):
        chosen=rng.choices(keys,k=len(keys))
        values.append(summarize([r for k in chosen for r in newgroups[k]])[key]-summarize([r for k in chosen for r in oldgroups[k]])[key])
    values.sort()
    return {'gain':summarize(new)[key]-summarize(old)[key],'ci95':[values[int(.025*samples)],values[int(.975*samples)]],'sources':len(keys)}
