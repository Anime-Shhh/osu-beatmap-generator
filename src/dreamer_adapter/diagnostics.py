"""Reference-dependent diagnostic rewards; never used to update generator weights."""
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from .contracts import write_json,digest
from .data import read_manifest,selected_rows,load_pair
from scripts.compare_native_rhythm import compare,events_of
from src.v2.mapspec import mapspec_from_osu
from .timing import clock_candidate,replace_timing


def flux_alignment(events,spec):
    flux=np.maximum(0,np.diff(spec,axis=-1,prepend=spec[:,:1])).mean(0)
    scale=np.quantile(flux,.95)+1e-8
    values=[]
    for e in events:
        i=round(e['start_ms']*16384/98000)
        if 0<=i<len(flux):values.append(float(np.max(flux[max(0,i-2):min(len(flux),i+3)])/scale))
    return float(np.mean(values)) if values else 0.


def score_components(events,truth,timing,spec):
    result=compare(events,truth,timing)
    timing_score=result['onsets']['10']['f1']
    coverage=1-np.mean([abs(s['generated']-s['reference'])/max(1,s['generated']+s['reference']) for s in result['segments']])
    pairs=result['short_circle_pairs'];pair_score=pairs['both_as_circles_50ms']/max(1,pairs['reference'])
    # Reference-relative movement is deliberately not a deployable musicality reward.
    from scripts.evaluate_rhythm_flow import match_onsets
    matched=match_onsets(events,truth,50)
    move=np.mean([math.hypot(events[i]['x']-truth[j]['x'],events[i]['y']-truth[j]['y']) for i,j in matched]) if matched else 640
    duration=np.mean([abs((events[i]['end_ms']-events[i]['start_ms'])-(truth[j]['end_ms']-truth[j]['start_ms'])) for i,j in matched]) if matched else 1000
    starts=[e['start_ms'] for e in events]
    invalid=(len(set(starts))!=len(starts) or any(not 0<=e['x']<=512 or not 0<=e['y']<=384 or e['end_ms']<e['start_ms'] for e in events))
    return dict(timing=timing_score,coverage=float(coverage),patterns=pair_score,movement=1/(1+move/50),
                duration=1/(1+duration/50),validity=float(not invalid),audio=flux_alignment(events,spec))


def corruptions(truth):
    for shift in (-100,-50,-25,25,50,100):
        e=copy.deepcopy(truth)
        for x in e:x['start_ms']+=shift;x['end_ms']+=shift
        yield f'shift{shift}','timing',e
    yield 'deleted_notes','coverage',[copy.deepcopy(e) for i,e in enumerate(truth) if i%3]
    if truth:
        start=truth[len(truth)//2]['start_ms']
        yield 'deleted_phrase','coverage',[copy.deepcopy(e) for e in truth if not start<=e['start_ms']<start+8000]
        yield 'duplicates','validity',sorted(copy.deepcopy(truth)+copy.deepcopy(truth[::5]),key=lambda e:e['start_ms'])
        e=copy.deepcopy(truth)
        for x in e:x['x'],x['y']=256.,192.
        yield 'constant_position','movement',e
        e=copy.deepcopy(truth)
        for i,x in enumerate(e):x['x']=100. if i%2 else 400.;x['y']=192.
        yield 'broken_repeated_positions','movement',e
        e=copy.deepcopy(truth)
        for x in e:
            if x['kind']==1:x['end_ms']=x['start_ms']+2*(x['end_ms']-x['start_ms'])
        yield 'double_slider_durations','duration',e
        e=copy.deepcopy(truth)
        for x in e:
            if x['kind']==0:x['kind']=1;x['end_ms']=x['start_ms']+100
        yield 'broken_circle_pairs','patterns',e
        grid=[]
        for t in np.arange(truth[0]['start_ms'],truth[-1]['start_ms'],250):
            grid.append(dict(start_ms=float(t),end_ms=float(t),kind=0,x=256.,y=192.))
        yield 'constant_density','timing',grid


def diagnostics(manifest,output,beat_checkpoint=None,beat_checkpoint_sha256=None,seconds=14400):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=read_manifest(manifest,verify=True);rows=selected_rows(m,'val');cases=[];started=time.monotonic()
    for row in rows:
        a,_,_=load_pair(Path(manifest).parent,m,row)
        native=mapspec_from_osu(Path(row['native_path']).read_text(encoding='utf-8-sig'))
        truth=events_of(native);base=score_components(truth,truth,native.timing_map,a.numpy())
        for name,component,events in corruptions(truth):
            score=score_components(events,truth,native.timing_map,a.numpy())[component]
            eligible=name!='broken_circle_pairs' or compare(truth,truth,native.timing_map)['short_circle_pairs']['reference']>0
            eligible=eligible and (name!='double_slider_durations' or any(e['kind']==1 for e in truth))
            cases.append(dict(group=row['group'],corruption=name,component=component,original=base[component],
                              corrupted=score,eligible=eligible,detected=bool(score<base[component]-1e-6)))
        from dataclasses import replace
        from src.v2.mapspec import compile_osu
        from src.v2.evaluation import RosuDifficultyCalculator
        calc=RosuDifficultyCalculator()
        original_text=Path(row['native_path']).read_text(encoding='utf-8-sig')
        original_stars=calc.calculate(original_text).stars
        altered=replace(native,difficulty=replace(native.difficulty,circle_size=max(0,native.difficulty.circle_size-2)))
        changed_stars=calc.calculate(compile_osu(altered,validate=False)).stars
        error=abs(changed_stars-original_stars)
        cases.append(dict(group=row['group'],corruption='difficulty_cs_changed',component='difficulty',
                          original=0.,corrupted=-error,eligible=True,detected=error>1e-6))
        wrong=min((r for r in rows if r['group']!=row['group']),key=lambda r:abs(m['audio'][r['audio_key']]['frames']-a.shape[-1]))
        other=load_pair(Path(manifest).parent,m,wrong)[0]
        from .model import intervene
        score=flux_alignment(truth,intervene(a,'wrong',other).numpy())
        cases.append(dict(group=row['group'],corruption='wrong_song',component='audio',original=base['audio'],
                          corrupted=score,eligible=True,detected=score<base['audio']-1e-6))
    summary={}
    for name in sorted({c['corruption'] for c in cases}):
        subset=[c for c in cases if c['corruption']==name and c['eligible']]
        rate=sum(c['detected'] for c in subset)/max(1,len(subset))
        summary[name]={'eligible_sources':len(subset),'detection_rate':rate,'passed':len(subset)>=12 and rate>=.9}
    write_json(output/'reward-validation.json',{'complete':True,'training_reward_enabled':False,
        'limitations':'Reference-relative components are diagnostic only. No instrument-selection or musical-taste claim.',
        'components':summary,'cases':cases})
    if beat_checkpoint is None:
        write_json(output/'timing-validation.json',{'complete':False,'reason':'pinned beat checkpoint not supplied'});return
    if digest(beat_checkpoint)!=beat_checkpoint_sha256:raise ValueError('beat tracker weights changed')
    from beat_this.inference import Audio2Beats
    from scripts.materialize_v2_data import hf_sources
    from src.v2.inference import load_audio
    tracker=Audio2Beats(checkpoint_path=beat_checkpoint,device='cuda',dbn=False)
    wanted={r['audio_key']:r for r in rows};results=[]
    import hashlib
    for audio,_,_ in hf_sources(m['revision'],1500):
        if time.monotonic()-started>seconds:raise TimeoutError('diagnostic budget')
        key=hashlib.sha256(audio).hexdigest()
        if key not in wanted:continue
        path=output/'input.audio';path.write_bytes(audio)
        try:
            wave,sr=load_audio(path);beats,downbeats=tracker(wave[0].numpy(),sr)
        finally:path.unlink(missing_ok=True)
        clock=clock_candidate(np.asarray(beats)*1000,np.asarray(downbeats)*1000)
        row=wanted.pop(key);text=Path(row['native_path']).read_text(encoding='utf-8-sig');spec=mapspec_from_osu(text)
        segments=spec.timing_map.segments;constant=all(abs(s.beat_length_ms-segments[0].beat_length_ms)<1e-6 and s.meter==4 for s in segments)
        record={'group':row['group'],'clock':clock,'reference_constant_4_4':constant}
        if constant and clock.get('accepted'):
            record['period_relative_error']=abs(clock['beat_length_ms']/segments[0].beat_length_ms-1)
            offset=segments[0].start_ms
            error=(offset-clock['offset_ms'])/clock['beat_length_ms']
            record['phase_error_ms']=abs(error-round(error))*clock['beat_length_ms']
            try:replace_timing(text,clock);record['duration_roundtrip_pass']=True
            except ValueError as exc:record['duration_roundtrip_pass']=False;record['error']=str(exc)
        results.append(record)
        if not wanted:break
    write_json(output/'timing-validation.json',{'complete':not wanted,'checkpoint_sha256':beat_checkpoint_sha256,'rows':results})
