"""Matched diagnostic evaluation of rhythm-flow checkpoints, without native geometry.

Thresholds are chosen on validation windows only. Test windows are fixed before
checkpoint evaluation; reported onset scores are not full-song beatmap scores.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from src.rhythm_flow.data import RhythmWindows
from src.rhythm_flow.model import RhythmFlow, RhythmFlowConfig, decode_rhythm, flow_loss, sample_full_song
from src.v2.inference import extract_audio_features, load_audio
from src.v2.preprocess import compact_profile


def match_onsets(predicted, reference, tolerance_ms):
    """Maximum-cardinality ordered one-to-one matching within a tolerance."""
    p = sorted(enumerate(predicted), key=lambda x: x[1]['start_ms'])
    r = sorted(enumerate(reference), key=lambda x: x[1]['start_ms'])
    i = j = 0
    pairs = []
    while i < len(p) and j < len(r):
        delta = p[i][1]['start_ms'] - r[j][1]['start_ms']
        if abs(delta) <= tolerance_ms:
            pairs.append((p[i][0], r[j][0])); i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return pairs


def reference_window(dataset, index):
    # Mirror the dataset's deterministic row/window selection, retaining native
    # event timestamps rather than recovering peaks from quantized target tensors.
    rng = random.Random(dataset.seed + dataset.epoch * 1000003 + index)
    group = dataset.groups[index // dataset.samples]
    row = group[rng.randrange(len(group))]
    audio = torch.load(dataset.root / row['audio_file'], weights_only=True)['audio']
    start = rng.randrange(max(1, len(audio) - dataset.window + 1))
    frames = min(dataset.window, len(audio) - start)
    events = [{'start_ms':float(s-start*10), 'end_ms':float(e-start*10), 'kind':int(k)}
              for s,e,k in row['events'] if start*10 <= s < (start+frames)*10]
    return {'source':row['source_component'], 'frames':frames, 'events':events,
            'chart_sha256':row['chart_sha256'], 'start_frame':start}


def score_windows(signals, references, threshold):
    counts = {t:Counter() for t in (25,50,100)}
    by_kind = {k:Counter() for k in range(3)}
    errors = []; hold_errors = []; group_counts = {}; gaps = []
    predicted_total = reference_total = 0; empty = 0; seconds = 0
    occupancy_intersection = torch.zeros(2); occupancy_union = torch.zeros(2)
    for signal, window in zip(signals, references):
        signal = signal[:,:window['frames']]
        predicted = decode_rhythm(signal, threshold=threshold)
        truth = window['events']; n = len(predicted); m = len(truth)
        predicted_total += n; reference_total += m; empty += not n
        seconds += window['frames']*.01
        gaps.extend(b['start_ms']-a['start_ms'] for a,b in zip(predicted,predicted[1:]))
        for tolerance, tally in counts.items():
            pairs = match_onsets(predicted, truth, tolerance)
            tally.update(tp=len(pairs), predicted=n, reference=m)
            if tolerance == 50:
                c = group_counts.setdefault(window['source'], Counter())
                c.update(tp=len(pairs), predicted=n, reference=m)
                for pi,ri in pairs:
                    p,r = predicted[pi],truth[ri]
                    tally['correct_kind'] += p['kind'] == r['kind']
                    errors.append(abs(p['start_ms']-r['start_ms']))
                    if p['kind']==r['kind'] and r['kind'] in (1,2):
                        hold_errors.append(abs((p['end_ms']-p['start_ms'])-(r['end_ms']-r['start_ms'])))
        for kind,tally in by_kind.items():
            p = [e for e in predicted if e['kind']==kind]
            r = [e for e in truth if e['kind']==kind]
            tally.update(tp=len(match_onsets(p,r,50)),predicted=len(p),reference=len(r))
        # Target tensors retain holds that began before the sampled window.
        target = window['target'][3:,:window['frames']] >= .5
        guessed = signal[3:] >= threshold
        occupancy_intersection += (target & guessed).sum(1)
        occupancy_union += (target | guessed).sum(1)
    def finish(c):
        p = c['tp']/c['predicted'] if c['predicted'] else 0.
        r = c['tp']/c['reference'] if c['reference'] else 0.
        return {**dict(c),'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
    result = {'threshold':threshold,'windows':len(references),'sampled_seconds':seconds,
              'onsets':{str(t):finish(c) for t,c in counts.items()},
              'kind_aware_50ms':{str(k):finish(c) for k,c in by_kind.items()},
              'kind_accuracy_among_50ms_matches':counts[50]['correct_kind']/max(1,counts[50]['tp']),
              'macro_source_f1_50ms':sum(finish(c)['f1'] for c in group_counts.values())/len(group_counts),
              'source_scores_50ms':{k:finish(c) for k,c in group_counts.items()},
              'predicted_events':predicted_total,'reference_events':reference_total,
              'predicted_events_per_second':predicted_total/seconds,
              'reference_events_per_second':reference_total/seconds,'empty_windows':empty,
              'occupancy_iou':{name:float(i/u) if u else None for name,i,u in zip(
                  ('slider','spinner'),occupancy_intersection,occupancy_union)}}
    for name,values in [('matched_onset_error_ms',errors),('matched_hold_duration_error_ms',hold_errors),('predicted_gap_ms',gaps)]:
        result[name] = {'count':len(values),'mean':sum(values)/len(values),
                        'median':float(torch.tensor(values).median())} if values else None
    return result


@torch.inference_mode()
def evaluate_windows(model, dataset, device, output, name, split):
    loader = DataLoader(dataset,batch_size=8,num_workers=0,shuffle=False)
    references = [reference_window(dataset,i) for i in range(len(dataset))]
    losses = Counter(); signals = []; targets = []; n = 0
    for i,b in enumerate(loader):
        audio,target,profile,mask = [b[k].to(device) for k in ('audio','target','profile','mask')]
        for mode in ('aligned','zero_audio','shift_1000ms'):
            a = audio if mode=='aligned' else (torch.zeros_like(audio) if mode=='zero_audio' else audio.roll(100,-1))
            gen = torch.Generator(device=device).manual_seed(1731+i)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = flow_loss(model,a,target,profile,mask,generator=gen)
            losses[mode] += float(loss)*len(audio)
        generated = sample_full_song(model,audio,profile,steps=32,
                                    generator=torch.Generator(device=device).manual_seed(91731+i)).cpu()
        signals.extend(generated.unbind()); targets.extend(b['target'].unbind()); n += len(audio)
        if i%8==0: print(json.dumps({'checkpoint':name,'split':split,'windows_completed':n}),flush=True)
    for window,target in zip(references,targets): window['target'] = target
    torch.save({'signals':signals,'windows':references},output/f'{name}-{split}-signals.pt')
    thresholds = [.2,.3,.4,.5,.6,.7,.8]
    grid = {str(t):score_windows(signals,references,t) for t in thresholds}
    chosen = max(thresholds,key=lambda t:grid[str(t)]['onsets']['50']['f1']) if split=='val' else None
    return {'loss':{k:v/n for k,v in losses.items()},'windows':n,
            'sources':len(dataset.groups),'threshold_grid':grid,'chosen_threshold':chosen,
            'window_identity_sha256':hashlib.sha256(json.dumps(
                [{k:w[k] for k in ('source','chart_sha256','start_frame','frames')} for w in references],sort_keys=True).encode()).hexdigest()}


def kaguya_summary(events,duration):
    starts = [e['start_ms'] for e in events]; ends = [e['end_ms'] for e in events]
    gaps = [b-a for a,b in zip(starts,starts[1:])]
    return {'events':len(events),'kinds':dict(Counter(str(e['kind']) for e in events)),
            'events_per_second':len(events)/(duration/1000),
            'first_event_ms':min(starts) if starts else None,
            'last_event_end_ms':max(ends) if ends else None,
            'unmapped_intro_ms':min(starts) if starts else duration,
            'unmapped_ending_ms':duration-max(ends) if ends else duration,
            'largest_start_gap_ms':max(gaps) if gaps else None,
            'median_start_gap_ms':float(torch.tensor(gaps).median()) if gaps else None,
            'zero_duration_holds':sum(e['kind']>0 and e['end_ms']==e['start_ms'] for e in events)}


@torch.inference_mode()
def generated_audio_controls(args):
    """Repeat test sampling with altered audio and exactly the original noise."""
    metrics=json.loads((args.output_dir/'metrics.json').read_text())
    result={}
    torch.set_num_threads(1)
    for name,path in [('trained',args.trained),('smoke',args.smoke)]:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==metrics['models'][name]['sha256']
        model=RhythmFlow(RhythmFlowConfig(**checkpoint['config'])).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        data=RhythmWindows(args.cache_dir,'test',samples_per_source=4)
        references=torch.load(args.output_dir/f'{name}-test-signals.pt',weights_only=True)['windows']
        result[name]={'aligned':metrics['models'][name]['test']['selected']}
        for mode in ('zero_audio','shift_1000ms'):
            signals=[]
            for i,b in enumerate(DataLoader(data,batch_size=8,num_workers=0,shuffle=False)):
                audio=b['audio'].cuda()
                audio=torch.zeros_like(audio) if mode=='zero_audio' else audio.roll(100,-1)
                generated=sample_full_song(model,audio,b['profile'].cuda(),steps=32,
                    generator=torch.Generator(device='cuda').manual_seed(91731+i)).cpu()
                signals.extend(generated.unbind())
            result[name][mode]=score_windows(signals,references,metrics['models'][name]['selected_threshold'])
            print(json.dumps({'checkpoint':name,'control':mode,
                'onsets_50ms':result[name][mode]['onsets']['50']}),flush=True)
        (args.output_dir/'generation-audio-controls.json').write_text(json.dumps(result,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trained',required=True); p.add_argument('--smoke',required=True)
    p.add_argument('--cache-dir',required=True); p.add_argument('--audio',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--generated-controls-only',action='store_true'); a=p.parse_args()
    if a.generated_controls_only:
        generated_audio_controls(a)
        return
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); device='cuda'; started=time.monotonic()
    waveform,sr = load_audio(a.audio); features,_ = extract_audio_features(waveform,sr)
    profile = compact_profile(target_star=4.5,approach_rate=9,circle_size=4,overall_difficulty=8,
                              hp_drain_rate=6,slider_multiplier=1.4,slider_tick_rate=1.)[None].to(device)
    audio = features.rhythm_features.T[None].to(device)
    results = {'protocol':{'frame_ms':10,'sampling_steps':32,'minimum_gap_ms':40,
        'window_frames':1024,'windows_per_source':4,'threshold_selection':'maximum validation micro F1 at 50 ms',
        'notes':['Window evaluation, not complete held-out songs. Windows may overlap.',
                 'Reference is one human map; alternative valid mappings count as errors.',
                 'Same noise, windows and precision for checkpoint comparisons.',
                 'Validation determines threshold; fixed 0.5 also reported.',
                 'No positions, native curves, realized stars or OSZ produced.']},'models':{}}
    for name,path in [('trained',a.trained),('smoke',a.smoke)]:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['feature_pipeline_key']!=features.feature_cache.pipeline_key: raise ValueError('pipeline mismatch')
        model=RhythmFlow(RhythmFlowConfig(**checkpoint['config'])).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        row={'checkpoint':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
             'global_step':checkpoint['global_step'],'training_manifest_sha256':checkpoint['manifest_sha256'],
             'parameters':sum(p.numel() for p in model.parameters()),'kaguya':[]}
        # Primary inference first, before slower held-out diagnostics.
        for seed in (1731,1732,1733):
            with torch.inference_mode():
                signal=sample_full_song(model,audio,profile,steps=32,
                    generator=torch.Generator(device=device).manual_seed(seed))[0].cpu()
            events=decode_rhythm(signal)
            item={'seed':seed,'threshold':.5,'audio_duration_ms':features.duration_ms,
                  'signal_frames':signal.shape[-1],**kaguya_summary(events,features.duration_ms),
                  'experimental':True,'native_geometry_generated':False,'events_detail':events}
            (a.output_dir/f'kaguya-{name}-{seed}-events.json').write_text(json.dumps(item,indent=2))
            torch.save({'signals':signal,'times_ms':features.rhythm_times_ms,
                        'audio_duration_ms':features.duration_ms},a.output_dir/f'kaguya-{name}-{seed}-rhythm.pt')
            row['kaguya'].append({k:v for k,v in item.items() if k!='events_detail'})
            print(json.dumps({'checkpoint':name,'kaguya':row['kaguya'][-1]}),flush=True)
        for split in ('val','test'):
            dataset=RhythmWindows(a.cache_dir,split,samples_per_source=4)
            if dataset.manifest['feature_pipeline_key']!=checkpoint['feature_pipeline_key']:raise ValueError('cache pipeline mismatch')
            row[split]=evaluate_windows(model,dataset,device,a.output_dir,name,split)
        threshold=row['val']['chosen_threshold']; row['selected_threshold']=threshold
        for split in ('val','test'):
            row[split]['selected']=row[split]['threshold_grid'][str(threshold)]
        primary=torch.load(a.output_dir/f'kaguya-{name}-1731-rhythm.pt',weights_only=True)['signals']
        calibrated=decode_rhythm(primary,threshold=threshold)
        (a.output_dir/f'kaguya-{name}-1731-calibrated-events.json').write_text(json.dumps({
            'threshold':threshold,'audio_duration_ms':features.duration_ms,'experimental':True,
            'native_geometry_generated':False,'summary':kaguya_summary(calibrated,features.duration_ms),
            'events':calibrated},indent=2))
        row['kaguya_calibrated']=kaguya_summary(calibrated,features.duration_ms)
        results['models'][name]=row
        (a.output_dir/'metrics.json').write_text(json.dumps(results,indent=2))
    for split in ('val','test'):
        assert results['models']['trained'][split]['window_identity_sha256']==results['models']['smoke'][split]['window_identity_sha256']
    results['seconds']=time.monotonic()-started
    (a.output_dir/'metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'complete':True,'seconds':results['seconds']}),flush=True)


if __name__=='__main__':main()
