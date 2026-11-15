"""Independent V2/Dreamer rollouts on fixed audio; no human-clock conditioning."""
import argparse
import json
import random
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from src.dreamer_adapter.contracts import artifact_path, digest, write_json
from src.dreamer_adapter.evaluate import decode
from src.dreamer_adapter.metrics import native_metrics, summarize
from src.v2.data import SourceIdentity
from src.v2.generation import GenerationRequest
from src.v2.inference import load_v2_models, load_audio, extract_audio_features, _sample_candidate_map
from src.v2.mapspec import mapspec_from_osu, compile_osu, MapMetadata
from scripts.compare_native_rhythm import events_of
from scripts.evaluate_rhythm_flow import match_onsets


def checked_songs(output):
    base=artifact_path('/common/users/asj102/osu_project/v2_runs/run_20261014_real')
    manifest=json.loads((base/'data_300/splits.json').read_text())
    dreamer=json.loads(artifact_path('/common/home/asj102/osu_flow_runs/pilot_20261105/cache-compact/manifest.json').read_text())
    train_keys=set()
    for row in manifest['records']+dreamer['rows']:
        if row['split']=='train':train_keys.update(SourceIdentity.from_dict(row['source']).split_keys)
    songs=json.loads((base/'inference_20261014/inputs/heldout-songs.json').read_text())
    for song in songs:
        source=next(r['source'] for r in manifest['records'] if r['source']['audio_sha256']==song['audio_sha256'])
        if train_keys.intersection(SourceIdentity.from_dict(source).split_keys):
            raise ValueError('evaluation source overlaps a training split: '+song['id'])
        folder=base/'inference_20261014_v2'/song['id']
        song['audio']=str(folder/'audio.mp3');song['reference']=str(folder/'human-reference.osu')
        if digest(song['audio'])!=song['audio_sha256']:raise ValueError('evaluation audio hash mismatch')
        song.pop('oracle',None)
    archive=artifact_path('/common/users/asj102/osu_project/1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size) [no video].osz')
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith('.osu') and 'Confession' in n]
        if len(names)!=1:raise ValueError('ambiguous Kaguya reference')
        text=z.read(names[0]).decode('utf-8-sig');ref=mapspec_from_osu(text)
        audio=z.read(str(Path(names[0]).parent/ref.general.audio_filename))
    folder=output/'kaguya-input';folder.mkdir();(folder/'reference.osu').write_text(text);(folder/'audio.mp3').write_bytes(audio)
    sha=digest(folder/'audio.mp3')
    if sha!='6ee5686d629635b982d5ccd97e4bcac5a7fa01969ab13c37587ef180b90bbe17':raise ValueError('Kaguya audio mismatch')
    for row in manifest['records']+dreamer['rows']:
        if row['split']=='train' and (row['source']['audio_sha256']==sha or row['source'].get('beatmap_set_id')=='1147361'):
            raise ValueError('Kaguya training leakage')
    songs.append(dict(id='kaguya-development',audio=str(folder/'audio.mp3'),reference=str(folder/'reference.osu'),audio_sha256=sha,target_star=4.6))
    write_json(output/'songs.json',songs)
    return songs


def type_diagnostics(text,reference):
    if not text:return {'matches25':0,'correct_types25':0,'confusion25':{}}
    pred=events_of(mapspec_from_osu(text));truth=events_of(reference);pairs=match_onsets(pred,truth,25)
    confusion={}
    for i,j in pairs:
        key=f"{truth[j]['kind']}->{pred[i]['kind']}";confusion[key]=confusion.get(key,0)+1
    return dict(matches25=len(pairs),correct_types25=sum(pred[i]['kind']==truth[j]['kind'] for i,j in pairs),confusion25=confusion)


@torch.inference_mode()
def run(v2_checkpoint,dreamer_checkpoint,output):
    from osu_dreamer.data.load_audio import make_spec,load_wave
    from osu_dreamer.models.inference.artifact import load_inference
    root=Path(output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    songs=checked_songs(root);results={}
    checkpoints={'v2-retrained':Path(v2_checkpoint),'dreamer-published':artifact_path('/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt'),'dreamer-finetuned':Path(dreamer_checkpoint)}
    for name,path in checkpoints.items():
        model=load_v2_models(path,device='cuda') if name=='v2-retrained' else load_inference(path).cuda().eval()
        rows=[];deliveries=[]
        for song in songs:
            folder=root/name/song['id'];folder.mkdir(parents=True)
            reference=mapspec_from_osu(Path(song['reference']).read_text());d=reference.difficulty
            if name=='v2-retrained':
                wave,sr=load_audio(song['audio']);features,clocks=extract_audio_features(wave,sr,models=model)
            else:features=torch.tensor(make_spec(load_wave(song['audio'])),device='cuda',dtype=torch.float32)
            candidates=[]
            for seed in (1731,1732,1733,1734):
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);started=time.monotonic();text='';error=None
                try:
                    if name=='v2-retrained':
                        request=GenerationRequest(audio_filename='audio.mp3',target_star=song['target_star'],seed=seed,candidate_count=1,
                            ar=d.approach_rate,od=d.overall_difficulty,cs=d.circle_size,hp=d.hp_drain_rate,
                            feature_revision=features.feature_cache.pipeline_revision,model_revision=model.model_revision)
                        spec=_sample_candidate_map(planner=model.planner,geometry=model.geometry,features=features,hypothesis=clocks[0],
                            request=request,seed=seed,temperature=.9,geometry_steps=32,minimum_start_gap_ms=10.,
                            metadata=MapMetadata(title=reference.metadata.title,artist=reference.metadata.artist,creator='osu research',version=name,beatmap_id=0,beatmap_set_id=-1))
                        text=compile_osu(spec)
                    else:
                        labels=torch.tensor([[song['target_star'],d.approach_rate,d.overall_difficulty,d.circle_size,d.hp_drain_rate]],device='cuda')
                        signals,settings=model.sample(features,labels,8)
                        text=decode(signals[0].cpu().numpy(),settings[0].cpu().numpy(),name)
                except (ValueError,ZeroDivisionError) as exc:error=repr(exc)
                metrics=native_metrics(text,reference,song['target_star'])
                row=dict(group=song['id'],seed=seed,metrics=metrics,seconds=time.monotonic()-started,error=error,types=type_diagnostics(text,reference))
                rows.append(row)
                if text and metrics['stars'] is not None:candidates.append((abs(metrics['stars']-song['target_star']),seed,text,metrics))
                write_json(folder/'metrics.json',[r for r in rows if r['group']==song['id']])
                print(json.dumps(dict(model=name,song=song['id'],seed=seed,seconds=row['seconds'],f1_25=metrics['onsets']['25']['f1'])),flush=True)
            if candidates:
                _,seed,text,metrics=min(candidates,key=lambda c:(c[0],c[1]))
                (folder/'selected.osu').write_text(text)
                with zipfile.ZipFile(folder/'selected.osz','w',zipfile.ZIP_DEFLATED) as z:
                    z.writestr('selected.osu',text);z.write(song['audio'],'audio.mp3')
                deliveries.append(dict(song=song['id'],seed=seed,stars=metrics['stars'],target_pass=abs(metrics['stars']-song['target_star'])<=.2,invalid=metrics['invalid'],sha256=digest(folder/'selected.osz')))
        heldout=[r for r in rows if r['group']!='kaguya-development'];kaguya=[r for r in rows if r['group']=='kaguya-development']
        results[name]=dict(checkpoint_sha256=digest(path),heldout_summary=summarize(heldout),kaguya_summary=summarize(kaguya),rows=rows,selected=deliveries)
        write_json(root/'comparison.json',dict(results=results,protocol='independent audio-only, four fixed seeds, no reference clocks or chart history',pretraining_overlap='unknown for published Dreamer',limitations=['V2 and Dreamer have different training data and pretraining budgets.','Three previously used held-out sources are a development audit, not a fresh test.']))
        del model;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v2-checkpoint',required=True);p.add_argument('--dreamer-checkpoint',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
