"""Matched original-audio Kaguya inference after an Amarel portability repeat."""
import argparse
import hashlib
import random
import zipfile
from pathlib import Path
import numpy as np
import torch
from src.dreamer_adapter.contracts import artifact_path, digest, write_json
from src.dreamer_adapter.evaluate import decode
from src.dreamer_adapter.metrics import native_metrics, summarize
from src.v2.evaluation import RosuDifficultyCalculator
from src.v2.mapspec import mapspec_from_osu


@torch.inference_mode()
def run(checkpoint, output, target_stars=4.6, candidates=4, fine_tuned_only=False, device="cuda", threads=1):
    from osu_dreamer.data.load_audio import make_spec, load_wave
    from osu_dreamer.models.inference.artifact import load_inference
    if not 0 < target_stars <= 15 or not 1 <= candidates <= 64:
        raise ValueError('invalid target difficulty or candidate count')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(threads)
    archive=artifact_path('/common/users/asj102/osu_project/1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size) [no video].osz')
    calc=RosuDifficultyCalculator();refs=[]
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('.osu'):
                text=z.read(name).decode('utf-8-sig');spec=mapspec_from_osu(text)
                if spec.general.mode==0:refs.append((abs(calc.calculate(text).stars-target_stars),name,text))
        _,name,text=min(refs);reference=mapspec_from_osu(text)
        audio=z.read(str(Path(name).parent/reference.general.audio_filename))
    expected='6ee5686d629635b982d5ccd97e4bcac5a7fa01969ab13c37587ef180b90bbe17'
    assert hashlib.sha256(audio).hexdigest()==expected,'original Kaguya audio changed'
    audio_path=output/'audio.mp3';audio_path.write_bytes(audio)
    features=torch.tensor(make_spec(load_wave(audio_path)),device=device,dtype=torch.float32)
    d=reference.difficulty;labels=torch.tensor([[target_stars,d.approach_rate,d.overall_difficulty,d.circle_size,d.hp_drain_rate]],device=device)
    result={}
    paths=[('finetuned',Path(checkpoint))] if fine_tuned_only else [('published',artifact_path('/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt')),('amarel-repeat',Path(checkpoint))]
    for name,path in paths:
        model=load_inference(path).to(device).eval();rows=[];texts={}
        for seed in range(1731,1731+candidates):
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            signals,settings=model.sample(features,labels,8)
            try:text=decode(signals[0].cpu().numpy(),settings[0].cpu().numpy(),name+' experimental')
            except (ZeroDivisionError,ValueError):text=''
            metrics=native_metrics(text,reference,target_stars)
            rows.append(dict(group='kaguya-development',seed=seed,metrics=metrics));texts[seed]=text
        measurable=[r for r in rows if r['metrics']['stars'] is not None and texts[r['seed']]]
        valid=[r for r in measurable if not r['metrics']['invalid']]
        measurable=valid or measurable
        chosen=min(measurable,key=lambda r:abs(r['metrics']['stars']-target_stars)) if measurable else None
        delivery=None
        if chosen:
            text=texts[chosen['seed']];stars=chosen['metrics']['stars']
            import re
            for key,value in [('Title','DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size)'),('Artist','Suzuki Masayuki'),('BeatmapID','0'),('BeatmapSetID','-1'),('Version',f'{name} {stars:.2f} stars ({target_stars:g} requested)')]:
                text=re.sub(r'(?m)^'+key+r'\s*:.*$',key+': '+value,text)
            (output/(name+'.osu')).write_text(text)
            packaged=output/(name+'.osz')
            with zipfile.ZipFile(packaged,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('audio.mp3',audio);z.writestr(name+'.osu',text)
            with zipfile.ZipFile(packaged) as z:assert z.testzip() is None and hashlib.sha256(z.read('audio.mp3')).hexdigest()==expected
            delivery=dict(seed=chosen['seed'],stars=stars,target_pass=abs(stars-target_stars)<=.2,sha256=digest(packaged),invalid=chosen['metrics']['invalid'])
        result[name]=dict(checkpoint_sha256=digest(path),summary=summarize(rows),rows=rows,selected=delivery)
        write_json(output/'comparison.json',dict(audio_sha256=expected,requested_stars=target_stars,target_tolerance=.2,results=result,pretraining_overlap='unknown',purpose='Amarel portability repeat; no model promotion'))
        print(name,result[name]['summary'],flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    audio_path.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--target-stars',type=float,default=4.6);p.add_argument('--candidates',type=int,default=4);p.add_argument('--fine-tuned-only',action='store_true');p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--threads',type=int,default=1);run(**vars(p.parse_args()))
