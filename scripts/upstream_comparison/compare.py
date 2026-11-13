"""Matched Kaguya inference; external models stay pinned and unchanged."""
import argparse,hashlib,json,random,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from src.dreamer_adapter.contracts import digest,write_json
from src.dreamer_adapter.evaluate import decode,score_signal
from src.dreamer_adapter.metrics import native_metrics,summarize
from src.v2.mapspec import mapspec_from_osu
from src.v2.evaluation import RosuDifficultyCalculator

@torch.inference_mode()
def run(root,output,dreamer_checkpoint=None,geometry_checkpoint=None):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    archive_path=Path('/common/users/asj102/osu_project/1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size) [no video].osz')
    calc=RosuDifficultyCalculator();references=[]
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.endswith('.osu'):
                text=archive.read(name).decode('utf-8-sig');spec=mapspec_from_osu(text)
                if spec.general.mode==0:references.append((abs(calc.calculate(text).stars-4.6),name,text,calc.calculate(text).stars))
        _,name,reference_text,reference_stars=min(references)
        reference=mapspec_from_osu(reference_text)
        audio=archive.read(str(Path(name).parent/reference.general.audio_filename))
    (output/'audio.mp3').write_bytes(audio);(output/'reference.osu').write_text(reference_text)
    from osu_dreamer.models.inference.artifact import load_inference
    from osu_dreamer.data.load_audio import make_spec,load_wave
    dreamer_path=dreamer_checkpoint or '/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt'
    dreamer=load_inference(dreamer_path).cuda().eval()
    features=torch.tensor(make_spec(load_wave(output/'audio.mp3')),device='cuda',dtype=torch.float32)
    d=reference.difficulty;labels=torch.tensor([[4.6,d.approach_rate,d.overall_difficulty,d.circle_size,d.hp_drain_rate]],device='cuda')
    sys.path.insert(0,str(root/'osu-diffusion'))
    from models import DiT_models
    from diffusion import create_diffusion
    from data_loading import beatmap_to_sequence,split_and_process_sequence_no_augment
    from export.create_beatmap import create_beatmap
    from slider import Beatmap
    gp=Path(geometry_checkpoint or root/'olibomby-ema.pt')
    state=torch.load(gp,map_location='cpu',weights_only=True)['ema'];classes=state['y_embedder.embedding_table.weight'].shape[0]-1
    geometry=DiT_models['DiT-B'](num_classes=classes,context_size=144).cuda().eval();geometry.load_state_dict(state)
    diffusion=create_diffusion('250',noise_schedule='squaredcos_cap_v2')
    reports={k:[] for k in ('dreamer','diffusion-human-rhythm','diffusion-dreamer-rhythm')};texts={k:{} for k in reports}
    for seed in (1731,1732,1733,1734):
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        signals,settings=dreamer.sample(features,labels,8)
        text=decode(signals[0].cpu().numpy(),settings[0].cpu().numpy(),f'Dreamer 4.6 request seed {seed}')
        texts['dreamer'][seed]=text
        (output/'input-generated.osu').write_text(text)
        for kind,path in [('dreamer',None),('diffusion-human-rhythm',output/'reference.osu'),('diffusion-dreamer-rhythm',output/'input-generated.osu')]:
            if path:
                torch.manual_seed(seed);bm=Beatmap.from_path(path);seq=beatmap_to_sequence(bm)
                (_,offset,condition),length=split_and_process_sequence_no_augment(seq)
                indices=torch.arange(length,device='cuda');mask=(indices[:,None]-indices[None,:]).abs()>128
                z=torch.randn(2,2,length,device='cuda');z[1]=z[0]
                kwargs=dict(o=(offset-offset[0])[None].repeat(2,1).cuda(),c=condition[None].repeat(2,1,1).cuda(),y=torch.full((2,),classes,device='cuda',dtype=torch.long),cfg_scale=1.,attn_mask=mask)
                sampled=diffusion.p_sample_loop(geometry.forward_with_cfg,z.shape,z,clip_denoised=True,model_kwargs=kwargs,device='cuda',progress=False)
                result=create_beatmap(torch.cat([sampled[0].cpu(),seq[2:]],0),bm,f'{kind} seed {seed}')
                temp=output/'temporary.osu';result.write_path(temp);text=temp.read_text();temp.unlink()
                texts[kind][seed]=text
            else:text=texts[kind][seed]
            metrics=native_metrics(text,reference,4.6)
            reports[kind].append({'group':'kaguya-development','seed':seed,'metrics':metrics})
            write_json(output/'progress.json',reports)
            print(json.dumps({'kind':kind,'seed':seed,'stars':metrics['stars'],'invalid':metrics['invalid']}),flush=True)
    summary={}
    for kind,rows in reports.items():
        selected=min(rows,key=lambda r:(r['metrics']['stars'] is None,abs((r['metrics']['stars'] or 0)-4.6)))
        seed=selected['seed'];text=texts[kind][seed];stars=selected['metrics']['stars']
        text=text.replace(reference.general.audio_filename,'audio.mp3')
        text=text.replace('AudioFilename: audio.mp3','AudioFilename: audio.mp3')
        # Preserve gameplay; replace audio declaration and display measured rather than promised difficulty.
        import re
        text=re.sub(r'(?m)^AudioFilename\s*:.*$','AudioFilename: audio.mp3',text)
        text=re.sub(r'(?m)^Version\s*:.*$',f'Version: {kind} {stars if stars is not None else "unmeasurable"} stars (4.6 requested)',text)
        (output/(kind+'.osu')).write_text(text)
        with zipfile.ZipFile(output/(kind+'.osz'),'w',zipfile.ZIP_DEFLATED) as z:z.writestr('audio.mp3',audio);z.writestr('map.osu',text)
        summary[kind]={'summary':summarize(rows),'selected_seed':seed,'selected_stars':stars,'target_pass':stars is not None and abs(stars-4.6)<=.2,'target_tolerance':.2,'osz_sha256':digest(output/(kind+'.osz'))}
    write_json(output/'comparison.json',{'requested_stars':4.6,'reference_version':reference.metadata.version,'reference_stars':reference_stars,'dreamer_checkpoint_sha256':digest(dreamer_path),'geometry_checkpoint_sha256':digest(gp),'audio_sha256':hashlib.sha256(audio).hexdigest(),'results':summary,'rows':reports,'caveat':'Human-rhythm geometry receives reference times, types and spacing. Its timing scores cannot measure audio-to-rhythm ability. Kaguya is development-only; pretraining overlap unknown.'})
    (output/'input-generated.osu').unlink();(output/'reference.osu').unlink();(output/'audio.mp3').unlink()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--dreamer-checkpoint');p.add_argument('--geometry-checkpoint');run(**vars(p.parse_args()))
