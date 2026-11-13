"""Bounded pretrained fine-tuning; fixed final update, no Kaguya model selection."""
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from src.dreamer_adapter.contracts import digest, write_json, require_space, bind
from src.dreamer_adapter.data import read_manifest, load_pair
from src.dreamer_adapter.model import load_upstream, pad
from src.dreamer_adapter.train import save_torch


def run(root, output, kind, steps=2000, seconds=5400, seed=1731, smoke=False):
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    require_space(); torch.set_num_threads(1); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.set_float32_matmul_precision('high')
    manifest=Path('/common/home/asj102/osu_flow_runs/pilot_20261105/cache-compact/manifest.json')
    m=read_manifest(manifest,verify=True); started=time.monotonic()
    training_rows=[r for r in m['rows'] if r['split']=='train']
    if smoke:training_rows=training_rows[:16]
    upstream='/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt'
    config=dict(kind=kind,smoke=smoke,seed=seed,max_steps=steps,max_seconds=seconds,batch=32,microbatch=8,lr=3e-5,
                precision='bf16',clip=1.,selection='fixed final update; no Kaguya selection',
                upstream_commit='0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc' if kind=='dreamer' else '532dc5d3026d23bf1578e7b1ba755ee4471b8d78')
    groups={}; failures=[]
    if kind=='dreamer':
        model=load_upstream(upstream,'cuda'); model.requires_grad_(False)
        net=model.diffusion.requires_grad_(True).train(); config.update(crop=152,ema=.99,codec='unchanged upstream RMS-normalized',weight_decay=.01)
        for i,r in enumerate(training_rows):
            if r['split']!='train':continue
            with torch.no_grad():
                a,x,_=load_pair(manifest.parent,m,r);valid=min(a.shape[-1],x.shape[-1])//27
                if valid<152:continue
                z,s=model.latent.encode_chart(pad(x[None].cuda()))
                _,h=model.latent.audio_encoder(pad(a[None].cuda()))
                item=(z[0,:,:valid].cpu().half(),h[0,:,:valid].cpu().half(),s[0].cpu().half())
            groups.setdefault(r['group'],[]).append(item)
            if i%100==0:print(json.dumps({'prepared':i,'seconds':time.monotonic()-started}),flush=True)
        initial=digest(upstream)
    else:
        sys.path.insert(0,str(root/'osu-diffusion'))
        from models import DiT_models
        from diffusion import create_diffusion
        from data_loading import beatmap_to_sequence,split_and_process_sequence,window_and_relative_time
        from slider import Beatmap
        p=root/'olibomby-ema.pt';state=torch.load(p,map_location='cpu',weights_only=True)['ema']
        classes=state['y_embedder.embedding_table.weight'].shape[0]-1
        net=DiT_models['DiT-B'](num_classes=classes,context_size=144).cuda();net.load_state_dict(state);del state
        # Existing beatmap-class IDs have unknown correspondence to this corpus.
        # Fine-tune generic/null-style generation, retaining and freezing the embedding table.
        net.y_embedder.requires_grad_(False);net.train();diffusion=create_diffusion('',noise_schedule='squaredcos_cap_v2')
        config.update(crop=128,ema=.9999,weight_decay=0.,conditioning='null style; real times/types/spacing',frozen='existing class embedding table')
        for i,r in enumerate(training_rows):
            if r['split']!='train':continue
            try:
                seq=beatmap_to_sequence(Beatmap.from_path(r['native_path']))
                if seq.shape[-1]<128:raise ValueError('fewer than 128 sequence elements')
                groups.setdefault(r['group'],[]).append(seq)
            except Exception as e:failures.append({'chart':r['sha256'],'error':repr(e)})
        initial=digest(p)
    if not groups:raise ValueError('no eligible training sources')
    config.update(sources=len(groups),charts=sum(map(len,groups.values())),initial_checkpoint_sha256=initial)
    write_json(output/'provenance.json',dict(config=config,binding=bind(m),conversion_failures=failures,gpu=torch.cuda.get_device_name(),cache='RAM only',kaguya_excluded=True))
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=config['lr'],weight_decay=config['weight_decay'])
    keys=sorted(groups); history=[]; step=0
    for step in range(1,steps+1):
        if time.monotonic()-started>seconds-120:
            step-=1;break
        opt.zero_grad(set_to_none=True);total=0.
        for _ in range(4):
            batch=[]
            for _ in range(8):
                item=random.choice(groups[random.choice(keys)])
                if kind=='dreamer':
                    z,h,s=item;start=random.randrange(z.shape[-1]-152+1)
                    batch.append((z[:,start:start+152],h[:,start:start+152],s))
                else:
                    seq,_=split_and_process_sequence(item);start=random.randrange(item.shape[-1]-128+1)
                    batch.append(window_and_relative_time(seq,start,start+128))
            values=[torch.stack([b[i] for b in batch]).cuda().float() for i in range(3)]
            if kind=='dreamer':
                target,h,s=values;noise=torch.randn_like(target)
                u=(torch.randperm(8,device='cuda')+torch.rand(8,device='cuda'))/8
                t=torch.special.ndtri(u.clamp(1e-6,1-1e-6)).sigmoid()[:,None,None]
                xt=torch.lerp(noise,target,t);dsq=(xt-target).square().sum(1).mean(1);utarget=(dsq+net.c0).sqrt()
                with torch.autocast('cuda',dtype=torch.bfloat16):distance,direction=net(h,s,xt)
                denoised=xt-distance[:,None,None]*direction
                loss=((denoised-target).square().sum(1).mean(1)/(dsq+net.c0)).mean()+30*(direction-(xt-target)/utarget[:,None,None]).square().sum(1).mean()
            else:
                x,o,c=values;t=torch.randint(0,diffusion.num_timesteps,(8,),device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=diffusion.training_losses(net,x,t,dict(o=o,c=c,y=torch.full((8,),classes,device='cuda',dtype=torch.long)))['loss'].mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            (loss/4).backward();total+=float(loss.detach())/4
        torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),net.parameters()):e.lerp_(p,1-config['ema'])
        if step%50==0 or step==steps:
            row=dict(step=step,loss=total,seconds=time.monotonic()-started);history.append(row);print(json.dumps(row),flush=True)
            write_json(output/'history.json',history)
    if step<1:raise RuntimeError('preparation exhausted time budget')
    if kind=='dreamer':
        model.diffusion.load_state_dict(ema.state_dict())
        artifact=torch.load(upstream,map_location='cpu',weights_only=True)
        artifact['state_dict']={k:v.cpu() for k,v in model.state_dict().items()}
        artifact['fine_tuning']=dict(config=config,binding=bind(m),step=step)
    else:artifact={'ema':{k:v.cpu() for k,v in ema.state_dict().items()},'fine_tuning':dict(config=config,binding=bind(m),step=step)}
    save_torch(output/'final.pt',artifact)
    save_torch(output/'last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,config=config,binding=bind(m),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()))
    write_json(output/'complete.json',dict(step=step,seconds=time.monotonic()-started,checkpoint_sha256=digest(output/'final.pt'),history=history,quality='not evaluated yet; no promotion'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['dreamer','geometry'],required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--seconds',type=int,default=5400);p.add_argument('--seed',type=int,default=1731);p.add_argument('--smoke',action='store_true');run(**vars(p.parse_args()))
