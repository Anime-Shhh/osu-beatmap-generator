import copy
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from .contracts import bind,digest,write_json,require_space,validate_binding
from .data import Crops,read_manifest,load_pair
from .model import Codec,load_upstream,load_codec,pad
from .evaluate import evaluate


def save_torch(path,state):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);os.replace(tmp,path)


def codec_score(summary):
    duration=summary['slider_duration_median_ms']
    return (summary['f1_10']+summary['type_f1_10']+summary['pair_recall'])/3-min(100,duration if duration is not None else 100)/1000


def train_codec(manifest,upstream,output,kind='ae',beta=0.,seed=1731,max_steps=10000,
                seconds=14400,evaluate_every=500,microbatch=4,resume=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=resume)
    torch.set_num_threads(1);torch.manual_seed(seed)
    m=read_manifest(manifest,verify=True);model=load_upstream(upstream,'cuda')
    codec=Codec(model.latent,kind).cuda().train()
    config=dict(kind=kind,beta=beta,seed=seed,lr=1e-4,batch=32,crop=2052,
                microbatch=microbatch,precision='bf16',clip=1.,weight_decay=.01,
                kl_ramp=2000,mask_fraction=.25,style_noise=.2,evaluate_every=evaluate_every)
    optimizer=torch.optim.AdamW((p for p in codec.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
    start_step=0;best=-float('inf');stale=0;history=[]
    if resume:
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        validate_binding(state['binding'],bind(m))
        if state['config']!=config:raise ValueError('resume config differs')
        codec.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_step=state['step'];best=state['best'];stale=state['stale'];history=state['history']
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    dataset=Crops(manifest,seed=seed)
    loader=iter(DataLoader(dataset,batch_size=microbatch,sampler=range(start_step*32,len(dataset)),num_workers=0))
    if 32%microbatch:raise ValueError('microbatch must divide 32')
    started=time.monotonic();stop=False
    for step in range(start_step+1,max_steps+1):
        if time.monotonic()-started>seconds-180:break
        optimizer.zero_grad(set_to_none=True)
        for _ in range(32//microbatch):
            a,x,_=next(loader);a,x=a.cuda(),x.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,details=codec.loss(a,x,step,beta)
            if not torch.isfinite(loss):raise ValueError('nonfinite codec loss')
            (loss/(32//microbatch)).backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(),1);optimizer.step()
        if step%50==0:print(json.dumps({'step':step,**details,'seconds':time.monotonic()-started}),flush=True)
        if step%evaluate_every==0 or step==max_steps:
            state=dict(schema='dreamer-codec/1',binding=bind(m),config=config,step=step,model=codec.state_dict(),
                       optimizer=optimizer.state_dict(),best=best,stale=stale,history=history,
                       rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            save_torch(output/'last.pt',state)
            report=evaluate(manifest,upstream,output/f'validation-{step}',codec_path=output/'last.pt',
                            modes=('reconstruction',),conditions=('aligned',),seconds=max(60,seconds-(time.monotonic()-started)))
            summary=report['summaries']['reconstruction/aligned'];score=codec_score(summary)
            history.append({'step':step,'score':score,'summary':summary,'seconds':time.monotonic()-started})
            if score>best+1e-4:
                best=score;stale=0
                # Weight-only best keeps the artifact/codec identity immutable until a true improvement.
                save_torch(output/'best.pt',{k:v for k,v in state.items() if k not in ('optimizer','rng','cuda_rng','history','best','stale')})
            else:stale+=1
            state.update(best=best,stale=stale,history=history);save_torch(output/'last.pt',state)
            write_json(output/'history.json',{'config':config,'history':history})
            codec.train()
            if stale>=4:stop=True
        if stop:break
    if not (output/'best.pt').exists():raise RuntimeError('budget ended before first validated checkpoint')
    write_json(output/'complete.json',{'validated_step':history[-1]['step'],'seconds':time.monotonic()-started,'early_stop':stop})


def latent_cache(manifest,upstream,codec_path,output):
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    m=read_manifest(manifest);binding=bind(m,digest(codec_path));meta=output/'manifest.json'
    if meta.exists():
        old=json.loads(meta.read_text());validate_binding(old['binding'],binding)
        for r in old['rows']:
            if digest(output/r['path'])!=r['sha256']:raise ValueError('latent cache changed')
        return old
    binding_file=output/'binding.json'
    if binding_file.exists():validate_binding(json.loads(binding_file.read_text()),binding)
    else:write_json(binding_file,binding)
    cache_bytes=sum(p.stat().st_size for d in output.parent.iterdir() if d.is_dir() and (d.name=='cache-compact' or d.name.endswith('-latents')) for p in d.rglob('*') if p.is_file())
    model=load_upstream(upstream,'cuda');codec,_=load_codec(model,codec_path,m,'cuda');codec.eval()
    rows=[];total=torch.zeros(6,dtype=torch.float64);squared=total.clone();count=0
    with torch.inference_mode():
        for r in m['rows']:
            if r['split']!='train':continue
            a,x,l=load_pair(Path(manifest).parent,m,r);a,x=pad(a.cuda()[None]),pad(x.cuda()[None])
            mu,lv,s=codec.encode(x);_,h=codec.latent.audio_encoder(a)
            # Cache posterior moments; sample VAE latents during generator training.
            # Exclude the incomplete final temporal chunk from targets and statistics.
            valid=m['audio'][r['audio_key']]['frames']//27
            mu,lv,h=mu[0,:,:valid],lv[0,:,:valid],h[0,:,:valid]
            total+=mu.double().sum(-1).cpu()
            second=mu.double().square()+(lv.double().exp() if codec.kind=='vae' else 0)
            squared+=second.sum(-1).cpu();count+=mu.shape[-1]
            path=output/(r['sha256']+'.npz')
            old_size=path.stat().st_size if path.exists() else 0
            np.savez_compressed(path,mu=mu.cpu().numpy().astype('float16'),lv=lv.cpu().numpy().astype('float16'),
                                h=h.cpu().numpy().astype('float16'),s=s[0].cpu().numpy().astype('float16'))
            cache_bytes+=path.stat().st_size-old_size
            if cache_bytes>4*1024**3:
                path.unlink();raise RuntimeError('combined compact and latent caches exceed 4 GiB')
            rows.append({'group':r['group'],'path':path.name,'sha256':digest(path),'frames':mu.shape[-1]})
    mean=total/count;std=(squared/count-mean.square()).clamp_min(1e-6).sqrt()
    result={'binding':binding,'kind':codec.kind,'normalization':{'mean':mean.tolist(),'std':std.tolist()},'rows':rows}
    write_json(meta,result);return result


def train_generator(manifest,upstream,codec_path,cache,output,seed=1731,max_steps=20000,seconds=21600,
                    evaluate_every=1000,microbatch=8,resume=False):
    import numpy as np
    import random
    output=Path(output);output.mkdir(parents=True,exist_ok=resume)
    torch.set_num_threads(1);torch.manual_seed(seed);rng=random.Random(seed)
    started=time.monotonic();m=read_manifest(manifest,verify=True)
    encoded=latent_cache(manifest,upstream,codec_path,cache);stats=encoded['normalization']
    model=load_upstream(upstream,'cuda');net=model.diffusion.train();ema=copy.deepcopy(net).eval().requires_grad_(False)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=.01)
    config=dict(seed=seed,lr=3e-5,batch=32,crop=152,codec_kind=encoded['kind'],
                microbatch=microbatch,precision='bf16',clip=1.,weight_decay=.01,
                ema=.99,evaluate_every=evaluate_every,osl_weight=1.,direction_weight=30.)
    binding=bind(m,digest(codec_path),stats);history=[];best=-1;start_step=0
    if resume:
        st=torch.load(output/'last.pt',map_location='cpu',weights_only=True);validate_binding(st['binding'],binding)
        if st['config']!=config:raise ValueError('resume config differs')
        net.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer'])
        start_step=st['step'];history=st['history'];best=st['best'];rng.setstate(st['python_rng'])
        torch.set_rng_state(st['rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
    groups={}
    for r in encoded['rows']:
        if r['frames']>=152:groups.setdefault(r['group'],[]).append(r)
    keys=sorted(groups)
    mean=torch.tensor(stats['mean'],device='cuda')[None,:,None];std=torch.tensor(stats['std'],device='cuda')[None,:,None]
    if 32%microbatch:raise ValueError('microbatch must divide32')
    for step in range(start_step+1,max_steps+1):
        if time.monotonic()-started>seconds-300:break
        opt.zero_grad(set_to_none=True)
        for _ in range(32//microbatch):
            batch=[]
            for _ in range(microbatch):
                r=rng.choice(groups[rng.choice(keys)]);start=rng.randrange(r['frames']-152+1)
                with np.load(Path(cache)/r['path']) as f:
                    batch.append([torch.tensor(f[k][:,start:start+152].copy()) for k in ('mu','lv','h')]+[torch.tensor(f['s'].copy())])
            mu,lv,h,s=[torch.stack([b[i] for b in batch]).cuda().float() for i in range(4)]
            target=mu+torch.exp(.5*lv)*torch.randn_like(mu) if encoded['kind']=='vae' else mu
            target=(target-mean)/std
            noise=torch.randn_like(target);u=(torch.randperm(microbatch,device='cuda')+torch.rand(microbatch,device='cuda'))/microbatch
            t=torch.special.ndtri(u.clamp(1e-6,1-1e-6)).sigmoid()[:,None,None]
            xt=torch.lerp(noise,target,t);dsq=(xt-target).square().sum(1).mean(1);utarget=(dsq+net.c0).sqrt()
            with torch.autocast('cuda',dtype=torch.bfloat16):distance,direction=net(h,s,xt)
            denoised=xt-distance[:,None,None]*direction
            osl=((denoised-target).square().sum(1).mean(1)/(dsq+net.c0)).mean()
            directional=(direction-(xt-target)/utarget[:,None,None]).square().sum(1).mean()
            loss=osl+30*directional
            if not torch.isfinite(loss):raise ValueError('nonfinite generator loss')
            (loss/(32//microbatch)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),net.parameters()):e.lerp_(p,.01)
        if step%50==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-started}),flush=True)
        if step%evaluate_every==0 or step==max_steps:
            st=dict(schema='dreamer-generator/1',binding=binding,normalization=stats,config=config,step=step,
                    model=net.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),history=history,best=best,
                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=rng.getstate())
            save_torch(output/'last.pt',st)
            report=evaluate(manifest,upstream,output/f'validation-{step}',codec_path=codec_path,generator_path=output/'last.pt',
                            modes=('generation',),conditions=('aligned',),seconds=max(60,seconds-(time.monotonic()-started)))
            score=report['summaries']['generation/aligned']['f1_25']
            history.append({'step':step,'summary':report['summaries']['generation/aligned'],'seconds':time.monotonic()-started})
            if score>best:
                best=score;save_torch(output/'best.pt',{k:v for k,v in st.items() if k not in ('optimizer','rng','cuda_rng','python_rng','history','best','model')})
            st.update(history=history,best=best);save_torch(output/'last.pt',st)
            write_json(output/'history.json',{'config':config,'history':history})
            net.train()
    if not (output/'best.pt').exists():raise RuntimeError('no validated generator checkpoint')
    write_json(output/'complete.json',{'validated_step':history[-1]['step'],'seconds':time.monotonic()-started})
