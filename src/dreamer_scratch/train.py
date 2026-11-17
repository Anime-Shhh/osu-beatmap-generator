"""Train all Dreamer components from random initialization on verified splits.

No published checkpoint is accepted. Parent weights must belong to this run.
"""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import random
import os
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer
from pytorch_lightning.loggers import CSVLogger
import yaml

from src.dreamer_adapter.contracts import digest, write_json
from src.dreamer_adapter.data import read_manifest, load_pair, selected_rows
from osu_dreamer.models.inference.artifact import dataclass_from_dict
from osu_dreamer.models.latent.model import LatentModelArgs
from osu_dreamer.models.latent.train import LatentTrainer
from osu_dreamer.models.diffusion.model import DiffusionModelArgs
from osu_dreamer.models.diffusion.train import DiffusionTrainer
from osu_dreamer.models.style.model import StyleModelArgs
from osu_dreamer.models.style.train import StyleTrainer
from osu_dreamer.common.lr_schedule import LRScheduleArgs
from osu_dreamer.data.modules.beatmap import pad_to_multiple

UPSTREAM = '0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc'
SCHEMA = 'dreamer-from-scratch-v1'


def signature(module):
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def check_checkpoint(path, binding, stage):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('scratch_binding') != binding or payload.get('scratch_stage') != stage:
        raise ValueError('checkpoint is not from this scratch experiment/stage')
    return payload


class Bound:
    def on_save_checkpoint(self, checkpoint):
        checkpoint['scratch_binding'] = self.scratch_binding
        checkpoint['scratch_stage'] = self.scratch_stage

    def log(self, name, value, *args, **kwargs):
        kwargs.setdefault('sync_dist', True)
        return super().log(name, value, *args, **kwargs)


class Codec(Bound, LatentTrainer):
    def plot_val(self, batch):
        pass  # CSV metrics; avoid TensorBoard-only plotting inside DDP.

    def on_validation_epoch_end(self):
        names = ['_on_pt', '_on_pp', '_on_tt', '_cur_res', '_cur_tot']
        v = torch.tensor([getattr(self, n) for n in names], device=self.device, dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(v)
        for n, x in zip(names, v.tolist()):
            setattr(self, n, x)
        super().on_validation_epoch_end()


class Generator(Bound, DiffusionTrainer):
    pass


class Style(Bound, StyleTrainer):
    def on_validation_epoch_end(self):
        if torch.distributed.is_initialized():
            # The frozen 24-source validation list divides evenly over 2/4 GPUs.
            self._val_s = [self.all_gather(torch.cat(self._val_s)).flatten(0, 1)]
            self._val_labels = [self.all_gather(torch.cat(self._val_labels)).flatten(0, 1)]
        super().on_validation_epoch_end()


def build(stage, upstream):
    folder = {'codec': 'latent', 'generator': 'diffusion', 'style': 'style'}[stage]
    cfg = yaml.safe_load((Path(upstream)/'osu_dreamer/models'/folder/'model.yml').read_text())['model']
    cfg = dict(cfg)
    cfg['schedule_args'] = dataclass_from_dict(LRScheduleArgs, cfg.get('schedule_args', {}))
    key, cls = {'codec': ('latent_args', LatentModelArgs), 'generator': ('diffusion_args', DiffusionModelArgs), 'style': ('style_args', StyleModelArgs)}[stage]
    cfg[key] = dataclass_from_dict(cls, cfg[key])
    return {'codec': Codec, 'generator': Generator, 'style': Style}[stage](**cfg)


class Charts(Dataset):
    def __init__(self, manifest, train, smoke=False):
        self.root=Path(manifest).parent; self.m=read_manifest(manifest); self.train=train
        self.rows=[r for r in self.m['rows'] if r['split']=='train'] if train else selected_rows(self.m,'val')
        if smoke: self.rows=self.rows[:8 if train else 4]
        self.groups={}
        for r in self.rows:self.groups.setdefault(r['group'],[]).append(r)
        self.keys=sorted(self.groups); self.smoke=smoke

    def __len__(self):
        return (64 if self.smoke else 32000) if self.train else len(self.rows)

    def __getitem__(self, i):
        r=random.choice(self.groups[random.choice(self.keys)]) if self.train else self.rows[i]
        a,x,l=load_pair(self.root,self.m,r)
        n=min(a.shape[-1],x.shape[-1]);a,x=a[:,:n],x[:,:n]
        if self.train or self.smoke:
            a,x=pad_to_multiple(a,2052),pad_to_multiple(x,2052)
            start=random.randrange(n-2052+1) if self.train and n>=2052 else 0
            a,x=a[:,start:start+2052],x[:,start:start+2052].clone()
        if self.train:
            if random.random()<.5:x[7]=1-x[7]
            if random.random()<.5:x[8]=1-x[8]
        return a,x,l


class Latents(Dataset):
    def __init__(self, root, stage, train, smoke=False):
        self.root=Path(root);self.stage=stage;self.train=train;self.smoke=smoke
        manifest=json.loads((self.root/'manifest.json').read_text())
        self.rows=[r for r in manifest['rows'] if r['split']==('train' if train else 'val')]
        self.groups={}
        for r in self.rows:self.groups.setdefault(r['group'],[]).append(r)
        self.keys=sorted(self.groups)

    def __len__(self):
        return (64 if self.smoke else 32000) if self.train else len(self.rows)

    def __getitem__(self,i):
        r=random.choice(self.groups[random.choice(self.keys)]) if self.train else self.rows[i]
        with np.load(self.root/r['file']) as f:
            s=torch.from_numpy(f['s'].copy()).float();labels=torch.from_numpy(f['labels'].copy()).float()
            if self.stage=='style': return torch.empty(0,1),torch.empty(0,1),s,labels
            h=torch.from_numpy(f['h'].copy()).float();z=torch.from_numpy(f['z'].copy()).float()
        if self.train or self.smoke:
            n=z.shape[-1];h,z=pad_to_multiple(h,152),pad_to_multiple(z,152)
            start=random.randrange(n-152+1) if n>=152 else 0
            h,z=h[:,start:start+152],z[:,start:start+152]
        return h,z,s,labels


class Progress(pl.Callback):
    def on_validation_end(self, trainer, module):
        if trainer.is_global_zero:
            print(json.dumps({'stage':module.scratch_stage,'step':trainer.global_step,'metrics':{k:float(v) for k,v in trainer.callback_metrics.items()}}),flush=True)


def fit(args, stage, binding):
    pl.seed_everything(args.seed,workers=True)
    model=build(stage,args.upstream)  # Constructors only: no pretrained-weight loader.
    stage_binding=dict(binding)
    if stage!='codec':stage_binding['codec_sha256']=digest(best(args.run,'codec'))
    model.scratch_binding=stage_binding;model.scratch_stage=stage
    out=Path(args.run)/stage;out.mkdir(parents=True,exist_ok=True)
    fresh=not (out/'last.ckpt').exists()
    if fresh:
        init={'stage':stage,'initialization':'random','parameter_sha256':signature(model),'parameters':sum(p.numel() for p in model.parameters()),'binding':binding}
        if not (out/'initialization.json').exists():
            if int(os.environ.get('LOCAL_RANK','0'))==0:write_json(out/'initialization.json',init)
        elif json.loads((out/'initialization.json').read_text())!=init:raise ValueError('initialization mismatch')
    else:check_checkpoint(out/'last.ckpt',stage_binding,stage)
    if stage!='codec':
        cache=json.loads((Path(args.run)/'latents/manifest.json').read_text())
        if cache['binding']!=binding or cache['codec_sha256']!=digest(best(args.run,'codec')):raise ValueError('latent cache provenance mismatch')
        for row in cache['rows']:
            if digest(Path(args.run)/'latents'/row['file'])!=row['sha256']:raise ValueError('cached latent changed')
    train=Charts(args.manifest,True,args.smoke) if stage=='codec' else Latents(Path(args.run)/'latents',stage,True,args.smoke)
    val=Charts(args.manifest,False,args.smoke) if stage=='codec' else Latents(Path(args.run)/'latents',stage,False,args.smoke)
    devices=torch.cuda.device_count()
    batch=2 if args.smoke else (8 if stage=='codec' else 32 if stage=='generator' else 128)
    effective=32 if stage=='codec' else 128 if stage=='generator' else 512
    if effective%(devices*batch):raise ValueError('GPU count does not divide effective batch')
    accumulation=1 if args.smoke else effective//(devices*batch)
    monitor={'codec':'eval/score','generator':'val/loss','style':'val/energy_dist'}[stage]
    mode='max' if stage=='codec' else 'min'
    checkpoint=ModelCheckpoint(dirpath=out,filename='best',monitor=monitor,mode=mode,save_top_k=1,save_last=True,enable_version_counter=False)
    callbacks=[checkpoint,Progress(),Timer(duration={'hours':47},interval='step')]
    if stage=='codec' and not args.smoke:callbacks.append(EarlyStopping(monitor=monitor,mode=mode,patience=10,min_delta=.001))
    steps=2 if args.smoke else {'codec':60000,'generator':60000,'style':20000}[stage]
    trainer=pl.Trainer(accelerator='gpu',devices=devices,strategy='ddp_find_unused_parameters_true' if devices>1 else 'auto',
        precision='bf16-mixed',max_steps=steps,max_epochs=-1,accumulate_grad_batches=accumulation,
        gradient_clip_val=1.,callbacks=callbacks,logger=CSVLogger(out,name='metrics'),
        enable_progress_bar=False,log_every_n_steps=1 if args.smoke else 50,
        num_sanity_val_steps=0,limit_train_batches=2 if args.smoke else 1.,
        check_val_every_n_epoch=1,deterministic=False)
    trainer.fit(model,DataLoader(train,batch_size=batch,num_workers=0 if args.smoke else 2,shuffle=True),
                DataLoader(val,batch_size=1,num_workers=0 if args.smoke else 1),ckpt_path=None if fresh else out/'last.ckpt')
    if trainer.is_global_zero:
        early=any(isinstance(c,EarlyStopping) and c.stopped_epoch>0 for c in callbacks)
        complete=trainer.global_step>=steps or early
        if not checkpoint.best_model_path:raise RuntimeError('no validation checkpoint')
        write_json(out/'result.json',dict(complete=complete,step=trainer.global_step,best=checkpoint.best_model_path,
            best_sha256=digest(checkpoint.best_model_path),binding=binding,smoke=args.smoke,
            initial_sha256=json.loads((out/'initialization.json').read_text())['parameter_sha256'],final_parameter_sha256=signature(model)))
        if not complete:raise RuntimeError('47-hour limit reached; resume this stage before advancing')


def best(run, stage):
    d=json.loads((Path(run)/stage/'result.json').read_text())
    if not d['complete']:raise ValueError('parent stage incomplete')
    p=Path(d['best'])
    if digest(p)!=d['best_sha256']:raise ValueError('parent checkpoint changed')
    return p


@torch.inference_mode()
def encode(args,binding):
    pl.seed_everything(args.seed,workers=True)
    model=build('codec',args.upstream)
    p=best(args.run,'codec');payload=check_checkpoint(p,binding,'codec')
    model.load_state_dict(payload['state_dict']);model=model.cuda().eval()
    root=Path(args.run)/'latents';root.mkdir(exist_ok=False)
    m=read_manifest(args.manifest);rows=[r for r in m['rows'] if r['split']=='train']+selected_rows(m,'val')
    if args.smoke:rows=[r for r in rows if r['split']=='train'][:8]+[r for r in rows if r['split']=='val'][:4]
    records=[];size=0
    for i,r in enumerate(rows):
        a,x,l=load_pair(Path(args.manifest).parent,m,r);n=min(a.shape[-1],x.shape[-1])
        if args.smoke:n=min(n,4104)
        a,x=pad_to_multiple(a[:,:n].cuda()[None],27),pad_to_multiple(x[:,:n].cuda()[None],27)
        z,s=model.latent.encode_chart(x);_,h=model.latent.audio_encoder(a)
        f=root/(r['sha256']+'.npz')
        np.savez_compressed(f,h=h[0].cpu().half().numpy(),z=z[0].cpu().half().numpy(),s=s[0].cpu().numpy(),labels=l.numpy())
        size+=f.stat().st_size
        if size>4*1024**3:raise ValueError('new latent cache exceeds 4 GiB; do not advance')
        records.append(dict(file=f.name,sha256=digest(f),split=r['split'],group=r['group']))
        if i%100==0:print(json.dumps({'encoded':i,'bytes':size}),flush=True)
    write_json(root/'manifest.json',dict(binding=binding,codec_sha256=digest(p),rows=records,bytes=size))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['codec','encode','generator','style','export'])
    for name in ('upstream','manifest','run'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=1731);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    # git binary is absent on Amarel compute nodes; read pinned checkout HEAD.
    git=Path(args.upstream)/'.git';head=(git/'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref=head[5:];rp=git/ref
        head=rp.read_text().strip() if rp.exists() else next(line.split()[0] for line in (git/'packed-refs').read_text().splitlines() if line.endswith(' '+ref))
    if head!=UPSTREAM:raise ValueError('upstream source revision mismatch')
    m=read_manifest(args.manifest,verify=True)
    binding=dict(schema=SCHEMA,upstream=UPSTREAM,manifest_sha256=digest(args.manifest),seed=args.seed,smoke=args.smoke,program_sha256=digest(__file__),config_sha256={name:digest(Path(args.upstream)/'osu_dreamer/models'/name/'model.yml') for name in ('latent','diffusion','style')})
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    bp=run/'binding.json'
    if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('run binding mismatch')
    if not bp.exists() and int(os.environ.get('LOCAL_RANK','0'))==0:write_json(bp,binding)
    if args.stage=='encode':encode(args,binding)
    elif args.stage=='export':
        paths=[best(run,s) for s in ('codec','generator','style')]
        parents=[]
        for path,stage in zip(paths,('codec','generator','style')):
            expected=dict(binding)
            if stage!='codec':expected['codec_sha256']=digest(paths[0])
            parents.append(check_checkpoint(path,expected,stage))
        codec,generator,style=parents
        keys=['emb_dim','style_dim','n_downs','stride','latent_args']
        hp={k:codec['hyper_parameters'][k] for k in keys}
        hp.update(diffusion_args=generator['hyper_parameters']['diffusion_args'],style_args=style['hyper_parameters']['style_args'])
        hp={k:asdict(v) if is_dataclass(v) else v for k,v in hp.items()}
        state={k:v for k,v in codec['state_dict'].items() if k.startswith('latent.')}
        for parent,prefix in [(generator,'diffusion'),(style,'style')]:
            stem=prefix+'_ema.module.'
            state.update({prefix+'.'+k.removeprefix(stem):v for k,v in parent['state_dict'].items() if k.startswith(stem)})
        torch.save(dict(hparams=hp,state_dict=state,from_scratch=binding),run/'scratch-inference.pt')
        from osu_dreamer.models.inference.artifact import load_inference
        for stage in ('codec','generator','style'):
            report=json.loads((run/stage/'result.json').read_text())
            if report['initial_sha256']==report['final_parameter_sha256']:raise ValueError('stage weights did not change')
        exported=load_inference(run/'scratch-inference.pt').cuda().eval()
        a,_,labels=load_pair(Path(args.manifest).parent,m,selected_rows(m,'val')[0])
        with torch.inference_mode():
            signal,out_labels=exported.sample(a[:,:2052].cuda(),labels[None].cuda(),8)
        if signal.shape!=(1,9,2052) or not torch.isfinite(signal).all() or not torch.isfinite(out_labels).all():raise ValueError('exported inference smoke failed')
        write_json(run/'export.json',dict(binding=binding,parents=[digest(p) for p in paths],sha256=digest(run/'scratch-inference.pt'),quality='not independently evaluated',pretrained_weights=False,inference_smoke='finite 2052-frame output'))
    else:fit(args,args.stage,binding)


if __name__=='__main__':main()
