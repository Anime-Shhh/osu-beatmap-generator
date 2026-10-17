"""Train the experimental rhythm flow with source-separated validation and DDP.

Example: torchrun --nproc_per_node=2 -m scripts.train_rhythm_flow --cache-dir ...
No native beatmap quality claim follows from this training loss or a smoke run.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler
from src.rhythm_flow.model import RhythmFlow,RhythmFlowConfig,flow_loss
from src.rhythm_flow.data import RhythmWindows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=0,help='bounded smoke; 0 runs the epoch budget')
    p.add_argument('--width',type=int,default=96);p.add_argument('--layers',type=int,default=10)
    p.add_argument('--val-batches',type=int,default=0,help='0 evaluates all validation windows')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1731)
    a=p.parse_args()
    if min(a.epochs,a.batch_size)<1 or min(a.max_steps,a.val_batches)<0: p.error('invalid budgets')
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if a.device=='cuda' else a.device)
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    # All ranks check before rank zero creates the fresh run directory.
    exists=torch.tensor(int(a.output_dir.exists()),device=device)
    if world>1: dist.all_reduce(exists,op=dist.ReduceOp.MAX)
    if exists.item(): raise FileExistsError('use a fresh experimental run directory')
    if rank==0:a.output_dir.mkdir(parents=True)
    if world>1:dist.barrier()
    cfg=RhythmFlowConfig(width=a.width,layers=a.layers)
    raw=RhythmFlow(cfg).to(device);ema=deepcopy(raw).eval()
    model=DistributedDataParallel(raw,device_ids=[local] if device.type=='cuda' else None) if world>1 else raw
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    train=RhythmWindows(a.cache_dir,'train',window=cfg.window_frames)
    val=RhythmWindows(a.cache_dir,'val',window=cfg.window_frames,samples_per_source=4)
    sampler=DistributedSampler(train,seed=a.seed) if world>1 else None
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,shuffle=sampler is None,num_workers=0,
                      generator=torch.Generator().manual_seed(a.seed+rank))
    # Validation is rank-zero only, without DDP calls or padded duplicates.
    validation=DataLoader(val,batch_size=a.batch_size,num_workers=0)
    manifest_hash=hashlib.sha256((a.cache_dir/'manifest.json').read_bytes()).hexdigest()
    torch.manual_seed(a.seed+rank)
    step=0;best=float('inf');history=[];started=time.monotonic()
    def autocast(): return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    for epoch in range(a.epochs):
        train.epoch=epoch
        if sampler:sampler.set_epoch(epoch)
        model.train();total=0.;count=0
        for batch in loader:
            b={k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
            opt.zero_grad(set_to_none=True)
            with autocast():loss=flow_loss(model,b['audio'],b['target'],b['profile'],b['mask'])
            if not torch.isfinite(loss):raise RuntimeError('non-finite flow loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('non-finite flow gradients')
            for group in opt.param_groups:group['lr']=3e-4*min(1.,(step+1)/100)
            opt.step()
            with torch.no_grad():
                decay=min(.999,(step+1)/(step+10))
                for e,m in zip(ema.parameters(),raw.parameters()):e.lerp_(m,1-decay)
            step+=1;total+=float(loss.detach())*len(b['audio']);count+=len(b['audio'])
            if a.max_steps and step>=a.max_steps:break
        totals=torch.tensor([total,count],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(totals)
        if rank==0:
            scores={'aligned':0.,'zero_audio':0.,'shift_1000ms':0.};n=0
            with torch.inference_mode():
                for i,batch in enumerate(validation):
                    if a.val_batches and i>=a.val_batches:break
                    b={k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
                    for mode in scores:
                        audio=b['audio'] if mode=='aligned' else (torch.zeros_like(b['audio']) if mode=='zero_audio' else b['audio'].roll(100,-1))
                        gen=torch.Generator(device=device).manual_seed(a.seed+i)
                        with autocast(): vl=flow_loss(ema,audio,b['target'],b['profile'],b['mask'],generator=gen)
                        scores[mode]+=float(vl)*len(audio)
                    n+=len(b['audio'])
            row={'epoch':epoch+1,'step':step,'train_loss':float(totals[0]/totals[1]),
                 'validation':{k:v/n for k,v in scores.items()},'validation_windows':n,'seconds':time.monotonic()-started}
            history.append(row);print(json.dumps(row),flush=True)
            improved=row['validation']['aligned']<best
            if improved:
                best=row['validation']['aligned']
                payload={'schema':'osu-rhythm-flow/1','config':asdict(cfg),'model':ema.state_dict(),
                    'global_step':step,'manifest_sha256':manifest_hash,'feature_pipeline_key':train.manifest['feature_pipeline_key'],
                    'validation':row,'experimental':True,'seed':a.seed}
                torch.save(payload,a.output_dir/'best.pt.tmp');(a.output_dir/'best.pt.tmp').replace(a.output_dir/'best.pt')
            (a.output_dir/'history.json').write_text(json.dumps({'history':history,'parameters':sum(p.numel() for p in raw.parameters()),
                'world_size':world,'bounded_smoke':bool(a.max_steps),'max_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None},indent=2))
        if world>1:dist.barrier()
        if a.max_steps and step>=a.max_steps:break
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
