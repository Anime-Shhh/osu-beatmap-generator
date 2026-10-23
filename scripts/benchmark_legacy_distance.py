"""Bounded, real-cache legacy training verification; writes no checkpoint."""
import argparse
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from src.checkpoints import load_legacy_state_dict
from src.dataset import OsuCachedDataset, collate_fn
from src.model import OsuMapper
from src.splits import load_split_manifest, resolve_split_shards
from src.train import train_one_epoch_legacy, set_seed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--split-file',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    if a.batch_size<1:raise ValueError('batch-size must be positive')
    torch.set_num_threads(2);set_seed(1731)
    paths=resolve_split_shards(a.data_dir,load_split_manifest(a.split_file))['train']
    samples=[]
    for path in paths[::max(1,len(paths)//8)][:8]:
        dataset=OsuCachedDataset('',shards=[path],shuffle=False)
        samples.extend(itertools.islice(dataset,16))
    if len(samples)<a.batch_size*4:raise ValueError('insufficient cached training examples')
    # Mix source shards within batches rather than measuring only one easy intro.
    permutation=torch.randperm(len(samples)).tolist()
    samples=[samples[i] for i in permutation]
    batches=[collate_fn(samples[i:i+a.batch_size]) for i in range(0,len(samples),a.batch_size)]
    print(json.dumps({'phase':'cached_batches_ready','samples':len(samples),
        'max_tokens':max(len(s['tokens']) for s in samples),
        'valid_distance_targets':sum(int(b['distance_mask'].sum()) for b in batches)}),flush=True)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model=OsuMapper(**checkpoint.get('legacy_model_config',{})).cuda()
    report=load_legacy_state_dict(model,checkpoint['model_state_dict'],initialize_distance_head=True)
    del checkpoint
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    args=SimpleNamespace(teacher_forcing_start=1.,teacher_forcing_end=1.,teacher_forcing_decay_epochs=1,
        difficulty=-1.,w_discrete=.8,w_residual=.2,distance_loss_weight=.1,accum_steps=1,
        max_grad_norm=1.,log_every=8,dry_run=False,max_updates=0)
    # Separate initialization/warmup from the measured training interval.
    warmup=train_one_epoch_legacy(model,batches[:4],optimizer,scaler,0,args,torch.device('cuda'))
    torch.cuda.synchronize();started=time.monotonic()
    result=train_one_epoch_legacy(model,batches,optimizer,scaler,0,args,torch.device('cuda'))
    torch.cuda.synchronize();seconds=time.monotonic()-started
    if result['train/distance_target_count']<=0 or result['train/optimizer_updates']<=4:
        raise RuntimeError('benchmark did not verify auxiliary targets and completed updates')
    payload={'gpu':torch.cuda.get_device_name(),'batch_size':a.batch_size,'examples':len(samples),
        'seconds':seconds,'examples_per_second':len(samples)/seconds,
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'migration':report,'metrics':result}
    with a.output.open('x') as handle:json.dump(payload,handle,indent=2)
    print(json.dumps(payload),flush=True)


if __name__=='__main__':main()
