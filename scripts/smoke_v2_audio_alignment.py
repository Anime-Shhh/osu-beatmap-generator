"""Bounded real-data GPU optimization check. Saves a report, never weights."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from src.v2.dataset import SparseSongDataset, collate_sparse_examples, move_batch_to_device
from src.v2.sparse_model import SparsePlannerConfig, SparseEventPlanner
from src.v2.training import CheckpointProvenance, initialize_planner, TrainingConfig, TrainingStage, loss_for_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--records-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);torch.manual_seed(2718)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    provenance=CheckpointProvenance.from_dict(payload['provenance'])
    config=replace(SparsePlannerConfig(**payload['planner_config']),local_audio_tokens=128)
    del payload
    planner=SparseEventPlanner(config)
    transferred=initialize_planner(args.checkpoint,planner,expected_provenance=provenance)
    planner=planner.to('cuda').train()
    ds=SparseSongDataset(args.records_dir,split='train',sampling_mode='bar',alignment_frames=128,
                        expected_pipeline_key=provenance.feature_pipeline_key)
    example=next(ds[i] for i in range(min(64,len(ds))) if ds[i]['target']['tokens'].numel()>2)
    batch=move_batch_to_device(collate_sparse_examples([example]),'cuda')
    optimizer=torch.optim.AdamW(planner.parameters(),lr=1e-5)
    training=TrainingConfig(stage=TrainingStage.PLANNER,profile_dropout=0,boundary_context_dropout=0)
    before=planner.local_audio_projection.weight.detach().clone()
    history=[];start=time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): losses=loss_for_batch(planner,None,batch,training)
        if not torch.isfinite(losses['loss']):raise FloatingPointError('nonfinite smoke loss')
        losses['loss'].backward()
        grad=torch.nn.utils.clip_grad_norm_(planner.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        history.append({**{k:float(v.detach()) for k,v in losses.items()},'gradient_norm':float(grad)})
    changed=not torch.equal(before,planner.local_audio_projection.weight)
    if not changed:raise AssertionError('new audio memory projection did not update')
    report={'kind':'three-step-training-path-check-not-quality-evaluation','source_component':example['metadata']['source_component_id'],
        'split':'train','history':history,'local_audio_projection_updated':changed,'weights_saved':False,
        'max_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,
        'initialization':transferred}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
