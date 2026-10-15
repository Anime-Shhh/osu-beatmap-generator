"""Explicit, checkpoint-free GPU comparison on authentic V2 bar batches."""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.v2.dataset import SparseSongDataset, collate_sparse_examples
from src.v2.geometry import ActiveEventGeometryDiffuser, GeometryDiffusionConfig
from src.v2.sparse_model import SparseEventPlanner, SparsePlannerConfig
from src.v2.training import TrainingConfig, TrainingStage, make_optimizer, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--planner-config', help='JSON overrides for the architecture being compared')
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl')
    overrides = {}
    if args.planner_config:
        with open(args.planner_config) as handle:
            overrides = json.load(handle)
    planner_config = SparsePlannerConfig(**overrides)
    dataset = SparseSongDataset(args.records_dir, split='train', sampling_mode='song',
                                alignment_frames=planner_config.local_audio_tokens)
    # Identical batches across devices/configurations isolate GPU/DDP costs.
    indices = [i * (len(dataset) - 1) // 31 for i in range(32)]
    samples = [sample for i in indices
               if (sample := dataset[i])['target']['event_kinds'].numel() > 0]
    batches = [collate_sparse_examples([samples[(i * args.batch_size + j) % len(samples)]
                                       for j in range(args.batch_size)]) for i in range(args.steps)]
    planner = SparseEventPlanner(planner_config).to(device)
    geometry = ActiveEventGeometryDiffuser(GeometryDiffusionConfig(context_dim=768)).to(device)
    for stage in (TrainingStage.PLANNER, TrainingStage.GEOMETRY):
        config = TrainingConfig(stage=stage, mixed_precision='bf16',
                                gradient_accumulation=args.gradient_accumulation)
        optimizer = make_optimizer(planner, geometry, config)
        trained = planner if stage is TrainingStage.PLANNER else geometry
        wrapped = DDP(trained, device_ids=[local], find_unused_parameters=True) if world > 1 else trained
        p = wrapped if stage is TrainingStage.PLANNER else planner
        g = wrapped if stage is TrainingStage.GEOMETRY else geometry
        run_epoch(planner=p, geometry=g, loader=batches[:2], optimizer=optimizer,
                  device=device, config=config)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        started = time.monotonic()
        metrics = run_epoch(planner=p, geometry=g, loader=batches, optimizer=optimizer,
                            device=device, config=config)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        if world > 1:
            value = torch.tensor(seconds, device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            seconds = value.item()
        if rank == 0:
            print(json.dumps(dict(stage=stage.value, gpu=torch.cuda.get_device_name(local),
                                  world_size=world, batch_size=args.batch_size, steps=args.steps,
                                  local_audio_tokens=planner_config.local_audio_tokens,
                                  seconds=seconds, examples_per_second=args.steps * args.batch_size * world / seconds,
                                  peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                                  diagnostic_only=True, metrics=metrics)), flush=True)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        started = time.monotonic()
        run_epoch(planner=p, geometry=g, loader=batches, optimizer=None, device=device, config=config)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        if world > 1:
            value = torch.tensor(seconds, device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            seconds = value.item()
        if rank == 0:
            print(json.dumps(dict(stage=stage.value, mode='validation', world_size=world,
                                  examples_per_second=args.steps * args.batch_size * world / seconds)), flush=True)
        if stage is TrainingStage.GEOMETRY and world > 1:
            empty_batch = dict(batches[0])
            if rank == 0:
                for name in ('event_mask', 'event_kinds', 'anchor_counts', 'event_positions',
                             'event_token_indices', 'geometry', 'event_features'):
                    empty_batch[name] = empty_batch[name][:, :0]
            run_epoch(planner=p, geometry=g, loader=[empty_batch, batches[0]],
                      optimizer=optimizer, device=device, config=config)
            dist.barrier()
            if rank == 0:
                print(json.dumps(dict(ddp_empty_rank_backward='passed', world_size=world)), flush=True)
        del wrapped, optimizer
        torch.cuda.empty_cache()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
