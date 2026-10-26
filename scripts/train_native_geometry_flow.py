"""Train native event topology and geometry flow on authentic source-split maps."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from src.beatmap_flow.geometry import NativeGeometryFlow, GeometryConfig, geometry_loss
from src.beatmap_flow.geometry_data import GeometryWindows
from scripts.train_onset_flow import atomic_save

SCHEMA = 'osu-native-geometry-flow/1'


def load_model(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != SCHEMA: raise ValueError('invalid geometry checkpoint')
    model = NativeGeometryFlow(GeometryConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    return model.eval(), payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--val-batches', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1731)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1 or min(a.max_steps, a.val_batches, a.workers) < 0:
        p.error('invalid budgets')
    a.output_dir.mkdir(parents=True, exist_ok=bool(a.resume))
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    config = GeometryConfig()
    model = NativeGeometryFlow(config).to(device)
    ema = deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    train = GeometryWindows(a.cache_dir, 'train', seed=a.seed)
    val = GeometryWindows(a.cache_dir, 'val', samples_per_source=4)
    fingerprint = hashlib.sha256((a.cache_dir/'geometry-manifest.json').read_bytes()).hexdigest()
    contract = {'config': asdict(config), 'manifest_sha256': fingerprint,
                'seed': a.seed, 'batch_size': a.batch_size, 'epochs_budget': a.epochs}
    start_epoch, step, best, history = 0, 0, float('inf'), []
    if a.resume:
        old = torch.load(a.resume, weights_only=True, map_location='cpu')
        if old['schema'] != SCHEMA or old['contract'] != contract or not old['epoch_complete']:
            raise ValueError('incompatible resume')
        model.load_state_dict(old['raw_model']); ema.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        start_epoch, step, best, history = old['epoch'], old['step'], old['best'], old['history']
        torch.set_rng_state(old['rng_cpu']); torch.cuda.set_rng_state(old['rng_cuda'], device)
    def move(b): return {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
    started = time.monotonic()
    for epoch in range(start_epoch, a.epochs):
        train.epoch = epoch
        loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                            generator=torch.Generator().manual_seed(a.seed+epoch))
        model.train(); total, n, complete = {}, 0, True
        for i, batch in enumerate(loader):
            b = move(batch)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): losses = geometry_loss(model, b)
            if not torch.isfinite(losses['loss']): raise RuntimeError('nonfinite geometry loss')
            losses['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite geometry gradients')
            progress = step/max(1, a.epochs*len(loader))
            for group in opt.param_groups:
                group['lr'] = 2e-4*min(1., (step+1)/100)*(.1+.9*(1+math.cos(math.pi*progress))/2)
            opt.step()
            with torch.no_grad():
                decay = min(.999, (step+1)/(step+10))
                for e, m in zip(ema.parameters(), model.parameters()): e.lerp_(m, 1-decay)
            step += 1; n += len(b['kinds'])
            for k, v in losses.items(): total[k] = total.get(k, 0.)+float(v.detach())*len(b['kinds'])
            if step % 50 == 0:
                print(json.dumps({'epoch': epoch+1, 'step': step, 'loss': float(losses['loss'].detach()),
                                  'seconds': time.monotonic()-started}), flush=True)
            if a.max_steps and step >= a.max_steps:
                complete = i+1 == len(loader)
                break
        sums, count = {}, 0
        with torch.inference_mode():
            for i, batch in enumerate(DataLoader(val, batch_size=a.batch_size, num_workers=a.workers)):
                if a.val_batches and i >= a.val_batches: break
                b = move(batch)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = geometry_loss(ema, b, generator=torch.Generator(device=device).manual_seed(1731+i))
                for k, v in losses.items(): sums[k] = sums.get(k, 0.)+float(v)*len(b['kinds'])
                count += len(b['kinds'])
        row = {'epoch': epoch+1, 'step': step, 'train': {k: v/n for k, v in total.items()},
               'validation': {k: v/count for k, v in sums.items()}, 'seconds': time.monotonic()-started}
        history.append(row)
        improved = row['validation']['loss'] < best
        best = min(best, row['validation']['loss'])
        payload = {'schema': SCHEMA, 'config': asdict(config), 'contract': contract,
                   'manifest_sha256': fingerprint, 'feature_pipeline_key': train.manifest['feature_pipeline_key'],
                   'rhythm_manifest_sha256': train.manifest['rhythm_manifest_sha256'],
                   'model': ema.state_dict(), 'raw_model': model.state_dict(), 'optimizer': opt.state_dict(),
                   'epoch': epoch+1, 'epoch_complete': complete, 'step': step, 'best': best, 'history': history,
                   'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(device)}
        atomic_save(payload, a.output_dir/'last.pt')
        if improved: atomic_save(payload, a.output_dir/'best.pt')
        (a.output_dir/'history.json').write_text(json.dumps({'contract': contract, 'history': history,
            'parameters': sum(p.numel() for p in model.parameters()),
            'max_gpu_memory_bytes': torch.cuda.max_memory_allocated(device)}, indent=2))
        print(json.dumps(row), flush=True)
        if a.max_steps and step >= a.max_steps: break


if __name__ == '__main__': main()
