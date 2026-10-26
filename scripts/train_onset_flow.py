"""Single-GPU source-separated onset-flow training with resumable snapshots."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from src.beatmap_flow.data import OnsetWindows
from src.beatmap_flow.rhythm import OnsetFlow, RhythmConfig, rhythm_loss

SCHEMA = 'osu-onset-flow/1'


def atomic_save(payload, path):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if checkpoint['schema'] != SCHEMA:
        raise ValueError('incompatible onset model')
    model = OnsetFlow(RhythmConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--initialize', type=Path, help='explicit weight transfer; new optimizer and experiment')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--context-layers', type=int, default=0)
    p.add_argument('--density-auxiliary', action='store_true')
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--val-batches', type=int, default=0)
    p.add_argument('--seed', type=int, default=1731)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.resume and a.initialize: p.error('resume and initialize are different operations')
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0: p.error('invalid learning rate')
    if min(a.epochs, a.batch_size, a.window) < 1 or min(a.workers, a.max_steps, a.val_batches) < 0:
        p.error('invalid training budgets')
    if a.output_dir.exists() and not a.resume:
        raise FileExistsError('new experiments require a fresh output directory')
    a.output_dir.mkdir(parents=True, exist_ok=bool(a.resume))
    device = torch.device('cuda:0' if a.device == 'cuda' else a.device)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    config = RhythmConfig(width=a.width, window_frames=a.window, overlap_frames=a.window//4,
                          context_layers=a.context_layers, density_auxiliary=a.density_auxiliary)
    raw = OnsetFlow(config).to(device)
    initialization = None
    if a.initialize:
        previous, weights = load_model(a.initialize)
        old_config = asdict(previous.config)
        changed = {k for k, v in asdict(config).items() if old_config[k] != v}
        if changed-{'context_layers', 'context_stride', 'density_auxiliary'}:
            raise ValueError('weight transfer cannot change existing architecture dimensions')
        if previous.config.context_layers and previous.config.context_layers != config.context_layers:
            raise ValueError('cannot silently truncate existing context layers')
        missing, unexpected = raw.load_state_dict(weights['model'], strict=False)
        if unexpected or any(not k.startswith(('phrase_context.', 'density_aux.')) for k in missing):
            raise ValueError('incompatible transfer state')
        initialization = {'checkpoint_sha256': hashlib.sha256(a.initialize.read_bytes()).hexdigest(),
            'manifest_sha256': weights['manifest_sha256'], 'feature_pipeline_key': weights['feature_pipeline_key'],
            'new_parameters': missing}
        del previous, weights
    ema = deepcopy(raw).eval()
    opt = torch.optim.AdamW(raw.parameters(), lr=a.learning_rate, weight_decay=.01)
    train = OnsetWindows(a.cache_dir, 'train', window=a.window, seed=a.seed)
    val = OnsetWindows(a.cache_dir, 'val', window=a.window, samples_per_source=4, seed=1731)
    fingerprint = hashlib.sha256((a.cache_dir/'manifest.json').read_bytes()).hexdigest()
    if initialization is not None:
        if initialization['feature_pipeline_key'] != train.manifest['feature_pipeline_key']:
            raise ValueError('weight transfer requires identical feature extraction')
        if initialization['manifest_sha256'] not in (fingerprint, train.manifest.get('parent_manifest_sha256')):
            raise ValueError('weight transfer requires the same corpus or its frozen-split expansion')
    contract = {'config': asdict(config), 'manifest_sha256': fingerprint,
                'seed': a.seed, 'batch_size': a.batch_size, 'epochs_budget': a.epochs,
                'learning_rate': a.learning_rate}
    start_epoch, step, best, history = 0, 0, float('inf'), []
    if a.resume:
        old = torch.load(a.resume, weights_only=True, map_location='cpu')
        if old['schema'] != SCHEMA or old['contract'] != contract or not old['epoch_complete']:
            raise ValueError('resume requires matching contract and a completed epoch')
        raw.load_state_dict(old['raw_model'])
        initialization = old.get('initialization')
        ema.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        start_epoch, step, best, history = old['epoch'], old['step'], old['best'], old['history']
        torch.set_rng_state(old['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(old['rng_cuda'], device)
    def autocast():
        return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    def move(batch):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    started = time.monotonic()
    for epoch in range(start_epoch, a.epochs):
        train.epoch = epoch
        loader = DataLoader(train, batch_size=a.batch_size, shuffle=True,
                            num_workers=a.workers, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(a.seed+epoch))
        raw.train()
        sums, n = {}, 0
        complete = True
        for i, batch in enumerate(loader):
            b = move(batch)
            opt.zero_grad(set_to_none=True)
            with autocast():
                losses = rhythm_loss(raw, b)
            if not torch.isfinite(losses['loss']):
                raise RuntimeError('nonfinite training loss')
            losses['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('nonfinite gradients')
            progress = step/max(1, a.epochs*len(loader))
            for group in opt.param_groups:
                group['lr'] = a.learning_rate*min(1., (step+1)/100)*(.1+.9*(1+math.cos(math.pi*progress))/2)
            opt.step()
            with torch.no_grad():
                decay = min(.999, (step+1)/(step+10))
                for e, m in zip(ema.parameters(), raw.parameters()):
                    e.lerp_(m, 1-decay)
            step += 1
            n += len(b['audio'])
            for k, value in losses.items():
                sums[k] = sums.get(k, 0.)+float(value.detach())*len(b['audio'])
            if step % 50 == 0:
                print(json.dumps({'epoch': epoch+1, 'step': step, 'loss': float(losses['loss'].detach()),
                                  'seconds': time.monotonic()-started}), flush=True)
            if a.max_steps and step >= a.max_steps:
                complete = i+1 == len(loader)
                break
        validation = {}
        with torch.inference_mode():
            for mode in ('aligned', 'zero_audio', 'shift_1000ms'):
                totals, count = {}, 0
                for i, batch in enumerate(DataLoader(val, batch_size=a.batch_size, num_workers=a.workers)):
                    if a.val_batches and i >= a.val_batches:
                        break
                    b = move(batch)
                    if mode == 'zero_audio': b['audio'].zero_()
                    if mode == 'shift_1000ms': b['audio'] = b['audio'].roll(100, -1)
                    with autocast():
                        losses = rhythm_loss(ema, b, generator=torch.Generator(device=device).manual_seed(1731+i))
                    for k, value in losses.items():
                        totals[k] = totals.get(k, 0.)+float(value)*len(b['audio'])
                    count += len(b['audio'])
                validation[mode] = {k: v/count for k, v in totals.items()}
        row = {'epoch': epoch+1, 'step': step, 'train': {k: v/n for k, v in sums.items()},
               'validation': validation, 'seconds': time.monotonic()-started}
        history.append(row)
        score = validation['aligned']['loss']
        improved = score < best
        best = min(best, score)
        payload = {'schema': SCHEMA, 'config': asdict(config), 'contract': contract,
                   'initialization': initialization,
                   'parent_manifest_sha256': train.manifest.get('parent_manifest_sha256'),
                   'feature_pipeline_key': train.manifest['feature_pipeline_key'],
                   'manifest_sha256': fingerprint, 'model': ema.state_dict(),
                   'raw_model': raw.state_dict(), 'optimizer': opt.state_dict(),
                   'epoch': epoch+1, 'epoch_complete': complete, 'step': step, 'best': best,
                   'history': history, 'rng_cpu': torch.get_rng_state(),
                   'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        atomic_save(payload, a.output_dir/'last.pt')
        if improved:
            atomic_save(payload, a.output_dir/'best.pt')
        (a.output_dir/'history.json').write_text(json.dumps({
            'contract': contract, 'history': history,
            'parameters': sum(v.numel() for v in raw.parameters()),
            'train_sources': len(train.groups), 'validation_sources': len(val.groups),
            'max_gpu_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}, indent=2))
        print(json.dumps(row), flush=True)
        if a.max_steps and step >= a.max_steps:
            break


if __name__ == '__main__':
    main()
