"""Paired audio interventions with identical reference prefixes and clocks.

This diagnoses dependence, not causal instrument attribution or free-run quality.
Validation is the default: keep the final test split out of iterative tuning.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.v2.dataset import SparseSongDataset, collate_sparse_examples, move_batch_to_device
from src.v2.inference import load_v2_models
from src.v2.training import forward_planner


def intervene(batch, mode):
    result = dict(batch)
    if mode == 'no_history':
        result['previous_bar_tokens'] = batch['previous_bar_tokens'][:, :0]
        result['previous_bar_mask'] = batch['previous_bar_mask'][:, :0]
        return result
    if mode == 'aligned':
        return result
    for stream in ('rhythm', 'semantic'):
        values = batch[f'{stream}_features'].clone()
        mask = batch[f'{stream}_mask']
        times = batch[f'{stream}_times_ms']
        for row in range(len(values)):
            valid = mask[row].nonzero().flatten()
            if mode == 'zero_audio':
                values[row, valid] = 0
            else:
                delay = float(mode.split('_')[1])
                if len(valid) > 1:
                    period = (times[row, valid[1:]] - times[row, valid[:-1]]).median().item()
                    steps = max(1, round(delay / period))
                    values[row, valid] = values[row, valid].roll(steps, dims=0)
        result[f'{stream}_features'] = values
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--records-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--bars-per-source', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.bars_per_source < 1 or args.batch_size < 1:
        p.error('bar count and batch size must be positive')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    torch.manual_seed(1731)
    models = load_v2_models(args.checkpoint, device=args.device)
    ds = SparseSongDataset(args.records_dir, split=args.split, sampling_mode='bar',
        expected_pipeline_key=models.checkpoint_provenance.feature_pipeline_key,
        alignment_frames=models.planner.config.local_audio_tokens if hasattr(models.planner.config, 'local_audio_tokens') else 0)
    groups = defaultdict(list)
    for i, (record_index, _) in enumerate(ds._index):
        groups[ds._record_indexes[record_index]['metadata']['source_component_id']].append(i)
    rng = random.Random(1731)
    indices = [i for key in sorted(groups) for i in rng.sample(groups[key], min(args.bars_per_source, len(groups[key])))]
    modes = ('aligned', 'shift_137', 'shift_997', 'zero_audio', 'no_history')
    totals = {mode: defaultdict(lambda: [0., 0, 0]) for mode in modes}
    family = [models.planner.vocabulary.family_of(i).value for i in range(len(models.planner.vocabulary))]
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, collate_fn=collate_sparse_examples)
    start = time.monotonic()
    with torch.inference_mode():
        for source in loader:
            batch = move_batch_to_device(source, args.device)
            labels = batch['labels']; valid = labels.ne(-100)
            for mode in modes:
                output = forward_planner(models.planner, intervene(batch, mode))
                logits = output['logits'].float()
                losses = F.cross_entropy(logits.transpose(1,2), labels, ignore_index=-100, reduction='none')
                for row in range(len(labels)):
                    key = source['metadata'][row]['source_component_id']
                    for loss, good, token in zip(losses[row,valid[row]].tolist(),
                            logits.argmax(-1)[row,valid[row]].eq(labels[row,valid[row]]).tolist(), labels[row,valid[row]].tolist()):
                        for group in ('all', family[token], 'source:'+key):
                            item = totals[mode][group]; item[0] += loss; item[1] += int(good); item[2] += 1
    metrics = {mode: {key: {'cross_entropy': v[0]/v[2], 'accuracy': v[1]/v[2], 'tokens': v[2]}
                     for key,v in rows.items()} for mode,rows in totals.items()}
    report = {'schema': 'v2-audio-interventions/1', 'model_revision': models.model_revision,
              'split': args.split, 'source_components': len(groups), 'bars': len(indices), 'seed':1731,
              'sample_indices': indices, 'metrics': metrics, 'seconds':time.monotonic()-start,
              'limitations': ['Reference current prefixes and oracle clocks remain fixed.',
                  'Zero features are an out-of-distribution control, not literal digital silence.',
                  'Shifts wrap within valid audio frames; shifts are rounded to each stream frame rate.',
                  'Sensitivity does not prove correct musical attribution; no instrument labels are available.']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({m:metrics[m]['all'] for m in modes}, indent=2))


if __name__ == '__main__': main()
