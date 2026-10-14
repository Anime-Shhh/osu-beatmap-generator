"""Read-only teacher-forced evaluation; this does not measure map playability."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.v2.dataset import SparseSongDataset, collate_sparse_examples, move_batch_to_device
from src.v2.geometry import geometry_diffusion_loss
from src.v2.grammar import TokenFamily
from src.v2.inference import load_v2_models
from src.v2.training import forward_planner, gather_event_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--records-dir', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1001340)
    torch.cuda.manual_seed_all(1001340)
    device = torch.device('cuda')
    models = load_v2_models(args.checkpoint, device=device)
    models.planner.eval()
    models.geometry.eval()
    dataset = SparseSongDataset(args.records_dir, split=args.split, sampling_mode='bar',
                                expected_pipeline_key=models.checkpoint_provenance.feature_pipeline_key)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1,
                        collate_fn=collate_sparse_examples, pin_memory=True)
    family_list = list(TokenFamily)
    lookup = torch.tensor([family_list.index(models.planner.vocabulary.family_of(i))
                           for i in range(len(models.planner.vocabulary))], device=device)
    correct = torch.zeros(len(family_list), dtype=torch.long, device=device)
    counts = torch.zeros_like(correct)
    token_loss_sum = 0.0
    batch_ce_sum = 0.0
    components = defaultdict(float)
    per_song = defaultdict(lambda: [0.0, 0, 0])
    started = time.monotonic()
    with torch.inference_mode():
        for step, source in enumerate(loader):
            batch = move_batch_to_device(source, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = forward_planner(models.planner, batch)
                logits = output['logits'].float()
                losses = F.cross_entropy(logits.transpose(1, 2), batch['labels'],
                                         ignore_index=-100, reduction='none')
                mask = batch['labels'].ne(-100)
                token_loss_sum += losses.sum().item()
                batch_ce_sum += losses.sum().item() / mask.sum().item()
                pred = logits.argmax(-1)
                families = lookup[batch['labels'][mask]]
                counts += torch.bincount(families, minlength=len(family_list))
                good = pred[mask].eq(batch['labels'][mask])
                correct += torch.bincount(families[good], minlength=len(family_list))
                if batch['event_mask'].any():
                    context = gather_event_context(output['decoder_hidden'], batch['event_token_indices'], batch['event_mask'])
                    result = geometry_diffusion_loss(models.geometry, context, batch['event_kinds'],
                        batch['anchor_counts'], batch['geometry'], batch['event_mask'],
                        event_features=batch['event_features'], event_positions=batch['event_positions'],
                        return_components=True)
                    for key, value in result.items():
                        components[key] += value.item()
            for i, metadata in enumerate(source['metadata']):
                row = per_song[metadata['source_component_id']]
                row[0] += losses[i].sum().item()
                row[1] += mask[i].sum().item()
                row[2] += (pred[i].eq(batch['labels'][i]) & mask[i]).sum().item()
            if step % 200 == 0:
                print(json.dumps({'batches_done': step + 1, 'batches_total': len(loader)}), flush=True)
    counts = counts.cpu().tolist()
    correct = correct.cpu().tolist()
    total = sum(counts)
    ce = token_loss_sum / total
    report = {
        'evaluation': 'teacher-forced-oracle-clock/v1', 'split': args.split,
        'checkpoint': args.checkpoint, 'model_revision': models.model_revision,
        'bars': len(dataset), 'audio_components': len(per_song), 'tokens': total,
        'token_weighted_cross_entropy': ce, 'token_perplexity': math.exp(ce),
        'batch_mean_cross_entropy': batch_ce_sum / len(loader),
        'token_accuracy': sum(correct) / total,
        'per_family': {family.value: {'tokens': counts[i], 'accuracy': correct[i] / counts[i] if counts[i] else None}
                       for i, family in enumerate(family_list)},
        'geometry_batch_mean': {key: value / len(loader) for key, value in components.items()},
        'per_audio': {key: {'cross_entropy': value[0] / value[1], 'tokens': value[1],
                           'token_accuracy': value[2] / value[1]} for key, value in per_song.items()},
        'planner_config': asdict(models.planner.config),
        'geometry_config': asdict(models.geometry.config),
        'seconds': time.monotonic() - started,
        'limitations': ['Ground-truth token prefixes and previous-bar tokens supplied.',
                        'Timing targets use source/oracle clocks.',
                        'Geometry conditioned on human event plans, not sampled planner output.',
                        'No autoregressive full-song rollout or playability test.'],
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'per_audio'}), flush=True)


if __name__ == '__main__':
    main()
