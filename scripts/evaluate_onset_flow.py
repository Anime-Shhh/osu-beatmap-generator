"""Generated-event comparison on fixed source-separated validation/test windows.

Decoder threshold selection uses validation only. Audio controls use the same
sampling seeds; latent dimensions differ between old and new architectures.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

from scripts.train_onset_flow import load_model
from scripts.evaluate_rhythm_flow import match_onsets, reference_window
from src.beatmap_flow.data import OnsetWindows
from src.beatmap_flow.rhythm import sample_rhythm, decode_events
from src.rhythm_flow.model import RhythmFlow, RhythmFlowConfig, sample_full_song, decode_rhythm


def finish(count):
    p = count['tp']/max(1, count['predicted'])
    r = count['tp']/max(1, count['reference'])
    return {**dict(count), 'precision': p, 'recall': r, 'f1': 2*p*r/(p+r) if p+r else 0.}


def score_events(predictions, references):
    tallies = {t: Counter() for t in (25, 50, 100)}
    kinds = {k: Counter() for k in range(3)}
    groups, duration_errors, onset_errors = {}, [], []
    correct = empty = 0
    for predicted, ref in zip(predictions, references):
        truth = ref['events']
        empty += not predicted
        for tolerance, tally in tallies.items():
            pairs = match_onsets(predicted, truth, tolerance)
            tally.update(tp=len(pairs), predicted=len(predicted), reference=len(truth))
            if tolerance == 50:
                groups.setdefault(ref['source'], Counter()).update(tp=len(pairs), predicted=len(predicted), reference=len(truth))
                for i, j in pairs:
                    a, b = predicted[i], truth[j]
                    correct += a['kind'] == b['kind']
                    onset_errors.append(abs(a['start_ms']-b['start_ms']))
                    if a['kind'] == b['kind'] and b['kind']:
                        duration_errors.append(abs((a['end_ms']-a['start_ms'])-(b['end_ms']-b['start_ms'])))
        for k, tally in kinds.items():
            a = [e for e in predicted if e['kind'] == k]
            b = [e for e in truth if e['kind'] == k]
            tally.update(tp=len(match_onsets(a, b, 50)), predicted=len(a), reference=len(b))
    def summarize(values):
        return {'count': len(values), 'mean': sum(values)/len(values),
                'median': float(torch.tensor(values).median())} if values else None
    seconds = sum(r['frames'] for r in references)*.01
    return {'onsets': {str(k): finish(v) for k, v in tallies.items()},
            'kind_aware_50ms': {str(k): finish(v) for k, v in kinds.items()},
            'source_scores_50ms': {k: finish(v) for k, v in groups.items()},
            'macro_source_f1_50ms': sum(finish(v)['f1'] for v in groups.values())/len(groups),
            'kind_accuracy_among_50ms_matches': correct/max(1, tallies[50]['tp']),
            'matched_duration_error_ms': summarize(duration_errors), 'matched_onset_error_ms': summarize(onset_errors),
            'predicted_events_per_second': tallies[50]['predicted']/seconds,
            'reference_events_per_second': tallies[50]['reference']/seconds,
            'empty_windows': empty, 'sampled_seconds': seconds, 'windows': len(references)}


@torch.inference_mode()
def generate(model, data, refs, *, legacy, mode, seed, steps):
    result = []
    for i, b in enumerate(DataLoader(data, batch_size=8, num_workers=2, shuffle=False)):
        audio, profile = b['audio'].cuda(), b['profile'].cuda()
        if mode == 'zero_audio': audio.zero_()
        elif mode == 'shift_1000ms': audio = audio.roll(100, -1)
        lengths = [r['frames'] for r in refs[i*8:i*8+len(audio)]]
        # Evaluate each common length separately: short-file padding is never audio.
        outputs = [None]*len(audio)
        for length in sorted(set(lengths)):
            ids = [j for j, n in enumerate(lengths) if n == length]
            gen = torch.Generator(device='cuda').manual_seed(seed+i*100+length)
            if legacy:
                signals = sample_full_song(model, audio[ids, :, :length], profile[ids], steps=steps, generator=gen)
                for j, signal in zip(ids, signals): outputs[j] = signal.cpu()
            else:
                onset, marks = sample_rhythm(model, audio[ids, :, :length], profile[ids], steps=steps, generator=gen)
                for j, a, m in zip(ids, onset, marks): outputs[j] = (a[0].cpu(), m.cpu())
        result.extend(outputs)
        if i % 8 == 0: print(json.dumps({'mode': mode, 'legacy': legacy, 'windows': len(result)}), flush=True)
    return result


def decode(outputs, threshold, legacy):
    if legacy: return [decode_rhythm(s, threshold=threshold) for s in outputs]
    return [decode_events(s, m, threshold=threshold) for s, m in outputs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--seeds', type=int, nargs='+', default=[91731, 91732, 91733])
    p.add_argument('--samples-per-source', type=int, default=4)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--allow-parent-checkpoint', action='store_true',
                   help='evaluate a checkpoint on its documented frozen-split corpus expansion')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, checkpoint = load_model(a.checkpoint, 'cuda')
    manifest = json.loads((a.cache_dir/'manifest.json').read_text())
    digest = hashlib.sha256((a.cache_dir/'manifest.json').read_bytes()).hexdigest()
    compatible = checkpoint['manifest_sha256'] == digest or (a.allow_parent_checkpoint and
        checkpoint['manifest_sha256'] == manifest.get('parent_manifest_sha256'))
    if not compatible: raise ValueError('checkpoint and evaluation cache differ')
    if checkpoint['feature_pipeline_key'] != manifest['feature_pipeline_key']:
        raise ValueError('evaluation feature pipeline differs from training')
    models = [('onset_flow', model, False, a.checkpoint)]
    if a.baseline:
        old = torch.load(a.baseline, weights_only=True, map_location='cpu')
        if old['manifest_sha256'] not in (digest, manifest.get('parent_manifest_sha256')):
            raise ValueError('baseline does not bind the frozen parent source splits')
        baseline = RhythmFlow(RhythmFlowConfig(**old['config'])).cuda().eval()
        baseline.load_state_dict(old['model'])
        models.append(('legacy_rhythm_flow', baseline, True, a.baseline))
    report = {'manifest_sha256': digest, 'steps': a.steps, 'seeds': a.seeds,
              'window_frames': 1024, 'threshold_selection': 'validation micro onset F1 at 50ms',
              'models': {}, 'started': time.time()}
    for name, model, legacy, path in models:
        result = {'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        threshold = None
        for split in (('val',) if a.validation_only else ('val', 'test')):
            data = OnsetWindows(a.cache_dir, split, window=1024, samples_per_source=a.samples_per_source)
            refs = [reference_window(data, i) for i in range(len(data))]
            refs_digest = hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest()
            split_result = {'sources': len(data.groups), 'reference_sha256': refs_digest, 'seeds': {}}
            if split == 'val':
                all_outputs, all_refs = [], []
                for seed in a.seeds:
                    outputs = generate(model, data, refs, legacy=legacy, mode='aligned', seed=seed, steps=a.steps)
                    all_outputs.extend(outputs); all_refs.extend(refs)
                    torch.save({'outputs': outputs, 'references': refs}, a.output_dir/f'{name}-{split}-{seed}.pt')
                grid = {str(round(t/100, 2)): score_events(decode(all_outputs, t/100, legacy), all_refs) for t in range(15, 81, 5)}
                threshold = float(max(grid, key=lambda t: grid[t]['onsets']['50']['f1']))
                split_result['threshold_grid'] = grid
                split_result['selected'] = grid[str(threshold)]
                result['threshold'] = threshold
            else:
                for mode in ('aligned', 'zero_audio', 'shift_1000ms'):
                    predictions, all_refs = [], []
                    for seed in a.seeds:
                        outputs = generate(model, data, refs, legacy=legacy, mode=mode, seed=seed, steps=a.steps)
                        events = decode(outputs, threshold, legacy)
                        split_result['seeds'][f'{mode}-{seed}'] = score_events(events, refs)
                        predictions.extend(events); all_refs.extend(refs)
                        (a.output_dir/f'{name}-{split}-{mode}-{seed}.json').write_text(json.dumps({'predictions': events, 'references': refs}))
                    split_result[mode] = score_events(predictions, all_refs)
            result[split] = split_result
            report['models'][name] = result
            (a.output_dir/'metrics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'model': name, 'threshold': threshold,
                          'validation_f1': result['val']['selected']['onsets']['50']['f1']}), flush=True)


if __name__ == '__main__': main()
