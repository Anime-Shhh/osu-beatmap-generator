"""Summarize completed paired validation and Kaguya inference runs."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from scripts.evaluate_rhythm_flow import match_onsets


def f1(counts):
    return 2*sum(c['tp'] for c in counts)/max(1, sum(c['predicted']+c['reference'] for c in counts))


def bootstrap_difference(old, new):
    keys = sorted(old)
    if keys != sorted(new): raise ValueError('validation sources differ')
    rng = random.Random(1731)
    differences = []
    for _ in range(20000):
        sample = rng.choices(keys, k=len(keys))
        differences.append(f1([new[k] for k in sample])-f1([old[k] for k in sample]))
    differences.sort()
    return {'source_count': len(keys), 'gain': f1(new.values())-f1(old.values()),
            'source_bootstrap_ci95': [differences[499], differences[19499]],
            'caveat': 'two generation seeds pooled within source; fixed trained checkpoints, not training-seed uncertainty'}


def spatial_pairs(events, beat_length_ms):
    pairs = [(a, b) for a, b in zip(events, events[1:]) if a['kind'] == b['kind'] == 0
             and 0 < b['start_ms']-a['start_ms'] <= .51*beat_length_ms]
    distances = [math.hypot(b['x']-a['x'], b['y']-a['y']) for a, b in pairs]
    return {'short_circle_pairs': len(pairs), 'overlapping_pairs_within_8px': sum(d <= 8 for d in distances),
            'median_short_pair_head_distance_px': statistics.median(distances) if distances else None,
            'definition': 'consecutive circles within 0.51 native beats; raw head distance, before client stacking'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    old = Path('/common/users/asj102/osu_project/flow_runs/run_20261025')
    result = {'corpus': json.loads((a.run/'recovered-corpus/summary.json').read_text()), 'models': {}}
    validations = {}
    for name in ('original', 'control', 'phrase'):
        output = a.run/f'kaguya-{name}'
        kaguya = json.loads((output/'rhythm-comparison.json').read_text())
        selected = json.loads((output/'selected-comparison/comparison.json').read_text())
        detail = next(r for r in selected['maps'] if r['version'] == selected['closest_original'])
        timeline = json.loads((output/'selected-comparison/timelines.json').read_text())
        truth = next(r['events'] for r in timeline['references'] if r['version'] == selected['closest_original'])
        predicted = timeline['generated']
        period = detail['timing_segments'][0]['beat_length_ms']
        detail['spatial_pairs'] = {'generated': spatial_pairs(predicted, period), 'reference': spatial_pairs(truth, period)}
        pairs = match_onsets(predicted, truth, 50)
        durations = [abs((predicted[i]['end_ms']-predicted[i]['start_ms'])-(truth[j]['end_ms']-truth[j]['start_ms']))
                     for i, j in pairs if predicted[i]['kind'] == truth[j]['kind'] and truth[j]['kind']]
        detail['generated_median_slider_duration_ms'] = statistics.median(
            e['end_ms']-e['start_ms'] for e in predicted if e['kind'] == 1)
        detail['matched_hold_duration_mae_ms'] = statistics.mean(durations) if durations else None
        validation_path = old/'original-full-song-validation/metrics.json' if name == 'original' else a.run/name/'full-song-validation/metrics.json'
        validation = json.loads(validation_path.read_text())
        validations[name] = validation['aggregate']['source_scores_50ms']
        summary = {'kaguya_across_seeds': kaguya['summary'], 'selected_candidate': kaguya['selected'],
            'threshold': kaguya['threshold'], 'checkpoint_sha256': kaguya['checkpoint_sha256'],
            'default_star_tolerance_passes': kaguya['default_tolerance_passes'],
            'selected_kaguya': detail, 'selected_stars': selected['generated_difficulty']['stars'],
            'full_song_validation': {k:v for k,v in validation['aggregate'].items() if k != 'source_scores_50ms'},
            'full_song_short_pairs': validation['short_circle_pairs']}
        if name != 'original':
            history = json.loads((a.run/name/'history.json').read_text())
            best = min(history['history'], key=lambda r: r['validation']['aligned']['loss'])
            summary['training'] = {'epochs': len(history['history']), 'best_epoch': best['epoch'],
                'steps': history['history'][-1]['step'], 'seconds': history['history'][-1]['seconds'],
                'parameters': history['parameters']}
            val = json.loads((a.run/name/'validation/metrics.json').read_text())
            summary['expanded_validation_window_f1_50ms'] = val['models']['onset_flow']['val']['selected']['onsets']['50']['f1']
        result['models'][name] = summary
    result['validation_gains'] = {name: bootstrap_difference(validations['original'], validations[name])
                                  for name in ('control', 'phrase')}
    result['context_vs_control'] = bootstrap_difference(validations['control'], validations['phrase'])
    result['selection_basis'] = 'highest full-song validation onset F1 at fixed threshold .5; Kaguya excluded from selection'
    result['recommended_model'] = max(('original', 'control', 'phrase'),
        key=lambda name: result['models'][name]['full_song_validation']['onsets']['50']['f1'])
    (a.run/'comparison-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'recommended_model': result['recommended_model'], 'validation_gains': result['validation_gains'],
                     'context_vs_control': result['context_vs_control']}), flush=True)


if __name__ == '__main__': main()
