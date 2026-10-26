"""Fixed validation-source full-song rhythm diagnostics; no Kaguya tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from scripts.train_onset_flow import load_model
from scripts.compare_native_rhythm import compare
from scripts.evaluate_onset_flow import score_events
from src.beatmap_flow.rhythm import sample_rhythm, decode_events
from src.v2.mapspec import mapspec_from_osu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--native-dir', type=Path, required=True)
    p.add_argument('--audio-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--seeds', type=int, nargs='+', default=[92731, 92732])
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = json.loads((a.cache_dir/'manifest.json').read_text())
    model, checkpoint = load_model(a.checkpoint, 'cuda')
    if checkpoint['feature_pipeline_key'] != manifest['feature_pipeline_key']:
        raise ValueError('feature pipeline mismatch')
    audios = {hashlib.sha256(path.read_bytes()).hexdigest() for path in a.audio_dir.glob('*.audio')}
    sources = {}
    for row in manifest['charts']:
        if row['split'] == 'val' and row['source']['audio_sha256'] in audios:
            sources.setdefault(row['source_component'], []).append(row)
    if not sources: raise ValueError('no frozen validation audio sources')
    # Select with native stars only, before observing any generated output.
    rows = [min(group, key=lambda r: (abs(r['profile'][0]*10-4.5), r['chart_sha256']))
            for _, group in sorted(sources.items())]
    report = {'role': 'frozen full-song validation', 'threshold': a.threshold, 'seeds': a.seeds,
        'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((a.cache_dir/'manifest.json').read_bytes()).hexdigest(),
        'selection': 'retained validation sources, chart nearest 4.5 stars; two fixed seeds', 'sources': []}
    all_predictions, all_references = [], []
    for row in rows:
        sha = row['chart_sha256']
        native = a.native_dir/'maps'/f'{sha}.osu'
        if hashlib.sha256(native.read_bytes()).hexdigest() != sha:
            raise ValueError('native hash mismatch')
        spec = mapspec_from_osu(native.read_text(encoding='utf-8-sig'))
        features = a.cache_dir/row['audio_file']
        if hashlib.sha256(features.read_bytes()).hexdigest() != row['audio_file_sha256']:
            raise ValueError('feature checksum mismatch')
        audio = torch.load(features, weights_only=True)['audio'].float().T[None].cuda()
        reference = [{'start_ms': s, 'end_ms': e, 'kind': k} for s, e, k in row['events']]
        item = {'source': row['source_component'], 'chart_sha256': sha,
                'profile': row['profile'], 'source_record_id': row['source']['source_record_id'], 'seeds': {}}
        for seed in a.seeds:
            onset, marks = sample_rhythm(model, audio, torch.tensor(row['profile'], device='cuda')[None],
                steps=48, generator=torch.Generator(device='cuda').manual_seed(seed))
            events = decode_events(onset[0, 0].cpu(), marks[0].cpu(), threshold=a.threshold,
                                   duration_ms=row['source']['audio_duration_ms'])
            item['seeds'][str(seed)] = compare(events, reference, spec.timing_map)
            all_predictions.append(events)
            all_references.append({'source': row['source_component'], 'frames': audio.shape[-1], 'events': reference})
            (a.output_dir/f'{sha}-{seed}.json').write_text(json.dumps(events))
        report['sources'].append(item)
        print(json.dumps({'sources_done': len(report['sources']), 'total': len(rows)}), flush=True)
        (a.output_dir/'metrics.json').write_text(json.dumps(report, indent=2))
    report['aggregate'] = score_events(all_predictions, all_references)
    report['short_circle_pairs'] = {key: sum(s['short_circle_pairs'][key]
        for r in report['sources'] for s in r['seeds'].values())
        for key in ('reference', 'both_onsets_recovered_50ms', 'both_as_circles_50ms')}
    (a.output_dir/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'aggregate': report['aggregate'], 'short_circle_pairs': report['short_circle_pairs']}), flush=True)


if __name__ == '__main__': main()
