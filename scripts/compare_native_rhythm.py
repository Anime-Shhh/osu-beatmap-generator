"""Compare a generated map against every native difficulty in an original OSZ.

This is a user-selected development diagnostic, not an untouched test score.
Native-map differences also reflect legitimate alternate musical interpretations.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from scripts.evaluate_rhythm_flow import match_onsets
from src.v2.mapspec import mapspec_from_osu
from src.v2.evaluation import RosuDifficultyCalculator


def events_of(spec):
    return [{'start_ms': spec.timing_map.beat_to_time(e.beat_position),
             'end_ms': spec.timing_map.beat_to_time(e.end_beat if e.end_beat is not None else e.beat_position),
             'kind': int(e.kind), 'x': e.x, 'y': e.y} for e in spec.events]


def summary(events):
    gaps = [b['start_ms']-a['start_ms'] for a, b in zip(events, events[1:])]
    return {'objects': len(events), 'kinds': dict(Counter(e['kind'] for e in events)),
            'gaps_ms': {'median': statistics.median(gaps) if gaps else None,
                        'under_80': sum(g < 80 for g in gaps),
                        'under_150': sum(g < 150 for g in gaps),
                        'under_250': sum(g < 250 for g in gaps)}}


def compare(generated, reference, timing):
    result = {'generated': summary(generated), 'reference': summary(reference), 'onsets': {}}
    for tolerance in (10, 25, 50, 100):
        pairs = match_onsets(generated, reference, tolerance)
        errors = [generated[i]['start_ms']-reference[j]['start_ms'] for i, j in pairs]
        result['onsets'][str(tolerance)] = {'matches': len(pairs),
            'precision': len(pairs)/max(1, len(generated)), 'recall': len(pairs)/max(1, len(reference)),
            'f1': 2*len(pairs)/max(1, len(generated)+len(reference)),
            'matched_mean_abs_error_ms': statistics.mean(map(abs, errors)) if errors else None,
            'matched_signed_error_ms': statistics.mean(errors) if errors else None}
    pairs = match_onsets(generated, reference, 50)
    matched_ref = {j for _, j in pairs}
    quarters = []
    for e in generated:
        beat = timing.time_to_beat(e['start_ms'])
        quarters.append(abs(beat-round(beat*4)/4)*timing.beat_length_at_time(e['start_ms']))
    result['generated_distance_to_reference_quarter_beat_ms'] = {
        'median': statistics.median(quarters) if quarters else None,
        'mean': statistics.mean(quarters) if quarters else None,
        'over_20ms': sum(x > 20 for x in quarters),
        'note': 'diagnostic only; native triplets and intentional unsnapped notes are possible'}
    # Consecutive circle pairs no longer than half of the active native beat.
    # This describes fast two-note patterns, not simultaneous clicks.
    pairs_ref = [j for j in range(len(reference)-1)
        if reference[j]['kind'] == reference[j+1]['kind'] == 0
        and 0 < reference[j+1]['start_ms']-reference[j]['start_ms'] <=
            .51*timing.beat_length_at_time(reference[j]['start_ms'])]
    matched_types = {j: generated[i]['kind'] for i, j in pairs}
    result['short_circle_pairs'] = {'reference': len(pairs_ref),
        'both_onsets_recovered_50ms': sum(j in matched_ref and j+1 in matched_ref for j in pairs_ref),
        'both_as_circles_50ms': sum(matched_types.get(j) == matched_types.get(j+1) == 0 for j in pairs_ref),
        'definition': 'consecutive circles, positive gap <= 0.51 native beat; overlapping pairs allowed'}
    segments = []
    last = max([e['end_ms'] for e in reference+generated], default=0)
    for start in range(0, int(last)+1, 8000):
        pred = [e for e in generated if start <= e['start_ms'] < start+8000]
        truth = [e for e in reference if start <= e['start_ms'] < start+8000]
        count = len(match_onsets(pred, truth, 50))
        segments.append({'start_ms': start, 'end_ms': start+8000, 'generated': len(pred),
                         'reference': len(truth), 'matched_50ms': count,
                         'missing_50ms': len(truth)-count, 'extra_50ms': len(pred)-count})
    result['segments'] = segments
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-osz', type=Path, required=True)
    p.add_argument('--generated-osu', type=Path, required=True)
    p.add_argument('--generated-audio', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-manifest', type=Path)
    p.add_argument('--target-stars', type=float, default=4.5)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    generated_text = a.generated_osu.read_text()
    generated_spec = mapspec_from_osu(generated_text)
    generated = events_of(generated_spec)
    calc = RosuDifficultyCalculator()
    audio_sha = hashlib.sha256(a.generated_audio.read_bytes()).hexdigest()
    report = {'role': 'user-selected development comparison', 'target_stars': a.target_stars,
        'generated_osu_sha256': hashlib.sha256(a.generated_osu.read_bytes()).hexdigest(),
        'original_osz_sha256': hashlib.sha256(a.original_osz.read_bytes()).hexdigest(),
        'generated_difficulty': calc.calculate(generated_text).to_dict(), 'maps': []}
    if a.cache_manifest:
        cache = json.loads(a.cache_manifest.read_text())
        report['matching_encoded_audio_in_cache'] = [
            {'split': r['split'], 'chart_sha256': r['chart_sha256']}
            for r in cache['charts'] if r['source']['audio_sha256'] == audio_sha]
    timeline = {'generated': generated, 'references': []}
    with zipfile.ZipFile(a.original_osz) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.osu'): continue
            raw = archive.read(name)
            text = raw.decode('utf-8-sig')
            spec = mapspec_from_osu(text)
            if spec.general.mode != 0: continue
            original_audio = archive.read(str(Path(name).parent/spec.general.audio_filename))
            if hashlib.sha256(original_audio).hexdigest() != audio_sha:
                raise ValueError('original and generated maps use different encoded audio; align before comparison')
            ref = events_of(spec)
            difficulty = calc.calculate(text).to_dict()
            item = {'filename': name, 'version': spec.metadata.version,
                    'chart_sha256': hashlib.sha256(raw).hexdigest(), 'difficulty': difficulty,
                    'timing_segments': [asdict(s) for s in spec.timing_map.segments],
                    **compare(generated, ref, spec.timing_map)}
            report['maps'].append(item)
            timeline['references'].append({'version': spec.metadata.version, 'events': ref, 'stars': difficulty['stars']})
    closest = min(report['maps'], key=lambda r: abs(r['difficulty']['stars']-a.target_stars))
    report['closest_original'] = closest['version']
    (a.output_dir/'comparison.json').write_text(json.dumps(report, indent=2))
    (a.output_dir/'timelines.json').write_text(json.dumps(timeline, separators=(',', ':')))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
