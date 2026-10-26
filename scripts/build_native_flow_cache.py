"""Join authentic native maps to verified shared audio without changing splits."""
import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path

import torch
from src.rhythm_flow.data import validate_manifest
from src.v2.mapspec import mapspec_from_osu
from src.v2.events import EventKind
from src.beatmap_flow.geometry import CURVES, event_condition


def build(audio_root, native_root, output):
    audio_root, native_root, output = map(Path, (audio_root, native_root, output))
    source = json.loads((audio_root/'manifest.json').read_text())
    validate_manifest(source)
    native = json.loads((native_root/'manifest.json').read_text())
    if native['frozen_manifest_sha256'] != hashlib.sha256((audio_root/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('native manifest does not bind the requested frozen audio cache')
    lookup = {r['chart_sha256']: r for r in source['charts']}
    if len(lookup) != len(source['charts']):
        raise ValueError('duplicate chart identities')
    output.mkdir(parents=True, exist_ok=False)
    (output/'geometry').mkdir()
    rows, geometries, errors, audio_cache = [], [], [], OrderedDict()
    checked_audio = set()
    for entry in native['charts']:
        row = dict(lookup[entry['sha256']])
        if entry['split'] != row['split'] or entry['group'] != row['source_component']:
            raise ValueError('native chart split mismatch')
        path = (native_root/entry['path']).resolve()
        if not path.is_relative_to(native_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('native chart path/checksum mismatch')
        spec = mapspec_from_osu(path.read_text(encoding='utf-8-sig'))
        timing = spec.timing_map
        events = list(spec.events)
        starts = torch.tensor([timing.beat_to_time(e.beat_position) for e in events])
        ends = torch.tensor([timing.beat_to_time(e.end_beat if e.end_beat is not None else e.beat_position) for e in events])
        kinds = torch.tensor([int(e.kind) for e in events])
        if not events or not torch.isfinite(starts).all() or not torch.isfinite(ends).all() or (starts[1:] < starts[:-1]).any():
            raise ValueError('invalid native event timeline')
        name = row['audio_file']
        if name not in checked_audio:
            if hashlib.sha256((audio_root/name).read_bytes()).hexdigest() != row['audio_file_sha256']:
                raise ValueError('audio hash mismatch')
            os.link(audio_root/name, output/name)
            checked_audio.add(name)
        row['events'] = [[float(s), float(e), int(k)] for s, e, k in zip(starts, ends, kinds)]
        rows.append(row)
        try:
            geometry = torch.zeros(len(events), 33, 2)
            counts, curves, repeats, lengths, combo = [], [], [], [], []
            for i, e in enumerate(events):
                if len(e.control_points) > 32 or e.slider_repeats > 8 or not (0 <= e.x <= 512 and 0 <= e.y <= 384):
                    raise ValueError('source exceeds declared geometry representation')
                geometry[i, 0] = torch.tensor((e.x/512, e.y/384))
                controls = e.control_points if e.kind == EventKind.SLIDER else ()
                for j, (x, y) in enumerate(controls): geometry[i, j+1] = torch.tensor((x/512, y/384))
                counts.append(len(controls)); curves.append(CURVES.index(e.curve_type) if controls else 0)
                repeats.append(e.slider_repeats if controls else 1)
                lengths.append(e.slider_length if controls else 0.); combo.append(float(e.new_combo))
            if not torch.isfinite(geometry).all(): raise ValueError('nonfinite geometry')
            if name not in audio_cache:
                audio_cache[name] = torch.load(audio_root/name, weights_only=True)['audio'].float()
                if len(audio_cache) > 2: audio_cache.popitem(last=False)
            audio_cache.move_to_end(name)
            condition = event_condition(audio_cache[name], starts, ends, kinds, torch.tensor(row['profile']))
            tensor = {'condition': condition, 'starts': starts, 'ends': ends, 'kinds': kinds,
                      'geometry': geometry, 'counts': torch.tensor(counts), 'curves': torch.tensor(curves),
                      'repeats': torch.tensor(repeats), 'lengths': torch.tensor(lengths), 'combo': torch.tensor(combo)}
            filename = 'geometry/'+row['chart_sha256']+'.pt'
            torch.save(tensor, output/filename)
            geometries.append({'file': filename, 'sha256': hashlib.sha256((output/filename).read_bytes()).hexdigest(),
                               'source_component': row['source_component'], 'split': row['split'],
                               'chart_sha256': row['chart_sha256'], 'events': len(events)})
        except ValueError as exc:
            errors.append({'chart_sha256': row['chart_sha256'], 'error': str(exc)})
        if len(rows) % 100 == 0: print(json.dumps({'charts': len(rows), 'geometry': len(geometries)}), flush=True)
    if len(rows) != len(source['charts']) or len({r['chart_sha256'] for r in rows}) != len(rows):
        raise ValueError('native collection must cover each frozen chart exactly once')
    manifest = {**source, 'charts': rows, 'label_source': 'native .osu millisecond start/end times',
                'parent_manifest_sha256': native['frozen_manifest_sha256']}
    validate_manifest(manifest)
    (output/'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    geometry_manifest = {'schema': 'osu-native-geometry-flow-cache/1', 'charts': geometries,
                        'feature_pipeline_key': source['feature_pipeline_key'],
                        'rhythm_manifest_sha256': hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()}
    (output/'geometry-manifest.json').write_text(json.dumps(geometry_manifest, separators=(',', ':')))
    audit = {'native_charts': len(rows), 'geometry_charts': len(geometries), 'audio_hashes_verified': len(checked_audio),
             'splits': dict(Counter(r['split'] for r in geometries)), 'excluded_geometry': errors,
             'source_status': dict(Counter(r['source']['beatmap_status'] for r in rows)),
             'rhythm_manifest_sha256': geometry_manifest['rhythm_manifest_sha256'],
             'geometry_manifest_sha256': hashlib.sha256((output/'geometry-manifest.json').read_bytes()).hexdigest()}
    (output/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-root', required=True)
    p.add_argument('--native-root', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    build(a.audio_root, a.native_root, a.output_dir)
