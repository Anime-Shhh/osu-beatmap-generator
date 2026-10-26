"""Expand authentic native labels/audio features with frozen historical splits.

Bound the scan and new disk use, retain every accepted native chart, and retain
a small source-separated validation audio set for full-song listening/evaluation.
The progress journal is deliberately not a trainable manifest.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from scripts.materialize_v2_data import hf_sources
from scripts.extend_rhythm_flow_cache import preserve_splits
from src.rhythm_flow.data import validate_manifest
from src.v2.data import SourceIdentity
from src.v2.evaluation import RosuDifficultyCalculator
from src.v2.inference import extract_audio_features, load_audio
from src.v2.mapspec import mapspec_from_osu
from src.v2.preprocess import compact_profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_index(rows, output, manifest_hash, revision, audio_paths):
    entries = []
    for row in rows:
        name = 'maps/'+row['chart_sha256']+'.osu'
        if digest(output/name) != row['chart_sha256']:
            raise ValueError('native chart checksum mismatch')
        entries.append({'path': name, 'sha256': row['chart_sha256'],
            'split': row['split'], 'group': row['source_component'], 'source': row['source'],
            'audio_path': audio_paths.get(row['source']['audio_sha256'])})
    return {'schema': 'osu-geometry-native/1', 'revision': revision,
            'frozen_manifest_sha256': manifest_hash, 'charts': entries,
            'counts': dict(Counter(r['split'] for r in rows)),
            'groups': {s: len({r['source_component'] for r in rows if r['split'] == s})
                       for s in ('train', 'val', 'test')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-cache', type=Path, required=True)
    p.add_argument('--base-native', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--hf-revision', required=True)
    p.add_argument('--start-track', type=int, default=500)
    p.add_argument('--max-tracks', type=int, default=1000)
    p.add_argument('--budget-mb', type=int, default=4096)
    p.add_argument('--validation-audio-sources', type=int, default=12)
    a = p.parse_args()
    if len(a.hf_revision) != 40 or any(c not in '0123456789abcdef' for c in a.hf_revision):
        p.error('immutable dataset SHA required')
    if min(a.max_tracks, a.budget_mb) < 1 or min(a.start_track, a.validation_audio_sources) < 0:
        p.error('invalid acquisition budgets')
    torch.set_num_threads(1)
    manifest = json.loads((a.base_cache/'manifest.json').read_text())
    validate_manifest(manifest)
    parent_hash = digest(a.base_cache/'manifest.json')
    base_native = json.loads((a.base_native/'manifest.json').read_text())
    if base_native['frozen_manifest_sha256'] not in (parent_hash, manifest.get('parent_manifest_sha256')):
        raise ValueError('native collection does not bind base cache or its verified parent')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    cache, native = a.output_dir/'audio-cache', a.output_dir/'native'
    cache.mkdir(); (native/'maps').mkdir(parents=True); (native/'audio').mkdir()
    rows = manifest['charts']
    old_count = len(rows)
    known = {r['chart_sha256']: r for r in rows}
    if len(known) != old_count:
        raise ValueError('duplicate base chart identity')
    native_lookup = {r['sha256']: r for r in base_native['charts']}
    for sha, row in known.items():
        entry = native_lookup[sha]
        if entry['split'] != row['split'] or entry['group'] != row['source_component']:
            raise ValueError('base native source/split mismatch')
        path = (a.base_native/entry['path']).resolve()
        if not path.is_relative_to(a.base_native.resolve()) or digest(path) != sha:
            raise ValueError('base native path/hash mismatch')
        os.link(path, native/'maps'/f'{sha}.osu')
    files = {r['audio_file']: r['audio_file_sha256'] for r in rows}
    for name, sha in files.items():
        if digest(a.base_cache/name) != sha:
            raise ValueError('base feature checksum mismatch')
        os.link(a.base_cache/name, cache/name)
    selected_val = list(dict.fromkeys(r['source_component'] for r in rows if r['split'] == 'val'))[:a.validation_audio_sources]
    val_hashes = {r['source']['audio_sha256'] for r in rows if r['source_component'] in selected_val}
    saved_audio, used, audio_bytes, rejected = {}, 0, 0, []
    calc = RosuDifficultyCalculator()
    scanned = 0
    with tempfile.TemporaryDirectory(dir=a.output_dir, prefix='.decode-') as tmp:
        path = Path(tmp)/'input.audio'
        # One pass also retrieves original audio for old, frozen validation songs.
        for index, (audio, maps, provenance) in enumerate(hf_sources(a.hf_revision, a.start_track+a.max_tracks)):
            scanned = index+1
            sha_audio = hashlib.sha256(audio).hexdigest()
            if sha_audio in val_hashes and sha_audio not in saved_audio:
                name = 'audio/'+sha_audio+'.audio'
                (native/name).write_bytes(audio)
                saved_audio[sha_audio] = name; audio_bytes += len(audio)
            if index < a.start_track or not maps:
                continue
            path.write_bytes(audio)
            try:
                wave, sr = load_audio(path)
            except RuntimeError as exc:
                rejected.append({'source': provenance['source_record_id'], 'reason': str(exc)})
                continue
            if wave.shape[-1]/sr > 600:
                rejected.append({'source': provenance['source_record_id'], 'reason': 'length budget >10min'})
                continue
            features, _ = extract_audio_features(wave, sr, audio_sha256=sha_audio)
            if features.feature_cache.pipeline_key != manifest['feature_pipeline_key']:
                raise ValueError('feature pipeline changed')
            key = features.feature_cache.canonical_audio.pcm_sha256
            name = key+'.pt'
            if name not in files and used+features.rhythm_features.numel()*2+8192 > a.budget_mb*1024**2:
                rejected.append({'source': provenance['source_record_id'], 'reason': 'feature budget reached'})
                break
            pending, contents = [], {}
            for raw in maps:
                sha = hashlib.sha256(raw['content'].encode()).hexdigest()
                if sha in known or sha in contents:
                    continue
                try:
                    spec = mapspec_from_osu(raw['content'])
                    difficulty = calc.calculate(raw['content'])
                    if spec.general.mode != 0 or not spec.events or not difficulty.stars:
                        raise ValueError('empty/nonstandard/unmeasurable chart')
                    events = [(spec.timing_map.beat_to_time(e.beat_position),
                        spec.timing_map.beat_to_time(e.end_beat if e.end_beat is not None else e.beat_position),
                        int(e.kind)) for e in spec.events]
                    if any(s < 0 or e < s or e > features.duration_ms+100 for s, e, _ in events):
                        raise ValueError('chart outside audio')
                    d = spec.difficulty
                    source = SourceIdentity(audio_sha256=sha_audio, canonical_audio_sha256=key,
                        audio_duration_ms=features.duration_ms,
                        beatmap_id=str(spec.metadata.beatmap_id) if spec.metadata.beatmap_id > 0 else None,
                        beatmap_set_id=str(spec.metadata.beatmap_set_id) if spec.metadata.beatmap_set_id > 0 else None,
                        mapper_id=str(raw['creator_id']) if raw.get('creator_id') else None,
                        **provenance, beatmap_status=raw['status'],
                        rights_status='research-only; rights retained by owners', rights_url='https://osu.ppy.sh/legal/terms')
                    profile = compact_profile(target_star=difficulty.stars, approach_rate=d.approach_rate,
                        circle_size=d.circle_size, overall_difficulty=d.overall_difficulty,
                        hp_drain_rate=d.hp_drain_rate, slider_multiplier=d.slider_multiplier,
                        slider_tick_rate=d.slider_tick_rate)
                    pending.append({'audio_key': key, 'audio_file': name, 'profile': profile.tolist(),
                        'events': events, 'source': source.to_dict(), 'chart_sha256': sha,
                        'label_source': 'native-osu-ms'})
                    contents[sha] = raw['content']
                except (ValueError, TypeError, KeyError) as exc:
                    rejected.append({'source': provenance['source_record_id'], 'chart': sha, 'reason': str(exc)})
            if pending:
                # Validate on copies: a rejected bridge must not mutate frozen rows.
                proposal = [dict(r) for r in rows+pending]
                try:
                    preserve_splits(proposal, old_count)
                except ValueError as exc:
                    rejected.append({'source': provenance['source_record_id'], 'reason': str(exc)})
                    continue
                if name not in files:
                    torch.save({'audio': features.rhythm_features.half()}, cache/name)
                    files[name] = digest(cache/name); used += (cache/name).stat().st_size
                rows = proposal
                for row in rows[-len(pending):]:
                    row['audio_file_sha256'] = files[name]
                    sha = row['chart_sha256']
                    (native/'maps'/f'{sha}.osu').write_text(contents[sha])
                    known[sha] = row
            progress = {'scanned': scanned, 'new_charts': len(rows)-old_count,
                        'new_feature_bytes': used, 'validation_audio_files': len(saved_audio)}
            print(json.dumps(progress), flush=True)
            journal = a.output_dir/'acquisition-progress.json'
            journal.with_suffix('.tmp').write_text(json.dumps({**progress, 'charts': rows, 'rejected': rejected}, separators=(',', ':')))
            journal.with_suffix('.tmp').replace(journal)
    preserve_splits(rows, old_count)
    manifest.update(charts=rows, audio_files=len(files), audio_bytes=manifest['audio_bytes']+used,
        label_source='native-osu-ms', parent_manifest_sha256=parent_hash,
        expansion={'revision': a.hf_revision, 'start_track': a.start_track, 'max_tracks': a.max_tracks,
                   'scanned': scanned, 'new_feature_bytes': used, 'rejected': rejected})
    validate_manifest(manifest)
    (cache/'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    collection = native_index(rows, native, digest(cache/'manifest.json'), a.hf_revision, saved_audio)
    (native/'manifest.json').write_text(json.dumps(collection, separators=(',', ':')))
    summary = {'charts': len(rows), 'new_charts': len(rows)-old_count, 'audio_files': len(files),
               'source_groups': collection['groups'], 'splits': collection['counts'],
               'new_feature_bytes': used, 'retained_validation_audio_bytes': audio_bytes,
               'validation_audio_files': len(saved_audio), 'rejected': len(rejected),
               'manifest_sha256': digest(cache/'manifest.json')}
    (a.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
