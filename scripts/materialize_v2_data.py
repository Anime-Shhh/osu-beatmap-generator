"""Materialize real audio + native .osu maps. Never fabricate ranked examples.

Use --osz-dir for local mapsets or --hf-revision for the pinned upstream corpus.
A fresh output directory is required. Rejected maps and source hashes are audited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

import torch

from src.v2.data import (SourceIdentity, SourceSplitAssignment, build_immutable_source_split_manifest,
                         source_component_groups)
from src.v2.evaluation import RosuDifficultyCalculator
from src.v2.inference import extract_audio_features, load_audio
from src.v2.mapspec import mapspec_from_osu
from src.v2.preprocess import build_song_record, write_song_record


def osz_sources(directory: Path):
    """Read members without extracting paths or claiming an unverified rank."""
    for archive in sorted(directory.glob('*.osz')):
        revision = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as bundle:
            grouped = {}
            for name in bundle.namelist():
                if not name.endswith('.osu'):
                    continue
                text = bundle.read(name).decode('utf-8-sig')
                audio_name = next((line.split(':', 1)[1].strip() for line in text.splitlines()
                                   if line.startswith('AudioFilename:')), None)
                if audio_name is None:
                    raise ValueError(f'{archive.name}:{name} has no AudioFilename')
                member = str(PurePosixPath(name).parent / audio_name.replace('\\', '/'))
                grouped.setdefault(member, []).append({'content': text, 'status': 'unverified-local'})
            for member, maps in grouped.items():
                yield bundle.read(member), maps, {
                    'source_dataset': 'local-osz', 'source_revision': revision,
                    'source_record_id': f'{revision}:{member}',
                }


def hf_sources(revision: str, limit: int, start: int = 0):
    from datasets import Audio, load_dataset
    dataset = load_dataset('project-riz/osu-beatmaps', 'original', revision=revision,
                           split='train', streaming=True)
    dataset = dataset.cast_column('mp3', Audio(decode=False))
    for index, example in enumerate(dataset):
        if index >= start + limit:
            break
        if index < start:
            continue
        metadata = example['json']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        audio = example['mp3']
        if isinstance(audio, dict):
            audio = audio.get('bytes')
        if not isinstance(audio, bytes):
            raise ValueError('Upstream record lacks original encoded audio bytes')
        # Original variant hashes encoded audio. Do not trust an identifier as
        # proof that bytes came from the named dataset revision.
        if hashlib.sha256(audio).hexdigest() != metadata['audio_hash']:
            raise ValueError('Upstream encoded audio hash mismatch')
        maps = []
        for beatmap in metadata.get('beatmaps', []):
            if str(beatmap.get('mode')) != '0' or str(beatmap.get('approved')) not in {'1', '2'}:
                continue
            maps.append({**beatmap, 'status': 'ranked' if str(beatmap['approved']) == '1' else 'approved'})
        yield audio, maps, {
            'source_dataset': 'project-riz/osu-beatmaps', 'source_revision': revision,
            'source_record_id': str(example['__key__']),
        }


def materialize(sources, output: Path, *, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory; historical datasets must not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    calculator = RosuDifficultyCalculator()
    entries, rejected = [], []
    curves = Counter()
    pipeline_key = None
    seen_maps = set()
    # iLab's node-local /tmp has a 2 GB per-user quota. Keep corpus staging
    # alongside the output instead; progressively release staged records so
    # finalization does not temporarily require two full corpus copies.
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=output) as temp:
        temp = Path(temp)
        for audio, maps, provenance in sources:
            if not maps:
                continue
            audio_hash = hashlib.sha256(audio).hexdigest()
            audio_path = temp / 'input.audio'
            audio_path.write_bytes(audio)
            waveform, sample_rate = load_audio(audio_path)
            features, _ = extract_audio_features(waveform, sample_rate, audio_sha256=audio_hash)
            pipeline_key = features.feature_cache.pipeline_key
            for raw in maps:
                text = raw['content']
                chart_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if (audio_hash, chart_hash) in seen_maps:
                    continue
                seen_maps.add((audio_hash, chart_hash))
                try:
                    spec = mapspec_from_osu(text)
                    if spec.general.mode != 0:
                        continue
                    difficulty = calculator.calculate(text)
                    if not difficulty.stars or not spec.events:
                        raise ValueError('Empty map or zero difficulty')
                    source = SourceIdentity(
                        audio_sha256=audio_hash,
                        canonical_audio_sha256=features.feature_cache.canonical_audio.pcm_sha256,
                        audio_duration_ms=features.duration_ms,
                        beatmap_id=str(spec.metadata.beatmap_id) if spec.metadata.beatmap_id > 0 else None,
                        beatmap_set_id=str(spec.metadata.beatmap_set_id) if spec.metadata.beatmap_set_id > 0 else None,
                        mapper_id=str(raw['creator_id']) if raw.get('creator_id') else None,
                        **provenance, beatmap_status=raw['status'],
                        rights_status='research-only; source audio and chart rights retained by owners',
                        rights_url='https://osu.ppy.sh/legal/terms',
                    )
                    record = build_song_record(
                        map_spec=spec, source_assignment=SourceSplitAssignment(source, source_component_groups([source])[source], 'train'),
                        feature_cache=features.feature_cache, rhythm_features=features.rhythm_features,
                        rhythm_times_ms=features.rhythm_times_ms, semantic_features=features.semantic_features,
                        semantic_times_ms=features.semantic_times_ms, target_star=difficulty.stars,
                    )
                    record['metadata']['materialization'] = {
                        'kind': 'real-audio-native-osu/v1', 'chart_sha256': chart_hash,
                        'calculator_id': calculator.calculator_id, 'measured_stars': difficulty.stars,
                    }
                    filename = f'{audio_hash[:16]}-{chart_hash[:16]}.pt'
                    torch.save(record, temp / filename)
                    entries.append((filename, source))
                    curves.update(event.curve_type for event in spec.events if event.control_points)
                except (ValueError, TypeError, KeyError) as exc:
                    rejected.append({'source': provenance['source_record_id'], 'chart_sha256': chart_hash,
                                     'error': str(exc)})
            print(json.dumps({'accepted': len(entries), 'rejected': len(rejected)}), flush=True)
        if not entries:
            (output / 'rejected.json').write_text(json.dumps(rejected, indent=2))
            raise ValueError('No authentic maps survived preprocessing; see rejected.json')
        manifest = build_immutable_source_split_manifest([s for _, s in entries], seed=seed)
        assignments = {SourceIdentity.from_dict(row['source']): row for row in manifest['records']}
        checksums = {}
        for filename, source in entries:
            record = torch.load(temp / filename, weights_only=True, map_location='cpu')
            row = assignments[source]
            record['metadata'].update(split=row['split'], source_component_id=row['component_id'])
            write_song_record(record, output / filename)
            checksums[filename] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
            (temp / filename).unlink()
    report = {
        'schema': 'osu-v2-real-corpus/1', 'feature_pipeline_key': pipeline_key,
        'calculator_id': calculator.calculator_id, 'records': len(entries),
        'source_components': {k: len(v) for k, v in manifest['splits'].items()},
        'slider_curves': dict(curves), 'record_sha256': checksums, 'rejected': rejected,
    }
    (output / 'splits.json').write_text(json.dumps(manifest, indent=2))
    (output / 'materialization.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {'record_sha256', 'rejected'}}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--osz-dir', type=Path)
    inputs.add_argument('--hf-revision', help='immutable 40-character dataset commit SHA')
    parser.add_argument('--max-tracks', type=int, default=1000)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_tracks < 1:
        parser.error('--max-tracks must be positive')
    if args.hf_revision and (len(args.hf_revision) != 40 or any(c not in '0123456789abcdef' for c in args.hf_revision)):
        parser.error('--hf-revision must be an immutable lowercase commit SHA')
    sources = osz_sources(args.osz_dir) if args.osz_dir else hf_sources(args.hf_revision, args.max_tracks)
    materialize(sources, args.output_dir, seed=args.seed)


if __name__ == '__main__':
    main()
