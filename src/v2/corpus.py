"""Verify dataset contents against the split manifest before optimization."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from .data import SourceIdentity, validate_source_split_manifest
from .dataset import validate_song_record


def audit_training_corpus(records_dir, split_manifest, *, allow_diagnostic_data=False):
    validate_source_split_manifest(split_manifest)
    expected = {SourceIdentity.from_dict(row['source']): (row['component_id'], row['split'])
                for row in split_manifest['records']}
    root = Path(records_dir)
    report_path = root / 'materialization.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    hashes = report.get('record_sha256', {})
    paths = sorted(root.glob('*.pt'))
    if hashes and set(hashes) != {p.name for p in paths}:
        raise ValueError('Materialization manifest and record files differ')
    seen, components = set(), {'train': set(), 'val': set(), 'test': set()}
    bar_counts = Counter()
    real = bool(paths) and report.get('schema') == 'osu-v2-real-corpus/1' and bool(hashes)
    legacy_synthetic = {hashlib.sha256(f'authentic_audio_track_{i}'.encode()).hexdigest() for i in range(6)}
    for path in paths:
        if hashes and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
            raise ValueError(f'{path.name}: contents differ from materialization manifest')
        record = torch.load(path, map_location='cpu', weights_only=True)
        metadata = record['metadata']
        source = SourceIdentity.from_dict(metadata['source'])
        if source.audio_sha256 in legacy_synthetic:
            raise ValueError('Legacy sine-wave fixture falsely labeled ranked: rebuild records from real audio/.osu sources')
        validate_song_record(record)
        membership = (metadata['source_component_id'], metadata['split'])
        if expected.get(source) != membership:
            raise ValueError(f'{path.name}: source/split differs from supplied split manifest')
        seen.add(source)
        components[membership[1]].add(membership[0])
        bar_counts[membership[1]] += len(record['bar_targets'])
        marker = metadata.get('materialization', {})
        real &= marker.get('kind') == 'real-audio-native-osu/v1'
        real &= marker.get('calculator_id') == report.get('calculator_id')
    if seen != set(expected):
        raise ValueError('Split manifest contains missing source records')
    # This separates a pipeline smoke test from a training run; it is not a
    # claim that this small minimum is sufficient for good generative quality.
    counts = {key: len(value) for key, value in components.items()}
    enough = counts['train'] >= 20 and counts['val'] >= 2 and counts['test'] >= 2
    ready = real and enough
    if not ready and not allow_diagnostic_data:
        raise ValueError(
            f'Unverified or smoke-sized corpus ({counts}). Materialize real maps with at least '
            '20 train / 2 val / 2 test source components. --allow-diagnostic-data is only for smoke tests.'
        )
    return {'production_ready': bool(ready), 'records': len(paths), 'source_components': counts,
            'bars': dict(bar_counts), 'calculator_id': report.get('calculator_id'),
            'materialization_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest() if report else None}
