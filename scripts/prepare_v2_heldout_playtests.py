"""Recover verified test audio and native reference charts from the pinned source."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from scripts.materialize_v2_data import hf_sources
from src.v2.mapspec import compile_osu, mapspec_from_osu
from src.v2.preprocess import oracle_timing_hypothesis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.records_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wanted_keys = {'000002', '000033', '000057'}
    groups = {}
    for path in sorted(root.glob('*.pt.index.json')):
        row = json.loads(path.read_text())['metadata']
        if row['split'] == 'test' and row['source']['source_record_id'] in wanted_keys:
            groups.setdefault(row['source']['source_record_id'], []).append(path.with_name(path.name.removesuffix('.index.json')))
    if set(groups) != wanted_keys:
        raise RuntimeError('Requested playtest sources are not all present in the test split')
    chosen = {}
    for key, paths in groups.items():
        records = [(p, torch.load(p, map_location='cpu', weights_only=True)) for p in paths]
        path, record = min(records, key=lambda item: abs(item[1]['metadata']['materialization']['measured_stars'] - 4.5))
        chosen[key] = (path, record)
    songs = []
    for audio, maps, provenance in hf_sources('b82117a5de13b7df5290482aed7cffc49cb76e6d', 58):
        key = provenance['source_record_id']
        if key not in chosen:
            continue
        path, record = chosen[key]
        source = record['metadata']['source']
        if hashlib.sha256(audio).hexdigest() != source['audio_sha256']:
            raise RuntimeError('Recovered audio differs from the held-out record')
        chart_hash = record['metadata']['materialization']['chart_sha256']
        text = next(m['content'] for m in maps if hashlib.sha256(m['content'].encode()).hexdigest() == chart_hash)
        spec = mapspec_from_osu(text)
        folder = output / f'heldout_{key}'
        folder.mkdir()
        audio_path = folder / 'audio.mp3'
        audio_path.write_bytes(audio)
        reference = replace(spec, general=replace(spec.general, audio_filename='audio.mp3'),
                            metadata=replace(spec.metadata, version='Human reference - ' + spec.metadata.version,
                                             beatmap_id=0, beatmap_set_id=-1))
        reference_path = folder / 'human-reference.osu'
        reference_path.write_text(compile_osu(reference), encoding='utf-8')
        song = dict(id=f'heldout_{key}', audio=str(audio_path), reference=str(reference_path),
                    title=spec.metadata.title, artist=spec.metadata.artist,
                    target_star=record['metadata']['materialization']['measured_stars'],
                    ar=spec.difficulty.approach_rate, cs=spec.difficulty.circle_size,
                    od=spec.difficulty.overall_difficulty, hp=spec.difficulty.hp_drain_rate,
                    slider_multiplier=spec.difficulty.slider_multiplier,
                    slider_tick_rate=spec.difficulty.slider_tick_rate,
                    oracle=oracle_timing_hypothesis(spec).to_dict(), split='test',
                    source_component_id=record['metadata']['source_component_id'], audio_sha256=source['audio_sha256'])
        songs.append(song)
        print(json.dumps({'prepared': song['id'], 'title': song['title'], 'target_star': song['target_star']}), flush=True)
        if len(songs) == len(chosen):
            break
    if len(songs) != len(chosen):
        raise RuntimeError('Could not recover all selected test songs')
    (output / 'heldout-songs.json').write_text(json.dumps(songs, indent=2))


if __name__ == '__main__':
    main()
