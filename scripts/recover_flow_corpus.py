"""Verify and finalize the last complete acquisition journal into a fresh corpus."""
import argparse
from collections import Counter
import hashlib
import json
import os
import errno
import shutil
from pathlib import Path

from scripts.expand_native_flow_corpus import native_index, digest
from scripts.extend_rhythm_flow_cache import preserve_splits
from src.rhythm_flow.data import validate_manifest


def retain(source, destination, *, feature=False):
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV: raise
        if feature:
            destination.symlink_to(source.resolve())
        else:
            shutil.copyfile(source, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial', type=Path, required=True)
    p.add_argument('--base-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = json.loads((a.base_cache/'manifest.json').read_text())
    journal = json.loads((a.partial/'acquisition-progress.json').read_text())
    rows = journal['charts']
    old = {r['chart_sha256']: r for r in base['charts']}
    if len({r['chart_sha256'] for r in rows}) != len(rows):
        raise ValueError('duplicate chart identities')
    if [r['chart_sha256'] for r in rows[:len(old)]] != list(old):
        raise ValueError('journal does not retain base chart ordering')
    for r in rows[:len(old)]:
        expected = old[r['chart_sha256']]
        if any(r[k] != expected[k] for k in ('split', 'source', 'events', 'profile', 'audio_file_sha256')):
            raise ValueError('journal changed historical labels, sources or splits')
    preserve_splits(rows, len(old))
    base.update(charts=rows, parent_manifest_sha256=digest(a.base_cache/'manifest.json'),
                label_source='native-osu-ms',
                recovery={'journal_sha256': digest(a.partial/'acquisition-progress.json'),
                          'scanned_records': journal['scanned'], 'complete_requested_scan': False,
                          'reason': 'acquisition job failed; recover only journaled, verified files'})
    validate_manifest(base)
    if any(r['source'].get('beatmap_set_id') == '1147361' for r in rows):
        raise ValueError('Kaguya development map appeared in recovery; isolate it before training')
    a.output.mkdir(parents=True, exist_ok=False)
    cache, native = a.output/'audio-cache', a.output/'native'
    cache.mkdir(); (native/'maps').mkdir(parents=True); (native/'audio').mkdir()
    files = {}
    for r in rows:
        name = r['audio_file']
        if name not in files:
            path = a.partial/'audio-cache'/name
            if digest(path) != r['audio_file_sha256']: raise ValueError('feature checksum mismatch')
            retain(path, cache/name, feature=True); files[name] = path.stat().st_size
        path = a.partial/'native'/'maps'/(r['chart_sha256']+'.osu')
        if digest(path) != r['chart_sha256']: raise ValueError('native checksum mismatch')
        retain(path, native/'maps'/path.name)
    audio_paths = {}
    allowed_audio = {r['source']['audio_sha256'] for r in rows if r['split'] == 'val'}
    for path in (a.partial/'native'/'audio').glob('*.audio'):
        sha = digest(path)
        if sha != path.stem or sha not in allowed_audio: raise ValueError('validation audio identity mismatch')
        name = 'audio/'+path.name
        retain(path, native/name); audio_paths[sha] = name
    base.update(audio_files=len(files), audio_bytes=sum(files.values()))
    (cache/'manifest.json').write_text(json.dumps(base, separators=(',', ':')))
    revisions = {r['source']['source_revision'] for r in rows}
    if len(revisions) != 1: raise ValueError('unexpected dataset revisions')
    index = native_index(rows, native, digest(cache/'manifest.json'), revisions.pop(), audio_paths)
    (native/'manifest.json').write_text(json.dumps(index, separators=(',', ':')))
    summary = {'charts': len(rows), 'added_charts': len(rows)-len(old), 'audio_files': len(files),
               'groups': index['groups'], 'splits': dict(Counter(r['split'] for r in rows)),
               'manifest_sha256': digest(cache/'manifest.json'), 'recovery': base['recovery'],
               'validation_audio_files': len(audio_paths)}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
