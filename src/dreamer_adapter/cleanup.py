"""Explicit download-cache cleanup and proof-carrying generated-intermediate deletion."""
import os
from pathlib import Path
import shutil
from .contracts import digest, write_json, quota_free_bytes


def cleanup(output, pip_cache=None, candidates=(), protected=()):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    before = quota_free_bytes()
    ledger = {'before_free_bytes': before, 'deleted': [], 'protected': list(map(str,protected))}
    write_json(output/'ledger.json', ledger)
    if pip_cache:
        p = Path(pip_cache)
        if p.is_symlink() or p.resolve() != Path('/common/home/asj102/.cache/pip'):
            raise ValueError('only the explicitly authorized pip download cache is allowed')
        # Installed environments are never inside pip's download cache.
        if p.exists():
            ledger['pip_logical_bytes'] = sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
            write_json(output/'ledger.json', ledger)
            shutil.rmtree(p)
            ledger['deleted'].append(str(p))
    for entry in candidates:
        p = Path(entry['path'])
        if p.is_symlink() or str(p.resolve()) in {str(Path(x).resolve()) for x in protected}:
            raise ValueError('protected artifact')
        if not entry.get('evaluation_complete') or not entry.get('seed') or not entry.get('source_revision'):
            raise ValueError('missing evaluation provenance')
        metrics = Path(entry['metrics'])
        if not metrics.is_file() or digest(metrics) != entry['metrics_sha256']:
            raise ValueError('metrics not retained')
        if p.suffix not in ('.osu','.json','.npy','.npz') or not p.name.startswith('candidate-'):
            raise ValueError('not an explicitly disposable candidate')
        if digest(p) != entry['sha256']: raise ValueError('candidate changed')
        ledger.setdefault('proofs', []).append(entry); write_json(output/'ledger.json', ledger)
        p.unlink(); ledger['deleted'].append(str(p)); write_json(output/'ledger.json', ledger)
    ledger['after_free_bytes'] = quota_free_bytes()
    ledger['actual_quota_bytes_freed'] = ledger['after_free_bytes']-before
    write_json(output/'ledger.json', ledger)
    return ledger
