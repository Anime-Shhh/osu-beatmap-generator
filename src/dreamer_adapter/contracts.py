import hashlib
import json
import os
from pathlib import Path
import subprocess

from . import UPSTREAM_COMMIT, UPSTREAM_WEIGHTS, FEATURE_KEY


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); os.replace(tmp, path)


def bind(manifest, codec='upstream', normalization=None):
    return dict(upstream=UPSTREAM_COMMIT, weights=UPSTREAM_WEIGHTS, features=FEATURE_KEY,
                manifest=identity(manifest), codec=codec, normalization=normalization)


def validate_binding(actual, expected):
    if actual != expected:
        raise ValueError('incompatible checkpoint/data/codec/normalization contract')


def quota_free_bytes():
    # NFS quotas, not filesystem free space. -w keeps filesystem and usage together.
    text = subprocess.run(['quota', '-w'], text=True, capture_output=True, check=False).stdout
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].endswith(':/common/home'):
            used, soft, hard = [int(x.rstrip('*')) for x in fields[1:4]]
            return (min(x for x in (soft, hard) if x > 0)-used)*1024
    raise RuntimeError('Could not determine home quota; refusing unbounded writes')


def require_space(gib=8):
    free = quota_free_bytes()
    if free < gib*1024**3: raise RuntimeError(f'Home quota free {free/1024**3:.2f} GiB < {gib}')
    return free


def storage_inventory():
    """Allocated blocks (du counts hardlinks once), plus quota; never delete on failure."""
    result={'home_free_bytes':quota_free_bytes(),'roots':{}}
    for path in ('/common/home/asj102','/common/users/asj102/osu_project'):
        run=subprocess.run(['du','-x','-B1','--max-depth=2',path],capture_output=True,text=True,timeout=180)
        result['roots'][path]={'allocated_bytes_by_path':run.stdout,'errors':run.stderr,'exit_code':run.returncode}
    return result


def validate_sources(rows):
    keys, groups = {}, {}
    for r in rows:
        s = r['source']; split = r['split']
        if split not in ('train','val','test'): raise ValueError('invalid split')
        if str(s.get('beatmap_set_id')) == '1147361': raise ValueError('Kaguya is development-only')
        if s.get('beatmap_status') != 'ranked': raise ValueError('non-ranked positive target')
        for key in (r['group'], s['audio_sha256'], s.get('canonical_audio_sha256'), s.get('beatmap_set_id')):
            if key is None: continue
            if key in keys and keys[key] != split: raise ValueError('source leakage')
            keys[key] = split
        if r['sha256'] in groups: raise ValueError('duplicate chart')
        groups[r['sha256']] = split


def evaluation_groups(rows, count=24):
    validate_sources(rows)
    result = {s: sorted({r['group'] for r in rows if r['split']==s})[:count] for s in ('val','test')}
    if any(len(v)!=count for v in result.values()): raise ValueError('insufficient frozen evaluation sources')
    return result
