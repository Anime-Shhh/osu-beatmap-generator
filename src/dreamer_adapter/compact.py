"""Losslessly compress only this pilot's new cache, preserving splits and native inputs."""
import json
from pathlib import Path
import numpy as np
from .data import read_manifest,freeze_references
from .contracts import digest,write_json


def compact(source,output):
    source=Path(source);output=Path(output)
    m=read_manifest(source/'manifest.json',verify=True)
    output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((source/'frozen-splits.json').read_text())
    write_json(output/'frozen-splits.json',frozen)
    m['reference_chart_hashes']=freeze_references(frozen,m['rows'])
    for a in m['audio'].values():
        original=np.load(source/a['path'])
        name=a['path'].replace('.npy','.npz');np.savez_compressed(output/name,spec=original)
        with np.load(output/name) as f:
            if not np.array_equal(original,f['spec']):raise ValueError('compression changed audio')
        a.update(path=name,sha256=digest(output/name))
    for r in m['rows']:
        with np.load(source/r['map_path']) as f:values={k:f[k] for k in f.files}
        np.savez_compressed(output/r['map_path'],**values)
        with np.load(output/r['map_path']) as f:
            if any(not np.array_equal(v,f[k]) for k,v in values.items()):raise ValueError('compression changed chart')
        r['map_sha256']=digest(output/r['map_path'])
    m['cache_bytes']=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    m['uncompressed_manifest_sha256']=digest(source/'manifest.json')
    write_json(output/'manifest.json',m)
    read_manifest(output/'manifest.json',verify=True)
    write_json(output/'compression-audit.json',{'source':str(source),'source_sha256':digest(source/'manifest.json'),
        'output_sha256':digest(output/'manifest.json'),'verified_equal':True,'bytes':m['cache_bytes']})
    return m
