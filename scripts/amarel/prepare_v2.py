"""Verify copied V2 records and make a ranked-only view without changing splits."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import torch
from src.dreamer_adapter.contracts import write_json, digest
from src.v2.corpus import audit_training_corpus
from src.v2.data import SourceIdentity, validate_source_split_manifest


def prepare(source,output):
    source=Path(source);output=Path(output)
    manifest=json.loads((source/'splits.json').read_text())
    audit=audit_training_corpus(source,manifest)
    report=json.loads((source/'materialization.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    kept=set();hashes={};excluded=[]
    for name,sha in report['record_sha256'].items():
        path=source/name;record=torch.load(path,map_location='cpu',weights_only=True)
        identity=SourceIdentity.from_dict(record['metadata']['source'])
        if identity.beatmap_status!='ranked':
            excluded.append(dict(filename=name,reason='not ranked',status=identity.beatmap_status));continue
        os.link(path,output/name)
        index=path.with_name(path.name+'.index.json')
        if index.exists():os.link(index,output/index.name)
        kept.add(identity);hashes[name]=sha
    filtered=dict(manifest)
    filtered['records']=[r for r in manifest['records'] if SourceIdentity.from_dict(r['source']) in kept]
    filtered['splits']={s:sorted({r['component_id'] for r in filtered['records'] if r['split']==s}) for s in ('train','val','test')}
    validate_source_split_manifest(filtered)
    report.update(records=len(hashes),record_sha256=hashes,source_components={s:len(v) for s,v in filtered['splits'].items()})
    write_json(output/'splits.json',filtered);write_json(output/'materialization.json',report)
    write_json(output/'preparation.json',dict(parent_audit=audit,parent_manifest_sha256=digest(source/'splits.json'),excluded=excluded,records=len(hashes),source_components=report['source_components'],policy='ranked only; original split ownership and tensor bytes unchanged; hardlinks to verified imported records'))
    print(json.dumps(dict(records=len(hashes),source_components=report['source_components'],excluded=excluded)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);prepare(**vars(p.parse_args()))
