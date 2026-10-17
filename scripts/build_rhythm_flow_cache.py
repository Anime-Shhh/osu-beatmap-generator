"""Convert verified V2 records to shared audio + lightweight event manifests.

Accepts multiple materialized corpora, rejecting cross-split source collisions.
Never changes historical records or silently recomputes their split assignment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from src.v2.alignment import reference_events
from src.v2.dataset import validate_song_record
from src.rhythm_flow.data import SCHEMA,validate_manifest


def build(roots,output,budget_bytes=1024**3):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];audio_hashes={};pipelines=set();seen={};used=0
    for root in map(Path,roots):
        audit=json.loads((root/'materialization.json').read_text())
        for filename,digest in sorted(audit['record_sha256'].items()):
            path=root/filename
            if path.name!=filename or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('source record checksum/path mismatch')
            record=torch.load(path,weights_only=True,map_location='cpu');validate_song_record(record)
            md=record['metadata'];pipelines.add(md['feature_pipeline_key'])
            if len(pipelines)>1: raise ValueError('cannot combine incompatible audio feature pipelines')
            if md.get('materialization',{}).get('kind')!='real-audio-native-osu/v1':
                raise ValueError('only audited authentic records may enter the cache')
            identity=md['source'];key=identity['canonical_audio_sha256']
            chart=md['materialization']['chart_sha256']
            if (key,chart) in seen:
                if seen[key,chart] != md['split']:
                    raise ValueError('duplicate chart has conflicting historical splits')
                continue
            seen[key,chart] = md['split']
            times=record['rhythm_times_ms']
            if not torch.allclose(times,torch.arange(len(times))*10.,atol=.01):
                raise ValueError('flow cache requires native 10 ms timestamps')
            name=key+'.pt'
            if name not in audio_hashes:
                audio=record['rhythm_features'].half()
                needed=audio.numel()*2+4096
                if used+needed>budget_bytes: raise ValueError('compact cache exceeds explicit storage budget')
                torch.save({'audio':audio},output/name)
                used+=(output/name).stat().st_size
                audio_hashes[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
            rows.append({'audio_key':key,'audio_file':name,'audio_file_sha256':audio_hashes[name],
                         'source_component':md['source_component_id'],'split':md['split'],
                         'profile':record['profile'].tolist(),'events':reference_events(record),
                         'source':identity,'chart_sha256':chart})
            if len(rows)%50 == 0:
                print(json.dumps({'cached_charts':len(rows),'audio_files':len(audio_hashes),'audio_bytes':used}),flush=True)
    manifest={'schema':SCHEMA,'frame_ms':10,'feature_pipeline_key':next(iter(pipelines)),
              'charts':rows,'audio_files':len(audio_hashes),'audio_bytes':used,
              'label_source':'existing V2 native-chart codec times (includes codec quantization)'}
    validate_manifest(manifest)
    (output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
    print(json.dumps({'charts':len(rows),'audio_files':len(audio_hashes),'audio_bytes':used}),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records-dir',nargs='+',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--budget-mb',type=int,default=1024)
    a=p.parse_args();build(a.records_dir,a.output_dir,a.budget_mb*1024**2)

if __name__=='__main__': main()
