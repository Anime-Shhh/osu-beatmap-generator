"""A bounded set of fresh examples; one sample per source, no reference-score selection."""
import hashlib
import time
from pathlib import Path
from .contracts import artifact_path
from .contracts import write_json,digest
from .data import read_manifest,selected_rows
from .generate import generate


def examples(manifest,upstream,output,codec_path=None,generator_path=None,seconds=3600):
    from scripts.materialize_v2_data import hf_sources
    from src.v2.mapspec import mapspec_from_osu
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=read_manifest(manifest);wanted={}
    # Select identities before inference; never choose by matching human events.
    for r in selected_rows(m,'val'):
        spec=mapspec_from_osu(artifact_path(r['native_path']).read_text(encoding='utf-8-sig'))
        if spec.general.audio_filename.lower().endswith('.mp3'):wanted[r['audio_key']]=r
        if len(wanted)==2:break
    if len(wanted)!=2:raise ValueError('two MP3 validation sources required')
    selected=list(wanted);reports=[];started=time.monotonic()
    for audio,_,_ in hf_sources(m['revision'],1500):
        if time.monotonic()-started>seconds:raise TimeoutError('example generation budget')
        key=hashlib.sha256(audio).hexdigest()
        if key not in wanted:continue
        path=output/'input.mp3';path.write_bytes(audio)
        try:
            audit=generate(path,4.5,upstream,output/key[:12],manifest,codec_path,generator_path)
            if audit['audio_sha256']!=key:raise ValueError('example source changed')
        finally:path.unlink(missing_ok=True)
        reports.append({'source':wanted.pop(key)['group'],'audit':audit})
        if not wanted:break
    write_json(output/'examples.json',{'complete':not wanted,'selected_audio':selected,
        'manifest_sha256':digest(manifest),'examples':reports,'ranked_ready':False})
    if wanted:raise RuntimeError('missing frozen example audio')
