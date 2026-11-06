"""Materialize compact v8 features from hash-verified original audio, with fixed splits."""
import io
import json
from pathlib import Path
import numpy as np
from .contracts import digest, identity, write_json, validate_sources, evaluation_groups
from . import FEATURE_KEY, UPSTREAM_COMMIT


def prepare(native_manifest, output, scan_limit=1500, cap_bytes=4*1024**3):
    from scripts.materialize_v2_data import hf_sources
    from osu_dreamer.data.load_audio import load_wave, make_spec, get_frame_times, write_spec
    from osu_dreamer.data.beatmap.encode import write_beatmap
    from osu_dreamer.osu.beatmap import Beatmap
    native_manifest = Path(native_manifest); output = Path(output)
    source = json.loads(native_manifest.read_text())
    rows = [r for r in source['charts'] if r['source']['beatmap_status']=='ranked']
    validate_sources(rows)
    selected = evaluation_groups(rows)
    # Freeze the exact source list before acquisition/training. Never substitute failed sources.
    wanted = [r for r in rows if r['split']=='train' or r['group'] in selected[r['split']]]
    frozen = dict(schema='dreamer-source-splits/1', upstream=UPSTREAM_COMMIT, features=FEATURE_KEY,
                  parent_sha256=digest(native_manifest), revision=source['revision'], groups=selected, rows=wanted)
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output/'frozen-splits.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen: raise ValueError('frozen splits changed')
    write_json(frozen_path,frozen)
    progress_path = output/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'charts':[], 'audio':{}, 'rejected':[]}
    done = {r['sha256'] for r in progress['charts']}
    by_audio = {}
    for r in wanted: by_audio.setdefault(r['source']['audio_sha256'],[]).append(r)
    used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    for audio_bytes, _, provenance in hf_sources(source['revision'], scan_limit):
        import hashlib
        audio_sha = hashlib.sha256(audio_bytes).hexdigest()
        if audio_sha not in by_audio or all(r['sha256'] in done for r in by_audio[audio_sha]): continue
        audio_path = output/'input.audio'; audio_path.write_bytes(audio_bytes)
        backend='torchcodec'
        print(json.dumps({'decoding_audio':audio_sha,'source':provenance['source_record_id']}),flush=True)
        try:
            try:wave=load_wave(audio_path)
            except RuntimeError as exc:
                import subprocess
                decoded=subprocess.run(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',str(audio_path),
                    '-ar','16384','-ac','1','-f','f32le','pipe:1'],capture_output=True,check=True)
                wave=np.frombuffer(decoded.stdout,dtype='<f4').copy()
                expected=by_audio[audio_sha][0]['source']['audio_duration_ms']
                actual=len(wave)/16384*1000
                if abs(actual-expected)>100:raise ValueError(f'fallback audio duration mismatch {actual} vs {expected}') from exc
                backend='ffmpeg-f32le-16384-mono'
            spec=make_spec(wave)
        finally: audio_path.unlink(missing_ok=True)
        if not np.isfinite(spec).all(): raise ValueError('nonfinite audio')
        encoded = io.BytesIO(); write_spec(encoded,spec); payload = encoded.getvalue()
        spec_path = output/(audio_sha+'.spec.npy')
        group_rows = []
        for r in by_audio[audio_sha]:
            if r['sha256'] in done: continue
            path = native_manifest.parent/r['path']
            if digest(path)!=r['sha256']: raise ValueError('native hash mismatch')
            try:
                bm = Beatmap(path.read_text(encoding='utf-8-sig'))
                if int(bm.mode)!=0:raise ValueError('nonstandard training target')
                out = io.BytesIO(); write_beatmap(out,bm,get_frame_times(spec.shape[-1]))
                group_rows.append((r,out.getvalue(),float(bm.sr)))
            except Exception as exc:
                progress['rejected'].append({'sha256':r['sha256'],'reason':repr(exc)})
        size = (0 if spec_path.exists() else len(payload))+sum(len(x[1]) for x in group_rows)
        if used+size > cap_bytes: raise RuntimeError('4 GiB cache cap reached; no silent partial corpus')
        if not spec_path.exists(): spec_path.write_bytes(payload); used += len(payload)
        progress['audio'][audio_sha] = {'path':spec_path.name,'sha256':digest(spec_path),'frames':spec.shape[-1],'decoder':backend}
        for r,payload,stars in group_rows:
            path=output/(r['sha256']+'.map.npz'); path.write_bytes(payload); used+=len(payload)
            progress['charts'].append({**r,'native_path':str((native_manifest.parent/r['path']).resolve()),
                'map_path':path.name,'map_sha256':digest(path),'audio_key':audio_sha,'stars':stars})
            done.add(r['sha256'])
        write_json(progress_path,progress)
        print(json.dumps({'cached_charts':len(done),'bytes':used}),flush=True)
    # A failed chart may be excluded, but never replace a frozen source with an easier one.
    for split in ('val','test'):
        found={r['group'] for r in progress['charts'] if r['split']==split}
        if found != set(selected[split]): raise RuntimeError(f'missing frozen {split} sources')
    rejected={r['sha256'] for r in progress['rejected']}
    missing={r['sha256'] for r in wanted}-done-rejected
    if missing: raise RuntimeError(f'{len(missing)} native charts lack original audio')
    validate_sources(progress['charts'])
    manifest={**frozen,'frozen_splits_sha256':digest(frozen_path),'rows':progress['charts'],
              'audio':progress['audio'],'rejected':progress['rejected'],'cache_bytes':used}
    write_json(output/'manifest.json',manifest)
    return manifest
