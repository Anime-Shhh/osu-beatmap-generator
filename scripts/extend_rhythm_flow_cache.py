"""Add pinned authentic audio/charts to a compact cache with frozen old splits.

Uses original audio (no compressed variant offset mismatch), native millisecond
chart labels, measured stars, and source-component deduplication. A hard byte
budget bounds new feature storage. Historical cache files remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from scripts.materialize_v2_data import hf_sources
from src.v2.data import SourceIdentity,source_component_groups
from src.v2.inference import load_audio,extract_audio_features
from src.v2.mapspec import mapspec_from_osu
from src.v2.preprocess import compact_profile
from src.v2.evaluation import RosuDifficultyCalculator
from src.rhythm_flow.data import validate_manifest


def preserve_splits(rows,old_count):
    sources=[SourceIdentity.from_dict(r['source']) for r in rows]
    groups=source_component_groups(sources)
    frozen={}
    for row,source in zip(rows[:old_count],sources[:old_count]):
        component=groups[source]
        if component in frozen and frozen[component]!=row['split']:
            raise ValueError('new source bridges historical train/validation/test groups; reject merge')
        frozen[component]=row['split']
    for row,source in zip(rows,sources):
        component=groups[source]
        bucket=int(hashlib.sha256(('rhythm-flow-1731:'+component).encode()).hexdigest()[:8],16)%10
        row['source_component']=component
        row['split']=frozen.get(component,'train' if bucket<8 else 'val' if bucket==8 else 'test')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--hf-revision',required=True);p.add_argument('--start-track',type=int,default=300)
    p.add_argument('--max-tracks',type=int,default=200);p.add_argument('--budget-mb',type=int,default=768)
    a=p.parse_args()
    if len(a.hf_revision)!=40 or any(c not in '0123456789abcdef' for c in a.hf_revision):p.error('immutable dataset SHA required')
    if min(a.max_tracks,a.budget_mb)<1 or a.start_track<0:p.error('invalid acquisition budget')
    torch.set_num_threads(1)
    manifest=json.loads((a.base_cache/'manifest.json').read_text());validate_manifest(manifest)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    rows=manifest['charts'];old_count=len(rows);seen={(r['audio_key'],r['chart_sha256']) for r in rows}
    known_files={r['audio_file']:r['audio_file_sha256'] for r in rows};used=0;rejected=[]
    for name,digest in known_files.items():
        source=a.base_cache/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('base cache hash mismatch')
        os.link(source,a.output_dir/name)
    calc=RosuDifficultyCalculator()
    with tempfile.TemporaryDirectory(dir=a.output_dir,prefix='.audio-') as temp:
        path=Path(temp)/'audio.bin'
        for index,(audio,maps,provenance) in enumerate(hf_sources(a.hf_revision,a.max_tracks,a.start_track)):
            if not maps:continue
            path.write_bytes(audio)
            try:
                waveform,sr=load_audio(path)
            except RuntimeError as exc:
                rejected.append({'source':provenance['source_record_id'],'reason':'undecodable audio: '+str(exc)})
                print(json.dumps({'scanned':index+1,'rejected_audio':provenance['source_record_id']}),flush=True)
                continue
            if waveform.shape[-1]/sr>600:
                rejected.append({'source':provenance['source_record_id'],'reason':'pilot length budget >10min'});continue
            features,_=extract_audio_features(waveform,sr,audio_sha256=hashlib.sha256(audio).hexdigest())
            if features.feature_cache.pipeline_key!=manifest['feature_pipeline_key']:raise ValueError('feature pipeline mismatch')
            key=features.feature_cache.canonical_audio.pcm_sha256;name=key+'.pt'
            if name not in known_files:
                needed=features.rhythm_features.numel()*2+4096
                if used+needed>a.budget_mb*1024**2:
                    rejected.append({'reason':'new feature storage budget reached'});break
            pending=[]
            for raw in maps:
                chart_hash=hashlib.sha256(raw['content'].encode()).hexdigest()
                if (key,chart_hash) in seen:continue
                try:
                    spec=mapspec_from_osu(raw['content']);difficulty=calc.calculate(raw['content'])
                    if spec.general.mode!=0 or not spec.events or not difficulty.stars:continue
                    events=[(spec.timing_map.beat_to_time(e.beat_position),
                             spec.timing_map.beat_to_time(e.end_beat if e.end_beat is not None else e.beat_position),int(e.kind)) for e in spec.events]
                    if any(start<0 or end<start or end>features.duration_ms+100 for start,end,_ in events):
                        raise ValueError('chart timing outside audio')
                    d=spec.difficulty
                    identity=SourceIdentity(audio_sha256=hashlib.sha256(audio).hexdigest(),canonical_audio_sha256=key,
                        audio_duration_ms=features.duration_ms,beatmap_id=str(spec.metadata.beatmap_id) if spec.metadata.beatmap_id>0 else None,
                        beatmap_set_id=str(spec.metadata.beatmap_set_id) if spec.metadata.beatmap_set_id>0 else None,
                        mapper_id=str(raw['creator_id']) if raw.get('creator_id') else None,**provenance,
                        beatmap_status=raw['status'],rights_status='research-only; rights retained by owners',rights_url='https://osu.ppy.sh/legal/terms')
                    profile=compact_profile(target_star=difficulty.stars,approach_rate=d.approach_rate,circle_size=d.circle_size,
                        overall_difficulty=d.overall_difficulty,hp_drain_rate=d.hp_drain_rate,slider_multiplier=d.slider_multiplier,slider_tick_rate=d.slider_tick_rate)
                    pending.append({'audio_key':key,'audio_file':name,'profile':profile.tolist(),'events':events,
                        'source':identity.to_dict(),'chart_sha256':chart_hash,'label_source':'native-osu-ms'})
                    seen.add((key,chart_hash))
                except (ValueError,TypeError,KeyError) as exc:
                    rejected.append({'source':provenance['source_record_id'],'chart':chart_hash,'reason':str(exc)})
            if pending:
                if name not in known_files:
                    torch.save({'audio':features.rhythm_features.half()},a.output_dir/name)
                    known_files[name]=hashlib.sha256((a.output_dir/name).read_bytes()).hexdigest();used+=(a.output_dir/name).stat().st_size
                for row in pending:row['audio_file_sha256']=known_files[name]
                rows.extend(pending)
            print(json.dumps({'scanned':index+1,'new_charts':len(rows)-old_count,'new_feature_bytes':used}),flush=True)
            # Preserve acquisition work if a later network/source error interrupts
            # the run. This is deliberately not a trainable split manifest.
            progress=a.output_dir/'acquisition-progress.json'
            progress.with_suffix('.tmp').write_text(json.dumps({'charts':rows,'rejected':rejected,
                'scanned':index+1,'new_feature_bytes':used},separators=(',',':')))
            progress.with_suffix('.tmp').replace(progress)
    preserve_splits(rows,old_count)
    manifest.update(audio_files=len(known_files),audio_bytes=manifest['audio_bytes']+used,charts=rows,
        label_source='existing codec times plus new native millisecond charts; see per-row label_source',
        expansion={'revision':a.hf_revision,'start_track':a.start_track,'max_tracks':a.max_tracks,
                   'new_charts':len(rows)-old_count,'new_feature_bytes':used,'rejected':rejected})
    validate_manifest(manifest)
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
    summary={'charts':len(rows),'audio_files':len(known_files),'new_charts':len(rows)-old_count,'new_feature_bytes':used,
             'source_groups':{s:len({r['source_component'] for r in rows if r['split']==s}) for s in ('train','val','test')}}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
