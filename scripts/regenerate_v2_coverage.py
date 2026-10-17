"""Bounded full-song generation with preserved empty-bar/coverage diagnostics."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from src.v2.inference import load_audio, extract_audio_features, load_v2_models, _sample_candidate_map, GeneratedDraft
from src.v2.evaluation import Candidate, StrictAuditPolicy, RosuDifficultyCalculator, rerank_candidates, build_generation_result
from src.v2.coverage import policy_for_audio
from src.v2.generation import GenerationRequest
from src.v2.mapspec import MapMetadata, compile_osu
from src.v2.packaging import write_osz_archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--candidates',type=int,default=8);p.add_argument('--seed',type=int,default=1601)
    p.add_argument('--target-star',type=float,default=4.5)
    args=p.parse_args()
    if args.candidates<1: p.error('positive candidate count required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    models=load_v2_models(args.checkpoint,device='cuda')
    w,sr=load_audio(args.audio);features,clocks=extract_audio_features(w,sr,models=models)
    # Keep the previously used base/half clock search for this matched repair.
    clocks=tuple(c for c in clocks if not c.hypothesis_id.startswith('double-'))
    clocks=tuple(sorted(clocks,key=lambda c: not c.hypothesis_id.startswith('half-')))
    req=GenerationRequest(audio_filename='audio.mp3',target_star=args.target_star,seed=args.seed,
        candidate_count=args.candidates,feature_revision=features.feature_cache.pipeline_revision,model_revision=models.model_revision)
    policy=policy_for_audio(StrictAuditPolicy(max_star_error=.5,minimum_start_gap_ms=10),features)
    calc=RosuDifficultyCalculator();candidates=[];report={'control':None,'candidates':[]}
    for i in range(-1,args.candidates):
        trace={};clock=clocks[max(i,0)%len(clocks)];seed=args.seed+max(i,0)
        spec=_sample_candidate_map(planner=models.planner,geometry=models.geometry,features=features,
            hypothesis=clock,request=req,seed=seed,temperature=.9,geometry_steps=32,
            minimum_start_gap_ms=10,empty_bar_retries=0 if i==-1 else 2,diagnostics=trace,
            metadata=MapMetadata(title='Kaguya-sama Opening',artist='Masayuki Suzuki',creator='osu-beatmap-generator V2',
                                 version='Coverage repair candidate',beatmap_id=0,beatmap_set_id=-1))
        candidate=Candidate(candidate_id='control' if i==-1 else f'repair-{i:02}',map_spec=spec,seed=seed,
                            provenance={'clock_hypothesis_id':clock.hypothesis_id,'empty_bar_retries':0 if i==-1 else 2})
        audit=rerank_candidates([candidate],target_star=args.target_star,calculator=calc,policy=policy)[0]
        entry={'audit':audit.to_manifest().to_dict(),'trace':trace}
        (args.output_dir/(candidate.candidate_id+'.osu')).write_text(compile_osu(spec,validate=False))
        if i==-1: report['control']=entry
        else:
            report['candidates'].append(entry);candidates.append(candidate)
        (args.output_dir/'audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'candidate':candidate.candidate_id,'valid':audit.valid,'stars':audit.difficulty.stars,
                          'errors':audit.errors,'metrics':audit.metrics}),flush=True)
        if i>=0 and audit.valid: break
    audits=rerank_candidates(candidates,target_star=args.target_star,calculator=calc,policy=policy)
    result=build_generation_result(req,audits,architecture_id=models.checkpoint_provenance.architecture_id,model_revision=models.model_revision)
    (args.output_dir/'result.json').write_text(json.dumps(result.to_dict(),indent=2))
    if result.selected_candidate_id:
        draft=GeneratedDraft(result=result,candidates=tuple(candidates),audits=audits)
        package=draft.package(args.output_dir/'selected',audio_source_path=args.audio)
        write_osz_archive(package,args.output_dir/'kaguya_coverage_fixed.osz')
    else:
        print('No candidate passed both full-song and target-star checks; no playable archive released.',flush=True)

if __name__=='__main__': main()
