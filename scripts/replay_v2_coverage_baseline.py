"""Read-only replay against the preserved pre-fix source via PYTHONPATH."""
import argparse,json
from pathlib import Path
import torch
from src.v2.inference import load_v2_models,load_audio,extract_audio_features,_sample_candidate_map
from src.v2.generation import GenerationRequest
from src.v2.mapspec import MapMetadata,compile_osu
from src.v2.grammar import BarEventCodec


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    root=Path('/common/users/asj102/osu_project')
    models=load_v2_models(root/'v2_runs/run_20261015_audio_alignment_precision/geometry/best.pt',device='cuda')
    w,sr=load_audio(root/'data/inference_input/Kaguya-sama_ Love Is War_ Opening Theme (Limited Time Only).mp3')
    f,clocks=extract_audio_features(w,sr,models=models);clock=next(c for c in clocks if c.hypothesis_id.startswith('half-'))
    req=GenerationRequest(audio_filename='audio.mp3',target_star=4.5,seed=1601,candidate_count=1,
                          feature_revision=f.feature_cache.pipeline_revision,model_revision=models.model_revision)
    original=models.planner.generate_bar;trace=[];codec=BarEventCodec(models.planner.vocabulary)
    def traced(*args,**kwargs):
        tokens=original(*args,**kwargs)
        trace.append({'bar':kwargs['target_bar_index'],'events':len(codec.event_context_positions(tokens))})
        return tokens
    models.planner.generate_bar=traced
    spec=_sample_candidate_map(planner=models.planner,geometry=models.geometry,features=f,hypothesis=clock,
        request=req,seed=1601,temperature=.9,geometry_steps=32,minimum_start_gap_ms=10,
        metadata=MapMetadata(title='Baseline replay',artist='Masayuki Suzuki',creator='V2',version='diagnostic'))
    times=[spec.timing_map.beat_to_time(e.beat_position) for e in spec.events]
    report={'audio_duration_ms':f.duration_ms,'clock':clock.hypothesis_id,'objects':len(times),
            'first_ms':min(times),'last_ms':max(times),'bars':trace,'device':torch.cuda.get_device_name()}
    (a.output_dir/'trace.json').write_text(json.dumps(report,indent=2));(a.output_dir/'replay.osu').write_text(compile_osu(spec,validate=False))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
