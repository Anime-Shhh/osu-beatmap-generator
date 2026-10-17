"""Export full-audio rhythm signals/events for evaluation, not a playable .osz."""
import argparse
import json
from pathlib import Path
import torch
from src.rhythm_flow.model import RhythmFlow,RhythmFlowConfig,sample_full_song,decode_rhythm
from src.v2.inference import load_audio,extract_audio_features
from src.v2.preprocess import compact_profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=32);p.add_argument('--seed',type=int,default=1731)
    p.add_argument('--target-star',type=float,default=4.5)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!='osu-rhythm-flow/1' or checkpoint.get('global_step',0)<1:
        raise ValueError('requires a trained experimental flow checkpoint')
    model=RhythmFlow(RhythmFlowConfig(**checkpoint['config'])).to(a.device).eval()
    model.load_state_dict(checkpoint['model'])
    w,sr=load_audio(a.audio);features,_=extract_audio_features(w,sr)
    if features.feature_cache.pipeline_key!=checkpoint['feature_pipeline_key']:
        raise ValueError('audio pipeline differs from training')
    profile=compact_profile(target_star=a.target_star,approach_rate=9,circle_size=4,overall_difficulty=8,
                            hp_drain_rate=6,slider_multiplier=1.4,slider_tick_rate=1.)[None].to(a.device)
    signals=sample_full_song(model,features.rhythm_features.T[None].to(a.device),profile,steps=a.steps,
                            generator=torch.Generator(device=a.device).manual_seed(a.seed))[0].cpu()
    events=decode_rhythm(signals)
    torch.save({'signals':signals,'times_ms':features.rhythm_times_ms,'audio_duration_ms':features.duration_ms},a.output_dir/'rhythm.pt')
    (a.output_dir/'events.json').write_text(json.dumps({'experimental':True,'native_geometry_generated':False,
        'audio_duration_ms':features.duration_ms,'signal_frames':signals.shape[-1],'events':events,
        'limitations':['Peak threshold is uncalibrated; no musicality claim.',
                       'Full tensor coverage does not guarantee event coverage or correct sound selection.']},indent=2))
    print(json.dumps({'audio_duration_ms':features.duration_ms,'signal_frames':signals.shape[-1],'events':len(events)}))

if __name__=='__main__':main()
