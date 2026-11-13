"""Re-evaluate retained AE/VAE pilot checkpoints on the identical Kaguya request."""
import argparse, json, random, zipfile
from pathlib import Path
import numpy as np
import torch
from src.dreamer_adapter.contracts import digest,write_json
from src.dreamer_adapter.data import read_manifest
from src.dreamer_adapter.evaluate import initialize,sample,decode,score_signal
from src.dreamer_adapter.metrics import summarize
from src.v2.mapspec import mapspec_from_osu
from src.v2.evaluation import RosuDifficultyCalculator

@torch.inference_mode()
def run(output):
    from osu_dreamer.data.load_audio import make_spec,load_wave
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    p=Path('/common/home/asj102/osu_flow_runs/pilot_20261105');m=read_manifest(p/'cache-compact/manifest.json')
    upstream='/common/home/asj102/osu_flow_runs/dreamer_v8_20261105/v8.pt'
    archive_path=Path('/common/users/asj102/osu_project/1147361 Suzuki Masayuki - DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size) [no video].osz')
    calc=RosuDifficultyCalculator();refs=[]
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.endswith('.osu'):
                text=archive.read(name).decode('utf-8-sig');spec=mapspec_from_osu(text)
                if spec.general.mode==0:refs.append((abs(calc.calculate(text).stars-4.6),name,text))
        _,name,text=min(refs);reference=mapspec_from_osu(text)
        (output/'audio.mp3').write_bytes(archive.read(str(Path(name).parent/reference.general.audio_filename)))
    features=torch.tensor(make_spec(load_wave(output/'audio.mp3')),device='cuda',dtype=torch.float32)
    d=reference.difficulty;labels=torch.tensor([[4.6,d.approach_rate,d.overall_difficulty,d.circle_size,d.hp_drain_rate]],device='cuda')
    reports={}
    for kind,codec_path,generator_path in [('adapted-ae',p/'codec-ae/best.pt',p/'generator-ae/best.pt'),('adapted-vae',p/'codec-vae01/best.pt',p/'generator-vae/best.pt')]:
        model,codec,gs=initialize(upstream,m,codec_path,generator_path);rows=[]
        for seed in (1731,1732,1733,1734):
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            pred,settings=sample(model,features,labels,codec=codec,generator_state=gs,return_labels=True)
            metrics=score_signal(pred.cpu().numpy(),settings.cpu().numpy(),reference,4.6)
            rows.append(dict(group='kaguya-development',seed=seed,metrics=metrics))
        reports[kind]=dict(rows=rows,summary=summarize(rows),codec_sha256=digest(codec_path),generator_sha256=digest(generator_path))
        write_json(output/'comparison.json',reports);print(json.dumps({'kind':kind,'summary':summarize(rows)}),flush=True)
        del model,codec,gs;torch.cuda.empty_cache()
    (output/'audio.mp3').unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))
