import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import torch
from .contracts import write_json,digest
from .data import read_manifest
from .evaluate import initialize,sample,decode
from .metrics import native_metrics
from src.v2.mapspec import mapspec_from_osu


def generate(audio,stars,upstream,output,manifest=None,codec_path=None,generator_path=None,seed=1731,
             ar=8.8,od=6.5,cs=4.,hp=6.5,steps=8,clock_json=None):
    from osu_dreamer.data.load_audio import load_wave,make_spec
    if not 0<stars<=12 or any(not 0<=x<=10 for x in (ar,od,cs,hp)):raise ValueError('invalid difficulty')
    if codec_path and manifest is None:raise ValueError('adaptation manifest required')
    m=read_manifest(manifest) if manifest else {}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed)
    model,codec,gs=initialize(upstream,m,codec_path,generator_path)
    a=torch.tensor(make_spec(load_wave(audio)),device='cuda',dtype=torch.float32)
    labels=torch.tensor([[stars,ar,od,cs,hp]],device='cuda',dtype=torch.float32)
    with torch.inference_mode():signal,out_labels=sample(model,a,labels,steps,codec,gs,return_labels=True)
    text=decode(signal.cpu().numpy(),out_labels.cpu().numpy(),'Experimental draft')
    clock_status='not evaluated; timing needs review'
    if clock_json:
        from .timing import replace_timing
        try:text=replace_timing(text,json.loads(Path(clock_json).read_text()));clock_status='audio-only constant 4/4 clock accepted'
        except ValueError as exc:clock_status=str(exc)
    metrics=native_metrics(text,mapspec_from_osu(text),stars)
    actual=metrics['stars']
    if actual is None:raise ValueError('unmeasurable native output; refusing to package')
    text=text.replace('Version: Experimental draft',f'Version: Experimental {actual:.2f} stars')
    (output/'map.osu').write_text(text)
    with zipfile.ZipFile(output/'map.osz','x',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(audio,'audio.mp3');archive.writestr('map.osu',text)
    with zipfile.ZipFile(output/'map.osz') as archive:
        if archive.testzip() is not None or hashlib.sha256(archive.read('audio.mp3')).hexdigest()!=digest(audio):raise ValueError('archive integrity failure')
    audit={'seed':seed,'audio_sha256':digest(audio),'upstream_sha256':digest(upstream),
           'codec_sha256':digest(codec_path) if codec_path else None,'generator_sha256':digest(generator_path) if generator_path else None,
           'requested_settings':[stars,ar,od,cs,hp],'predicted_settings':out_labels.cpu().tolist(),'requested_stars':stars,'measured_stars':actual,'difficulty_pass':abs(actual-stars)<=.5,
           'native_errors':metrics['errors'],'timing_status':clock_status,'ranked_ready':False,
           'osu_sha256':digest(output/'map.osu'),'osz_sha256':digest(output/'map.osz')}
    write_json(output/'audit.json',audit);return audit
