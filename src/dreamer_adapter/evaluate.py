from .contracts import artifact_path
import contextlib
import functools
import io
import json
import time
from pathlib import Path
import random
import numpy as np
import torch
from .contracts import digest, write_json, bind, validate_binding
from .data import read_manifest,selected_rows,load_pair
from .model import load_upstream,load_codec,pad,intervene
from .metrics import native_metrics,summarize


def preserve_rng(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        python_state=random.getstate();numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng():return function(*args,**kwargs)
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state)
    return wrapped


def decode(signal,labels,version='experimental'):
    from osu_dreamer.data.beatmap.decode import decode_beatmap,Metadata
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        return decode_beatmap(Metadata('audio.mp3','Evaluation','Unknown',version),labels,signal)


def score_signal(signal,labels,reference,requested):
    try:return native_metrics(decode(signal,labels),reference,requested)
    except ZeroDivisionError as exc:
        result=native_metrics('',reference,requested)
        result.update(invalid=True,stars=None,star_error=12.,errors=['upstream decoder: '+repr(exc)])
        return result


def sample(model,audio,labels,steps=8,codec=None,generator_state=None,return_labels=False):
    if codec is None:
        charts,out_labels=model.sample(audio,labels,steps)
        return (charts[0],out_labels[0]) if return_labels else charts[0]
    length=audio.shape[-1];a=pad(audio[None])
    skips,h=codec.latent.audio_encoder(a)
    style=model.style.sample(labels)
    z=model.diffusion.sample(h,style,steps)
    if generator_state is None:raise ValueError('adapted codec requires a bound adapted generator')
    stats=generator_state['normalization']
    mean=z.new_tensor(stats['mean'])[None,:,None];std=z.new_tensor(stats['std'])[None,:,None]
    logits=codec.latent.decode_logits(z*std+mean,style,skips=skips)
    chart=torch.cat([logits[:,:7].sigmoid(),logits[:,7:]],dim=1)[0,:,:length]
    out_labels=codec.latent.label_predictor(style).clamp(0,10)[0]
    return (chart,out_labels) if return_labels else chart


def initialize(upstream,manifest,codec_path=None,generator_path=None,device='cuda'):
    model=load_upstream(upstream,device);codec=None;gs=None
    if codec_path:codec,_=load_codec(model,codec_path,manifest,device)
    if generator_path:
        gs=torch.load(generator_path,map_location='cpu',weights_only=True)
        if not codec_path:raise ValueError('generator requires bound codec')
        validate_binding(gs['binding'],bind(manifest,digest(codec_path),gs['normalization']))
        model.diffusion.load_state_dict(gs['ema'],strict=True)
    return model,codec,gs


@preserve_rng
def evaluate(manifest,upstream,output,codec_path=None,generator_path=None,split='val',
             modes=('roundtrip','reconstruction','generation'),conditions=('aligned','zero','shift+500','shift-500','wrong'),
             seeds=(1731,1732,1733,1734),seconds=14400,test_authorization=None):
    from src.v2.mapspec import mapspec_from_osu
    if split=='test':
        if not test_authorization:raise ValueError('locked test requires selection decision')
        decision=json.loads(Path(test_authorization).read_text())
        if decision['selected_generator'] != (digest(generator_path) if generator_path else 'upstream'):
            raise ValueError('test decision mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);m=read_manifest(manifest,verify=True);root=Path(manifest).parent
    model,codec,gs=initialize(upstream,m,codec_path,generator_path)
    rows=selected_rows(m,split);all_results=[];started=time.monotonic()
    from .data import load_pair
    for row in rows:
        audio,x,labels=load_pair(root,m,row);audio,x=audio.cuda(),x.cuda()
        reference=mapspec_from_osu(artifact_path(row['native_path']).read_text(encoding='utf-8-sig'))
        # Match the target's actual difficulty/profile; no chart events or clock enter generation.
        request=labels[None].cuda()
        otherrow=min((r for r in rows if r['group']!=row['group']),key=lambda r:abs(m['audio'][r['audio_key']]['frames']-audio.shape[-1]))
        other=load_pair(root,m,otherrow)[0].cuda()
        for mode in modes:
            variants=conditions if mode=='generation' else ('aligned',)
            for condition in variants:
                for seed in (seeds if mode=='generation' else (seeds[0],)):
                    if time.monotonic()-started>seconds:raise TimeoutError('evaluation budget exhausted; partial report not promotable')
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    a=intervene(audio,condition,other)
                    with torch.inference_mode():
                        extra={};export_labels=labels.numpy()
                        if mode=='roundtrip':pred=x
                        elif mode=='reconstruction':
                            if codec is None:
                                z,s=model.latent.encode_chart(pad(x[None]))
                                pred=model.latent.decode(z,s,audio=pad(a[None]))[0][0,:,:x.shape[-1]]
                                shuffled=model.latent.decode(z[...,torch.randperm(z.shape[-1],device=z.device)],s,audio=pad(a[None]))[0][0,:,:x.shape[-1]]
                            else:
                                pred,mu,lv=codec.reconstruct(a[None],x[None]);pred=pred[0]
                                shuffled=codec.reconstruct(a[None],x[None],shuffle=True)[0][0]
                                extra['latent_variance_min']=float(mu.float().var((0,2)).min())
                                extra['posterior_variance_mean']=float(lv.float().exp().mean())
                            extra['cursor_mse']=float((pred[7:]-x[7:]).square().mean())
                            extra['shuffled_cursor_mse']=float((shuffled[7:]-x[7:]).square().mean())
                            extra['shuffled_metrics']=score_signal(shuffled.cpu().numpy(),labels.numpy(),reference,float(labels[0]))
                            if codec is not None and codec.kind=='vae':
                                sampled=codec.reconstruct(a[None],x[None],sample=True)[0][0]
                                extra['sampled_metrics']=score_signal(sampled.cpu().numpy(),labels.numpy(),reference,float(labels[0]))
                        else:
                            pred,out_labels=sample(model,a,request,codec=codec,generator_state=gs,return_labels=True)
                            export_labels=out_labels.cpu().numpy()
                    if not torch.isfinite(pred).all():raise ValueError('nonfinite output')
                    metrics=score_signal(pred.cpu().numpy(),export_labels,reference,float(labels[0]))
                    item={'group':row['group'],'chart':row['sha256'],'seed':seed,'mode':mode,'condition':condition,
                          'metrics':metrics,**extra}
                    all_results.append(item)
                    with (output/'records.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps({'completed_source':row['group'],'records':len(all_results)}),flush=True)
    report={'complete':True,'binding':bind(m,digest(codec_path) if codec_path else 'upstream'),
            'generator_sha256':digest(generator_path) if generator_path else 'upstream',
            'seeds':list(seeds),'sampler_steps':8,'conditions':list(conditions),'modes':list(modes),
            'split':split,'pretraining_overlap':'unknown','elapsed_seconds':time.monotonic()-started,'rows':all_results,'summaries':{}}
    for mode in modes:
        for condition in conditions if mode=='generation' else ('aligned',):
            chosen=[r for r in all_results if r['mode']==mode and r['condition']==condition]
            report['summaries'][mode+'/'+condition]=summarize(chosen)
    write_json(output/'metrics.json',report)
    return report
