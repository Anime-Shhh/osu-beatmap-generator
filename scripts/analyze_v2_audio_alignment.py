"""Plot audio, spectral attacks, optional aligned stems, and chart event lanes.

This is an inspection tool, not a claim that a chart must follow every attack.
It accepts FL Studio/Demucs stem exports; their timestamps must already match
its audio. Frequency bands are not labeled as instruments.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
import zipfile

import numpy as np
import torch
from src.v2.inference import load_audio
from src.v2.mapspec import mapspec_from_osu
from src.v2.evaluation import RosuDifficultyCalculator
from src.v2.events import EventKind


def acoustic_lanes(waveform, sample_rate):
    signal = waveform.flatten().float()
    if len(signal) < 1024:
        signal = torch.nn.functional.pad(signal, (0,1024-len(signal)))
    magnitude = torch.stft(signal,n_fft=1024,hop_length=240,window=torch.hann_window(1024),return_complex=True).abs()
    # Log compression emphasizes modest attacks without treating amplitude as
    # a direct proxy for mapping importance. Absolute timing remains at 10 ms.
    power = torch.log1p(magnitude)
    flux = torch.nn.functional.pad((power[:,1:]-power[:,:-1]).clamp_min(0),(1,0))
    frequencies = torch.fft.rfftfreq(1024,1/sample_rate)
    bounds=(30,180,600,2000,6000,sample_rate/2+1)
    lanes={f'{lo:g}–{min(hi,sample_rate/2):g} Hz':flux[(frequencies>=lo)&(frequencies<hi)].mean(0).numpy()
           for lo,hi in zip(bounds,bounds[1:])}
    lanes['mixture attacks']=flux.mean(0).numpy()
    for name, values in lanes.items():
        lanes[name]=values/max(float(np.quantile(values,.99)),1e-6)
    return np.arange(power.shape[1])*240/sample_rate, power.numpy(), lanes


def peak_times(times, values):
    values=np.asarray(values)
    indices=np.where((values[1:-1]>values[:-2]) & (values[1:-1]>=values[2:]) & (values[1:-1]>.15))[0]+1
    selected=[]
    for index in sorted(indices,key=lambda i: values[i],reverse=True):
        if all(abs(times[index]-times[j])>=.04 for j in selected): selected.append(index)
    return np.array(sorted(times[selected]))


def summarize_alignment(starts, peaks):
    if not len(starts) or not len(peaks):
        return {'objects':len(starts),'median_nearest_attack_ms':None,'within_50ms':None}
    distances=np.abs(np.asarray(starts)[:,None]-peaks[None,:]).min(axis=1)*1000
    return {'objects':len(starts),'median_nearest_attack_ms':float(np.median(distances)),
            'within_50ms':float((distances<=50).mean())}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--osz',type=Path)
    source.add_argument('--audio',type=Path)
    p.add_argument('--maps',type=Path,nargs='*',default=[])
    p.add_argument('--stems',nargs='*',default=[],help='name=/path/to/aligned.wav (optional FL Studio/Demucs exports)')
    p.add_argument('--start',type=float,default=0)
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args(argv)
    if args.start < 0 or args.seconds <= 0: p.error('invalid plot window')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='osu-alignment-') as temporary:
        if args.osz:
            with zipfile.ZipFile(args.osz) as archive:
                charts=[(name,archive.read(name).decode('utf-8-sig')) for name in archive.namelist() if name.endswith('.osu')]
                charts=[(name,text) for name,text in charts if mapspec_from_osu(text).general.mode==0]
                if not charts: raise ValueError('archive has no standard maps')
                # Require one exact audio asset; comparing unrelated recordings
                # is not an alignment experiment. Never extract archive paths.
                from pathlib import PurePosixPath
                assets={str(PurePosixPath(name).parent / mapspec_from_osu(text).general.audio_filename.replace('\\','/')) for name,text in charts}
                if len(assets)!=1: raise ValueError('archive maps do not share one audio asset')
                audio=archive.read(next(iter(assets)))
            audio_path=Path(temporary)/'audio.bin';audio_path.write_bytes(audio)
        else:
            audio_path=args.audio;audio=audio_path.read_bytes()
            charts=[(path.name,path.read_text(encoding='utf-8-sig')) for path in args.maps]
        waveform,sr=load_audio(audio_path)
        duration=waveform.shape[-1]/sr
        if args.start>=duration: raise ValueError('plot starts after the audio ends')
        times,spectrum,lanes=acoustic_lanes(waveform,sr)
        stems={}
        for entry in args.stems:
            name,filename=entry.split('=',1)
            if not name or name in stems or name in lanes: raise ValueError('stem names must be nonempty and unique')
            stem,stem_sr=load_audio(filename)
            if stem_sr!=sr or abs(stem.shape[-1]-waveform.shape[-1])/sr>.02:
                raise ValueError(f'{name}: stem duration/sample rate differs; preserve original offset and duration')
            st,_,sl=acoustic_lanes(stem,sr)
            lanes[name]=np.interp(times,st,sl['mixture attacks'])
            stems[name]={'path':str(Path(filename).resolve()),'sha256':hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
        calc=RosuDifficultyCalculator()
        reports=[]; event_rows=[]
        peaks={name:peak_times(times,values) for name,values in lanes.items()}
        for name,text in charts:
            spec=mapspec_from_osu(text)
            events=[(spec.timing_map.beat_to_time(e.beat_position)/1000,
                     spec.timing_map.beat_to_time(e.end_beat or e.beat_position)/1000,e.kind.name) for e in spec.events]
            starts=np.array([e[0] for e in events])
            reports.append({'file':name,'version':spec.metadata.version,'stars':calc.calculate(text).stars,
                            'events':events,'acoustic_proximity':{n:summarize_alignment(starts,ps) for n,ps in peaks.items()}})
            for start,end,kind in events:
                row={'map':name,'start_ms':start*1000,'end_ms':end*1000,'kind':kind}
                for lane,values in lanes.items():
                    row[lane+' strength']=float(np.interp(start,times,values))
                    row[lane+' nearest_attack_ms']=float(np.abs(peaks[lane]-start).min()*1000) if len(peaks[lane]) else None
                event_rows.append(row)
        report={'audio_pairing': 'shared archive asset' if args.osz else 'user-supplied pairing; not independently verified',
            'audio_sha256':hashlib.sha256(audio).hexdigest(),'duration_seconds':duration,
            'stems':stems,'calculator':calc.calculator_id,'maps':reports,
            'limitations':['Spectral proximity is not evidence of mapper intention or model attribution.',
                'No event is assigned an instrument label; stem estimates can contain bleed/artifacts.',
                'Multiple valid maps can select different musical layers; unmapped attacks are not necessarily errors.',
                'Nearest-attack distances are many-to-one diagnostics, not chart-onset F1.',
                'External stems require verified zero-offset exports; matching lengths cannot establish phase alignment.']}
        (args.output_dir/'alignment.json').write_text(json.dumps(report,indent=2)+'\n')
        if event_rows:
            with (args.output_dir/'events.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=event_rows[0].keys());writer.writeheader();writer.writerows(event_rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        end=min(duration,args.start+args.seconds);selection=(times>=args.start)&(times<=end)
        fig,axes=plt.subplots(3+len(reports),1,figsize=(16,7+1.2*len(reports)),sharex=True,
                            gridspec_kw={'height_ratios':[1,2,1.6]+[1]*len(reports)})
        signal=waveform.flatten().numpy(); decimate=max(1,sr//1000)
        sample_times=np.arange(0,len(signal),decimate)/sr
        axes[0].plot(sample_times,signal[::decimate],lw=.4,color='#506580');axes[0].set_ylabel('Waveform')
        axes[1].imshow(spectrum[:,selection],origin='lower',aspect='auto',extent=[times[selection][0],times[selection][-1],0,sr/2],cmap='magma')
        axes[1].set_ylim(0,8000);axes[1].set_ylabel('Frequency (Hz)')
        shown=['mixture attacks',*stems] if stems else list(lanes)
        for lane in shown: axes[2].plot(times[selection],lanes[lane][selection],lw=.8,label=lane)
        axes[2].legend(loc='upper right',ncol=3,fontsize=8);axes[2].set_ylabel('Relative attacks')
        colors={'CIRCLE':'#356ddb','SLIDER':'#e07824','SPINNER':'#8345b8'}
        for ax,report_map in zip(axes[3:],reports):
            for start,stop,kind in report_map['events']:
                if stop<args.start or start>end: continue
                ax.vlines(start,0,.8,color=colors[kind],lw=1)
                if stop>start: ax.hlines(.4,start,stop,color=colors[kind],lw=3)
            label=textwrap.fill(report_map["version"],width=24,max_lines=2,placeholder="…")
            ax.set_ylabel(f"{label}\n{report_map['stars']:.2f}★",fontsize=8,rotation=0,ha="right",va="center")
            ax.set_ylim(-.1,1);ax.set_yticks([])
        axes[-1].set_xlabel('Audio time (seconds)')
        axes[-1].set_xlim(args.start,end)
        fig.suptitle('Audio/chart alignment — blue: circle; orange: slider; purple: spinner\nAttack proximity is a diagnostic, not an instrument attribution',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.95])
        for extension in ('png','svg'): fig.savefig(args.output_dir/f'alignment.{extension}',dpi=130)
        plt.close(fig)
    print(json.dumps({'output':str(args.output_dir),'maps':len(reports),'events':len(event_rows),'stems':len(stems)}))


if __name__=='__main__': main()
