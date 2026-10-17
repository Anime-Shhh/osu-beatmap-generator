"""Conditional flow matching over chart signals on the actual audio timeline.

Five channels: circle/slider/spinner onsets and slider/spinner occupancy.
No oracle clock, chart-prefix teacher forcing, latent codec, or EOS token.
This experiment predicts rhythm only; native geometry is still a separate task.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RhythmFlowConfig:
    audio_dim: int = 101
    profile_dim: int = 6
    channels: int = 5
    width: int = 96
    layers: int = 10
    window_frames: int = 1024
    overlap_frames: int = 256
    frame_ms: float = 10.0

    def __post_init__(self):
        if self.channels != 5 or self.width%8 or self.width<8 or self.layers<1:
            raise ValueError('invalid rhythm flow configuration')
        if not 0<=self.overlap_frames<self.window_frames or self.frame_ms!=10:
            raise ValueError('requires valid overlap and the versioned 10 ms feature pipeline')


class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.norm=nn.GroupNorm(8,width)
        self.conv=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.mod=nn.Linear(width,2*width)
        self.out=nn.Conv1d(width,width,1)
    def forward(self,x,condition):
        scale,shift=self.mod(condition).chunk(2,dim=-1)
        h=self.norm(x)*(1+scale[:,:,None])+shift[:,:,None]
        return x+self.out(F.silu(self.conv(F.silu(h))))/math.sqrt(2)


class RhythmFlow(nn.Module):
    def __init__(self,config=RhythmFlowConfig()):
        super().__init__();self.config=config
        self.audio=nn.Conv1d(config.audio_dim,config.width,3,padding=1)
        self.input=nn.Conv1d(config.channels,config.width,1)
        self.condition=nn.Sequential(nn.Linear(config.profile_dim+32,config.width),nn.SiLU(),nn.Linear(config.width,config.width))
        self.blocks=nn.ModuleList(Block(config.width,2**(i%8)) for i in range(config.layers))
        self.output=nn.Conv1d(config.width,config.channels,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,audio,profile,t,mask=None):
        if x.shape[:2]!=(audio.shape[0],self.config.channels) or x.shape[-1]!=audio.shape[-1]:
            raise ValueError('rhythm/audio timeline mismatch')
        frequencies=torch.exp(torch.linspace(0,math.log(1000),16,device=x.device))
        angles=t.float()[:,None]*frequencies[None,:]*math.pi
        condition=self.condition(torch.cat((profile,angles.sin(),angles.cos()),dim=1))
        if mask is not None:
            x=x*mask[:,None];audio=audio*mask[:,None]
        h=self.input(x)+self.audio(audio)
        for block in self.blocks:
            h=block(h,condition)
            if mask is not None: h=h*mask[:,None]
        result=self.output(F.silu(h))
        return result if mask is None else result*mask[:,None]


def flow_loss(model,audio,target,profile,mask,*,generator=None):
    """Learn a noise-to-chart vector field; retain empty windows as negatives."""
    noise=torch.randn(target.shape,device=target.device,generator=generator)
    t=torch.rand(target.shape[0],device=target.device,generator=generator)
    clean=2*target-1
    x=(1-t[:,None,None])*noise+t[:,None,None]*clean
    prediction=model(x,audio,profile,t,mask)
    # Fixed event emphasis prevents vast empty-frame counts dominating the task.
    weight=(1+4*target)*mask[:,None]
    loss=((prediction-(clean-noise)).float().square()*weight).sum()/weight.sum().clamp_min(1)
    return loss


def window_starts(length,window,overlap):
    if length<1 or not 0<=overlap<window: raise ValueError('invalid timeline/window')
    starts=list(range(0,max(1,length-window+1),window-overlap))
    if starts[-1]+window<length: starts.append(max(0,length-window))
    return sorted(set(starts))


@torch.no_grad()
def sample_full_song(model,audio,profile,*,steps=32,generator=None):
    """Average overlapping vector fields at every ODE step on one global noise.

    Output length is exactly the audio feature length. This avoids independent
    window seams and cannot stop early via EOS; it can still predict no events.
    """
    if steps<1 or audio.ndim!=3 or audio.shape[-1]<1: raise ValueError('invalid sampling arguments')
    if model.training: raise ValueError('sampling requires eval mode')
    cfg=model.config;length=audio.shape[-1]
    x=torch.randn((audio.shape[0],cfg.channels,length),device=audio.device,generator=generator)
    starts=window_starts(length,cfg.window_frames,cfg.overlap_frames)
    for i in range(steps):
        velocity=torch.zeros_like(x);weights=torch.zeros(length,device=x.device)
        for start in starts:
            end=min(length,start+cfg.window_frames);n=end-start
            # No appended padded timeline at inference; all positions are real.
            weight=torch.hann_window(n+2,periodic=False,device=x.device)[1:-1].clamp_min(.05)
            v=model(x[:,:,start:end],audio[:,:,start:end],profile,
                    torch.full((audio.shape[0],),i/steps,device=x.device))
            velocity[:,:,start:end]+=v*weight;weights[start:end]+=weight
        x=x+velocity/weights[None,None]/steps
    return ((x+1)/2).clamp(0,1)


def decode_rhythm(signals,*,frame_ms=10.,threshold=.5,min_gap_ms=40.):
    """Diagnostic peak decoder. Thresholds must be tuned on validation only."""
    if signals.ndim!=2 or signals.shape[0]!=5: raise ValueError('expected five rhythm channels')
    strength,kind=signals[:3].max(0);candidates=[]
    for i in range(len(strength)):
        if strength[i]>=threshold and (i==0 or strength[i]>strength[i-1]) and (i+1==len(strength) or strength[i]>=strength[i+1]):
            candidates.append(i)
    selected=[]
    for i in sorted(candidates,key=lambda j:float(strength[j]),reverse=True):
        if all(abs(j-i)*frame_ms>=min_gap_ms for j in selected): selected.append(i)
    selected.sort();events=[]
    for index,i in enumerate(selected):
        k=int(kind[i]);end=i
        if k:
            cap=selected[index+1] if index+1<len(selected) else signals.shape[1]
            while end+1<cap and signals[2+k,end+1]>=threshold: end+=1
        events.append({'start_ms':i*frame_ms,'end_ms':end*frame_ms,'kind':k,'confidence':float(strength[i])})
    return events
