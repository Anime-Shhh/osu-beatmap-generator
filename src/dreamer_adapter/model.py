"""Frozen pretrained features with matched AE and Gaussian temporal-VAE adapters."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .contracts import digest, validate_binding
from . import UPSTREAM_WEIGHTS


def load_upstream(path, device='cpu'):
    if digest(path)!=UPSTREAM_WEIGHTS: raise ValueError('unexpected upstream weights')
    import importlib.util,json
    from pathlib import Path
    root=Path(importlib.util.find_spec('osu_dreamer').submodule_search_locations[0]).parent
    expected=json.loads(Path(__file__).with_name('upstream-files.json').read_text())
    for name,sha in expected.items():
        if digest(root/name)!=sha:raise ValueError('upstream source changed: '+name)
    from osu_dreamer.models.inference.artifact import load_inference
    return load_inference(path).to(device).eval()


def pad(x, multiple=27):
    n=(-x.shape[-1])%multiple
    return F.pad(x,(0,n),mode='replicate') if n else x


def masked_kl(mu, logvar, mask=None):
    terms=.5*(mu.float().square()+logvar.float().exp()-1-logvar.float())
    if mask is None: return terms.mean()
    weights=mask[:,None,:].to(terms)
    return (terms*weights).sum()/(weights.sum()*terms.shape[1]).clamp_min(1)


def mask_latents(z, max_fraction=.25):
    b,_,length=z.shape
    span=(torch.rand(b,device=z.device)*max_fraction*length).long()
    start=(torch.rand(b,device=z.device)*(length-span).clamp_min(1)).long()
    indices=torch.arange(length,device=z.device)[None]
    mask=(indices>=start[:,None])&(indices<(start+span)[:,None])
    return z.masked_fill(mask[:,None],0),mask


class Codec(nn.Module):
    def __init__(self, latent, kind='ae'):
        super().__init__()
        if kind not in ('ae','vae'): raise ValueError(kind)
        self.latent=latent;self.kind=kind
        latent.requires_grad_(False)
        self.mean=nn.Conv1d(6,6,1)
        with torch.no_grad():
            self.mean.weight.copy_(torch.eye(6)[:,:,None]);self.mean.bias.zero_()
        if kind=='vae':
            self.logvar=nn.Conv1d(6,6,1)
            nn.init.zeros_(self.logvar.weight);nn.init.constant_(self.logvar.bias,math.log(.04))
        for name in ('temporal_layer','temporal_head','proj_emb','decoder','proj_out'):
            getattr(latent,name).requires_grad_(True)
        self.register_buffer('loss_scale',torch.ones(10))
        self.register_buffer('loss_initialized',torch.tensor(False))

    def train(self,mode=True):
        super().train(mode)
        # Freeze running style centering as well as its parameters.
        for name in ('chart_encoder','audio_encoder','style_head','label_predictor'):
            getattr(self.latent,name).eval()
        return self

    def encode(self,chart):
        with torch.no_grad():
            _,h=self.latent.chart_encoder(chart)
            style=self.latent.style_head(h)
        z=self.latent.temporal_head(self.latent.temporal_layer(h,style))
        mu=self.mean(z)
        logvar=self.logvar(z).clamp(-10,2) if self.kind=='vae' else torch.full_like(mu,math.log(.04))
        return mu,logvar,style

    def reconstruct(self,audio,chart,sample=False,shuffle=False):
        length=chart.shape[-1];audio,chart=pad(audio),pad(chart)
        mu,lv,s=self.encode(chart)
        z=mu+torch.exp(.5*lv)*torch.randn_like(mu) if sample else mu
        if shuffle: z=z[...,torch.randperm(z.shape[-1],device=z.device)]
        pred,_=self.latent.decode(z,s,audio=audio)
        return pred[...,:length],mu,lv

    def loss(self,audio,chart,step,beta):
        # Each crop has two paired halves, mirroring upstream cross-half style supervision.
        b,d,length=chart.shape
        audio=audio.reshape(b,audio.shape[1],2,length//2).permute(0,2,1,3).flatten(0,1)
        chart=chart.reshape(b,d,2,length//2).permute(0,2,1,3).flatten(0,1)
        mu,lv,s=self.encode(chart)
        s=s.reshape(b,2,-1).flip(1).reshape(b*2,-1)+.2*torch.randn_like(s)
        z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
        z,_=mask_latents(z)
        # Use detached audio skips; gradients still pass through trainable chart decoder.
        with torch.no_grad(): skips,_=self.latent.audio_encoder(audio)
        logits=self.latent.decode_logits(z,s,skips=skips)
        floor=-torch.special.xlogy(chart[:,:7],chart[:,:7])-torch.special.xlogy(1-chart[:,:7],1-chart[:,:7])
        hit=(F.binary_cross_entropy_with_logits(logits[:,:7].float(),chart[:,:7],reduction='none')-floor).mean((0,2))
        cursor=[F.mse_loss(logits[:,7:].float().diff(n=i),chart[:,7:].diff(n=i)) for i in range(3)]
        losses=torch.cat([hit,torch.stack(cursor)])
        with torch.no_grad():
            if not self.loss_initialized:
                self.loss_scale.copy_(losses.detach());self.loss_initialized.fill_(True)
            else:self.loss_scale.lerp_(losses.detach(),.01)
        weights=losses.new_tensor([1]*7+[2]*3)
        reconstruction=(weights*losses/self.loss_scale.clamp_min(1e-8)).sum()
        kl=masked_kl(mu,lv) if self.kind=='vae' else losses.new_zeros(())
        loss=reconstruction+beta*min(1,step/2000)*kl
        # Style and label heads are frozen: their losses are constants, not trainable objectives.
        return loss,{'reconstruction':float(reconstruction.detach()),'kl':float(kl.detach()),
                     'mu_variance_min':float(mu.detach().float().var((0,2)).min())}


def load_codec(upstream,path,manifest,device='cpu'):
    from .contracts import bind
    state=torch.load(path,map_location='cpu',weights_only=True)
    validate_binding(state['binding'],bind(manifest))
    codec=Codec(upstream.latent,state['config']['kind']).to(device)
    codec.load_state_dict(state['model'],strict=True)
    return codec.eval(),state


def intervene(audio,kind,other=None):
    if kind=='aligned':return audio
    if kind=='zero':return torch.zeros_like(audio)
    if kind=='wrong':
        if other is None:raise ValueError('wrong-song audio required')
        return F.interpolate(other[None] if other.ndim==2 else other,size=audio.shape[-1],mode='linear',align_corners=False).squeeze(0)
    if kind not in ('shift+500','shift-500'):raise ValueError(kind)
    shift=round(.5*16384/98);out=torch.zeros_like(audio)
    if shift>=audio.shape[-1]:return out
    if kind=='shift+500':out[...,shift:]=audio[...,:-shift]
    else:out[...,:-shift]=audio[...,shift:]
    return out
