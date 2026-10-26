"""Factor onset existence from object type and duration.

The flow transports one onset signal; supervised mark heads read audio and an
onset neighborhood. No reference timing grid or autoregressive chart prefix is
required at inference. Durations are explicit, not occupancy-threshold walks.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.rhythm_flow.model import window_starts


@dataclass(frozen=True)
class RhythmConfig:
    audio_dim: int = 101
    profile_dim: int = 6
    width: int = 128
    audio_layers: int = 8
    flow_layers: int = 12
    window_frames: int = 2048
    overlap_frames: int = 512
    frame_ms: float = 10.0
    context_layers: int = 0
    context_stride: int = 8
    density_auxiliary: bool = False

    def __post_init__(self):
        if self.width < 16 or min(self.audio_layers, self.flow_layers) < 1:
            raise ValueError('invalid model dimensions')
        if not 0 <= self.overlap_frames < self.window_frames or self.frame_ms != 10:
            raise ValueError('invalid window or incompatible feature clock')
        if self.context_layers < 0 or self.context_stride < 1 or (self.context_layers and self.width % 4):
            raise ValueError('invalid pooled audio attention dimensions')


class PhraseContext(nn.Module):
    """Masked audio attention at 80ms; retain the 10ms path for precise attacks."""
    def __init__(self, width, layers, stride):
        super().__init__()
        self.stride = stride
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, 4, 3*width, dropout=.1, activation='gelu', batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, mask):
        length = h.shape[-1]
        pad = (-length) % self.stride
        valid = F.pad(mask[:, None].to(h.dtype), (0, pad))
        count = F.avg_pool1d(valid, self.stride)*self.stride
        pooled = F.avg_pool1d(F.pad(h*mask[:, None], (0, pad)), self.stride)*self.stride
        pooled = (pooled/count.clamp_min(1)).transpose(1, 2)
        keep = count[:, 0] > 0
        safe = keep.clone()
        safe[:, 0] |= ~safe.any(-1)
        frequencies = torch.exp(torch.linspace(0, -math.log(10000), h.shape[1]//2, device=h.device))
        angle = torch.arange(pooled.shape[1], device=h.device)[:, None]*frequencies[None]
        position = torch.cat((angle.sin(), angle.cos()), -1).to(h.dtype)
        context = self.layers(pooled+position, src_key_padding_mask=~safe)
        context = self.out(context)*keep[..., None]
        context = context.transpose(1, 2).repeat_interleave(self.stride, -1)[..., :length]
        return (h+context)*mask[:, None]


class FrameNorm(nn.Module):
    """Channel-only normalization; appended padding cannot change statistics."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = FrameNorm(width)
        self.depth = nn.Conv1d(width, width, 5, padding=2*dilation,
                               dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, 2*width, 1)
        self.out = nn.Conv1d(width, width, 1)
        self.condition = nn.Linear(width, 2*width)

    def forward(self, x, condition, mask):
        scale, shift = self.condition(condition).chunk(2, -1)
        h = (self.norm(x)*(1+scale[..., None])+shift[..., None])*mask
        a, b = self.mix(self.depth(h)).chunk(2, 1)
        return (x + self.out(F.silu(a)*torch.sigmoid(b))/math.sqrt(2))*mask


class OnsetFlow(nn.Module):
    def __init__(self, config=RhythmConfig()):
        super().__init__()
        self.config = config
        w = config.width
        self.profile = nn.Sequential(nn.Linear(config.profile_dim, w), nn.SiLU(), nn.Linear(w, w))
        self.audio_in = nn.Conv1d(config.audio_dim, w, 3, padding=1)
        self.audio_blocks = nn.ModuleList(TemporalBlock(w, 2**(i % 8)) for i in range(config.audio_layers))
        self.phrase_context = PhraseContext(w, config.context_layers, config.context_stride) if config.context_layers else None
        self.onset_aux = nn.Conv1d(w, 1, 1)
        self.density_aux = nn.Conv1d(w, 1, 1) if config.density_auxiliary else None
        self.noise_in = nn.Conv1d(1, w, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(32, w), nn.SiLU(), nn.Linear(w, w))
        self.flow_blocks = nn.ModuleList(TemporalBlock(w, 2**(i % 9)) for i in range(config.flow_layers))
        self.velocity = nn.Conv1d(w, 1, 1)
        self.mark_in = nn.Conv1d(1, w, 5, padding=2)
        self.mark_blocks = nn.ModuleList(TemporalBlock(w, 2**i) for i in range(4))
        self.marks = nn.Conv1d(w, 4, 1)  # type logits, log(1+duration_ms/100)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

    def encode_audio(self, audio, profile, mask):
        mask = mask[:, None].to(audio.dtype)
        condition = self.profile(profile)
        h = self.audio_in(audio*mask)*mask
        for block in self.audio_blocks:
            h = block(h, condition, mask)
        if self.phrase_context is not None:
            h = self.phrase_context(h, mask[:, 0].bool())
        return h, condition

    def field(self, x, encoded, condition, t, mask):
        frequencies = torch.exp(torch.linspace(0, math.log(1000), 16, device=x.device))
        angle = t.float()[:, None]*frequencies[None]*math.pi
        condition = condition+self.time(torch.cat((angle.sin(), angle.cos()), -1))
        mask = mask[:, None].to(x.dtype)
        h = (self.noise_in(x*mask)+encoded)*mask
        for block in self.flow_blocks:
            h = block(h, condition, mask)
        return self.velocity(F.silu(h))*mask

    def mark(self, onset, encoded, condition, mask):
        m = mask[:, None].to(onset.dtype)
        h = (encoded+self.mark_in(onset*m))*m
        for block in self.mark_blocks:
            h = block(h, condition, m)
        return self.marks(h)*m

    def forward(self, audio, profile, mask, x, t, onset):
        h, condition = self.encode_audio(audio, profile, mask)
        auxiliary = self.onset_aux(h)
        if self.density_aux is not None:
            auxiliary = torch.cat((auxiliary, self.density_aux(h)), 1)
        return self.field(x, h, condition, t, mask), auxiliary, self.mark(onset, h, condition, mask)


def rhythm_loss(model, batch, *, generator=None):
    target, mask = batch['onset'], batch['mask']
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    t = torch.rand(len(target), device=target.device, generator=generator)
    clean = 2*target-1
    x = (1-t[:, None, None])*noise+t[:, None, None]*clean
    # Shift chart context up to 20ms during training to tolerate sampled peaks.
    onset = target
    if model.training:
        shift = int(torch.randint(-2, 3, (), device=target.device, generator=generator))
        onset = F.pad(target, (2, 2))[..., 2+shift:2+shift+target.shape[-1]]
    velocity, auxiliary, marks = model(batch['audio'], batch['profile'], mask, x, t, onset)
    valid = mask[:, None]
    flow = ((velocity.float()-(clean-noise)).square()*valid).sum()/valid.sum().clamp_min(1)
    aux = F.binary_cross_entropy_with_logits(auxiliary[:, :1].float(), target,
                pos_weight=target.new_tensor(5.), reduction='none')
    aux = (aux*valid).sum()/valid.sum().clamp_min(1)
    eligible = (batch['kind'] >= 0) & mask
    kind = F.cross_entropy(marks[:, :3].float(), batch['kind'].clamp_min(0), reduction='none')
    kind = (kind*eligible).sum()/eligible.sum().clamp_min(1)
    holds = eligible & (batch['kind'] > 0)
    duration = F.smooth_l1_loss(marks[:, 3].float(), batch['duration'], reduction='none')
    duration = (duration*holds).sum()/holds.sum().clamp_min(1)
    density = target.new_zeros(())
    if model.config.density_auxiliary:
        # Count actual native starts in 1.28s phrases, not the area of soft peaks.
        pad = (-target.shape[-1]) % 128
        sizes = F.avg_pool1d(F.pad(valid.float(), (0, pad)), 128)*128
        counts = F.avg_pool1d(F.pad(batch['onset_count']*valid, (0, pad)), 128)*128
        prediction = F.avg_pool1d(F.pad(auxiliary[:, 1:2].float()*valid, (0, pad)), 128)*128
        prediction = prediction/sizes.clamp_min(1)
        target_density = torch.log1p(counts*100/sizes.clamp_min(1))
        eligible_density = sizes > 0
        density = (F.smooth_l1_loss(prediction, target_density, reduction='none')*eligible_density).sum()/eligible_density.sum().clamp_min(1)
    return {'loss': flow+.25*aux+.5*kind+.25*duration+.1*density,
            'flow': flow, 'audio_onset_bce': aux, 'type_ce': kind, 'duration_huber': duration,
            'phrase_density_huber': density}


@torch.inference_mode()
def sample_rhythm(model, audio, profile, *, steps=32, generator=None):
    """Single full-length noise state with overlap blending at every ODE step."""
    if model.training or steps < 1 or audio.ndim != 3 or audio.shape[-1] < 1:
        raise ValueError('sampling requires eval model, audio, positive steps')
    cfg = model.config
    length = audio.shape[-1]
    x = torch.randn((len(audio), 1, length), device=audio.device, generator=generator)
    windows = []
    weights = torch.zeros(length, device=audio.device)
    for start in window_starts(length, cfg.window_frames, cfg.overlap_frames):
        end = min(length, start+cfg.window_frames)
        mask = torch.ones((len(audio), end-start), device=audio.device, dtype=torch.bool)
        h, condition = model.encode_audio(audio[..., start:end], profile, mask)
        weight = torch.hann_window(end-start+2, periodic=False, device=audio.device)[1:-1].clamp_min(.05)
        windows.append((start, end, mask, h, condition, weight))
        weights[start:end] += weight
    for i in range(steps):
        velocity = torch.zeros_like(x)
        t = torch.full((len(audio),), i/steps, device=audio.device)
        for start, end, mask, h, condition, weight in windows:
            velocity[..., start:end] += model.field(x[..., start:end], h, condition, t, mask)*weight
        x += velocity/weights/steps
    onset = ((x+1)/2).clamp(0, 1)
    marks = audio.new_zeros((len(audio), 4, length))
    for start, end, mask, h, condition, weight in windows:
        marks[..., start:end] += model.mark(onset[..., start:end], h, condition, mask)*weight
    return onset, marks/weights


def decode_events(onset, marks, *, threshold=.4, min_gap_ms=40., duration_ms=None):
    if onset.ndim != 1 or marks.shape != (4, len(onset)) or not 0 < threshold <= 1:
        raise ValueError('invalid onset/mark tensors or threshold')
    candidates = [i for i in range(len(onset)) if onset[i] >= threshold
                  and (i == 0 or onset[i] > onset[i-1])
                  and (i+1 == len(onset) or onset[i] >= onset[i+1])]
    chosen = []
    for i in sorted(candidates, key=lambda j: float(onset[j]), reverse=True):
        if all(abs(i-j)*10 >= min_gap_ms for j in chosen):
            chosen.append(i)
    chosen.sort()
    end_of_audio = len(onset)*10 if duration_ms is None else duration_ms
    result = []
    for j, i in enumerate(chosen):
        kind = int(marks[:3, i].argmax())
        duration = max(0., math.expm1(float(marks[3, i].clamp(0, math.log1p(300))))) * 100
        start = i*10.
        cap = min(end_of_audio, chosen[j+1]*10-20 if j+1 < len(chosen) else end_of_audio)
        end = min(start+duration, cap) if kind else start
        if kind and end-start < 40:
            kind, end = 0, start
        result.append({'start_ms': start, 'end_ms': end, 'kind': kind, 'confidence': float(onset[i])})
    return result
