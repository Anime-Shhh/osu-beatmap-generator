"""Native head/control geometry flow conditioned on events rather than planner states."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from src.v2.geometry import (ActiveEventGeometryDiffuser, GeometryDiffusionConfig,
    encode_geometry, decode_geometry, geometry_point_mask, geometry_pattern_losses)

SPACE = 'head-logit-relative-anchors-v1'
CURVES = ('L', 'B', 'P', 'C')


def event_condition(audio, starts, ends, kinds, profile):
    """Uniform 10ms audio timestamps, used identically for training and inference.

    Native millisecond intervals avoid oracle BPM dependence in the learned path.
    The two audio views are onset-local and a 400ms neighborhood average.
    """
    if audio.ndim != 2 or audio.shape[1] != 101:
        raise ValueError('expected [frames,101] native audio features')
    ids = (starts/10).round().long().clamp(0, len(audio)-1)
    offsets = torch.arange(-20, 21, device=audio.device)
    neighbors = (ids[:, None]+offsets).clamp(0, len(audio)-1)
    local = audio[neighbors].mean(1)
    previous = torch.cat((starts[:1], starts[:-1]))
    following = torch.cat((starts[1:], ends[-1:]))
    temporal = torch.stack(((starts-previous).clamp_min(0),
                            (following-starts).clamp_min(0), (ends-starts).clamp_min(0)), -1)
    temporal = torch.log1p(temporal/100)/math.log(301)
    return torch.cat((audio[ids], local, profile.expand(len(ids), -1),
                      F.one_hot(kinds, 3).float(), temporal), -1)


@dataclass(frozen=True)
class GeometryConfig:
    condition_dim: int = 214
    width: int = 192
    heads: int = 6
    context_layers: int = 3
    flow_layers: int = 4
    max_anchors: int = 32
    window_events: int = 64
    overlap_events: int = 16


class NativeGeometryFlow(nn.Module):
    def __init__(self, config=GeometryConfig()):
        super().__init__()
        self.config = config
        w = config.width
        self.input = nn.Linear(config.condition_dim, w)
        layer = nn.TransformerEncoderLayer(w, config.heads, 4*w, dropout=.1,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, config.context_layers, enable_nested_tensor=False)
        # Curve, control count (1..32), repeat count (1..8), log length, combo.
        self.attributes = nn.Linear(w, 4+32+8+2)
        self.topology = nn.Linear(4+1+1+1, w)
        self.flow = ActiveEventGeometryDiffuser(GeometryDiffusionConfig(
            context_dim=w, hidden_dim=w, heads=config.heads, layers=config.flow_layers,
            max_anchors=config.max_anchors, event_feature_dim=4, dropout=.1))

    def encode(self, condition, starts, mask):
        relative = (starts-starts[:, :1])/1000
        position = self.flow._time_embedding(relative*10, self.config.width)
        h = (self.input(condition)+position)*mask[..., None]
        attention = mask.clone()
        attention[~attention.any(1), 0] = True
        h = self.context(h, src_key_padding_mask=~attention)*mask[..., None]
        return h, self.attributes(h)

    def field(self, h, state, t, kinds, counts, curves, repeats, lengths, mask, features, positions):
        topo = torch.cat((F.one_hot(curves, 4).float(), counts[..., None]/32,
                          repeats[..., None]/8, torch.log1p(lengths[..., None]/100)), -1)
        return self.flow(h+self.topology(topo), kinds, counts, state, t,
                         event_mask=mask, event_features=features, event_positions=positions)

    def forward(self, batch, state, t):
        h, attributes = self.encode(batch['condition'], batch['starts'], batch['mask'])
        velocity = self.field(h, state, t, batch['kinds'], batch['counts'], batch['curves'],
                              batch['repeats'], batch['lengths'], batch['mask'],
                              batch['features'], batch['positions'])
        return velocity, attributes


def geometry_loss(model, b, *, generator=None):
    mask = geometry_point_mask(b['mask'], b['kinds'], b['counts'])
    clean = encode_geometry(b['geometry'], SPACE)*mask[..., None]
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)*mask[..., None]
    t = torch.rand(len(clean), device=clean.device, generator=generator)
    t4 = t[:, None, None, None]
    state = (1-t4)*noise+t4*clean
    v, attributes = model(b, state, t)
    error = (v.float()-(clean-noise)).square().mean(-1)
    head = (error[..., 0]*b['mask']).sum()/b['mask'].sum().clamp_min(1)
    anchors = (error[..., 1:]*mask[..., 1:]).sum()/mask[..., 1:].sum().clamp_min(1)
    sliders = b['mask'] & b['kinds'].eq(1)
    def masked_mean(values, valid):
        return (values*valid).sum()/valid.sum().clamp_min(1)
    def ce(logits, labels):
        return masked_mean(F.cross_entropy(logits.transpose(1, 2).float(), labels, reduction='none'), sliders)
    curve = ce(attributes[..., :4], b['curves'])
    count = ce(attributes[..., 4:36], (b['counts']-1).clamp(0, 31))
    repeat = ce(attributes[..., 36:44], (b['repeats']-1).clamp(0, 7))
    length = masked_mean(F.smooth_l1_loss(attributes[..., 44].float(), torch.log1p(b['lengths']/100), reduction='none'), sliders)
    combo = masked_mean(F.binary_cross_entropy_with_logits(attributes[..., 45].float(), b['combo'], reduction='none'), b['mask'])
    estimate = decode_geometry(state+(1-t4)*v.float(), SPACE)
    patterns = geometry_pattern_losses(estimate, b['geometry'], b['kinds'], b['counts'],
                                      b['mask'], reliability=t.square(), event_positions=b['positions'])
    loss = head+anchors+.15*(curve+count+repeat+length+combo)
    loss = loss+.1*patterns['spacing_loss']+.05*patterns['turn_loss']+.1*patterns['slider_shape_loss']
    return {'loss': loss, 'head_flow': head, 'anchor_flow': anchors, 'curve_ce': curve,
            'count_ce': count, 'repeat_ce': repeat, 'length_huber': length, 'combo_bce': combo, **patterns}


@torch.inference_mode()
def sample_geometry_flow(model, batch, *, steps=48, generator=None, known=None, known_mask=None):
    if model.training or steps < 1:
        raise ValueError('sampling requires eval mode and positive steps')
    point_mask = geometry_point_mask(batch['mask'], batch['kinds'], batch['counts'])[..., None]
    shape = (*batch['mask'].shape, 33, 2)
    noise = torch.randn(shape, device=batch['condition'].device, generator=generator)*point_mask
    state = noise.clone()
    h, _ = model.encode(batch['condition'], batch['starts'], batch['mask'])
    if known is not None:
        if known_mask is None or known.shape != state.shape:
            raise ValueError('known geometry and mask must match sample')
        fixed = known_mask[..., None, None] & point_mask
        clean_known = encode_geometry(known, SPACE)
    for i in range(steps):
        level = i/steps
        if known is not None:
            state = torch.where(fixed, (1-level)*noise+level*clean_known, state)
        t = torch.full((len(state),), level, device=state.device)
        velocity = model.field(h, state, t, batch['kinds'], batch['counts'], batch['curves'],
                               batch['repeats'], batch['lengths'], batch['mask'], batch['features'], batch['positions'])
        state = (state+velocity/steps)*point_mask
    result = decode_geometry(state, SPACE)*point_mask
    return torch.where(fixed, known, result) if known is not None else result
