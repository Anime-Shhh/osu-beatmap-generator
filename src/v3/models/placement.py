"""[G] Placement: rectified-flow DiT over the object sequence.

State per object (normalized): head (x/256-1, y/192-1) and, for sliders, 32 unit
heading vectors of equal arc-length steps; the pixel length is *given* (from SV x
duration), so ``path = head + cumsum(heading * length / 32)`` is exact in length.
Velocity prediction, self-conditioning, t=1 is data. During the last part of
sampling the predicted clean state is constrained (exact stacks, combo-group
translation into the playfield) and re-noised consistently.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Backbone, ConvFrontend, FourierScalar, gather_frames

H = 32
STATE = 2 + 2 * H
SCALE = torch.tensor([256.0, 192.0])

# The legal footprint, shared by projection, validation and the eval checker (src.v3.eval.metrics).
# Heads: every ranked human head lies inside the 512x384 playfield. Slider bodies: humans routinely run
# them past the playfield into osu!'s visible 4:3 area (val300: 220/300 charts, median 8 px, max 57 px
# left / 34 right / 22 top / 7.6 bottom), so bodies are bounded by that area (the playfield is drawn
# 8 px low, hence the smaller bottom margin). How far bodies overshoot is a style distribution, not a fault.
HEAD_LO, HEAD_HI = (0.0, 0.0), (512.0, 384.0)
BODY_LO, BODY_HI = (-64.0, -56.0), (576.0, 424.0)


def out_of_bounds(hp, pts, kind, tol: float = 1.0):
    """Per object: head outside the playfield, or (sliders) body centerline outside the visible area."""
    lo_h, hi_h = hp.new_tensor(HEAD_LO) - tol, hp.new_tensor(HEAD_HI) + tol
    lo_b, hi_b = pts.new_tensor(BODY_LO) - tol, pts.new_tensor(BODY_HI) + tol
    head = ((hp < lo_h) | (hp > hi_h)).any(-1)
    body = ((pts < lo_b) | (pts > hi_b)).any(-1).any(-1) & (kind == 1)
    return (head | body) & (kind != 2)


class PlacementModel(nn.Module):
    def __init__(self, *, d: int = 512, depth: int = 12, heads: int = 8, audio_dim: int = 130, audio_feat: int = 48,
                 frontend_width: int = 128, cond_dim: int = 256, n_scal: int = 6, dropout: float = 0.1):
        super().__init__()
        self.frontend = ConvFrontend(audio_dim, frontend_width, audio_feat, depth=5)
        self.audio_in = nn.Linear(5 * audio_feat, d)
        self.audio_null = nn.Parameter(torch.zeros(d))
        self.embeds = nn.ModuleDict({"kind": nn.Embedding(3, d), "nc": nn.Embedding(2, d), "stack": nn.Embedding(3, d),
                                     "spans": nn.Embedding(33, d), "tick": nn.Embedding(48, d), "bim": nn.Embedding(17, d)})
        self.scal_in = nn.Linear(n_scal, d)
        self.state_in = nn.Linear(2 * STATE, d)
        self.t_embed = FourierScalar(cond_dim, nullable=False)   # flow time is never dropped
        self.cond_scalars = nn.ModuleDict({k: FourierScalar(cond_dim) for k in ("stars", "ar", "od", "cs", "hp", "sm")})
        self.era_embed = nn.Embedding(10, cond_dim)
        self.source_embed = nn.Embedding(3, cond_dim)                         # human, teacher, null
        self.backbone = Backbone(d, depth, heads, cond_dim, dropout, rope_base=1000.0)
        self.out = nn.Linear(d, STATE)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def condition(self, batch, t, keep=None):
        keep = keep or {}
        scaled = {"stars": batch["stars"] / 10, "ar": batch["ar"] / 10, "od": batch["od"] / 10,
                  "cs": batch["cs"] / 10, "hp": batch["hp"] / 10, "sm": batch["sm"] / 4}
        c = sum(self.cond_scalars[k](v, keep.get(k)) for k, v in scaled.items()) + self.t_embed(t)
        era = batch["era"].clamp(0, 8)
        if "era" in keep:
            era = torch.where(keep["era"], era, torch.full_like(era, 9))
        source = batch["source"]
        if "source" in keep:
            source = torch.where(keep["source"], source, torch.full_like(source, 2))
        return c + self.era_embed(era) + self.source_embed(source)

    def encode(self, batch, audio_keep=None):
        feats = self.frontend(batch["audio"])
        a = self.audio_in(gather_frames(feats, batch["apos"]).flatten(2))
        if audio_keep is not None:
            a = torch.where(audio_keep[:, None, None], a, self.audio_null.to(a.dtype))
        tok = a + self.scal_in(batch["scal"]) + sum(self.embeds[k](batch[k]) for k in self.embeds)
        return tok

    def forward(self, x_t, x_sc, t, tokens, cond, batch):
        h = tokens + self.state_in(torch.cat([x_t, x_sc], -1))
        h = self.backbone(h, cond, batch["beat_pos"], batch["mask"])
        return self.out(h)


# ------------------------------------------------------------------ state ----

def pack(head, path):
    return torch.cat([head, path.flatten(-2)], -1)


def unpack(x):
    return x[..., :2], x[..., 2:].unflatten(-1, (H, 2))


def state_mask(batch):
    """[B, N, STATE] dims that carry signal (heads of non-spinners, paths of sliders)."""
    m = torch.zeros(*batch["mask"].shape, STATE, dtype=torch.bool, device=batch["mask"].device)
    m[..., :2] = (batch["kind"] != 2)[..., None] & batch["mask"][..., None]
    m[..., 2:] = (batch["kind"] == 1)[..., None] & batch["mask"][..., None]
    return m


def to_pixels(head, path, length):
    """Normalized state -> head px [B,N,2], path px [B,N,H+1,2] (exact length)."""
    scale = SCALE.to(head.device, head.dtype)
    hp = (head + 1) * scale
    u = F.normalize(path.float(), dim=-1)
    steps = u * (length[..., None, None] / H)
    pts = hp[..., None, :] + torch.cat([torch.zeros_like(steps[..., :1, :]), steps.cumsum(-2)], -2)
    return hp, pts


def exits(hp, pts, spans):
    odd = (spans % 2 == 1)[..., None]
    return torch.where(odd, pts[..., -1, :], hp)


# ----------------------------------------------------------- constraints ----

def project_playfield(hp, pts, kind, nc, valid):
    """Translate combo groups (or single objects if a group cannot fit) into the legal footprint
    (heads in 512x384, slider bodies in the visible area; see HEAD_LO / BODY_LO).

    Returns translated head/path pixels and the per-object displacement (px).
    """
    hp, pts = hp.clone(), pts.clone()
    disp = torch.zeros(hp.shape[:2], device=hp.device)
    for b in range(hp.shape[0]):
        idx = torch.nonzero(valid[b] & (kind[b] != 2)).flatten().tolist()
        if not idx:
            continue
        starts = [i for i in idx if nc[b, i] == 1 or i == idx[0]]
        bounds = starts[1:] + [idx[-1] + 1]
        for s, e in zip(starts, bounds):
            members = [i for i in idx if s <= i < e]
            fits = _shift(pts[b, members], kind[b, members])
            if fits is None:  # group larger than the playfield: fit objects individually
                for i in members:
                    shift = _shift(pts[b, [i]], kind[b, [i]])
                    if shift is not None:
                        _apply(hp, pts, disp, b, [i], shift)
            else:
                _apply(hp, pts, disp, b, members, fits)
    return hp, pts, disp


def _shift(pts, kind):
    """Smallest translation putting every head in the playfield and every slider body in the visible
    area; None when no single translation satisfies both. pts[:, 0] is each object's head."""
    heads = pts[:, 0]
    lo = pts.new_tensor(HEAD_LO) - heads.min(0).values
    hi = pts.new_tensor(HEAD_HI) - heads.max(0).values
    body = pts[kind == 1].reshape(-1, 2)
    if body.numel():
        lo = torch.maximum(lo, pts.new_tensor(BODY_LO) - body.min(0).values)
        hi = torch.minimum(hi, pts.new_tensor(BODY_HI) - body.max(0).values)
    if (lo > hi).any():
        return None
    return torch.clamp(torch.zeros(2, device=pts.device, dtype=pts.dtype), lo, hi)


def _apply(hp, pts, disp, b, members, shift):
    hp[b, members] += shift
    pts[b, members] += shift
    disp[b, members] += shift.norm()


def from_pixels(hp, path_norm):
    scale = SCALE.to(hp.device, hp.dtype)
    return hp / scale - 1, path_norm


@torch.no_grad()
def sample(model, batch, steps: int = 32, project_from: float = 0.7, cfg: float = 1.0,
           generator: torch.Generator | None = None, constrain: bool = True):
    """Euler sampling (t: 0 noise -> 1 data). Returns head px, path px, pre-projection displacement."""
    dev = batch["mask"].device
    b, n = batch["mask"].shape
    sm = state_mask(batch).float()
    x = torch.randn(b, n, STATE, device=dev, generator=generator) * sm
    x_sc = torch.zeros_like(x)
    tokens = model.encode(batch)
    total_disp = torch.zeros(b, n, device=dev)
    for i in range(steps):
        t = torch.full((b,), i / steps, device=dev)
        cond = model.condition(batch, t)
        v = model(x, x_sc, t, tokens, cond, batch)
        if cfg != 1.0:
            cond_u = model.condition(batch, t, keep={k: torch.zeros(b, dtype=torch.bool, device=dev)
                                                     for k in ("stars", "ar", "od", "cs", "hp", "sm", "era")})
            v_u = model(x, x_sc, t, tokens, cond_u, batch)
            v = v_u + cfg * (v - v_u)
        tf = float(t[0])
        x1 = x + (1 - tf) * v
        if constrain and tf >= project_from:
            x1, disp = _constrain(x1, batch)
            total_disp = torch.maximum(total_disp, disp)
        x1 = x1 * sm
        x_sc = x1
        x0 = (x - tf * x1) / max(1 - tf, 1e-4)
        tn = (i + 1) / steps
        x = ((1 - tn) * x0 + tn * x1) * sm
    head, path = unpack(x)
    path = F.normalize(path.float(), dim=-1)
    hp, pts = to_pixels(head, path, batch["length"])
    pre = hp.clone()
    if constrain:
        hp, pts = _stacks(hp, pts, batch)
        hp, pts, disp = project_playfield(hp, pts, batch["kind"], batch["nc"], batch["mask"])
        total_disp = torch.maximum(total_disp, disp)
    return {"head_px": hp, "path_px": pts, "displacement": total_disp, "head_px_unprojected": pre}


def _stacks(hp, pts, batch):
    """Exact stacks: head = previous head (stack 1) or previous slider exit (stack 2)."""
    hp, pts = hp.clone(), pts.clone()
    ex = exits(hp, pts, batch["spans"])
    st = batch["stack"]
    for i in range(1, hp.shape[1]):
        target = torch.where((st[:, i] == 1)[:, None], hp[:, i - 1], ex[:, i - 1])
        on = (st[:, i] > 0) & batch["mask"][:, i]
        delta = torch.where(on[:, None], target - hp[:, i], torch.zeros_like(target))
        hp[:, i] += delta
        pts[:, i] += delta[:, None]
        ex[:, i] = exits(hp[:, i:i + 1], pts[:, i:i + 1], batch["spans"][:, i:i + 1])[:, 0]
    return hp, pts


def _constrain(x1, batch):
    head, path = unpack(x1)
    path = F.normalize(path.float(), dim=-1).to(x1.dtype)
    hp, pts = to_pixels(head, path, batch["length"])
    hp, pts = _stacks(hp, pts, batch)
    hp, pts, disp = project_playfield(hp, pts, batch["kind"], batch["nc"], batch["mask"])
    head_n, _ = from_pixels(hp, path)
    return pack(head_n.to(x1.dtype), path), disp
