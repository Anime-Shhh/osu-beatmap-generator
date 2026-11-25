"""[R] Rhythm: masked discrete diffusion over beat tokens (one token per beat).

State per beat: ``pattern`` (which of the 24 slots hold onsets, from a learned-
free vocabulary of human patterns), then per onset k < popcount(pattern) the
attributes kind / duration class / spans / new-combo / stack, and the beat's
slider-velocity class. Attributes are only revealed after their pattern.
Audio enters as features sampled at the 24 slot times of each beat (own conv
front end on 5 ms features), so a 1/4 double is an explicit token, not a peak.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Backbone, ConvFrontend, FourierScalar, gather_frames

ATTRS = ("kind", "dur", "spans", "nc", "stack")


class RhythmModel(nn.Module):
    def __init__(self, patterns: list[int], K: int, n_dur: int, n_sv: int, *, spinner_beats: int = 32,
                 d: int = 512, depth: int = 12, heads: int = 8, audio_dim: int = 130, slot_dim: int = 32,
                 frontend_width: int = 192, attr_dim: int = 64, cond_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        masks = torch.tensor([0] + list(patterns), dtype=torch.long)
        bits = ((masks[:, None] >> torch.arange(24)) & 1).bool()
        self.register_buffer("pattern_bits", bits, persistent=False)          # [P, 24]
        self.register_buffer("pattern_pop", bits.sum(1), persistent=False)    # [P]
        self.P, self.K = len(masks), K
        self.sizes = {"kind": 3, "dur": n_dur + spinner_beats, "spans": 32, "nc": 2, "stack": 3}
        self.n_sv = n_sv
        self.frontend = ConvFrontend(audio_dim, frontend_width, slot_dim, depth=5)
        self.audio_in = nn.Sequential(nn.Linear(24 * slot_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.audio_null = nn.Parameter(torch.zeros(d))
        self.pattern_embed = nn.Embedding(self.P + 1, d)                      # + MASK
        self.attr_embed = nn.ModuleDict({k: nn.Embedding(n + 2, attr_dim) for k, n in self.sizes.items()})  # + NA, MASK
        self.k_embed = nn.Parameter(torch.randn(K, attr_dim) * 0.02)
        self.attr_in = nn.Linear(K * attr_dim, d)
        self.sv_embed = nn.Embedding(n_sv + 2, d)                             # + NA, MASK
        self.bim_embed = nn.Embedding(17, d)                                  # 16 = unknown/dropped
        self.valid_in = nn.Linear(24, d)
        self.cond_scalars = nn.ModuleDict({k: FourierScalar(cond_dim) for k in ("stars", "ar", "od", "cs", "hp", "bpm")})
        self.era_embed = nn.Embedding(10, cond_dim)                           # 9 buckets + null
        self.backbone = Backbone(d, depth, heads, cond_dim, dropout, rope_base=1000.0)
        self.out_pattern = nn.Linear(d, self.P)
        self.out_attr = nn.Linear(d, K * sum(self.sizes.values()))
        self.out_sv = nn.Linear(d, n_sv)
        self.out_sal = nn.Linear(d, 24)

    # ids for special tokens
    def mask_id(self, name: str) -> int:
        return {"pattern": self.P, "sv": self.n_sv + 1}.get(name, self.sizes.get(name, 0) + 1)

    def na_id(self, name: str) -> int:
        return self.n_sv if name == "sv" else self.sizes[name]

    def encode_audio(self, audio, slot_pos):
        feats = self.frontend(audio)                                          # [B, T, slot_dim]
        g = gather_frames(feats, slot_pos)                                    # [B, L, 24, slot_dim]
        return self.audio_in(g.flatten(2))

    def condition(self, batch, keep: dict[str, torch.Tensor] | None = None):
        keep = keep or {}
        scaled = {"stars": batch["stars"] / 10, "ar": batch["ar"] / 10, "od": batch["od"] / 10,
                  "cs": batch["cs"] / 10, "hp": batch["hp"] / 10,
                  "bpm": torch.log2(batch["bpm"].clamp(30, 480) / 30) / 4}
        c = sum(self.cond_scalars[k](v, keep.get(k)) for k, v in scaled.items())
        era = batch["era"].clamp(0, 8)
        if "era" in keep:
            era = torch.where(keep["era"], era, torch.full_like(era, 9))
        return c + self.era_embed(era)

    def forward(self, state, audio_emb, audio_keep, cond, batch, bim_keep=None):
        """state: pattern [B,L], attrs [B,L,K,5], sv [B,L] (ids incl. NA/MASK)."""
        b, l = state["pattern"].shape
        a = torch.where(audio_keep[:, None, None], audio_emb, self.audio_null.to(audio_emb.dtype))
        attr = sum(self.attr_embed[name](state["attrs"][..., j]) for j, name in enumerate(ATTRS)) + self.k_embed
        bim = batch["bim"].clamp(0, 15)
        if bim_keep is not None:
            bim = torch.where(bim_keep[:, None], bim, torch.full_like(bim, 16))
        h = (a + self.pattern_embed(state["pattern"]) + self.attr_in(attr.flatten(2)) + self.sv_embed(state["sv"])
             + self.bim_embed(bim) + self.valid_in(batch["valid"].float()))
        h = self.backbone(h, cond, batch["beat_pos"], batch["mask"])
        attr_logits = self.out_attr(h).view(b, l, self.K, -1)
        split = torch.split(attr_logits, list(self.sizes.values()), dim=-1)
        out = {"pattern": self.out_pattern(h), "sv": self.out_sv(h), "salience": self.out_sal(h)}
        out.update({name: split[j] for j, name in enumerate(ATTRS)})
        # Patterns that place onsets outside a partial beat are impossible.
        bad = (self.pattern_bits[None, None] & ~batch["valid"][:, :, None, :]).any(-1)   # [B, L, P]
        out["pattern"] = out["pattern"].masked_fill(bad, -1e4)
        return out

    # ------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, batch, steps: int = 48, cfg: float = 1.5, temperature: float = 1.0,
               generator: torch.Generator | None = None):
        dev = batch["mask"].device
        b, l = batch["mask"].shape
        K = self.K
        emb = self.encode_audio(batch["audio"], batch["slot_pos"])
        cond = self.condition(batch)
        on = torch.ones(b, dtype=torch.bool, device=dev)
        state = {"pattern": torch.full((b, l), self.mask_id("pattern"), device=dev, dtype=torch.long),
                 "attrs": torch.stack([torch.full((b, l, K), self.mask_id(n), device=dev, dtype=torch.long)
                                       for n in ATTRS], -1),
                 "sv": torch.full((b, l), self.mask_id("sv"), device=dev, dtype=torch.long)}
        p_masked = batch["mask"].clone()
        a_masked = batch["mask"][:, :, None, None].expand(b, l, K, len(ATTRS)).clone()
        s_masked = batch["mask"].clone()
        logp = {"sv": torch.zeros(b, l, self.n_sv, device=dev)}
        kk = torch.arange(K, device=dev)

        def draw(logits):
            probs = F.softmax(logits.float() / temperature, -1)
            flat = probs.reshape(-1, probs.shape[-1])
            return torch.multinomial(flat, 1, generator=generator).view(probs.shape[:-1])

        for s in range(steps):
            t_now, t_next = 1 - s / steps, 1 - (s + 1) / steps
            out_c = self(state, emb, on, cond, batch)
            if cfg != 1.0:
                out_u = self(state, emb, ~on, cond, batch)
                out = {k: out_u[k] + cfg * (out_c[k] - out_u[k]) for k in out_c}
            else:
                out = out_c
            reveal = (t_now - t_next) / t_now
            last = s == steps - 1
            r = torch.rand(b, l, generator=generator, device=dev) < reveal
            rev = p_masked & (r | last)
            state["pattern"] = torch.where(rev, draw(out["pattern"]), state["pattern"])
            p_masked &= ~rev
            known = ~p_masked & batch["mask"]
            pc = self.pattern_pop[state["pattern"].clamp(max=self.P - 1)]
            exists = known[..., None] & (kk[None, None] < pc[..., None])                       # [B, L, K]
            for j, name in enumerate(ATTRS):
                na = known[..., None] & ~exists & a_masked[..., j]
                state["attrs"][..., j] = torch.where(na, torch.full_like(state["attrs"][..., j], self.na_id(name)),
                                                     state["attrs"][..., j])
                a_masked[..., j] &= ~na
                ra = torch.rand(b, l, K, generator=generator, device=dev) < reveal
                take = exists & a_masked[..., j] & (ra | last)
                state["attrs"][..., j] = torch.where(take, draw(out[name]), state["attrs"][..., j])
                a_masked[..., j] &= ~take
            rs = s_masked & ((torch.rand(b, l, generator=generator, device=dev) < reveal) | last)
            logp["sv"] = torch.where(rs[..., None], F.log_softmax(out["sv"].float(), -1), logp["sv"])
            state["sv"] = torch.where(rs, draw(out["sv"]), state["sv"])
            s_masked &= ~rs
        return state, logp
