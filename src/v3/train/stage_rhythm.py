"""[R] training stage: loss (masked discrete diffusion) and generated-metric validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.v3.common import io
from src.v3.data.audio import FRAME_MS
from src.v3.data.grid import SLOT_TICKS
from src.v3.eval.match import onset_scores, short_pair_recall
from src.v3.models.rhythm import ATTRS, RhythmModel
from src.v3.train.data import DATA_ROOT, RhythmData, collate_rhythm
from src.v3.train.loop import sample_rng

SPAN_MASK_P = 0.25
AUDIO_DROP = 0.10
COND_KEEP = {"stars": 0.9, "ar": 0.7, "od": 0.7, "cs": 0.7, "hp": 0.7, "bpm": 0.8, "era": 0.7}


class RhythmStage:
    def __init__(self, root: Path = DATA_ROOT, size: str = "base", seed: int = 0, val_charts: int = 32,
                 max_beats: int = 384):
        self.root, self.size, self.seed, self.val_n, self.max_beats = Path(root), size, seed, val_charts, max_beats
        self.train = RhythmData(self.root, "train", max_beats=max_beats, seed=seed)
        self.vocab = io.read_json(self.root / "manifest" / "vocab.json")

    def model_kwargs(self) -> dict[str, Any]:
        sizes = {"tiny": dict(d=128, depth=2, heads=4, frontend_width=64, attr_dim=16, cond_dim=64),
                 "small": dict(d=384, depth=8, heads=6, frontend_width=128),
                 "base": dict(d=512, depth=12, heads=8, frontend_width=192),
                 "large": dict(d=768, depth=16, heads=12, frontend_width=256)}
        return sizes[self.size]

    def build_model(self):
        return RhythmModel(self.vocab["patterns"], self.train.K, self.train.n_dur, self.train.n_sv,
                           **self.model_kwargs())

    def dataset(self):
        return self.train

    def collate(self, items):
        return collate_rhythm(items)

    def binding(self):
        return {"stage": "rhythm", "size": self.size, "model": self.model_kwargs(), "max_beats": self.max_beats,
                "manifest": io.sha256_file(self.root / "manifest" / "charts.jsonl"),
                "vocab": io.sha256_file(self.root / "manifest" / "vocab.json")}

    # --------------------------------------------------------------- loss
    def loss(self, model, batch, step):
        raw = model.module if hasattr(model, "module") else model
        b, l = batch["pattern"].shape
        K = batch["attrs"].shape[2]
        dev = batch["pattern"].device
        mask = batch["mask"]
        t = torch.rand(b, device=dev).clamp(0.02, 1.0)
        mp = torch.rand(b, l, device=dev) < t[:, None]
        # span masking: whole phrases hidden (teaches inpainting / continuation)
        span = torch.rand(b, device=dev) < SPAN_MASK_P
        if span.any():
            starts = (torch.rand(b, device=dev) * l).long()
            lengths = (8 + torch.rand(b, device=dev) * 56).long()
            pos = torch.arange(l, device=dev)[None]
            mp |= span[:, None] & (pos >= starts[:, None]) & (pos < (starts + lengths)[:, None])
        mp &= mask
        ma = mp[..., None, None] | (torch.rand(b, l, K, len(ATTRS), device=dev) < t[:, None, None, None])
        ms = (torch.rand(b, l, device=dev) < t[:, None]) & mask
        state = {"pattern": torch.where(mp, raw.mask_id("pattern"), batch["pattern"].clamp(min=0)),
                 "sv": torch.where(ms, raw.mask_id("sv"),
                                   torch.where(batch["sv"] < 0, raw.na_id("sv"), batch["sv"]))}
        attrs = []
        for j, name in enumerate(ATTRS):
            tgt = batch["attrs"][..., j]
            ids = torch.where(tgt < 0, raw.na_id(name), tgt)
            attrs.append(torch.where(ma[..., j], raw.mask_id(name), ids))
        state["attrs"] = torch.stack(attrs, -1)
        keep = {k: torch.rand(b, device=dev) < p for k, p in COND_KEEP.items()}
        cond = raw.condition(batch, keep)
        audio_keep = torch.rand(b, device=dev) >= AUDIO_DROP
        emb = raw.encode_audio(batch["audio"], batch["slot_pos"])
        out = model(state, emb, audio_keep, cond, batch, bim_keep=torch.rand(b, device=dev) > 0.2)

        def ce(logits, target, where):
            if where.sum() == 0:
                return logits.sum() * 0.0
            return F.cross_entropy(logits[where].float(), target[where])

        l_pat = ce(out["pattern"], batch["pattern"], mp & (batch["pattern"] >= 0))
        l_attr = {}
        for j, name in enumerate(ATTRS):
            tgt = batch["attrs"][..., j]
            l_attr[name] = ce(out[name], tgt, ma[..., j] & (tgt >= 0) & mask[..., None])
        l_sv = ce(out["sv"], batch["sv"], ms & (batch["sv"] >= 0))
        vm = batch["valid"] & mask[..., None]
        l_sal = F.binary_cross_entropy_with_logits(out["salience"].float()[vm], batch["salience"][vm])
        loss = l_pat + sum(l_attr.values()) / len(ATTRS) + 0.5 * l_sv + 0.1 * l_sal
        logs = {"loss": loss.detach(), "pattern": l_pat.detach(), "sv": l_sv.detach(), "salience": l_sal.detach()}
        logs.update({f"attr_{k}": v.detach() for k, v in l_attr.items()})
        return loss, logs

    # ---------------------------------------------------------- validation
    def _val_items(self):
        """Fixed validation crops (human grid), each starting just before the chart's first onset."""
        if not hasattr(self, "_val"):
            data = RhythmData(self.root, "val", max_beats=self.max_beats, seed=self.seed, augment=False)
            order = sorted(range(len(data.rows)), key=lambda i: data.rows[i]["chart_sha256"])
            picks = order[:: max(1, len(order) // self.val_n)][: self.val_n]
            items = []
            for i in picks:
                beats, ons = data.chart_arrays(data.rows[i])
                first = int(ons["beat"][0]) if len(ons["beat"]) else 0
                start = min(max(0, first - 4), max(0, len(beats["start_ms"]) - self.max_beats))
                items.append(data.make(i, sample_rng(self.seed, i, 99), start=start))
            self._val = items
        return self._val

    def validate(self, model, step):
        dev = next(model.parameters()).device
        items = self._val_items()
        bits = model.pattern_bits.cpu().numpy()
        frac = np.asarray(SLOT_TICKS) / 48.0
        scores = {"f1_25": [], "f1_10": [], "density": [], "pairs": [0, 0], "val_loss": []}
        g = torch.Generator(device=dev).manual_seed(1234)
        for start in range(0, len(items), 8):
            chunk = items[start:start + 8]
            batch = {k: v.to(dev) for k, v in collate_rhythm(chunk).items()}
            torch.manual_seed(4321 + start)
            loss, _ = self.loss(model, batch, step)
            scores["val_loss"].append(float(loss))
            state, _ = model.sample(batch, steps=32, cfg=1.5, generator=g)
            for i, it in enumerate(chunk):
                n = len(it["pattern"])
                ref_on = bits[it["pattern"]]
                gen_on = bits[state["pattern"][i, :n].cpu().numpy()]
                times = it["slot_pos"] * FRAME_MS          # no jitter/stretch in validation crops
                ref_t, gen_t = times[ref_on], times[gen_on]
                ref_b = (it["beat_pos"][:, None] + frac[None, :])[ref_on]
                s25 = onset_scores(gen_t, ref_t, 25.0)
                scores["f1_25"].append(s25["f1"])
                scores["f1_10"].append(onset_scores(gen_t, ref_t, 10.0)["f1"])
                scores["density"].append(len(gen_t) / max(1, len(ref_t)))
                got, tot = short_pair_recall(gen_t, ref_t, ref_b, 25.0)
                scores["pairs"][0] += got
                scores["pairs"][1] += tot
        f1 = float(np.mean(scores["f1_25"]))
        return {"select": f1, "f1_25": f1, "f1_10": float(np.mean(scores["f1_10"])),
                "density_ratio": float(np.median(scores["density"])),
                "short_pair_recall": scores["pairs"][0] / max(1, scores["pairs"][1]),
                "val_loss": float(np.mean(scores["val_loss"]))}
