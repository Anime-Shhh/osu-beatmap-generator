"""[G] training stage: rectified-flow loss with geometry auxiliaries; distribution validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.v3.common import io
from src.v3.models.placement import PlacementModel, pack, sample, state_mask, to_pixels, exits, unpack, out_of_bounds
from src.v3.train.data import DATA_ROOT, PlacementData, collate_placement
from src.v3.train.loop import sample_rng

COND_KEEP = {"stars": 0.9, "ar": 0.7, "od": 0.7, "cs": 0.8, "hp": 0.7, "sm": 0.8, "era": 0.7, "source": 0.9}
AUX_W = 0.1


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0, 1, 101)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def geometry_stats(hp: np.ndarray, kind: np.ndarray, time_ms: np.ndarray) -> dict[str, np.ndarray]:
    """Spacing per ms of gap and signed turn angles over consecutive non-spinner heads."""
    keep = kind != 2
    h, t = hp[keep], time_ms[keep]
    if len(h) < 3:
        return {"spacing": np.zeros(0), "angle": np.zeros(0)}
    seg = np.diff(h, axis=0)
    dist = np.linalg.norm(seg, axis=1)
    gap = np.maximum(np.diff(t), 1.0)
    a = seg[:-1]
    b = seg[1:]
    ok = (dist[:-1] > 8) & (dist[1:] > 8)
    ang = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))[ok]
    return {"spacing": dist / gap, "angle": ang}


class PlacementStage:
    def __init__(self, root: Path = DATA_ROOT, size: str = "base", seed: int = 0, val_charts: int = 32,
                 teacher_p: float = 0.3, max_objects: int = 512):
        self.root, self.size, self.seed, self.val_n = Path(root), size, seed, val_charts
        self.teacher_p, self.max_objects = teacher_p, max_objects
        self.train = PlacementData(self.root, "train", max_objects=max_objects, seed=seed, teacher_p=teacher_p)

    def model_kwargs(self):
        return {"tiny": dict(d=128, depth=2, heads=4, frontend_width=64, cond_dim=64),
                "small": dict(d=384, depth=8, heads=6),
                "base": dict(d=512, depth=12, heads=8),
                "large": dict(d=768, depth=16, heads=12)}[self.size]

    def build_model(self):
        return PlacementModel(**self.model_kwargs())

    def dataset(self):
        return self.train

    def collate(self, items):
        return collate_placement(items)

    def binding(self):
        return {"stage": "placement", "size": self.size, "model": self.model_kwargs(), "teacher_p": self.teacher_p,
                "max_objects": self.max_objects,
                "manifest": io.sha256_file(self.root / "manifest" / "charts.jsonl")}

    def loss(self, model, batch, step):
        raw = model.module if hasattr(model, "module") else model
        b, n = batch["mask"].shape
        dev = batch["mask"].device
        sm = state_mask(batch).to(torch.float32)
        x1 = pack(batch["head"], batch["path"]) * sm
        x0 = torch.randn_like(x1) * sm
        t = torch.sigmoid(torch.randn(b, device=dev))
        tt = t[:, None, None]
        xt = (1 - tt) * x0 + tt * x1
        keep = {k: torch.rand(b, device=dev) < p for k, p in COND_KEEP.items()}
        cond = raw.condition(batch, t, keep)
        tokens = raw.encode(batch, audio_keep=torch.rand(b, device=dev) >= 0.1)
        x_sc = torch.zeros_like(xt)
        if torch.rand(()) < 0.5:
            with torch.no_grad():   # rank-local coin flip: bypass the DDP wrapper (no collectives)
                v0 = raw(xt, x_sc, t, tokens.detach(), cond.detach(), batch)
                x_sc = (xt + (1 - tt) * v0) * sm
        v = model(xt, x_sc, t, tokens, cond, batch)
        err = (v.float() - (x1 - x0)) ** 2
        mh = (batch["m_head"] & batch["mask"]).float()
        mp = (batch["m_path"] & batch["mask"]).float()
        l_head = (err[..., :2].sum(-1) * mh).sum() / (2 * mh.sum()).clamp(min=1)
        l_path = (err[..., 2:].sum(-1) * mp).sum() / (64 * mp.sum()).clamp(min=1)
        # geometry auxiliaries on the predicted clean state (reliable as t -> 1)
        x1h = xt + (1 - tt) * v.float()
        hh, ph = unpack(x1h)
        ht, pt = unpack(x1)
        hp_h, pts_h = to_pixels(hh, ph, batch["length"])
        hp_t, pts_t = to_pixels(ht, pt, batch["length"])
        w = (t ** 2)[:, None]
        pair = mh[:, 1:] * mh[:, :-1]
        d_h = (hp_h[:, 1:] - hp_h[:, :-1]).norm(dim=-1)
        d_t = (hp_t[:, 1:] - hp_t[:, :-1]).norm(dim=-1)
        l_space = ((d_h - d_t).abs() / 100 * pair * w).sum() / pair.sum().clamp(min=1)
        ex_h, ex_t = exits(hp_h, pts_h, batch["spans"]), exits(hp_t, pts_t, batch["spans"])
        sl = mp[:, :-1] * mh[:, 1:]
        e_h = (hp_h[:, 1:] - ex_h[:, :-1]).norm(dim=-1)
        e_t = (hp_t[:, 1:] - ex_t[:, :-1]).norm(dim=-1)
        l_exit = ((e_h - e_t).abs() / 100 * sl * w).sum() / sl.sum().clamp(min=1)
        seg_h, seg_t = hp_h[:, 1:] - hp_h[:, :-1], hp_t[:, 1:] - hp_t[:, :-1]
        tri = pair[:, 1:] * pair[:, :-1] * (d_t[:, 1:] > 8).float() * (d_t[:, :-1] > 8).float()

        def turn(s):
            a, c = F.normalize(s[:, :-1], dim=-1), F.normalize(s[:, 1:], dim=-1)
            return torch.stack([(a * c).sum(-1), a[..., 0] * c[..., 1] - a[..., 1] * c[..., 0]], -1)

        l_turn = (((turn(seg_h) - turn(seg_t)) ** 2).sum(-1) * tri * w).sum() / tri.sum().clamp(min=1)
        loss = l_head + l_path + AUX_W * (l_space + l_exit + l_turn)
        return loss, {"loss": loss.detach(), "head": l_head.detach(), "path": l_path.detach(),
                      "spacing": l_space.detach(), "exit": l_exit.detach(), "turn": l_turn.detach()}

    def _val_items(self):
        if not hasattr(self, "_val"):
            data = PlacementData(self.root, "val", max_objects=self.max_objects, seed=self.seed, augment=False,
                                 teacher_p=0.0)
            order = sorted(range(len(data.rows)), key=lambda i: data.rows[i]["chart_sha256"])
            picks = order[:: max(1, len(order) // self.val_n)][: self.val_n]
            self._val = [data.make(i, sample_rng(self.seed, i, 99), start=0) for i in picks]
        return self._val

    def validate(self, model, step):
        dev = next(model.parameters()).device
        items = self._val_items()
        gen = {"spacing": [], "angle": []}
        ref = {"spacing": [], "angle": []}
        oob, total, losses = 0, 0, []
        g = torch.Generator(device=dev).manual_seed(1234)
        for s in range(0, len(items), 8):
            chunk = items[s:s + 8]
            batch = {k: v.to(dev) for k, v in collate_placement(chunk).items()}
            torch.manual_seed(4321 + s)
            losses.append(float(self.loss(model, batch, step)[0]))
            out = sample(model, batch, steps=32, generator=g, constrain=False)
            for i, it in enumerate(chunk):
                n = len(it["kind"])
                hp = out["head_px"][i, :n].float().cpu().numpy()
                pts = out["path_px"][i, :n].float().cpu().numpy()
                kind = it["kind"]
                live = kind != 2
                bad = out_of_bounds(torch.as_tensor(hp), torch.as_tensor(pts), torch.as_tensor(kind))
                oob += int(bad.sum())   # illegal footprint only: human-style body overshoot is not penalized
                total += int(live.sum())
                rh = (it["head"] + 1) * np.array([256.0, 192.0])
                for d, h in ((gen, hp), (ref, rh)):
                    st = geometry_stats(h, kind, it["time_ms"])
                    d["spacing"].append(st["spacing"])
                    d["angle"].append(st["angle"])
        w_sp = _w1(np.concatenate(gen["spacing"]), np.concatenate(ref["spacing"]))
        w_an = _w1(np.concatenate(gen["angle"]), np.concatenate(ref["angle"]))
        rate = oob / max(1, total)
        return {"select": w_sp / 0.1 + w_an / np.pi + 2 * rate, "w1_spacing_px_per_ms": w_sp,
                "w1_turn_rad": w_an, "oob_rate_unprojected": rate, "val_loss": float(np.mean(losses))}
