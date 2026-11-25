"""Training datasets over the Phase 0 caches (index-addressed, deterministic).

``__getitem__(i)`` is a pure function of (seed, i): the training loop's exact
resume depends on it. Audio features are memory-mapped per song and normalized
with train-split statistics; augmentations act on features and *sampling times*,
never on beat-space labels.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.v3.common import io
from src.v3.data.audio import FRAME_MS
from src.v3.data.grid import SLOT_TICKS, TICKS
from src.v3.train.loop import sample_rng

SLOT_FRAC = np.asarray(SLOT_TICKS, np.float64) / TICKS
DATA_ROOT = Path(os.environ.get("V3_DATA_ROOT", "/scratch/asj102/osu_project/datasets/v3-phase0"))


class FieldStore:
    """Lazily memory-mapped per-shard field arrays (safe across DataLoader workers)."""

    def __init__(self, base: Path):
        self.base = base
        self._maps: dict[tuple[str, str], np.ndarray] = {}

    def get(self, shard: str, field: str) -> np.ndarray:
        key = (shard, field)
        if key not in self._maps:
            self._maps[key] = np.load(self.base / shard / f"{field}.npy", mmap_mode="r")
        return self._maps[key]

    def slice(self, shard: str, field: str, span: list[int]) -> np.ndarray:
        a, n = span
        return np.asarray(self.get(shard, field)[a:a + n])


def load_index(root: Path, stage: str) -> list[dict[str, Any]]:
    rows = []
    for d in sorted((root / "cache" / stage).glob("shard-*")):
        for r in io.read_jsonl(d / "index.jsonl"):
            r["shard"] = d.name
            rows.append(r)
    return rows


def window_bounds(n_frames: int, start_ms: float, end_ms: float) -> tuple[int, int]:
    """Frame range [a, b) covering [start, end), never empty. A request past the end of the audio gets
    its last frames (gather_frames clamps positions beyond them) instead of a zero-length array, which
    crashed a loader worker (and with it the first R launch) on a chart timed past its audio."""
    a = max(0, int(np.floor(start_ms / FRAME_MS)))
    b = min(n_frames, int(np.ceil(end_ms / FRAME_MS)) + 1)
    if b - a < 2:
        a = max(0, min(a, n_frames - 2))
        b = min(n_frames, a + 2)
    return a, b


class Audio:
    def __init__(self, root: Path):
        self.root = root
        norm = io.read_json(root / "cache" / "audio_norm.json")
        self.mean = np.asarray(norm["mean"], np.float32)
        self.std = np.asarray(norm["std"], np.float32)

    def frames(self, sha: str) -> np.ndarray:
        """Memory-mapped [frames, 130] features (opening reads only the header)."""
        return np.load(self.root / "audio" / sha[:2] / f"{sha}.npy", mmap_mode="r")

    def window(self, sha: str, start_ms: float, end_ms: float) -> tuple[np.ndarray, float]:
        """Normalized float32 [T, 130] covering [start, end) plus the crop's first-frame time."""
        f = self.frames(sha)
        a, b = window_bounds(len(f), start_ms, end_ms)
        x = (np.asarray(f[a:b], np.float32) - self.mean) / self.std
        return x, a * FRAME_MS


def augment_audio(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    x[:, :128] += rng.normal(0, 0.3)                                         # gain (normalized units)
    x[:, :128] += np.linspace(-1, 1, 128, dtype=np.float32) * rng.normal(0, 0.3)  # EQ tilt
    for _ in range(rng.integers(0, 3)):                                      # frequency masks
        w = rng.integers(2, 12)
        s = rng.integers(0, 128 - w)
        x[:, s:s + w] = 0.0
    return x


def stretch(x: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch features by ``rate`` (>1 = slower: more frames)."""
    n = max(2, int(round(len(x) * rate)))
    src = np.arange(n, dtype=np.float64) / rate
    lo = np.clip(np.floor(src).astype(np.int64), 0, len(x) - 1)
    hi = np.clip(lo + 1, 0, len(x) - 1)
    w = (src - lo).astype(np.float32)[:, None]
    return x[lo] * (1 - w) + x[hi] * w


def era_bucket(set_id, cuts: list[float]) -> int:
    return int(np.searchsorted(cuts, float(set_id or 0)))


def cond_vector(row: dict[str, Any], cuts: list[float], bpm: float | None = None) -> dict[str, float]:
    return {"stars": float(row.get("stars") or 0.0), "ar": float(row["ar"]), "od": float(row["od"]),
            "cs": float(row["cs"]), "hp": float(row["hp"]), "era": era_bucket(row.get("beatmap_set_id"), cuts),
            "bpm": float(bpm if bpm is not None else (row.get("main_bpm") or 120.0)),
            "sm": float(row.get("slider_multiplier") or 1.4)}


# ------------------------------------------------------------------ rhythm --

class RhythmData(torch.utils.data.Dataset):
    def __init__(self, root: Path = DATA_ROOT, split: str = "train", max_beats: int = 384,
                 max_seconds: float = 180.0, seed: int = 0, augment: bool = True):
        self.root, self.split, self.seed, self.augment = root, split, seed, augment
        self.max_beats, self.max_seconds = max_beats, max_seconds
        self.rows = [r for r in load_index(root, "rhythm") if r["split"] == split]
        self.store = FieldStore(root / "cache" / "rhythm")
        self.audio = Audio(root)
        vocab = io.read_json(root / "manifest" / "vocab.json")
        self.patterns = np.array([0] + vocab["patterns"], np.int64)
        self.K = int(max(bin(int(m)).count("1") for m in vocab["patterns"]))
        self.n_dur, self.n_sv = len(vocab["span_duration_ticks"]), len(vocab["sv_classes"])
        self.meta = io.read_json(root / "cache" / "meta.json")
        groups = collections.defaultdict(list)
        self.siblings = collections.defaultdict(list)
        for i, r in enumerate(self.rows):
            groups[r["group"]].append(i)
            self.siblings[(r["audio_sha256"], r["red_signature"])].append(i)
        self.groups = [groups[g] for g in sorted(groups)]
        w = np.sqrt([len(g) for g in self.groups])
        self.group_p = w / w.sum()

    def __len__(self):
        return 1 << 40

    def pick(self, i: int) -> int:
        rng = sample_rng(self.seed, i, 1)
        g = self.groups[rng.choice(len(self.groups), p=self.group_p)]
        return g[rng.integers(len(g))]

    def chart_arrays(self, row):
        sh, b, o = row["shard"], row["beats"], row["onsets"]
        beats = {k: self.store.slice(sh, k, b) for k in ("start_ms", "beat_ms", "frac", "bim", "pattern", "sv")}
        ons = {k: self.store.slice(sh, f"o_{k}", o) for k in ("beat", "slot", "kind", "dur", "spans", "nc", "stack")}
        return beats, ons

    def salience(self, row_idx: int, s: int, e: int) -> np.ndarray:
        row = self.rows[row_idx]
        sib = self.siblings[(row["audio_sha256"], row["red_signature"])]
        sal = np.zeros((e - s, 24), np.float32)
        for j in sib:
            r = self.rows[j]
            ob = self.store.slice(r["shard"], "o_beat", r["onsets"])
            os_ = self.store.slice(r["shard"], "o_slot", r["onsets"])
            m = (ob >= s) & (ob < e)
            sal[ob[m] - s, os_[m]] += 1.0
        return sal / len(sib)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.make(self.pick(i), sample_rng(self.seed, i, 2))

    @staticmethod
    def usable_beats(row, start_ms: np.ndarray) -> int:
        """Beats a crop may cover. Grids run to the last red line, which a few charts (10 of 145k, up to
        2740 beats) place after the audio ends; crops stay on beats that start inside the audio."""
        return max(1, min(len(start_ms), int(np.searchsorted(start_ms, row["frames"] * FRAME_MS))))

    def make(self, row_idx: int, rng: np.random.Generator, start: int | None = None) -> dict[str, Any]:
        row = self.rows[row_idx]
        beats, ons = self.chart_arrays(row)
        nb = self.usable_beats(row, beats["start_ms"])
        length = min(self.max_beats, nb)
        s = int(rng.integers(0, nb - length + 1)) if start is None else start
        e = s + length
        while e - s > 8 and beats["start_ms"][e - 1] - beats["start_ms"][s] > self.max_seconds * 1000:
            e -= 1
        st, bl, fr = beats["start_ms"][s:e], beats["beat_ms"][s:e].astype(np.float64), beats["frac"][s:e]
        slot_ms = st[:, None] + SLOT_FRAC[None, :] * bl[:, None]
        valid = SLOT_FRAC[None, :] < fr[:, None] - 1e-6
        valid[:, 0] = True
        rate = float(rng.uniform(0.92, 1.08)) if self.augment and rng.random() < 0.5 else 1.0
        x, t0 = self.audio.window(row["audio_sha256"], st[0] - 300.0, st[-1] + bl[-1] * fr[-1] + 300.0)
        if self.augment:
            x = augment_audio(x, rng)
            slot_ms = slot_ms + rng.normal(0, 1.5) + np.linspace(0, rng.normal(0, 2.0), len(st))[:, None]
        pos = (slot_ms - t0) / FRAME_MS
        if rate != 1.0:
            x = stretch(x, rate)
            pos = pos * rate
        # targets
        K = self.K
        attrs = np.full((e - s, K, 5), -1, np.int64)
        m = (ons["beat"] >= s) & (ons["beat"] < e)
        ob = ons["beat"][m] - s
        k_idx = np.zeros(len(ob), np.int64)
        if len(ob):
            first = np.r_[True, ob[1:] != ob[:-1]]
            run_start = np.maximum.accumulate(np.where(first, np.arange(len(ob)), 0))
            k_idx = np.arange(len(ob)) - run_start
        keep = k_idx < K
        kind = ons["kind"][m].astype(np.int64)
        for j, key in enumerate(("kind", "dur", "spans", "nc", "stack")):
            vals = ons[key][m].astype(np.int64)
            if key == "dur":     # slider span class, or spinner length (beats) after the slider classes
                vals = np.where(kind == 1, vals, np.where(kind == 2, self.n_dur + vals, -1))
            elif key == "spans":  # repeat count only exists for sliders
                vals = np.where(kind == 1, np.clip(vals, 1, 32) - 1, -1)
            attrs[ob[keep], k_idx[keep], j] = vals[keep]
        # OOV beats were mapped to the nearest vocab pattern: drop attributes beyond its onset count.
        pop = np.array([bin(int(p)).count("1") for p in self.patterns[beats["pattern"][s:e]]])
        attrs[np.arange(K)[None, :] >= pop[:, None]] = -1
        bpm = 60000.0 / float(np.median(bl)) / rate
        return {
            "audio": x.astype(np.float32), "slot_pos": pos.astype(np.float32), "valid": valid,
            "beat_pos": np.arange(s, e, dtype=np.float32), "bim": beats["bim"][s:e].astype(np.int64),
            "pattern": beats["pattern"][s:e].astype(np.int64), "sv": beats["sv"][s:e].astype(np.int64),
            "attrs": attrs, "salience": self.salience(row_idx, s, e),
            "cond": cond_vector(row, self.meta["era_cuts_set_id"], bpm),
        }


HEADINGS = 32
AUDIO_OFFSETS_MS = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])


class PlacementData(torch.utils.data.Dataset):
    """Objects of one chart crop: conditioning features + flow-matching state target."""

    def __init__(self, root: Path = DATA_ROOT, split: str = "train", max_objects: int = 512, seed: int = 0,
                 augment: bool = True, teacher_p: float = 0.3):
        self.root, self.split, self.seed, self.augment = root, split, seed, augment
        self.max_objects, self.teacher_p = max_objects, teacher_p
        rows = [r for r in load_index(root, "placement") if r["split"] == split]
        self.rows = [r for r in rows if not r["synthetic"]]
        teacher = {r["parent"]: r for r in rows if r["synthetic"]}
        self.teacher = [teacher.get(r["chart_sha256"]) for r in self.rows]
        self.store = FieldStore(root / "cache" / "placement")
        self.audio = Audio(root)
        self.meta = io.read_json(root / "cache" / "meta.json")
        groups = collections.defaultdict(list)
        for i, r in enumerate(self.rows):
            groups[r["group"]].append(i)
        self.groups = [groups[g] for g in sorted(groups)]
        w = np.sqrt([len(g) for g in self.groups])
        self.group_p = w / w.sum()

    def __len__(self):
        return 1 << 40

    def __getitem__(self, i: int) -> dict[str, Any]:
        rng = sample_rng(self.seed, i, 1)
        g = self.groups[rng.choice(len(self.groups), p=self.group_p)]
        idx = g[rng.integers(len(g))]
        use_teacher = self.teacher[idx] is not None and rng.random() < self.teacher_p
        return self.make(idx, sample_rng(self.seed, i, 2), teacher=use_teacher)

    def make(self, idx: int, rng: np.random.Generator, teacher: bool = False, start: int | None = None):
        row = self.teacher[idx] if teacher else self.rows[idx]
        sh, span = row["shard"], row["objects"]
        f = lambda k: self.store.slice(sh, k, span)
        n_all = span[1]
        n = min(self.max_objects, n_all)
        s = int(rng.integers(0, n_all - n + 1)) if start is None else start
        sl = slice(s, s + n)
        o = {k: f(k)[sl] for k in ("time_ms", "kind", "beat", "tick", "bim", "beat_ms", "spans", "length", "dur_ms",
                                     "nc", "stack", "edge", "x", "y", "path_ok", "poly_ratio")}
        head = np.stack([o["x"] / 256.0 - 1.0, o["y"] / 192.0 - 1.0], -1).astype(np.float32)
        path = f("headings")[sl].astype(np.float32)                       # [n, 32, 2]
        kind = o["kind"].astype(np.int64)
        m_head = kind != 2
        m_path = (kind == 1) & (o["path_ok"] == 1)
        if teacher:  # teacher artifacts: objects pushed onto an edge, extended/truncated paths
            m_head &= o["edge"] == 0
            m_path &= (o["edge"] == 0) & (o["poly_ratio"] >= 0.9)
        if self.augment:
            if rng.random() < 0.5:
                head[:, 0] *= -1
                path[..., 0] *= -1
            if rng.random() < 0.5:
                head[:, 1] *= -1
                path[..., 1] *= -1
        t = o["time_ms"]
        bl = np.maximum(o["beat_ms"].astype(np.float64), 1.0)
        prev_gap = np.diff(t, prepend=t[0] - 1000.0)
        end = t + np.where(kind == 0, 0.0, o["dur_ms"])
        next_gap = np.r_[t[1:] - end[:-1], 1000.0]
        scal = np.stack([np.log1p(np.maximum(prev_gap, 0) / 100.0), np.log1p(np.maximum(next_gap, 0) / 100.0),
                         np.log2(np.maximum(prev_gap, 1) / bl + 1 / 48), np.log2(np.maximum(next_gap, 1) / bl + 1 / 48),
                         np.log1p(np.where(kind == 0, 0.0, o["dur_ms"]) / bl),
                         np.where(kind == 1, o["length"] / 100.0, 0.0)], -1).astype(np.float32)
        beat_pos = (np.maximum(o["beat"], 0) + o["tick"] / 48.0).astype(np.float32)
        x, t0 = self.audio.window(row["audio_sha256"], t[0] - 400.0, end[-1] + 400.0)
        if self.augment:
            x = augment_audio(x, rng)
        apos = ((t[:, None] + AUDIO_OFFSETS_MS[None]) - t0) / FRAME_MS
        return {"audio": x.astype(np.float32), "apos": apos.astype(np.float32), "kind": kind,
                "nc": o["nc"].astype(np.int64), "stack": o["stack"].astype(np.int64),
                "spans": np.clip(o["spans"], 0, 32).astype(np.int64), "tick": np.clip(o["tick"], 0, 47).astype(np.int64),
                "bim": np.clip(o["bim"], 0, 15).astype(np.int64), "scal": scal, "beat_pos": beat_pos,
                "length": np.where(kind == 1, o["length"], 0.0).astype(np.float32),
                "head": head, "path": path, "m_head": m_head, "m_path": m_path,
                "cond": cond_vector(row, self.meta["era_cuts_set_id"]) | {"source": 1 if teacher else 0},
                "time_ms": t.astype(np.float64)}


def collate_placement(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    b = len(items)
    n = max(len(it["kind"]) for it in items)
    T = max(len(it["audio"]) for it in items)
    out = {"audio": torch.zeros(b, T, items[0]["audio"].shape[1]), "apos": torch.zeros(b, n, 5),
           "scal": torch.zeros(b, n, items[0]["scal"].shape[1]), "beat_pos": torch.zeros(b, n),
           "length": torch.zeros(b, n), "head": torch.zeros(b, n, 2), "path": torch.zeros(b, n, HEADINGS, 2),
           "m_head": torch.zeros(b, n, dtype=torch.bool), "m_path": torch.zeros(b, n, dtype=torch.bool),
           "mask": torch.zeros(b, n, dtype=torch.bool)}
    for k in ("kind", "nc", "stack", "spans", "tick", "bim"):
        out[k] = torch.zeros(b, n, dtype=torch.long)
    for i, it in enumerate(items):
        m = len(it["kind"])
        out["audio"][i, :len(it["audio"])] = torch.from_numpy(it["audio"])
        for k in ("apos", "scal", "beat_pos", "length", "head", "path", "m_head", "m_path",
                  "kind", "nc", "stack", "spans", "tick", "bim"):
            out[k][i, :m] = torch.from_numpy(np.asarray(it[k]))
        out["mask"][i, :m] = True
    for key in ("stars", "ar", "od", "cs", "hp", "sm"):
        out[key] = torch.tensor([it["cond"][key] for it in items], dtype=torch.float32)
    out["era"] = torch.tensor([it["cond"]["era"] for it in items], dtype=torch.long)
    out["source"] = torch.tensor([it["cond"]["source"] for it in items], dtype=torch.long)
    return out


def collate_rhythm(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    b = len(items)
    L = max(len(it["pattern"]) for it in items)
    T = max(len(it["audio"]) for it in items)
    K = items[0]["attrs"].shape[1]
    out = {
        "audio": torch.zeros(b, T, items[0]["audio"].shape[1]), "slot_pos": torch.zeros(b, L, 24),
        "valid": torch.zeros(b, L, 24, dtype=torch.bool), "beat_pos": torch.zeros(b, L),
        "bim": torch.zeros(b, L, dtype=torch.long), "pattern": torch.full((b, L), -1, dtype=torch.long),
        "sv": torch.full((b, L), -1, dtype=torch.long), "attrs": torch.full((b, L, K, 5), -1, dtype=torch.long),
        "salience": torch.zeros(b, L, 24), "mask": torch.zeros(b, L, dtype=torch.bool),
    }
    for i, it in enumerate(items):
        n, t = len(it["pattern"]), len(it["audio"])
        out["audio"][i, :t] = torch.from_numpy(it["audio"])
        for k in ("slot_pos", "valid", "beat_pos", "bim", "pattern", "sv", "attrs", "salience"):
            out[k][i, :n] = torch.from_numpy(np.asarray(it[k]))
        out["mask"][i, :n] = True
    for key in ("stars", "ar", "od", "cs", "hp", "bpm", "sm"):
        out[key] = torch.tensor([it["cond"][key] for it in items], dtype=torch.float32)
    out["era"] = torch.tensor([it["cond"]["era"] for it in items], dtype=torch.long)
    return out
