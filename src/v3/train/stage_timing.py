"""[T] training stage: beat/downbeat/tempo supervision from consensus osu! red lines."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.v3.common import io
from src.v3.data.audio import FRAME_MS
from src.v3.data.grid import build_grid
from src.v3.data.osu import RedLine
from src.v3.infer.tempo import RedLineFit, drift_report, fit_red_lines
from src.v3.models.timing import TimingModel, tempo_bin
from src.v3.train.data import DATA_ROOT, Audio, augment_audio, load_index, stretch
from src.v3.train.loop import sample_rng

CROP = 6000           # 30 s of 5 ms frames
SIGMA = 1.5           # target width in frames


def songs_for(root: Path, split: str) -> list[dict]:
    rows = [r for r in load_index(root, "rhythm") if r["split"] == split and r.get("timing_ok")]
    by_song = collections.defaultdict(list)
    for r in rows:
        by_song[r["audio_sha256"]].append(r)
    songs = []
    for sha, rs in sorted(by_song.items()):
        sig = collections.Counter(r["red_signature"] for r in rs).most_common(1)[0][0]
        rs = [r for r in rs if r["red_signature"] == sig]
        corr = rs[0]["offset_correction_ms"]
        reds = [(t + corr, b, m) for t, b, m in json.loads(sig)]
        region = (min(r["first_ms"] for r in rs), max(r["last_ms"] for r in rs))
        spans = [((reds[i + 1][0] if i + 1 < len(reds) else region[1]) - reds[i][0], reds[i][1]) for i in range(len(reds))]
        songs.append({"audio": sha, "red_lines": reds, "region": region, "frames": rs[0]["frames"],
                      "downbeats_ok": any(r["format_version"] == 14 for r in rs),
                      "main_bpm": 60000.0 / max(spans)[1]})
    return songs


class TimingData(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, seed: int, augment: bool = True):
        self.songs, self.seed, self.augment = songs_for(root, split), seed, augment
        self.audio = Audio(root)

    def __len__(self):
        return 1 << 40

    def __getitem__(self, i):
        rng = sample_rng(self.seed, i, 1)
        return self.make(self.songs[rng.integers(len(self.songs))], sample_rng(self.seed, i, 2))

    @staticmethod
    def start_range(song) -> tuple[float, float]:
        """Crop starts: the mapped region +-5 s, ending at the audio (objects can run past it)."""
        lo = max(0.0, song["region"][0] - 5000)
        hi = min(song["region"][1] + 5000, song["frames"] * FRAME_MS)
        return lo, max(lo + 1, hi - CROP * FRAME_MS)

    def make(self, song, rng, start_ms=None):
        length_ms = CROP * FRAME_MS
        if start_ms is None:
            start_ms = float(rng.uniform(*self.start_range(song)))
        x, t0 = self.audio.window(song["audio"], start_ms, start_ms + length_ms)
        rate = float(rng.uniform(0.92, 1.08)) if self.augment and rng.random() < 0.5 else 1.0
        if self.augment:
            x = augment_audio(x, rng)
        reds = [RedLine(*r) for r in song["red_lines"]]
        grid = build_grid(reds, min(t0, reds[0].time_ms) - 1.0, t0 + length_ms + 2000)
        if rate != 1.0:
            x = stretch(x, rate)
        n = len(x) - len(x) % 4
        x = x[:n]
        frames = np.arange(n) * FRAME_MS / rate + t0          # original-time of each (stretched) frame
        beat = np.zeros(n, np.float32)
        down = np.zeros(n, np.float32)
        for s, bim in zip(grid.start_ms, grid.beat_in_measure):
            c = (s - t0) * rate / FRAME_MS
            if -5 < c < n + 5:
                a, b = max(0, int(c - 5)), min(n, int(c + 6))
                g = np.exp(-0.5 * ((np.arange(a, b) - c) / SIGMA) ** 2)
                beat[a:b] = np.maximum(beat[a:b], g)
                if bim == 0:
                    down[a:b] = np.maximum(down[a:b], g)
        region = song["region"]
        valid = (frames >= region[0] - 2000) & (frames <= region[1] + 2000)
        return {"audio": x.astype(np.float32), "beat": beat, "down": down, "valid": valid,
                "down_ok": song["downbeats_ok"], "tempo": tempo_bin(song["main_bpm"] / rate)}


def collate_timing(items):
    b = len(items)
    t = max(len(it["audio"]) for it in items)
    out = {"audio": torch.zeros(b, t, items[0]["audio"].shape[1]), "beat": torch.zeros(b, t), "down": torch.zeros(b, t),
           "valid": torch.zeros(b, t, dtype=torch.bool), "frame_mask": torch.zeros(b, t, dtype=torch.bool)}
    for i, it in enumerate(items):
        n = len(it["audio"])
        out["audio"][i, :n] = torch.from_numpy(it["audio"])
        out["beat"][i, :n] = torch.from_numpy(it["beat"])
        out["down"][i, :n] = torch.from_numpy(it["down"])
        out["valid"][i, :n] = torch.from_numpy(it["valid"])
        out["frame_mask"][i, :n] = True
    out["down_ok"] = torch.tensor([it["down_ok"] for it in items])
    out["tempo"] = torch.tensor([it["tempo"] for it in items], dtype=torch.long)
    return out


class TimingStage:
    def __init__(self, root: Path = DATA_ROOT, size: str = "base", seed: int = 0, val_songs: int = 24):
        self.root, self.size, self.seed, self.val_n = Path(root), size, seed, val_songs
        self.train = TimingData(self.root, "train", seed)

    def model_kwargs(self):
        return {"tiny": dict(d=96, depth=2, heads=4, width=64), "small": dict(d=256, depth=6, heads=4),
                "base": dict(d=384, depth=8, heads=6), "large": dict(d=512, depth=12, heads=8)}[self.size]

    def build_model(self):
        return TimingModel(**self.model_kwargs())

    def dataset(self):
        return self.train

    def collate(self, items):
        return collate_timing(items)

    def binding(self):
        return {"stage": "timing", "size": self.size, "model": self.model_kwargs(),
                "manifest": io.sha256_file(self.root / "manifest" / "charts.jsonl")}

    def loss(self, model, batch, step):
        logits, tempo = model(batch["audio"], batch["frame_mask"])
        logits = logits.float()
        v = batch["valid"].float()
        w_beat = (1 + 9 * batch["beat"]) * v
        l_beat = (F.binary_cross_entropy_with_logits(logits[..., 0], batch["beat"], reduction="none") * w_beat).sum() / w_beat.sum().clamp(min=1)
        vd = v * batch["down_ok"][:, None].float()
        w_down = (1 + 9 * batch["down"]) * vd
        l_down = (F.binary_cross_entropy_with_logits(logits[..., 1], batch["down"], reduction="none") * w_down).sum() / w_down.sum().clamp(min=1)
        l_tempo = F.cross_entropy(tempo.float(), batch["tempo"], label_smoothing=0.05)
        loss = l_beat + 0.5 * l_down + 0.2 * l_tempo
        return loss, {"loss": loss.detach(), "beat": l_beat.detach(), "down": l_down.detach(), "tempo": l_tempo.detach()}

    def validate(self, model, step):
        dev = next(model.parameters()).device
        if not hasattr(self, "_val"):
            songs = songs_for(self.root, "val")
            const = [s for s in songs if len(s["red_lines"]) == 1]
            self._val = const[:: max(1, len(const) // self.val_n)][: self.val_n]
        audio = Audio(self.root)
        passed, octave, med = 0, 0, []
        for s in self._val:
            x, t0 = audio.window(s["audio"], 0.0, s["frames"] * FRAME_MS)
            beat, down, tempo = model.activations(x, dev)
            lines = fit_red_lines(beat, down, tempo, s["region"])
            human = [RedLineFit(*r) for r in s["red_lines"]]
            rep = drift_report(lines, human, s["region"])
            good_ratio = rep["ok"] and abs(rep["ratio"] - 1) < 0.01
            octave += int(rep["ok"] and not good_ratio)
            if good_ratio:
                med.append(rep["median_ms"])
                passed += int(rep["max_ms"] <= 5.0 and rep["median_ms"] <= 2.0)
        n = max(1, len(self._val))
        return {"select": passed / n, "drift_pass_rate": passed / n, "tempo_ratio_errors": octave / n,
                "median_err_ms": float(np.median(med)) if med else float("nan")}
