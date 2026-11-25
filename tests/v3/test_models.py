import unittest

import numpy as np
import torch

from src.v3.data.audio import FRAME_MS
from src.v3.infer.tempo import RedLineFit, drift_report, fit_red_lines
from src.v3.models.placement import PlacementModel, project_playfield, sample as g_sample, to_pixels
from src.v3.models.rhythm import ATTRS, RhythmModel
from src.v3.models.timing import TEMPO_BINS, TimingModel, tempo_bin


def _activation(beats_ms, n, sigma=1.5, jitter=1.0, seed=0):
    rng = np.random.default_rng(seed)
    act = np.zeros(n, np.float32)
    idx = np.arange(n)
    for b in beats_ms:
        c = (b + rng.normal(0, jitter)) / FRAME_MS
        lo, hi = max(0, int(c) - 6), min(n, int(c) + 7)
        act[lo:hi] = np.maximum(act[lo:hi], 0.9 * np.exp(-0.5 * ((idx[lo:hi] - c) / sigma) ** 2))
    for s in rng.integers(0, n, 40):  # spurious weak peaks
        act[s] = max(act[s], 0.35)
    return act


class TempoTest(unittest.TestCase):
    def test_constant_tempo_exact(self):
        period, phase = 600.0, 123.0
        n = int(120000 / FRAME_MS)
        beats = phase + period * np.arange(0, 199)
        act = _activation(beats, n)
        down = _activation(beats[::4], n, seed=1)
        tempo = np.full(TEMPO_BINS, 1e-3)
        tempo[tempo_bin(100.0)] = 1.0
        lines = fit_red_lines(act, down, tempo / tempo.sum(), (1000.0, 110000.0))
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(60000 / lines[0].beat_ms, 100.0, places=6)
        rep = drift_report(lines, [RedLineFit(phase, period, 4)], (1000.0, 110000.0))
        self.assertLess(abs(rep["ratio"] - 1), 1e-3)
        self.assertLessEqual(rep["max_ms"], 2.0)
        off = (lines[0].time_ms - phase) % (4 * period)           # timing line sits on a downbeat
        self.assertLessEqual(min(off, 4 * period - off), 1.5)     # (offsets are whole milliseconds)

    def test_tempo_change_segments(self):
        n = int(70000 / FRAME_MS)
        a = 200.0 + 500.0 * np.arange(60)            # 120 BPM until 30.2 s
        b = a[-1] + 500.0 + 400.0 * np.arange(95)    # then 150 BPM
        act = _activation(np.r_[a, b], n)
        tempo = np.full(TEMPO_BINS, 1.0 / TEMPO_BINS)
        lines = fit_red_lines(act, act * 0, tempo, (500.0, 68000.0))
        self.assertGreaterEqual(len(lines), 2)
        human = [RedLineFit(200.0, 500.0, 4), RedLineFit(a[-1] + 500.0, 400.0, 4)]
        rep = drift_report(lines, human, (500.0, 68000.0))
        self.assertLessEqual(rep["median_ms"], 3.0)


def _rhythm_batch(b=2, l=40, K=4, T=3000):
    torch.manual_seed(0)
    valid = torch.ones(b, l, 24, dtype=torch.bool)
    valid[0, -1, 12:] = False  # a partial final beat
    return {"audio": torch.randn(b, T, 130), "slot_pos": torch.rand(b, l, 24) * (T - 1),
            "valid": valid, "beat_pos": torch.arange(l).float()[None].expand(b, -1).contiguous(),
            "bim": torch.randint(0, 4, (b, l)), "pattern": torch.randint(0, 6, (b, l)),
            "sv": torch.randint(-1, 3, (b, l)), "attrs": torch.randint(-1, 2, (b, l, K, 5)),
            "salience": torch.rand(b, l, 24), "mask": torch.ones(b, l, dtype=torch.bool),
            "stars": torch.tensor([3.0, 5.5]), "ar": torch.tensor([8.0, 9.0]), "od": torch.tensor([7.0, 8.0]),
            "cs": torch.tensor([4.0, 4.0]), "hp": torch.tensor([5.0, 6.0]), "bpm": torch.tensor([120.0, 180.0]),
            "sm": torch.tensor([1.4, 1.8]), "era": torch.tensor([3, 7])}


class RhythmModelTest(unittest.TestCase):
    def test_loss_and_sample(self):
        from src.v3.train.stage_rhythm import RhythmStage
        patterns = [1, 1 | (1 << 12), 1 | (1 << 6) | (1 << 12) | (1 << 18), 1 << 12, (1 << 12) | (1 << 21)]
        model = RhythmModel(patterns, K=4, n_dur=8, n_sv=3, d=64, depth=2, heads=4, frontend_width=32,
                            attr_dim=8, cond_dim=32)
        batch = _rhythm_batch()
        stage = RhythmStage.__new__(RhythmStage)
        loss, logs = RhythmStage.loss(stage, model, batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
        model.eval()
        state, _ = model.sample(batch, steps=6, cfg=1.5, generator=torch.Generator().manual_seed(0))
        pc = model.pattern_pop[state["pattern"]]
        kind = state["attrs"][..., 0]
        exists = torch.arange(4)[None, None] < pc[..., None]
        self.assertTrue((kind[exists] < 3).all())                      # real attribute ids where onsets exist
        self.assertTrue((kind[~exists] == model.na_id("kind")).all())  # NA elsewhere
        last_bits = model.pattern_bits[state["pattern"][0, -1]]
        self.assertFalse(last_bits[12:].any())                          # partial beat respected


def _placement_batch(b=2, n=24, T=4000):
    torch.manual_seed(1)
    kind = torch.randint(0, 2, (b, n))
    kind[:, 5] = 2
    return {"audio": torch.randn(b, T, 130), "apos": torch.rand(b, n, 5) * (T - 1), "kind": kind,
            "nc": (torch.rand(b, n) < 0.2).long(), "stack": torch.zeros(b, n, dtype=torch.long),
            "spans": torch.randint(1, 3, (b, n)), "tick": torch.randint(0, 48, (b, n)), "bim": torch.randint(0, 4, (b, n)),
            "scal": torch.randn(b, n, 6), "beat_pos": torch.arange(n).float()[None].expand(b, -1).contiguous(),
            "length": torch.where(kind == 1, torch.rand(b, n) * 250 + 40, torch.zeros(b, n)),
            "head": torch.rand(b, n, 2) * 2 - 1, "path": torch.nn.functional.normalize(torch.randn(b, n, 32, 2), dim=-1),
            "m_head": kind != 2, "m_path": kind == 1, "mask": torch.ones(b, n, dtype=torch.bool),
            "stars": torch.tensor([3.0, 6.0]), "ar": torch.tensor([8.0, 9.5]), "od": torch.tensor([7.0, 9.0]),
            "cs": torch.tensor([4.0, 4.2]), "hp": torch.tensor([5.0, 5.0]), "sm": torch.tensor([1.4, 1.9]),
            "era": torch.tensor([2, 8]), "source": torch.tensor([0, 1])}


class PlacementTest(unittest.TestCase):
    def test_loss_and_constrained_sample(self):
        from src.v3.train.stage_placement import PlacementStage
        model = PlacementModel(d=64, depth=2, heads=4, frontend_width=32, cond_dim=32)
        batch = _placement_batch()
        batch["stack"][:, 3] = 1
        loss, logs = PlacementStage.loss(PlacementStage.__new__(PlacementStage), model, batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.eval()
        out = g_sample(model, batch, steps=6, generator=torch.Generator().manual_seed(0))
        hp, pts = out["head_px"], out["path_px"]
        live = batch["kind"] != 2
        self.assertTrue(((hp[live] >= -1e-3) & (hp[live] <= torch.tensor([512.0, 384.0]) + 1e-3)).all())
        sl = batch["kind"] == 1
        self.assertTrue(((pts[sl] >= torch.tensor([-64.0, -56.0]) - 1e-3) & (pts[sl] <= torch.tensor([576.0, 424.0]) + 1e-3)).all())
        seg = (pts[sl][:, 1:] - pts[sl][:, :-1]).norm(dim=-1).sum(-1)
        torch.testing.assert_close(seg, batch["length"][sl], rtol=1e-4, atol=1e-2)
        self.assertTrue(torch.allclose(hp[:, 3], hp[:, 2], atol=1e-3) or (batch["kind"][:, 2] == 2).any())

    def test_projection_moves_groups(self):
        hp = torch.tensor([[[600.0, 100.0], [620.0, 120.0]]])
        pts = hp[:, :, None, :].expand(1, 2, 33, 2).clone()
        out_hp, _, disp = project_playfield(hp, pts, torch.zeros(1, 2, dtype=torch.long),
                                            torch.tensor([[1, 0]]), torch.ones(1, 2, dtype=torch.bool))
        self.assertLessEqual(float(out_hp[..., 0].max()), 512.0)
        self.assertAlmostEqual(float(out_hp[0, 1, 0] - out_hp[0, 0, 0]), 20.0, places=4)  # group shape kept
        self.assertGreater(float(disp.max()), 0.0)

    def test_projection_heads_in_playfield_bodies_may_overshoot(self):
        from src.v3.models.placement import out_of_bounds
        # a slider whose head is legal and whose body runs 30 px past the right edge: human style, untouched
        hp = torch.tensor([[[490.0, 100.0]]])
        pts = (hp[:, :, None, :] + torch.linspace(0, 52, 33)[None, None, :, None] * torch.tensor([1.0, 0.0])).clone()
        kind = torch.ones(1, 1, dtype=torch.long)
        out_hp, out_pts, disp = project_playfield(hp, pts, kind, torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(float(disp.max()), 0.0)
        self.assertFalse(bool(out_of_bounds(out_hp, out_pts, kind).any()))
        # the same body 100 px past the edge leaves the visible area: translated just enough, head kept legal
        pts2 = (hp[:, :, None, :] + torch.linspace(0, 122, 33)[None, None, :, None] * torch.tensor([1.0, 0.0])).clone()
        out_hp, out_pts, disp = project_playfield(hp, pts2, kind, torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(float(out_pts[..., 0].max()), 576.0, places=3)
        self.assertAlmostEqual(float(disp.max()), 36.0, places=3)
        self.assertFalse(bool(out_of_bounds(out_hp, out_pts, kind).any()))


class TimingModelTest(unittest.TestCase):
    def test_forward_and_activations(self):
        model = TimingModel(d=64, depth=2, heads=4, width=32)
        x = torch.randn(2, 400, 130)
        logits, tempo = model(x, torch.ones(2, 400, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (2, 400, 2))
        self.assertEqual(tuple(tempo.shape), (2, TEMPO_BINS))
        beat, down, tp = model.activations(np.random.randn(5000, 130).astype(np.float32), torch.device("cpu"),
                                           chunk_frames=2000, overlap=400)
        self.assertEqual(len(beat), 5000)
        self.assertAlmostEqual(float(tp.sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
