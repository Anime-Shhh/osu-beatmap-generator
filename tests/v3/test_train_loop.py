import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.v3.train.loop import Trainer, TrainConfig, lr_at, sample_rng


class _Data(torch.utils.data.Dataset):
    def __len__(self):
        return 1 << 30

    def __getitem__(self, i):
        r = sample_rng(7, i)
        x = r.normal(size=8).astype(np.float32)
        return torch.from_numpy(x), torch.tensor([x.sum() * 0.5], dtype=torch.float32)


class _Stage:
    def __init__(self, stop_at=None, run=None):
        self.stop_at, self.run = stop_at, run

    def build_model(self):
        return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Dropout(0.2), torch.nn.GELU(), torch.nn.Linear(32, 1))

    def dataset(self):
        return _Data()

    def collate(self, items):
        return {"x": torch.stack([a for a, _ in items]), "y": torch.stack([b for _, b in items])}

    def loss(self, model, batch, step):
        if self.stop_at is not None and step == self.stop_at:
            Path(self.run, "STOP").touch()
        loss = torch.nn.functional.mse_loss(model(batch["x"]), batch["y"])
        return loss, {"loss": loss.detach()}

    def validate(self, model, step):
        with torch.no_grad():
            x = torch.ones(4, 8)
            return {"select": -float(model(x).float().pow(2).mean())}

    def binding(self):
        return {"dummy": 1}


def _cfg(run):
    return TrainConfig(run_dir=run, max_steps=30, batch_size=4, accum=2, lr=1e-2, warmup=5, val_every=10,
                       log_every=5, num_workers=0, bf16=False, ckpt_minutes=1e9)


class TrainLoopTest(unittest.TestCase):
    def setUp(self):
        os.environ["V3_AUTO_REQUEUE"] = "0"

    def test_stop_and_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            Trainer(_Stage(), _cfg(a)).fit()
            straight = torch.load(Path(a) / "last.pt", weights_only=False)
            Trainer(_Stage(stop_at=13, run=b), _cfg(b)).fit()      # stops after step 14
            mid = torch.load(Path(b) / "last.pt", weights_only=False)
            self.assertLess(mid["step"], 30)
            self.assertEqual(json_state(b), "incomplete")
            Trainer(_Stage(), _cfg(b)).fit()                        # resumes to the end
            resumed = torch.load(Path(b) / "last.pt", weights_only=False)
            self.assertEqual(resumed["step"], 30)
            for k in straight["model"]:
                torch.testing.assert_close(straight["model"][k], resumed["model"][k], rtol=0, atol=0)
                torch.testing.assert_close(straight["ema"][k], resumed["ema"][k], rtol=0, atol=0)
            self.assertTrue((Path(b) / "best.pt").exists())
            self.assertEqual(json_state(b), "complete")

    def test_truncated_last_falls_back_to_prev(self):
        with tempfile.TemporaryDirectory() as a:
            Trainer(_Stage(stop_at=23, run=a), _cfg(a)).fit()
            (Path(a) / "last.pt").write_bytes(b"truncated")
            Trainer(_Stage(), _cfg(a)).fit()
            self.assertEqual(torch.load(Path(a) / "last.pt", weights_only=False)["step"], 30)

    def test_binding_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as a:
            Trainer(_Stage(stop_at=3, run=a), _cfg(a)).fit()
            cfg = _cfg(a)
            cfg.lr = 5e-3
            with self.assertRaises(RuntimeError):
                Trainer(_Stage(), cfg).fit()

    def test_budget_sets_max_steps_once(self):
        with tempfile.TemporaryDirectory() as a:
            cfg = _cfg(a)
            cfg.max_steps, cfg.warmup, cfg.val_every = 0, 60, 10 ** 6
            cfg.extra = {"budget_gpu_hours": 0.0004}
            Trainer(_Stage(), cfg).fit()
            plan = __import__("json").load(open(Path(a) / "plan.json"))
            self.assertGreater(plan["max_steps"], 60)
            self.assertEqual(torch.load(Path(a) / "last.pt", weights_only=False)["step"], plan["max_steps"])
            self.assertEqual(json_state(a), "complete")

    def test_lr_schedule(self):
        cfg = TrainConfig(run_dir="x", max_steps=100, batch_size=1, lr=1.0, warmup=10, decay_frac=0.2, min_lr_ratio=0.1)
        self.assertAlmostEqual(lr_at(0, cfg), 0.1)
        self.assertAlmostEqual(lr_at(50, cfg), 1.0)
        self.assertAlmostEqual(lr_at(100, cfg), 0.1)


def json_state(run):
    import json
    return json.load(open(Path(run) / "status.json"))["state"]


class TorchrunArgvTest(unittest.TestCase):
    """torchrun prefix-matches every "--opt" in its argv against its own options (argparse quirk),
    even those meant for the training script: "--run" once aborted all four production launches."""

    def test_every_trainer_option_passes_through_torchrun(self):
        from torch.distributed.run import get_args_parser
        from src.v3.train.run import parser
        tail = ["timing"]
        for action in parser()._actions:
            for opt in action.option_strings:
                if opt.startswith("--") and opt != "--help":
                    tail += [opt] if action.nargs == 0 else [opt, "1"]
        launch = ["--standalone", "--nproc_per_node=2", "-m", "src.v3.train.run"]
        args = get_args_parser().parse_args(launch + tail)
        self.assertEqual(args.training_script, "src.v3.train.run")
        self.assertEqual(args.training_script_args, tail)
        self.assertEqual(parser().parse_args(tail).run, "1")


class AutocastTest(unittest.TestCase):
    """bf16 autocast caches weight casts regardless of grad mode; a no_grad pass must not starve the
    grad-enabled pass that follows (killed the first G/G0 launch at step 0)."""

    def test_no_grad_pass_inside_region_keeps_weight_gradients(self):
        from src.v3.train.loop import train_autocast
        a, b = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        with train_autocast("cpu", True):
            h = a(torch.randn(4, 8))
            with torch.no_grad():
                b(h)
            b(h).float().sum().backward()
        self.assertIsNotNone(b.weight.grad)

    def test_placement_loss_reaches_every_parameter_under_bf16(self):
        from src.v3.models.placement import PlacementModel
        from src.v3.train.loop import train_autocast
        from src.v3.train.stage_placement import PlacementStage
        from tests.v3.test_models import _placement_batch
        model = PlacementModel(d=64, depth=2, heads=4, frontend_width=32, cond_dim=32)
        stage, batch = PlacementStage.__new__(PlacementStage), _placement_batch()
        for seed in range(6):   # self-conditioning is a per-step coin flip: cover both branches
            torch.manual_seed(seed)
            model.zero_grad(set_to_none=True)
            with train_autocast("cpu", True):
                loss, _ = stage.loss(model, batch, 0)
                loss.backward()
            gradless = [n for n, p in model.named_parameters() if p.grad is None]
            self.assertEqual(gradless, [], f"seed {seed}")


class CropBoundsTest(unittest.TestCase):
    def test_window_is_never_empty(self):
        from src.v3.data.audio import FRAME_MS
        from src.v3.train.data import window_bounds
        self.assertEqual(window_bounds(1000, 0.0, 10 * FRAME_MS), (0, 11))
        self.assertEqual(window_bounds(1000, 100 * FRAME_MS, 2000 * FRAME_MS), (100, 1000))
        for start in (999 * FRAME_MS, 1000 * FRAME_MS, 9e6):   # at / past the end of a 1000-frame song
            self.assertEqual(window_bounds(1000, start, start + 2000.0), (998, 1000))

    def test_rhythm_crops_stay_on_audible_beats(self):
        from src.v3.data.audio import FRAME_MS
        from src.v3.train.data import RhythmData
        start_ms = np.arange(0, 20000, 500.0)       # 40 beats; audio ends at 8 s
        self.assertEqual(RhythmData.usable_beats({"frames": int(8000 / FRAME_MS)}, start_ms), 16)
        self.assertEqual(RhythmData.usable_beats({"frames": 10 ** 6}, start_ms), 40)
        self.assertEqual(RhythmData.usable_beats({"frames": 0}, start_ms), 1)


if __name__ == "__main__":
    unittest.main()
