import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.v3.common import io
from src.v3.data import audio as A
from src.v3.data.charts import HEADINGS, convert_chart, headings_from_path
from src.v3.data.grid import SLOT_TICKS, TICKS, build_grid, locate, slot_time
from src.v3.data.osu import RedLine, parse_chart


def osu_text(timing: str, objects: str, version: int = 14, sm: float = 1.4) -> str:
    return (f"osu file format v{version}\n\n[General]\nMode: 0\nStackLeniency: 0.7\n\n"
            f"[Metadata]\nTitle:t\nArtist:a\nCreator:c\nVersion:v\n\n"
            f"[Difficulty]\nHPDrainRate:5\nCircleSize:4\nOverallDifficulty:8\nApproachRate:9\n"
            f"SliderMultiplier:{sm}\nSliderTickRate:1\n\n[TimingPoints]\n{timing}\n\n[HitObjects]\n{objects}\n")


class GridTest(unittest.TestCase):
    def test_slots_are_union_of_16th_and_12th(self):
        self.assertEqual(len(SLOT_TICKS), 24)
        for t in SLOT_TICKS:
            self.assertTrue(t % 3 == 0 or t % 4 == 0)

    def test_reset_and_partial_beat(self):
        # 120 BPM from 0; a new red line 750 ms in (mid-beat at 1.5 beats).
        reds = [RedLine(0.0, 500.0, 4), RedLine(750.0, 400.0, 3)]
        grid = build_grid(reds, 0.0, 2000.0)
        np.testing.assert_allclose(grid.start_ms[:4], [0, 500, 750, 1150])
        self.assertAlmostEqual(grid.frac[1], 0.5)
        self.assertEqual(grid.beat_in_measure[2], 0)  # counter reset at the red line
        self.assertEqual(int(grid.meter[2]), 3)
        loc = locate(grid, np.array([750.0, 950.0, 625.0]))
        self.assertEqual(list(loc.beat), [2, 2, 1])
        self.assertEqual(list(loc.tick), [0, 24, 12])
        self.assertTrue((loc.slot >= 0).all())

    def test_slot_outside_partial_beat_rolls_forward(self):
        reds = [RedLine(0.0, 500.0, 4), RedLine(750.0, 400.0, 4)]
        grid = build_grid(reds, 0.0, 2000.0)
        loc = locate(grid, np.array([749.5]))  # nearest valid candidate is next beat start
        self.assertEqual(int(loc.beat[0]), 2)
        self.assertEqual(int(loc.tick[0]), 0)

    def test_off_grid_and_round_trip(self):
        grid = build_grid([RedLine(100.0, 333.333, 4)], 0.0, 5000.0)
        on = slot_time(grid, np.array([3, 5]), np.array([5, 17]))
        loc = locate(grid, np.concatenate([on, [on[0] + 7.0]]))
        np.testing.assert_allclose(slot_time(grid, loc.beat[:2], loc.slot[:2]), on, atol=1e-6)
        self.assertEqual(int(loc.slot[2]), -1)  # 7 ms away from any 1/16 or 1/12 tick at 180 BPM

    def test_extends_before_first_red_line(self):
        grid = build_grid([RedLine(1000.0, 500.0, 4)], 0.0, 3000.0)
        self.assertLessEqual(grid.start_ms[0], 0.0)
        self.assertEqual(int(grid.beat_in_measure[list(grid.start_ms).index(1000.0)]), 0)


class ParserTest(unittest.TestCase):
    def test_legacy_offset_and_sv(self):
        text = osu_text("0,500,4,1,0,100,1,0\n1000,-50,4,1,0,100,0,0",
                        "256,192,1000,2,0,L|356:192,1,140", version=4)
        chart = parse_chart(text)
        self.assertEqual(chart.red_lines[0].time_ms, 24.0)
        slider = chart.objects[0]
        self.assertEqual(slider.time_ms, 1024.0)
        self.assertAlmostEqual(slider.sv, 2.0)
        # 140 px / (1.4 * 100 * 2.0) beats * 500 ms
        self.assertAlmostEqual(slider.end_ms - slider.time_ms, 250.0)

    def test_red_resets_sv_and_same_time_green_wins(self):
        text = osu_text("0,500,4,1,0,100,1,0\n0,-200,4,1,0,100,0,0\n2000,500,4,1,0,100,1,0",
                        "0,0,100,2,0,L|100:0,1,70\n0,0,2100,2,0,L|100:0,1,70")
        chart = parse_chart(text)
        self.assertAlmostEqual(chart.objects[0].sv, 0.5)
        self.assertAlmostEqual(chart.objects[1].sv, 1.0)


class ChartTest(unittest.TestCase):
    def test_convert_snaps_and_measures_paths(self):
        text = osu_text("0,500,4,1,0,100,1,0",
                        "100,100,0,1,0\n100,100,125,1,0\n200,200,250,6,0,P|250:150|300:200,2,157\n"
                        "256,192,1500,12,0,2500")
        meta, cols = convert_chart(text, descriptors=False)
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["off_grid_frac"], 0.0)
        self.assertEqual(int(cols["stack_head"][1]), 1)
        self.assertEqual(int(cols["spans"][2]), 2)
        self.assertLess(float(cols["path_err"][2]), 1.0)
        np.testing.assert_allclose(np.linalg.norm(cols["headings"][2], axis=1), 1.0, atol=1e-5)
        self.assertEqual(int(cols["end_tick"][2]) % 3, 0)

    def test_heading_length_is_exact(self):
        path = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
        h, err, total = headings_from_path(path, 200.0)
        recon_end = path[0] + (h * (200.0 / HEADINGS)).sum(0)
        self.assertAlmostEqual(total, 200.0)
        self.assertLess(np.linalg.norm(recon_end - path[-1]), 15.0)  # corner is cut, not lost
        self.assertLess(err, 10.0)

    def test_reject_non_standard_mode(self):
        meta, cols = convert_chart(osu_text("0,500,4,1,0,100,1,0", "1,1,0,1,0").replace("Mode: 0", "Mode: 3"),
                                   descriptors=False)
        self.assertEqual(meta["status"], "rejected")
        self.assertIsNone(cols)


class DownstreamTest(unittest.TestCase):
    def test_beat_patterns(self):
        from src.v3.data.manifest import beat_patterns
        masks = beat_patterns(np.array([0, 0, 2, 2, 2, -1]), np.array([0, 12, 0, 6, -1, 3]))
        self.assertEqual(list(masks), [(1 << 0) | (1 << 12), (1 << 0) | (1 << 6)])

    def test_reliable_red_lines(self):
        from src.v3.data.manifest import reliable_red_lines
        self.assertTrue(reliable_red_lines([[0, 500, 4]]))
        self.assertTrue(reliable_red_lines([[0, 500, 4], [6000, 400, 4]]))
        self.assertFalse(reliable_red_lines([[0, 500, 4], [1000, 400, 4]]))

    def test_offset_lag_recovery(self):
        import torch
        from src.v3.data.offset_audit import chart_lags
        rng = np.random.default_rng(0)
        times = np.sort(rng.uniform(1000, 60000, 300))
        env = np.zeros(int(70000 / A.FRAME_MS))
        true_lag = 23.0
        idx = np.rint((times + true_lag) / A.FRAME_MS).astype(int)
        env[idx] = 10.0
        lag, sharp = chart_lags(torch.tensor(env, dtype=torch.float32), times, device="cpu")
        self.assertLess(abs(lag - true_lag), A.FRAME_MS / 2 + 1)
        self.assertGreater(sharp, 4.0)


class IoTest(unittest.TestCase):
    def test_done_marker_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.npy"
            io.write_npy_atomic(f, np.arange(5))
            io.mark_done(Path(tmp) / "done.json", [f], binding={"x": 1})
            self.assertTrue(io.is_done(Path(tmp) / "done.json", expect={"binding": {"x": 1}}))
            self.assertFalse(io.is_done(Path(tmp) / "done.json", expect={"binding": {"x": 2}}))
            f.write_bytes(b"corrupt")
            self.assertFalse(io.is_done(Path(tmp) / "done.json"))

    def test_sniff_containers(self):
        self.assertEqual(A.sniff(b"OggS" + b"\0" * 30 + b"OpusHead").container, "opus")
        self.assertEqual(A.sniff(b"OggS" + b"\0" * 60).container, "ogg")
        self.assertEqual(A.sniff(b"\0\0\0" + b"\xff\xfb" + b"\0" * 40).leading_zero_bytes, 3)
        self.assertEqual(A.sniff(b"ID3" + b"\0" * 40).container, "mp3")


if __name__ == "__main__":
    unittest.main()
