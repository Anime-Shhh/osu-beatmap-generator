import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.v2.events import (
    EventKind,
    build_beat_grid_targets,
    events_from_osu,
    events_to_hitobjects,
    serialize_hitobjects,
    validate_events,
)
from src.v2.timing import parse_timing_map


OSU_CONTENT = """osu file format v14

[TimingPoints]
0,500,4,2,0,70,1,0
1000,400,4,2,0,70,1,0
1200,-50,4,2,0,70,0,0

[HitObjects]
64,96,500,1,2,0:0:0:0:
128,192,1200,2,4,B|256:192|320:240,2,180
256,192,1600,8,8,2200
"""


class TimingAndEventTests(unittest.TestCase):
    def test_piecewise_timing_map_round_trips(self):
        timing = parse_timing_map(OSU_CONTENT)
        self.assertEqual(len(timing.segments), 2)
        self.assertAlmostEqual(timing.time_to_beat(1000.0), 2.0)
        self.assertAlmostEqual(timing.time_to_beat(1400.0), 3.0)
        self.assertAlmostEqual(timing.beat_to_time(3.0), 1400.0)
        self.assertAlmostEqual(timing.beat_length_at_time(1300.0), 400.0)
        self.assertAlmostEqual(timing.slider_velocity_at_time(1300.0), 2.0)

    def test_events_preserve_slider_and_spinner_semantics(self):
        timing = parse_timing_map(OSU_CONTENT)
        events = events_from_osu(OSU_CONTENT, timing)
        self.assertEqual([event.kind for event in events], [
            EventKind.CIRCLE,
            EventKind.SLIDER,
            EventKind.SPINNER,
        ])
        slider = events[1]
        self.assertEqual(slider.slider_repeats, 2)
        self.assertEqual(slider.slider_length, 180.0)
        self.assertEqual(slider.control_points, ((256.0, 192.0), (320.0, 240.0)))
        self.assertIsNotNone(slider.end_beat)
        self.assertIsNotNone(events[2].end_beat)
        self.assertTrue(validate_events(events, timing).valid)

        reconstructed = events_to_hitobjects(events, timing)
        self.assertAlmostEqual(reconstructed[1]["time"], 1200.0)
        self.assertAlmostEqual(reconstructed[2]["end_time"], 2200.0)
        slider_line = serialize_hitobjects(events, timing)[1]
        self.assertIn(",2,4,B|256:192|320:240,2,180.000000", slider_line)

    def test_grid_targets_use_multiple_lanes_without_overwriting(self):
        timing = parse_timing_map(OSU_CONTENT)
        events = events_from_osu(OSU_CONTENT, timing)
        if torch is None:
            self.skipTest("PyTorch is not installed in this lightweight test environment")
        targets = build_beat_grid_targets(events, subdivisions_per_beat=16, max_events_per_slot=2)
        self.assertEqual(targets.occupancy.ndim, 2)
        self.assertEqual(int(targets.occupancy.sum().item()), len(events))

        duplicate = list(events) + [events[0]]
        with self.assertRaises(ValueError):
            build_beat_grid_targets(duplicate, subdivisions_per_beat=16, max_events_per_slot=1)


if __name__ == "__main__":
    unittest.main()
