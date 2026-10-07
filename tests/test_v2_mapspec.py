"""CPU-only tests for canonical v2 map parsing, validation, and compilation."""

from dataclasses import replace
import unittest

from src.v2.events import HitSample
from src.v2.mapspec import (
    MapSpec,
    compile_osu,
    mapspec_from_osu,
    validate_map_spec,
)


RICH_OSU = """osu file format v14

[General]
AudioFilename: song.mp3
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Soft
StackLeniency: 0.7
Mode: 0

[Metadata]
Title: Test Song
TitleUnicode: Test Song
Artist: Test Artist
ArtistUnicode: Test Artist
Creator: Test Creator
Version: Generated Draft
Source:
Tags: generated test
BeatmapID: 0
BeatmapSetID: -1

[Difficulty]
HPDrainRate: 6
CircleSize: 4
OverallDifficulty: 8
ApproachRate: 9
SliderMultiplier: 1.4
SliderTickRate: 1

[TimingPoints]
0,500,4,2,0,70,1,0
400,-50,4,2,0,70,0,1
1000,-100,4,2,0,70,0,0
2000,600,3,2,0,70,1,8

[HitObjects]
64,96,0,1,2,1:2:3:70:
128,192,500,2,4,B|256:192|320:240,1,140,0|2,0:0|1:2,1:2:0:60:
256,192,1000,8,8,1500,0:0:0:0:
512,384,2000,1,0,0:0:0:0:
"""


class MapSpecTests(unittest.TestCase):
    def test_parser_preserves_native_timing_meter_kiai_sv_and_hit_samples(self):
        spec = mapspec_from_osu(RICH_OSU)
        timing = spec.timing_map

        self.assertEqual(len(spec.timing_points), 4)
        self.assertEqual(timing.meter_at_time(2100.0), 3)
        self.assertTrue(timing.kiai_at_time(500.0))
        self.assertFalse(timing.kiai_at_time(1200.0))
        self.assertEqual(
            [(segment.start_ms, segment.enabled) for segment in timing.kiai_segments()],
            [(400.0, True), (1000.0, False)],
        )
        self.assertAlmostEqual(timing.slider_velocity_at_time(500.0), 2.0)

        circle, slider, spinner, _ = spec.events
        self.assertEqual(circle.hit_sample, HitSample(1, 2, 3, 70, ""))
        self.assertEqual(slider.slider_edge_sounds, (0, 2))
        self.assertEqual(slider.slider_edge_sets, ((0, 0), (1, 2)))
        self.assertEqual(slider.hit_sample, HitSample(1, 2, 0, 60, ""))
        self.assertIsNotNone(spinner.end_beat)
        self.assertTrue(validate_map_spec(spec).valid)

    def test_compiler_round_trip_preserves_gameplay_semantics(self):
        original = mapspec_from_osu(RICH_OSU)
        compiled = compile_osu(original)
        reparsed = mapspec_from_osu(compiled)

        self.assertIn("[Events]", compiled)
        self.assertIn("[TimingPoints]", compiled)
        self.assertIn("[HitObjects]", compiled)
        self.assertEqual(reparsed.general.audio_filename, "song.mp3")
        self.assertEqual(reparsed.difficulty.slider_multiplier, 1.4)
        self.assertEqual(reparsed.timing_points, original.timing_points)
        self.assertEqual(len(reparsed.events), len(original.events))
        self.assertEqual(reparsed.events[1].slider_edge_sets, ((0, 0), (1, 2)))
        self.assertTrue(validate_map_spec(reparsed).valid)

    def test_strict_validator_rejects_serialized_collision_and_slider_occupancy(self):
        original = mapspec_from_osu(RICH_OSU)
        collided = replace(
            original,
            events=(
                original.events[0],
                replace(original.events[1], beat_position=0.0001),
                *original.events[2:],
            ),
        )
        collision_report = validate_map_spec(collided)
        self.assertFalse(collision_report.valid)
        self.assertTrue(any("serialized timestamp" in error for error in collision_report.errors))

        overlapping = replace(
            original,
            events=(
                original.events[0],
                original.events[1],
                replace(original.events[2], beat_position=1.25, end_beat=2.0),
                *original.events[3:],
            ),
        )
        overlap_report = validate_map_spec(overlapping)
        self.assertFalse(overlap_report.valid)
        self.assertTrue(any("starts during slider" in error for error in overlap_report.errors))
        with self.assertRaises(ValueError):
            compile_osu(overlapping)

    def test_strict_validator_disallows_custom_sample_assets(self):
        original = mapspec_from_osu(RICH_OSU)
        unsupported = replace(
            original,
            events=(
                replace(original.events[0], hit_sample=HitSample(filename="custom.wav")),
                *original.events[1:],
            ),
        )
        report = validate_map_spec(unsupported)
        self.assertFalse(report.valid)
        self.assertTrue(any("custom hit sample" in error for error in report.errors))

    def test_strict_validator_rejects_bezier_length_beyond_control_path(self):
        original = mapspec_from_osu(RICH_OSU)
        impossible = replace(
            original,
            events=(
                original.events[0],
                replace(original.events[1], slider_length=1_000.0),
                *original.events[2:],
            ),
        )
        report = validate_map_spec(impossible)
        self.assertFalse(report.valid)
        self.assertTrue(any("control-path feasibility" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
