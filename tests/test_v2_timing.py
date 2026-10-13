from typing import Any
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.v2.events import (
    BeatEvent,
    EventKind,
    build_beat_grid_targets,
    events_from_osu,
    events_to_hitobjects,
    serialize_hitobjects,
    validate_events,
)
from src.v2.clock import MeterSegment, TimingHypothesis, jitter_timing_hypothesis
from src.v2.inference import (
    reproject_events_for_compiled_timing,
    timing_points_from_hypothesis,
)
from src.v2.mapspec import (
    DifficultySettings,
    GeneralSettings,
    MapMetadata,
    MapSpec,
    compile_osu,
    mapspec_from_osu,
)
from src.v2.preprocess import reproject_events_absolute
from src.v2.timing import (
    TimingPoint,
    parse_timing_map,
    parse_timing_points,
    timing_map_from_points,
)


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

    def test_f04_red_timing_point_resets_slider_velocity(self):
        # Fixture from F04: red 0/500, green 0/-50 (SV=2.0), red 1000/500.
        timing = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -50.0, uninherited=False),
            TimingPoint(1000.0, 500.0, uninherited=True),
        ])
        # Before 1000ms, SV is 2.0
        self.assertAlmostEqual(timing.slider_velocity_at_time(500.0), 2.0)
        # At and after 1000ms, red point resets SV to 1.0
        self.assertAlmostEqual(timing.slider_velocity_at_time(1500.0), 1.0)
        # 140px one-span slider at multiplier 1.4: duration should be 500ms (SV=1.0), not 250ms (SV=2.0)
        duration = timing.slider_duration_ms(1500.0, 140.0, 1, 1.4)
        self.assertAlmostEqual(duration, 500.0)

    def test_f04_same_timestamp_red_and_green_precedence(self):
        # Green point after red point at same time -> SV is 2.0
        timing1 = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -50.0, uninherited=False),
        ])
        self.assertAlmostEqual(timing1.slider_velocity_at_time(0.0), 2.0)

        # Red point after green point at same time -> green point still overrides difficulty point
        timing2 = timing_map_from_points([
            TimingPoint(0.0, -50.0, uninherited=False),
            TimingPoint(0.0, 500.0, uninherited=True),
        ])
        self.assertAlmostEqual(timing2.slider_velocity_at_time(0.0), 2.0)

        # Multiple green points at same time -> last green point wins
        timing3 = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -50.0, uninherited=False),   # SV 2.0
            TimingPoint(0.0, -200.0, uninherited=False),  # SV 0.5
        ])
        self.assertAlmostEqual(timing3.slider_velocity_at_time(0.0), 0.5)

    def test_f04_slider_velocity_clamping_and_non_negative_inherited(self):
        # SV clamped to [0.1, 10.0]
        timing_high = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -5.0, uninherited=False),  # SV 20 -> clamp 10.0
        ])
        self.assertAlmostEqual(timing_high.slider_velocity_at_time(100.0), 10.0)

        timing_low = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -2000.0, uninherited=False),  # SV 0.05 -> clamp 0.1
        ])
        self.assertAlmostEqual(timing_low.slider_velocity_at_time(100.0), 0.1)

        # Inherited point with non-negative beat length -> speedMultiplier = 1.0
        timing_pos = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(100.0, 500.0, uninherited=False),
        ])
        self.assertAlmostEqual(timing_pos.slider_velocity_at_time(200.0), 1.0)

    def test_f15_timing_points_from_hypothesis_meter_transitions(self):
        # Constant 500ms beats, meter 4 -> 3 at beat 8 (4000ms)
        hypothesis = TimingHypothesis(
            "meter-change",
            timing_map_from_points([TimingPoint(0.0, 500.0)]),
            meter_segments=(MeterSegment(0.0, 4), MeterSegment(8.0, 3)),
        )
        points = timing_points_from_hypothesis(hypothesis)
        red_points = [p for p in points if p.uninherited]
        self.assertEqual(len(red_points), 2)
        self.assertEqual(red_points[0].offset_ms, 0.0)
        self.assertEqual(red_points[0].meter, 4)
        self.assertEqual(red_points[1].offset_ms, 4000.0)
        self.assertEqual(red_points[1].meter, 3)

        compiled_timing = timing_map_from_points(points)
        self.assertEqual(compiled_timing.meter_at_time(1000.0), 4)
        self.assertEqual(compiled_timing.meter_at_time(4500.0), 3)

    def test_f15_meter_change_preserves_active_slider_velocity(self):
        # Meter change at 4000ms while active SV is 2.0 from 0ms
        timing_map = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -50.0, uninherited=False),  # SV 2.0
        ])
        hypothesis = TimingHypothesis(
            "meter-sv-interaction",
            timing_map,
            meter_segments=(MeterSegment(0.0, 4), MeterSegment(8.0, 3)),
        )
        points = timing_points_from_hypothesis(hypothesis)
        compiled_timing = timing_map_from_points(points)

        # Meter updated at 4500ms
        self.assertEqual(compiled_timing.meter_at_time(4500.0), 3)
        # Active SV preserved across inserted red point
        self.assertAlmostEqual(compiled_timing.slider_velocity_at_time(4500.0), 2.0)
        # Slider duration at 4500ms should be 250ms (SV=2.0)
        duration = compiled_timing.slider_duration_ms(4500.0, 140.0, 1, 1.4)
        self.assertAlmostEqual(duration, 250.0)

    def test_f15_non_quarter_meter_denominator_conversion_and_rejection(self):
        # 6/8 meter: 6 beats of eighth-notes (scale = 0.5)
        hypothesis_68 = TimingHypothesis(
            "meter-6-8",
            timing_map_from_points([TimingPoint(0.0, 500.0)]),
            meter_segments=(MeterSegment(0.0, 6, 8),),
        )
        points_68 = timing_points_from_hypothesis(hypothesis_68)
        self.assertEqual(len(points_68), 1)
        self.assertEqual(points_68[0].meter, 6)
        self.assertAlmostEqual(points_68[0].beat_length_ms, 250.0)

        # 2/2 meter: 2 beats of half-notes (scale = 2.0)
        hypothesis_22 = TimingHypothesis(
            "meter-2-2",
            timing_map_from_points([TimingPoint(0.0, 500.0)]),
            meter_segments=(MeterSegment(0.0, 2, 2),),
        )
        points_22 = timing_points_from_hypothesis(hypothesis_22)
        self.assertEqual(len(points_22), 1)
        self.assertEqual(points_22[0].meter, 2)
        self.assertAlmostEqual(points_22[0].beat_length_ms, 1000.0)

    def test_f15r_six_eight_preserves_circle_and_repeated_slider_absolute_times(self):
        source_timing = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(0.0, -50.0, uninherited=False),
        ])
        hypothesis = TimingHypothesis(
            "meter-6-8-events",
            source_timing,
            meter_segments=(MeterSegment(0.0, 6, 8),),
        )
        difficulty = DifficultySettings(slider_multiplier=1.4)
        source_events = (
            BeatEvent(EventKind.CIRCLE, 1.0, 64.0, 96.0, source_object_index=0),
            BeatEvent(
                EventKind.SLIDER,
                2.0,
                128.0,
                192.0,
                end_beat=4.0,
                control_points=((256.0, 192.0),),
                slider_repeats=2,
                slider_length=280.0,
                source_object_index=1,
            ),
        )
        points = timing_points_from_hypothesis(hypothesis)
        projected = reproject_events_for_compiled_timing(
            source_events, hypothesis, difficulty, points
        )
        compiled = compile_osu(
            MapSpec(
                format_version=14,
                general=GeneralSettings("song.mp3"),
                metadata=MapMetadata("Song", "Artist", "Creator", "6/8"),
                difficulty=difficulty,
                timing_points=points,
                events=projected,
            ),
            validate=False,
        )
        parsed = mapspec_from_osu(compiled)
        self.assertAlmostEqual(parsed.timing_map.beat_to_time(parsed.events[0].beat_position), 500.0)
        slider = parsed.events[1]
        self.assertAlmostEqual(parsed.timing_map.beat_to_time(slider.beat_position), 1000.0)
        assert slider.end_beat is not None
        self.assertAlmostEqual(parsed.timing_map.beat_to_time(slider.end_beat), 2000.0)
        self.assertEqual(slider.slider_repeats, 2)
        self.assertAlmostEqual(parsed.timing_map.slider_velocity_at_time(1000.0), 2.0)

    def test_clock_jitter_reprojection_round_trips_absolute_milliseconds(self):
        source = TimingHypothesis(
            "source",
            timing_map_from_points([TimingPoint(0.0, 500.0)]),
            is_oracle=True,
        )
        jittered = jitter_timing_hypothesis(
            source, offset_ms=15.0, time_scale=1.01, confidence=0.7
        )
        events = (
            BeatEvent(EventKind.CIRCLE, 1.0, 64.0, 96.0, source_object_index=0),
            BeatEvent(
                EventKind.SLIDER,
                2.0,
                128.0,
                192.0,
                end_beat=3.0,
                control_points=((256.0, 192.0),),
                slider_length=140.0,
                source_object_index=1,
            ),
        )
        projected = reproject_events_absolute(
            events,
            source_timing=source.timing_map,
            target_timing=jittered.timing_map,
        )
        self.assertAlmostEqual(jittered.timing_map.beat_to_time(projected[0].beat_position), 500.0)
        self.assertAlmostEqual(jittered.timing_map.beat_to_time(projected[1].beat_position), 1000.0)
        assert projected[1].end_beat is not None
        self.assertAlmostEqual(jittered.timing_map.beat_to_time(projected[1].end_beat), 1500.0)

    def test_f15_combined_bpm_meter_sv_transitions(self):
        timing_map = timing_map_from_points([
            TimingPoint(0.0, 500.0, uninherited=True),
            TimingPoint(2000.0, 400.0, uninherited=True),
            TimingPoint(2000.0, -66.666667, uninherited=False),  # SV 1.5
        ])
        hypothesis = TimingHypothesis(
            "combined-change",
            timing_map,
            meter_segments=(MeterSegment(0.0, 4), MeterSegment(4.0, 3)),  # 4 quarter beats = 2000ms
        )
        points = timing_points_from_hypothesis(hypothesis)
        compiled_timing = timing_map_from_points(points)

        self.assertAlmostEqual(compiled_timing.beat_length_at_time(2500.0), 400.0)
        self.assertEqual(compiled_timing.meter_at_time(2500.0), 3)
        self.assertAlmostEqual(compiled_timing.slider_velocity_at_time(2500.0), 1.5, places=4)

    def test_differential_against_upstream_csharp_decoder_semantics(self):
        # Reference implementation of upstream-LegacyBeatmapDecoder.cs and upstream-Slider.cs
        def reference_legacy_decoder(points: list[TimingPoint]) -> dict[str, Any]:
            # Simulate handleTimingPoint + addControlPoint + flushPendingPoints
            # Points grouped by timestamp:
            unique_times = sorted(set(p.offset_ms for p in points))
            timing_control_points: list[tuple[float, float, int]] = []  # time, beatLength, meter
            difficulty_control_points: list[tuple[float, float]] = []  # time, speedMultiplier

            for t in unique_times:
                points_at_t = [p for p in points if p.offset_ms == t]
                # In C#, timingChange points are Insert(0, p), non-timingChange are Add(p)
                # When flushed backwards, last non-timingChange point is seen first.
                # If no non-timingChange point, last timingChange point is seen.
                last_timing_point = None
                last_diff_from_green = None
                last_diff_from_red = None

                for p in points_at_t:
                    beat_length = p.beat_length_ms
                    speed_mult = 100.0 / -beat_length if beat_length < 0 else 1.0
                    speed_mult = max(0.1, min(10.0, speed_mult))
                    if p.uninherited:
                        last_timing_point = (t, beat_length, p.meter)
                        last_diff_from_red = (t, 1.0)
                    else:
                        last_diff_from_green = (t, speed_mult)

                if last_timing_point is not None:
                    timing_control_points.append(last_timing_point)

                if last_diff_from_green is not None:
                    difficulty_control_points.append(last_diff_from_green)
                elif last_diff_from_red is not None:
                    difficulty_control_points.append(last_diff_from_red)

            def get_timing_point_at(time: float):
                active = timing_control_points[0]
                for cp in timing_control_points:
                    if cp[0] <= time:
                        active = cp
                    else:
                        break
                return active

            def get_diff_point_at(time: float):
                active = (0.0, 1.0)
                for cp in difficulty_control_points:
                    if cp[0] <= time:
                        active = cp
                    else:
                        break
                return active

            def slider_duration(time: float, length: float, repeats: int, mult: float):
                t_point = get_timing_point_at(time)
                d_point = get_diff_point_at(time)
                beat_len = t_point[1]
                sv = d_point[1]
                velocity = (100.0 * mult) / (beat_len / sv)
                return (repeats * length) / velocity

            return {
                "get_timing_point_at": get_timing_point_at,
                "get_diff_point_at": get_diff_point_at,
                "slider_duration": slider_duration,
            }

        fixtures = [
            # Fixture 1: F04 baseline
            [
                TimingPoint(0.0, 500.0, uninherited=True),
                TimingPoint(0.0, -50.0, uninherited=False),
                TimingPoint(1000.0, 500.0, uninherited=True),
            ],
            # Fixture 2: Green before red at same time
            [
                TimingPoint(0.0, -50.0, uninherited=False),
                TimingPoint(0.0, 500.0, uninherited=True),
                TimingPoint(500.0, -100.0, uninherited=False),
                TimingPoint(1000.0, 400.0, uninherited=True),
            ],
            # Fixture 3: Cascading SV changes with red resets
            [
                TimingPoint(0.0, 500.0, uninherited=True),
                TimingPoint(200.0, -25.0, uninherited=False),   # SV 4.0
                TimingPoint(400.0, -50.0, uninherited=False),   # SV 2.0
                TimingPoint(600.0, 300.0, uninherited=True),    # Red resets SV to 1.0
                TimingPoint(800.0, -200.0, uninherited=False),  # SV 0.5
                TimingPoint(1000.0, 500.0, uninherited=True),   # Red resets SV to 1.0
            ],
            # Fixture 4: Clamped SV extremes
            [
                TimingPoint(0.0, 500.0, uninherited=True),
                TimingPoint(0.0, -2.0, uninherited=False),     # SV 50 -> clamp 10.0
                TimingPoint(500.0, 400.0, uninherited=True),   # Reset SV 1.0
                TimingPoint(1000.0, -5000.0, uninherited=False), # SV 0.02 -> clamp 0.1
            ],
        ]

        query_times = [0.0, 100.0, 200.0, 350.0, 500.0, 600.0, 750.0, 800.0, 1000.0, 1500.0]
        for fixture_idx, fixture in enumerate(fixtures):
            ref = reference_legacy_decoder(fixture)
            timing_map = timing_map_from_points(fixture)

            for q_time in query_times:
                ref_diff = ref["get_diff_point_at"](q_time)
                repo_sv = timing_map.slider_velocity_at_time(q_time)
                self.assertAlmostEqual(
                    repo_sv,
                    ref_diff[1],
                    places=6,
                    msg=f"Fixture {fixture_idx} at {q_time}ms: SV mismatch",
                )

                for length in [100.0, 140.0, 250.0]:
                    for repeats in [1, 2, 3]:
                        for sm in [1.0, 1.4, 2.0]:
                            ref_dur = ref["slider_duration"](q_time, length, repeats, sm)
                            repo_dur = timing_map.slider_duration_ms(q_time, length, repeats, sm)
                            self.assertAlmostEqual(
                                repo_dur,
                                ref_dur,
                                places=6,
                                msg=f"Fixture {fixture_idx} at {q_time}ms: duration mismatch",
                            )



if __name__ == "__main__":
    unittest.main()
