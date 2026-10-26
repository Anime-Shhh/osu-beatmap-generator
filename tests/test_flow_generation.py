import unittest
import torch
from src.beatmap_flow.generate import native_plan, compile_plan, profile_for_request
from src.v2.timing import TimingPoint
from src.v2.mapspec import DifficultySettings, compile_osu, mapspec_from_osu, validate_map_spec


class FlowGenerationTests(unittest.TestCase):
    def test_native_duration_and_round_trip(self):
        events = [{'start_ms': 500., 'end_ms': 500., 'kind': 0},
                  {'start_ms': 1000., 'end_ms': 1500., 'kind': 1},
                  {'start_ms': 2000., 'end_ms': 3500., 'kind': 2}]
        attributes = torch.zeros(3, 46)
        attributes[:, 0] = 10.  # linear
        attributes[:, 4] = 10.  # one endpoint
        attributes[:, 36] = 10. # one span
        attributes[:, 44] = torch.log1p(torch.tensor(1.))
        attributes[:, 45] = 1.
        plan, points, constrained = native_plan(events, attributes, (TimingPoint(0., 400.),), DifficultySettings())
        geometry = torch.zeros(3, 33, 2)
        geometry[:, 0] = torch.tensor([.5, .5])
        geometry[1, 1] = torch.tensor([.5+100/512, .5])
        spec = compile_plan(plan, geometry, points, DifficultySettings(), audio_filename='audio.mp3', title='test', stars=4.)
        parsed = mapspec_from_osu(compile_osu(spec))
        self.assertTrue(validate_map_spec(parsed).valid)
        self.assertAlmostEqual(parsed.timing_map.beat_to_time(parsed.events[1].end_beat), 1500., places=3)
        self.assertEqual(parsed.events[2].x, 256.)
        self.assertEqual(constrained, 0)

    def test_topology_constraints_keep_perfect_two_controls_and_sv_legal(self):
        attributes = torch.zeros(1, 46)
        attributes[0, 2] = 10.
        attributes[0, 35] = 10.
        attributes[0, 44] = 10.
        plan, points, constrained = native_plan([{'start_ms': 100., 'end_ms': 200., 'kind': 1}],
            attributes, (TimingPoint(0., 500.),), DifficultySettings())
        self.assertEqual(int(plan['counts'][0]), 2)
        self.assertEqual(constrained, 1)
        self.assertLessEqual(points[-1].slider_velocity_multiplier, 10)

    def test_fractional_tempo_slider_velocity_boundary(self):
        events = [{'start_ms': 930., 'end_ms': 1230., 'kind': 1}]
        attributes = torch.zeros(1, 46)
        attributes[0, 44] = torch.log1p(torch.tensor(1.))
        plan, points, _ = native_plan(events, attributes,
            (TimingPoint(120., 393.44262295081967),), DifficultySettings())
        geometry = torch.zeros(1, 33, 2)
        geometry[0, 0] = torch.tensor([.5, .5])
        geometry[0, 1] = torch.tensor([.5+100/512, .5])
        spec = compile_plan(plan, geometry, points, DifficultySettings(),
                            audio_filename='audio.mp3', title='test', stars=4.)
        parsed = mapspec_from_osu(compile_osu(spec))
        self.assertTrue(validate_map_spec(parsed).valid)
        self.assertAlmostEqual(parsed.timing_map.beat_to_time(parsed.events[0].end_beat), 1230., places=3)

    def test_invalid_star_request(self):
        for value in (0, -1, float('nan'), 13):
            with self.assertRaises(ValueError): profile_for_request(value)

    def test_non_slider_unused_topology_matches_training(self):
        attributes = torch.zeros(1, 46)
        attributes[0, 3] = 10.
        plan, _, _ = native_plan([{'start_ms': 100., 'end_ms': 100., 'kind': 0}],
                                 attributes, (TimingPoint(0., 500.),), DifficultySettings())
        self.assertEqual(int(plan['curves'][0]), 0)
        self.assertEqual(int(plan['counts'][0]), 0)
        self.assertEqual(float(plan['lengths'][0]), 0.)


if __name__ == '__main__': unittest.main()
