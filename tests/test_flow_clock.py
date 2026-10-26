import math
import unittest
from src.beatmap_flow.clock import fit_global_grid


class FlowClockTests(unittest.TestCase):
    def test_quantized_beats_do_not_accumulate_median_rounding_drift(self):
        period = 60000/137.3
        beats = [round((123+i*period)/20)*20 for i in range(210) if i not in (14, 15, 60)]
        beats += [123+8.5*period, 123+50.5*period]
        result = fit_global_grid(beats)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['bpm'], 137.3, delta=.01)
        self.assertAlmostEqual(result['offset_ms'], 123, delta=2)

    def test_variable_tempo_is_not_forced_to_one_grid(self):
        beats = [0.]
        for i in range(200): beats.append(beats[-1]+500-i*.4)
        self.assertFalse(fit_global_grid(beats)['accepted'])
        self.assertFalse(fit_global_grid([100]*30)['accepted'])
        with self.assertRaises(ValueError): fit_global_grid([float('nan')]*30)


if __name__ == '__main__': unittest.main()
