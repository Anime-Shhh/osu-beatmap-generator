import unittest

import torch

from src.representation import NUM_SIGNAL_CHANNELS, decode_signal_to_osu, encode_osu_content_to_signal


OSU_CONTENT = """osu file format v14

[General]
AudioFilename:test.mp3
Mode: 0

[Difficulty]
SliderMultiplier:1.4

[TimingPoints]
0,500,4,2,0,70,1,0

[HitObjects]
256,192,500,1,0,0:0:0:0:
128,192,1000,2,0,B|256:192|384:192,1,280
"""


class RepresentationTest(unittest.TestCase):
    def test_encode_shape(self):
        waveform = torch.zeros(1, 22050 * 3)
        signal = encode_osu_content_to_signal(OSU_CONTENT, waveform=waveform)
        self.assertEqual(signal.shape[0], NUM_SIGNAL_CHANNELS)
        self.assertGreater(signal.shape[1], 0)
        self.assertTrue(torch.isfinite(signal).all())

    def test_decode_returns_playfield_objects(self):
        waveform = torch.zeros(1, 22050 * 3)
        signal = encode_osu_content_to_signal(OSU_CONTENT, waveform=waveform)
        objects = decode_signal_to_osu(signal, bpm=120.0, offset_ms=0.0, star_rating=4.0)
        self.assertGreaterEqual(len(objects), 1)
        for obj in objects:
            self.assertGreaterEqual(obj["x"], 0.0)
            self.assertLessEqual(obj["x"], 512.0)
            self.assertGreaterEqual(obj["y"], 0.0)
            self.assertLessEqual(obj["y"], 384.0)


if __name__ == "__main__":
    unittest.main()
