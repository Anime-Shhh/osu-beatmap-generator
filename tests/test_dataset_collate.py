import unittest

import torch

from src.dataset import collate_fn
from src.tokenizer import Residuals


class DatasetCollateTest(unittest.TestCase):
    def test_collate_with_signal(self):
        batch = [
            {
                "mel": torch.ones(1, 129, 10),
                "tokens": [1, 2, 3],
                "residuals": [Residuals(), Residuals(), Residuals()],
                "difficulty_id": 2,
                "bpm": 120.0,
                "star_rating": 4.0,
                "signal": torch.ones(9, 12),
                "sample_type": "window",
                "frame_rate": 1000.0 / 6.0,
                "target_start_ms": 0.0,
                "beatmap_status": "ranked",
            },
            {
                "mel": torch.ones(1, 129, 8),
                "tokens": [1, 2],
                "residuals": [Residuals(), Residuals()],
                "difficulty_id": 3,
                "bpm": 180.0,
                "star_rating": 5.0,
                "signal": torch.ones(9, 10),
                "sample_type": "window",
                "frame_rate": 1000.0 / 6.0,
                "target_start_ms": 1500.0,
                "beatmap_status": "ranked",
            },
        ]
        collated = collate_fn(batch)
        self.assertEqual(tuple(collated["signal"].shape), (2, 9, 12))
        self.assertEqual(tuple(collated["signal_mask"].shape), (2, 12))
        self.assertEqual(tuple(collated["tokens"].shape), (2, 3))

    def test_collate_without_signal(self):
        batch = [
            {
                "mel": torch.ones(1, 129, 10),
                "tokens": [1, 2],
                "residuals": [Residuals(), Residuals()],
                "difficulty_id": 2,
                "bpm": 120.0,
                "star_rating": 4.0,
                "signal": None,
                "sample_type": "window",
                "frame_rate": 1000.0 / 6.0,
                "target_start_ms": 0.0,
                "beatmap_status": None,
            }
        ]
        collated = collate_fn(batch)
        self.assertIsNone(collated["signal"])
        self.assertIsNone(collated["signal_mask"])


if __name__ == "__main__":
    unittest.main()
