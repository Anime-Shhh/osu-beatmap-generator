"""Small CPU-only contracts for v2 sparse boundary and geometry conditioning."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight discovery host
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SparseArchitectureTests(unittest.TestCase):
    def _planner(self):
        from src.v2.sparse_model import SparseEventPlanner, SparsePlannerConfig

        return SparseEventPlanner(
            SparsePlannerConfig(
                rhythm_dim=4,
                semantic_dim=3,
                model_dim=32,
                local_layers=1,
                global_layers=1,
                decoder_layers=1,
                heads=4,
                ff_dim=64,
                dropout=0.0,
                max_bars=8,
                max_decoder_tokens=32,
            )
        ).eval()

    def test_previous_bar_summary_accepts_mixed_empty_and_nonempty_rows(self):
        planner = self._planner()
        vocabulary = planner.vocabulary
        result = planner(
            rhythm_features=torch.randn(2, 12, 4),
            rhythm_times_ms=torch.arange(12, dtype=torch.float32).repeat(2, 1),
            semantic_features=torch.randn(2, 4, 3),
            semantic_times_ms=(torch.arange(4, dtype=torch.float32) * 3.0).repeat(2, 1),
            bar_mid_times_ms=torch.tensor([[2.0, 6.0], [2.0, 6.0]]),
            profile=torch.full((2, 6), 0.5),
            clock_features=torch.full((2, 2, 4), 0.5),
            decoder_tokens=torch.full((2, 1), vocabulary.bos_id, dtype=torch.long),
            target_bar_index=torch.tensor([0, 1]),
            previous_bar_tokens=torch.tensor(
                [[vocabulary.bos_id, vocabulary.eos_id], [vocabulary.bos_id, 0]], dtype=torch.long
            ),
            previous_bar_mask=torch.tensor([[True, True], [True, False]]),
        )
        self.assertEqual(tuple(result["logits"].shape[:2]), (2, 1))
        self.assertTrue(torch.isfinite(result["logits"]).all())

    def test_incremental_decoder_matches_full_teacher_forced_eval_decode(self):
        planner = self._planner()
        vocabulary = planner.vocabulary
        tokens = torch.tensor(
            [[vocabulary.bos_id, vocabulary.event_id, vocabulary.eos_id]], dtype=torch.long
        )
        memory = torch.randn(1, 2, 32)
        target_bar_index = torch.tensor([1])
        full = planner.decode_hidden(tokens, memory, target_bar_index=target_bar_index)
        cache = None
        steps = []
        for position in range(tokens.shape[1]):
            hidden, cache = planner.decode_step(
                tokens[:, position : position + 1],
                memory,
                position=position,
                target_bar_index=target_bar_index,
                cache=cache,
            )
            steps.append(hidden)
        incremental = torch.cat(steps, dim=1)
        torch.testing.assert_close(incremental, full, rtol=1e-5, atol=1e-5)

    def test_geometry_absolute_positions_do_not_alias_at_the_old_table_limit(self):
        from src.v2.events import EventKind
        from src.v2.geometry import ActiveEventGeometryDiffuser, GeometryDiffusionConfig

        diffuser = ActiveEventGeometryDiffuser(
            GeometryDiffusionConfig(
                context_dim=32,
                hidden_dim=32,
                heads=4,
                layers=1,
                dropout=0.0,
                max_events=2,
            )
        ).eval()
        output = diffuser(
            context=torch.randn(1, 3, 32),
            event_kinds=torch.tensor([[int(EventKind.CIRCLE)] * 3]),
            anchor_counts=torch.zeros((1, 3), dtype=torch.long),
            noisy_geometry_logits=torch.randn(1, 3, 9, 2),
            noise_level=torch.tensor([0.5]),
            event_positions=torch.tensor([[0, 2_048, 100_000]]),
        )
        self.assertEqual(tuple(output.shape), (1, 3, 9, 2))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
