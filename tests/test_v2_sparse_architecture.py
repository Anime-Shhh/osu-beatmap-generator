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

    def test_local_audio_memory_impulse_relocation_produces_differing_representations(self):
        from src.v2.sparse_model import TimestampedAudioEncoder, SparsePlannerConfig

        config = SparsePlannerConfig(
            rhythm_dim=4,
            semantic_dim=3,
            model_dim=32,
            heads=4,
            local_layers=2,
            dropout=0.0,
        )
        encoder = TimestampedAudioEncoder(config).eval()

        # Create two 8s audio sequences with an isolated rhythm impulse at 500ms vs 1000ms
        n_frames = 800  # 8 seconds at 10ms/frame
        times = torch.arange(n_frames, dtype=torch.float32) * 10.0  # ms

        rhythm_500 = torch.zeros(1, n_frames, 4)
        rhythm_500[0, 50, :] = 10.0  # frame 50 is 500ms

        rhythm_1000 = torch.zeros(1, n_frames, 4)
        rhythm_1000[0, 100, :] = 10.0  # frame 100 is 1000ms

        semantic = torch.zeros(1, 80, 3)
        sem_times = torch.arange(80, dtype=torch.float32) * 100.0

        bar_queries = torch.tensor([[2000.0, 6000.0]])

        _, local_500 = encoder(
            rhythm_500, times.unsqueeze(0), semantic, sem_times.unsqueeze(0), bar_queries, return_local_frames=True
        )
        _, local_1000 = encoder(
            rhythm_1000, times.unsqueeze(0), semantic, sem_times.unsqueeze(0), bar_queries, return_local_frames=True
        )

        diff = (local_500 - local_1000).abs().max().item()
        self.assertGreater(diff, 1e-3, "Local frame audio memory must distinguish 500ms vs 1000ms impulses")

    def test_event_lanes_construction(self):
        from src.v2.events import BeatEvent, EventKind, build_event_lanes
        from src.v2.timing import TimingMap, TimingSegment

        timing_map = TimingMap((TimingSegment(0.0, 500.0, 0.0, 4),), ())
        events = [
            BeatEvent(kind=EventKind.CIRCLE, beat_position=1.0, x=256.0, y=192.0),
            BeatEvent(
                kind=EventKind.SLIDER,
                beat_position=2.0,
                x=256.0,
                y=192.0,
                slider_length=140.0,
                slider_repeats=1,
            ),
            BeatEvent(kind=EventKind.SPINNER, beat_position=4.0, end_beat=6.0, x=256.0, y=192.0),
        ]
        frame_times = torch.arange(400, dtype=torch.float32) * 10.0
        lanes = build_event_lanes(events, timing_map, frame_times_ms=frame_times, slider_multiplier=1.4)
        tensor = lanes.as_tensor()
        self.assertEqual(tensor.shape, (400, 5))
        # Circle at beat 1.0 = 500ms = frame 50
        self.assertEqual(lanes.circle_impulses[50].item(), 1.0)
        # Slider at beat 2.0 = 1000ms = frame 100
        self.assertEqual(lanes.slider_onsets[100].item(), 1.0)
        # Slider length 140 / (1.4 * 100 * 1.0) * 500 = 500ms duration -> end at 1500ms = frame 150
        self.assertEqual(lanes.slider_tails[150].item(), 1.0)
        self.assertTrue((lanes.slider_spans[100:151] == 1.0).all())
        # Spinner at beat 4.0 = 2000ms (frame 200) to beat 6.0 = 3000ms (frame 300)
        self.assertTrue((lanes.spinner_spans[200:301] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
