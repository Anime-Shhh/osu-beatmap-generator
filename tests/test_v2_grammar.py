"""CPU-only tests for the sparse meter-aware event grammar."""

from __future__ import annotations

import unittest

from src.v2.events import BeatEvent, EventKind
from src.v2.grammar import (
    BarEventCodec,
    GrammarState,
    MAX_DURATION_BEATS,
    RESIDUAL_TICKS_PER_BEAT,
    TICKS_PER_BEAT,
)


class SparseGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = BarEventCodec()

    def test_binary_and_triplet_fine_snaps_round_trip_without_grid_aliasing(self):
        events = (
            BeatEvent(EventKind.CIRCLE, 1.0 / 48.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0 / 32.0, 192.0, 192.0),
            BeatEvent(EventKind.CIRCLE, 1.0 / 3.0, 320.0, 96.0),
        )
        tokens = self.codec.encode(events, bar_start_beat=0.0, meter=4.0)
        decoded = self.codec.decode(tokens, bar_start_beat=0.0, meter=4.0)
        self.assertEqual([event.beat_position for event in decoded], [1.0 / 48.0, 1.0 / 32.0, 1.0 / 3.0])
        positions = self.codec.event_context_positions(tokens)
        self.assertEqual(len(positions), len(events))
        self.assertTrue(all(tokens[position] != self.codec.vocabulary.event_id for position in positions))
        self.assertEqual(TICKS_PER_BEAT, 96)

    def test_same_tick_events_are_rejected_by_the_grammar(self):
        events = (
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 0.0, 320.0, 192.0),
        )
        with self.assertRaises(ValueError):
            self.codec.encode(events, bar_start_beat=0.0, meter=4.0)

    def test_fractional_quarter_note_meter_and_long_spinner_are_explicit(self):
        spinner = BeatEvent(
            EventKind.SPINNER,
            0.25,
            256.0,
            192.0,
            end_beat=0.25 + 16.0,
        )
        tokens = self.codec.encode((spinner,), bar_start_beat=0.0, meter=1.5)  # 3/8 bar
        decoded = self.codec.decode(tokens, bar_start_beat=0.0, meter=1.5)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].kind, EventKind.SPINNER)
        self.assertAlmostEqual(decoded[0].duration_beats, 16.0)

    def test_duration_overflow_is_not_silently_clipped(self):
        spinner = BeatEvent(
            EventKind.SPINNER,
            0.0,
            256.0,
            192.0,
            end_beat=MAX_DURATION_BEATS + 1.0,
        )
        with self.assertRaises(ValueError):
            self.codec.encode((spinner,), bar_start_beat=0.0, meter=4.0)

    def test_bounded_microtiming_residual_round_trips_instead_of_silent_rounding(self):
        original = 1.0 / 3.0 + 3.0 / RESIDUAL_TICKS_PER_BEAT
        tokens = self.codec.encode(
            (BeatEvent(EventKind.CIRCLE, original, 128.0, 96.0),),
            bar_start_beat=0.0,
            meter=4.0,
        )
        decoded = self.codec.decode(tokens, bar_start_beat=0.0, meter=4.0)
        self.assertAlmostEqual(decoded[0].beat_position, original)
        self.assertIn(
            self.codec.vocabulary.id_for("residual:3"),
            tokens,
        )

    def test_rare_off_grid_timing_uses_the_1_96_base_without_large_rounding(self):
        original = 0.05
        tokens = self.codec.encode(
            (BeatEvent(EventKind.CIRCLE, original, 128.0, 96.0),),
            bar_start_beat=0.0,
            meter=4.0,
        )
        decoded = self.codec.decode(tokens, bar_start_beat=0.0, meter=4.0)
        self.assertAlmostEqual(
            decoded[0].beat_position,
            original,
            delta=0.5 / RESIDUAL_TICKS_PER_BEAT + 1e-9,
        )
        self.assertIn(self.codec.vocabulary.id_for("snap:96"), tokens)

    def test_meter_aware_generation_mask_never_offers_out_of_bar_offsets(self):
        grammar = self.codec.grammar
        vocabulary = self.codec.vocabulary
        state = grammar.transition(
            grammar.transition(GrammarState(), vocabulary.bos_id), vocabulary.event_id
        )
        allowed = grammar.allowed_ids_for_bar(state, meter=1.5)
        self.assertTrue(allowed)
        self.assertTrue(all(vocabulary.offset_tick(token) < 144 for token in allowed))


if __name__ == "__main__":
    unittest.main()
