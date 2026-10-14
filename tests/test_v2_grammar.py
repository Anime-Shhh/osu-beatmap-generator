"""CPU-only tests for the sparse meter-aware event grammar."""

from __future__ import annotations

import random
import unittest

from src.v2.events import BeatEvent, EventKind
from src.v2.grammar import (
    BarEventCodec,
    DecodePhase,
    GrammarState,
    MAX_DURATION_BEATS,
    RESIDUAL_TICKS_PER_BEAT,
    SNAP_DENOMINATORS,
    TICKS_PER_BEAT,
    TokenFamily,
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

    def test_generation_gap_mask_has_viable_residuals_and_covers_bar_boundary(self):
        grammar, vocab = self.codec.grammar, self.codec.vocabulary
        for last_units in (8, -8):
            state = GrammarState(DecodePhase.EVENT_OFFSET, last_time_units=last_units)
            allowed = grammar.allowed_ids_for_bar(state, meter=4, minimum_start_gap_beats=.025)
            self.assertNotIn(vocab.id_for('offset:0'), allowed)
            for token in allowed:
                tick = vocab.offset_tick(token)
                residual_state = GrammarState(DecodePhase.EVENT_RESIDUAL,
                                              last_time_units=last_units, current_offset_tick=tick)
                residuals = grammar.allowed_ids_for_bar(residual_state, meter=4, minimum_start_gap_beats=.025)
                self.assertTrue(residuals)
                for residual in residuals:
                    beat = tick / TICKS_PER_BEAT + vocab.residual_steps(residual) / RESIDUAL_TICKS_PER_BEAT
                    self.assertGreaterEqual(beat - last_units / RESIDUAL_TICKS_PER_BEAT, .025 - 1e-9)

        exhausted = GrammarState(DecodePhase.BETWEEN_EVENTS, last_offset_tick=383, last_time_units=6140)
        self.assertEqual(grammar.allowed_ids_for_bar(exhausted, meter=4, minimum_start_gap_beats=.025), (vocab.eos_id,))

    def test_grammar_snap_filtering_matches_transition_invariants(self):
        grammar = self.codec.grammar
        vocabulary = self.codec.vocabulary
        for offset_tick in (0, 24, 32, 48, 72, 95, 96, 120):
            state = GrammarState(
                phase=DecodePhase.EVENT_SNAP,
                last_offset_tick=-1,
                last_time_units=-1_000_000_000,
                current_offset_tick=offset_tick,
            )
            allowed = grammar.allowed_ids(state)
            allowed_denominators = {vocabulary.snap_denominator(token_id) for token_id in allowed}
            for denominator in SNAP_DENOMINATORS:
                step = TICKS_PER_BEAT // denominator
                token_id = vocabulary.id_for(f"snap:{denominator}")
                if offset_tick % step == 0:
                    self.assertIn(denominator, allowed_denominators)
                    next_state = grammar.transition(state, token_id)
                    self.assertEqual(next_state.phase, DecodePhase.EVENT_RESIDUAL)
                else:
                    self.assertNotIn(denominator, allowed_denominators)
                    with self.assertRaises(ValueError):
                        grammar.transition(state, token_id)

    def test_meter_aware_grammar_final_offset_and_dead_end_freedom(self):
        grammar = self.codec.grammar
        vocabulary = self.codec.vocabulary
        for meter in (0.5, 1.0, 1.5, 3.0, 4.0, 5.25):
            max_tick = int(meter * TICKS_PER_BEAT)
            # Final legal offset in the bar.
            final_tick = max_tick - 1
            state_at_final = GrammarState(
                phase=DecodePhase.BETWEEN_EVENTS,
                last_offset_tick=final_tick,
                last_time_units=final_tick * 16,
            )
            allowed_final = grammar.allowed_ids_for_bar(state_at_final, meter=meter)
            self.assertEqual(allowed_final, (vocabulary.eos_id,))

            # Non-final offset where at least one next offset tick exists.
            if final_tick > 0:
                state_before_final = GrammarState(
                    phase=DecodePhase.BETWEEN_EVENTS,
                    last_offset_tick=final_tick - 1,
                    last_time_units=(final_tick - 1) * 16,
                )
                allowed_before = grammar.allowed_ids_for_bar(state_before_final, meter=meter)
                self.assertIn(vocabulary.event_id, allowed_before)
                self.assertIn(vocabulary.eos_id, allowed_before)
                next_state = grammar.transition(state_before_final, vocabulary.event_id)
                next_allowed = grammar.allowed_ids_for_bar(next_state, meter=meter)
                self.assertTrue(len(next_allowed) > 0)

    def test_bounded_legal_grammar_rollouts_always_terminate_validly(self):
        grammar = self.codec.grammar
        vocabulary = self.codec.vocabulary
        rng = random.Random(42)
        for meter in (1.0, 1.5, 3.0, 4.0, 6.0):
            for _ in range(20):
                tokens = [vocabulary.bos_id]
                state = grammar.transition(GrammarState(), tokens[0])
                for _ in range(200):
                    allowed = grammar.allowed_ids_for_bar(state, meter=meter)
                    self.assertTrue(len(allowed) > 0, f"dead end encountered in meter {meter}")
                    next_token = rng.choice(allowed)
                    tokens.append(next_token)
                    state = grammar.transition(state, next_token)
                    if state.phase == DecodePhase.START:
                        break
                self.assertEqual(state.phase, DecodePhase.START)
                decoded = self.codec.decode(tokens, bar_start_beat=0.0, meter=meter)
                for event in decoded:
                    self.assertTrue(0.0 <= event.beat_position < meter)


if __name__ == "__main__":
    unittest.main()
