"""CPU-only contract tests for v2 cache, split, and clock foundations."""

from __future__ import annotations

import hashlib
import math
import unittest

from src.v2.audio import (
    CanonicalAudioManifest,
    FeatureCacheManifest,
    FeatureStreamManifest,
    apply_scalar_alignment,
    build_timestamp_alignment,
    regular_timestamps_ms,
)
from src.v2.clock import (
    ClockEvidence,
    MeterMap,
    MeterSegment,
    StaticClockResolver,
    TimingHypothesis,
    jitter_timing_hypothesis,
)
from src.v2.data import (
    SourceIdentity,
    SourceSplitAssignment,
    assert_source_split_disjoint,
    build_source_split_assignments,
    source_component_groups,
)
from src.v2.timing import TimingMap, TimingSegment


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AudioCacheAndAlignmentTests(unittest.TestCase):
    def test_cache_manifest_is_round_trippable_and_relocatable(self):
        source = SourceIdentity(
            audio_sha256=digest("encoded song"),
            canonical_audio_sha256=digest("canonical pcm"),
            audio_duration_ms=12_000.0,
        )
        canonical = CanonicalAudioManifest(
            source=source,
            pcm_sha256=digest("canonical pcm"),
            frame_count=288_000,
        )
        stream = FeatureStreamManifest(
            name="mert_v1_25hz",
            extractor="mert-v1-95m",
            extractor_revision="layer-8-projection-v1",
            feature_dim=128,
            frame_count=300,
            dtype="float16",
            feature_path="features/mert.npy",
            feature_sha256=digest("mert features"),
            timestamps_path="timestamps/mert.npy",
            timestamps_sha256=digest("mert timestamps"),
            nominal_frame_rate_hz=25.0,
        )
        manifest = FeatureCacheManifest(
            canonical_audio=canonical,
            streams=(stream,),
            pipeline_revision="v2-multirate-audio-1",
        )
        restored = FeatureCacheManifest.from_json(manifest.to_json())
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.cache_key, manifest.cache_key)

        relocated_stream = FeatureStreamManifest(
            **{**stream.to_dict(), "feature_path": "other/mert.npy", "timestamps_path": "other/t.npy"}
        )
        relocated = FeatureCacheManifest(
            canonical_audio=canonical,
            streams=(relocated_stream,),
            pipeline_revision=manifest.pipeline_revision,
        )
        self.assertEqual(relocated.cache_key, manifest.cache_key)

    def test_timestamp_alignment_is_explicit_about_boundary_padding(self):
        alignment = build_timestamp_alignment(
            (5.0, 15.0, 25.0), (0.0, 5.0, 10.0, 25.0, 40.0)
        )
        self.assertEqual(alignment.left_indices, (0, 0, 0, 2, 2))
        self.assertEqual(alignment.right_indices, (0, 0, 1, 2, 2))
        self.assertEqual(alignment.valid, (False, True, True, True, False))
        self.assertEqual(apply_scalar_alignment((0.0, 10.0, 20.0), alignment), (0.0, 0.0, 5.0, 20.0, 20.0))
        self.assertEqual(
            regular_timestamps_ms(3, first_timestamp_ms=5.0, frame_hop_ms=10.0),
            (5.0, 15.0, 25.0),
        )
        stacked = build_timestamp_alignment((5.0, 15.0), (10.0, 10.0, 5.0))
        self.assertEqual(stacked.left_indices, (0, 0, 0))
        self.assertEqual(stacked.right_indices, (1, 1, 0))
        with self.assertRaises(ValueError):
            build_timestamp_alignment((5.0, 15.0), (0.0,), out_of_range="error")


class SourceSplitTests(unittest.TestCase):
    def test_transitive_canonical_and_fingerprint_components_never_split(self):
        first = SourceIdentity(
            audio_sha256=digest("encoded-a"),
            canonical_audio_sha256=digest("pcm-shared"),
        )
        reencode = SourceIdentity(
            audio_sha256=digest("encoded-b"),
            canonical_audio_sha256=digest("pcm-shared"),
            acoustic_fingerprint="cluster-7",
            fingerprint_algorithm="chromaprint-v1",
        )
        acoustic_match = SourceIdentity(
            audio_sha256=digest("encoded-c"),
            acoustic_fingerprint="cluster-7",
            fingerprint_algorithm="chromaprint-v1",
        )
        groups = source_component_groups((acoustic_match, first, reencode))
        self.assertEqual(groups[first], groups[reencode])
        self.assertEqual(groups[reencode], groups[acoustic_match])

        assignments = build_source_split_assignments((first, reencode, acoustic_match), seed=73)
        self.assertEqual(len({assignment.component_id for assignment in assignments}), 1)
        self.assertEqual(len({assignment.split for assignment in assignments}), 1)
        self.assertEqual(
            assignments,
            build_source_split_assignments((acoustic_match, reencode, first), seed=73),
        )

        component = groups[first]
        with self.assertRaises(ValueError):
            assert_source_split_disjoint(
                (
                    SourceSplitAssignment(first, component, "train"),
                    SourceSplitAssignment(reencode, component, "val"),
                )
            )


class ClockContractTests(unittest.TestCase):
    def test_hypothesis_serializes_bar_phase_and_coherent_jitter(self):
        timing = TimingMap(
            (
                TimingSegment(0.0, 500.0, 0.0),
                TimingSegment(1_000.0, 400.0, 2.0),
            )
        )
        hypothesis = TimingHypothesis(
            hypothesis_id="oracle-fixture",
            timing_map=timing,
            meter_segments=(MeterSegment(0.0, 4, 4), MeterSegment(4.0, 3, 4)),
            confidence=1.0,
            score=2.0,
            source="fixture",
            is_oracle=True,
        )
        self.assertAlmostEqual(hypothesis.beat_phase(125.0)[0], 1.0)
        self.assertAlmostEqual(hypothesis.beat_phase(125.0)[1], 0.0, places=6)
        bar = hypothesis.bar_position(2_200.0)  # beat 5: second 3/4 bar, beat 1.
        self.assertEqual(bar.bar_index, 1)
        self.assertAlmostEqual(bar.phase, 1.0 / 3.0)
        self.assertEqual(
            TimingHypothesis.from_dict(hypothesis.to_dict()).to_dict(), hypothesis.to_dict()
        )

        jittered = jitter_timing_hypothesis(
            hypothesis, offset_ms=20.0, time_scale=1.1, confidence=0.4
        )
        self.assertFalse(jittered.is_oracle)
        self.assertEqual(jittered.source, "jittered_oracle")
        self.assertAlmostEqual(jittered.timing_map.beat_to_time(2.0), 1_120.0)
        self.assertAlmostEqual(jittered.confidence, 0.4)

        evidence = ClockEvidence(
            audio_duration_ms=10_000.0,
            beat_times_ms=(0.0, 500.0, 1_000.0),
            beat_confidences=(0.9, 0.8, 0.9),
            source="fixture-beats",
            extractor_revision="fixture-1",
        )
        self.assertEqual(StaticClockResolver((hypothesis, jittered)).resolve(evidence, max_hypotheses=1), (hypothesis,))

    def test_midbar_meter_change_requires_explicit_bar_number(self):
        with self.assertRaises(ValueError):
            MeterMap((MeterSegment(0.0), MeterSegment(2.0, 3, 4)))
        meter = MeterMap((MeterSegment(0.0), MeterSegment(2.0, 3, 4, start_bar=1)))
        position = meter.bar_position(2.0)
        self.assertEqual(position.bar_index, 1)
        self.assertTrue(math.isclose(position.phase, 0.0))


if __name__ == "__main__":
    unittest.main()
