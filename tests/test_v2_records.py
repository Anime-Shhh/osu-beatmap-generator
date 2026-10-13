"""CPU-only record/cache compatibility tests; no optimizer step is run."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SongRecordTests(unittest.TestCase):
    # Keep tensor-dependent imports inside the skipped test class's helper so
    # CPU/login-node test discovery works without importing the train-only
    # dataset module at all.
    def _cache_and_assignment(self):
        from src.v2.audio import CanonicalAudioManifest, FeatureCacheManifest, FeatureStreamManifest
        from src.v2.data import SourceIdentity, SourceSplitAssignment, source_component_groups
        source = SourceIdentity(
            audio_sha256=digest("encoded"),
            canonical_audio_sha256=digest("pcm"),
            audio_duration_ms=4_000.0,
            source_record_id="fixture-record",
            source_revision="fixture-revision",
            beatmap_status="ranked",
            rights_status="test-fixture-only",
            rights_url="https://example.invalid/test-rights",
        )
        assignment = SourceSplitAssignment(source, source_component_groups((source,))[source], "train")
        canonical = CanonicalAudioManifest(source, digest("pcm"), 96_000)
        rhythm = FeatureStreamManifest(
            name="rhythm_100hz",
            extractor="fixture",
            extractor_revision="1",
            feature_dim=101,
            frame_count=400,
            dtype="float32",
            feature_path="rhythm.npy",
            feature_sha256=digest("rhythm"),
            timestamps_path="rhythm-times.npy",
            timestamps_sha256=digest("rhythm-times"),
            nominal_frame_rate_hz=100.0,
        )
        semantic = FeatureStreamManifest(
            name="mert_25hz_128d",
            extractor="fixture-mert",
            extractor_revision="pca-train-only-1",
            feature_dim=128,
            frame_count=100,
            dtype="float16",
            feature_path="semantic.npy",
            feature_sha256=digest("semantic"),
            timestamps_path="semantic-times.npy",
            timestamps_sha256=digest("semantic-times"),
            nominal_frame_rate_hz=25.0,
        )
        return FeatureCacheManifest(canonical, (rhythm, semantic), "fixture-v2"), assignment

    def _map_spec(self):
        from src.v2.events import BeatEvent, EventKind
        from src.v2.mapspec import DifficultySettings, GeneralSettings, MapMetadata, MapSpec
        from src.v2.timing import TimingPoint
        return MapSpec(
            format_version=14,
            general=GeneralSettings("song.mp3"),
            metadata=MapMetadata("Song", "Artist", "Creator", "Draft"),
            difficulty=DifficultySettings(),
            timing_points=(TimingPoint(0.0, 500.0),),
            events=(
                BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0, source_object_index=0),
                BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0, source_object_index=1),
            ),
        )

    def test_song_record_binds_song_hash_but_checkpoints_use_shared_pipeline_key(self):
        from src.v2.dataset import (
            SparseSongDataset,
            collate_sparse_examples,
            song_record_index_path,
            validate_song_record,
        )
        from src.v2.preprocess import build_song_record, write_song_record
        cache, assignment = self._cache_and_assignment()
        record = build_song_record(
            map_spec=self._map_spec(),
            source_assignment=assignment,
            feature_cache=cache,
            rhythm_features=torch.zeros((400, 101)),
            rhythm_times_ms=torch.arange(400, dtype=torch.float32) * 10.0,
            semantic_features=torch.zeros((100, 128)),
            semantic_times_ms=torch.arange(100, dtype=torch.float32) * 40.0,
            target_star=4.5,
        )
        validate_song_record(record)
        self.assertEqual(record["metadata"]["cache_key"], cache.cache_key)
        self.assertEqual(record["metadata"]["feature_pipeline_key"], cache.pipeline_key)
        self.assertEqual(len(record["clock_variants"]), 1)  # unsafe target-invariant jitter is disabled
        self.assertEqual(record["bar_targets"][0]["tokens"][0].item(), 0)  # BOS is stable

        with tempfile.TemporaryDirectory() as directory:
            record_path = write_song_record(record, Path(directory) / "fixture.pt")
            self.assertTrue(song_record_index_path(record_path).is_file())
            # Corpus discovery must consume the tiny sidecar rather than
            # deserializing every full song tensor at DataLoader startup.
            with patch("src.v2.dataset._read_record", side_effect=AssertionError("full record was read")):
                indexed = SparseSongDataset(
                    directory,
                    split="train",
                    expected_pipeline_key=cache.pipeline_key,
                    clock_mode="mixed",
                )
                self.assertEqual(len(indexed), 1)
            dataset = SparseSongDataset(
                directory,
                split="train",
                expected_pipeline_key=cache.pipeline_key,
                clock_mode="mixed",
            )
            dataset.set_epoch(1)
            example = dataset[0]
            batch = collate_sparse_examples((example,))
            self.assertEqual(batch["rhythm_features"].shape[-1], 101)
            self.assertEqual(batch["semantic_features"].shape[-1], 128)
            self.assertEqual(batch["clock_features"].shape[-1], 4)
            self.assertEqual(batch["labels"].shape[0], 1)
            self.assertTrue(batch["previous_bar_mask"].any())
            self.assertEqual(record_path.suffix, ".pt")

    def _make_record_with_bars(self, bars: int):
        assert torch is not None
        from src.v2.events import BeatEvent, EventKind
        from src.v2.preprocess import build_song_record
        cache, assignment = self._cache_and_assignment()
        duration = float(bars * 2000)
        source = dataclasses.replace(assignment.source, audio_duration_ms=duration)
        from src.v2.data import source_component_groups
        assignment = dataclasses.replace(
            assignment,
            source=source,
            component_id=source_component_groups((source,))[source],
        )
        cache = dataclasses.replace(
            cache,
            canonical_audio=dataclasses.replace(
                cache.canonical_audio, source=source, frame_count=int(duration * 24)
            ),
        )
        spec = dataclasses.replace(
            self._map_spec(),
            events=(BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0, source_object_index=0),),
        )
        rhythm_count = math.ceil(duration / 10)
        semantic_count = math.ceil(duration / 40)
        cache = dataclasses.replace(
            cache,
            streams=tuple(
                dataclasses.replace(
                    stream,
                    frame_count=rhythm_count if stream.name == "rhythm_100hz" else semantic_count,
                )
                for stream in cache.streams
            ),
        )
        return build_song_record(
            map_spec=spec,
            source_assignment=assignment,
            feature_cache=cache,
            rhythm_features=torch.zeros(rhythm_count, 101),
            rhythm_times_ms=torch.arange(rhythm_count, dtype=torch.float32) * 10.0,
            semantic_features=torch.zeros(semantic_count, 128),
            semantic_times_ms=torch.arange(semantic_count, dtype=torch.float32) * 40.0,
            target_star=4.5,
        )

    def test_sparse_dataset_rotation_and_bar_coverage_across_configurations(self):
        assert torch is not None
        from src.v2.dataset import SparseSongDataset
        from src.v2.preprocess import write_song_record

        for bars_by_map in ([100, 100], [100] * 4, [99] * 3, [12, 15]):
            with self.subTest(bars=bars_by_map), tempfile.TemporaryDirectory() as directory:
                for i, bars in enumerate(bars_by_map):
                    record = self._make_record_with_bars(bars)
                    write_song_record(record, Path(directory) / f"{i:02d}-a.pt")

                    # Interleave records from another component to ensure noncontiguous indices.
                    from src.v2.data import SourceIdentity, source_component_groups
                    other = copy.deepcopy(record)
                    other_source = SourceIdentity(
                        audio_sha256=digest("encoded-z"),
                        canonical_audio_sha256=digest("pcm-z"),
                        audio_duration_ms=record["metadata"]["source"]["audio_duration_ms"],
                        source_record_id="fixture-record-z",
                        source_revision="fixture-revision",
                        beatmap_status="ranked",
                        rights_status="test-fixture-only",
                        rights_url="https://example.invalid/test-rights",
                    )
                    other["metadata"]["source"] = other_source.to_dict()
                    other["metadata"]["source_component_id"] = source_component_groups((other_source,))[other_source]
                    write_song_record(other, Path(directory) / f"{i:02d}-z.pt")

                dataset = SparseSongDataset(directory, clock_mode="oracle")
                self.assertEqual(len(dataset), 2)  # Two source components: source:a and source:z

                cycle = len(bars_by_map) * math.lcm(*bars_by_map)
                selected_indices = dataset._song_groups[0]
                seen = {i: set() for i in selected_indices}

                for epoch in range(cycle):
                    dataset.set_epoch(epoch)
                    sample = dataset[0]
                    selected = dataset._song_groups[0][epoch % len(bars_by_map)]
                    bar_idx = sample["target"]["bar_index"]
                    seen[selected].add(bar_idx)

                    if bar_idx > 0:
                        prior = dataset._record(selected)["bar_targets"][bar_idx - 1]["tokens"]
                        self.assertTrue(torch.equal(prior, sample["previous_bar_tokens"]))
                    else:
                        self.assertEqual(sample["previous_bar_tokens"].numel(), 0)

                # Full 100% coverage across all bars of all maps.
                self.assertEqual([len(seen[i]) for i in selected_indices], list(bars_by_map))

                # Deterministic resume check.
                dataset.set_epoch(7)
                sample_epoch_7 = dataset[0]
                dataset.set_epoch(0)
                dataset.set_epoch(7)
                sample_epoch_7_resumed = dataset[0]
                self.assertEqual(
                    sample_epoch_7["target"]["bar_index"],
                    sample_epoch_7_resumed["target"]["bar_index"],
                )
                self.assertEqual(
                    sample_epoch_7["metadata"]["source_component_id"],
                    sample_epoch_7_resumed["metadata"]["source_component_id"],
                )

    def test_dataset_preflight_rejects_duplicate_audio_across_splits(self):
        from src.v2.dataset import RecordSchemaError, SparseSongDataset
        from src.v2.preprocess import write_song_record

        record = self._make_record_with_bars(2)
        leaked = copy.deepcopy(record)
        leaked["metadata"]["split"] = "val"
        leaked["metadata"]["source_component_id"] = "source:" + "f" * 64
        leaked["metadata"]["source"]["acoustic_fingerprint"] = "same-family"
        leaked["metadata"]["source"]["fingerprint_algorithm"] = "chromaprint-v1"
        record["metadata"]["source"]["acoustic_fingerprint"] = "same-family"
        record["metadata"]["source"]["fingerprint_algorithm"] = "chromaprint-v1"
        from src.v2.data import SourceIdentity, source_component_groups
        source = SourceIdentity.from_dict(record["metadata"]["source"])
        record["metadata"]["source_component_id"] = source_component_groups((source,))[source]

        with tempfile.TemporaryDirectory() as directory:
            write_song_record(record, Path(directory) / "train.pt")
            write_song_record(leaked, Path(directory) / "val.pt")
            with self.assertRaisesRegex(RecordSchemaError, "split preflight"):
                SparseSongDataset(directory, split="train", clock_mode="oracle")

    def test_authentic_sample_ingest_and_record_build_smoke(self):
        assert torch is not None
        from src.v2.audio import (
            CanonicalAudioManifest,
            FeatureCacheManifest,
            FeatureStreamManifest,
        )
        from src.v2.inference import extract_audio_features
        from src.v2.data import (
            SourceIdentity,
            SourceSplitAssignment,
            audio_sha256,
            build_immutable_source_split_manifest,
            source_component_groups,
            validate_source_split_manifest,
        )
        from src.v2.dataset import SparseSongDataset, validate_song_record
        from src.v2.mapspec import mapspec_from_osu
        from src.v2.preprocess import (
            build_song_record,
            ranked_standard_source_filter,
            write_song_record,
        )

        # Authentic metadata matching pinned public HF pair (kamome sano - dawn)
        osu_content = (
            "osu file format v14\n\n"
            "[General]\n"
            "AudioFilename: 000001.mp3\n"
            "Mode: 0\n\n"
            "[Metadata]\n"
            "Title:dawn(ginkiha hardcore remix)\n"
            "Artist:kamome sano\n"
            "Creator:wendao\n"
            "Version:Beginner\n"
            "BeatmapID:443939\n"
            "BeatmapSetID:185766\n\n"
            "[Difficulty]\n"
            "HPDrainRate:1.6\n"
            "CircleSize:2.8\n"
            "OverallDifficulty:1.8\n"
            "ApproachRate:2.4\n"
            "SliderMultiplier:0.8\n"
            "SliderTickRate:1\n\n"
            "[TimingPoints]\n"
            "1740,331.491712707182,4,2,0,70,1,0\n"
            "15002,331.491712707182,4,2,0,70,1,0\n\n"
            "[HitObjects]\n"
            "256,192,1740,1,0,0:0:0:0:\n"
            "320,192,2403,1,0,0:0:0:0:\n"
            "256,128,3066,2,0,L|384:128,1,120\n"
        )
        raw_source_record = {
            "mode": 0,
            "status": "ranked",
            "beatmap_id": 443939,
            "beatmapset_id": 185766,
            "stars": 1.65866,
        }
        self.assertTrue(ranked_standard_source_filter(raw_source_record))

        # Synthetic 5-second 24kHz audio simulating decoded authentic track
        sample_rate = 24000
        duration_s = 5.0
        t = torch.linspace(0, duration_s, int(sample_rate * duration_s), dtype=torch.float32).unsqueeze(0)
        signal = 0.2 * torch.sin(2 * math.pi * 440.0 * t)
        encoded_hash = "0003560e88d0dc6d0678fd2d1a9a19d44c1270a318b8fefa7273a839612cde25"
        pcm_hash = hashlib.sha256(bytes(signal.contiguous().untyped_storage())).hexdigest()

        source_id = SourceIdentity(
            audio_sha256=encoded_hash,
            canonical_audio_sha256=pcm_hash,
            audio_duration_ms=duration_s * 1000.0,
            beatmap_id="443939",
            beatmap_set_id="185766",
            mapper_id="1843939",
            source_dataset="project-riz/osu-beatmaps",
            source_record_id="000001",
            source_revision="b82117a5de13b7df5290482aed7cffc49cb76e6d",
            beatmap_status="ranked",
            rights_status="research-only; audio rights remain with owners",
            rights_url="https://osu.ppy.sh/legal/terms",
        )
        source_id.validate_training_provenance()

        # Build group assignments and immutable manifest
        comp_id = source_component_groups((source_id,))[source_id]
        assignment = SourceSplitAssignment(source_id, comp_id, "train")
        manifest = build_immutable_source_split_manifest((source_id,), seed=42)
        validate_source_split_manifest(manifest)

        # Feature extraction
        bundle, _ = extract_audio_features(signal, sample_rate_hz=sample_rate)
        rhythm_feats = bundle.rhythm_features
        rhythm_times = bundle.rhythm_times_ms
        semantic_feats = bundle.semantic_features
        semantic_times = bundle.semantic_times_ms

        canonical_manifest = CanonicalAudioManifest(
            source=source_id,
            pcm_sha256=pcm_hash,
            frame_count=signal.shape[-1],
        )
        rhythm_stream = FeatureStreamManifest(
            name="rhythm_100hz",
            extractor="v2-rhythm-fft",
            extractor_revision="v2-rhythm-1",
            feature_dim=101,
            frame_count=rhythm_feats.shape[0],
            dtype="float32",
            feature_path="rhythm.pt",
            feature_sha256=digest("rhythm_feat"),
            timestamps_path="rhythm_times.pt",
            timestamps_sha256=digest("rhythm_time"),
            nominal_frame_rate_hz=100.0,
        )
        semantic_stream = FeatureStreamManifest(
            name="mert_25hz_128d",
            extractor="mert-v1-95m",
            extractor_revision="layer-8-pca-1",
            feature_dim=128,
            frame_count=semantic_feats.shape[0],
            dtype="float32",
            feature_path="semantic.pt",
            feature_sha256=digest("semantic_feat"),
            timestamps_path="semantic_times.pt",
            timestamps_sha256=digest("semantic_time"),
            nominal_frame_rate_hz=25.0,
        )
        feature_cache = FeatureCacheManifest(
            canonical_audio=canonical_manifest,
            streams=(rhythm_stream, semantic_stream),
            pipeline_revision="v2-multirate-audio-1",
        )

        map_spec = mapspec_from_osu(osu_content)
        record = build_song_record(
            map_spec=map_spec,
            source_assignment=assignment,
            feature_cache=feature_cache,
            rhythm_features=rhythm_feats,
            rhythm_times_ms=rhythm_times,
            semantic_features=semantic_feats,
            semantic_times_ms=semantic_times,
            target_star=1.65866,
        )
        validate_song_record(record)
        self.assertEqual(record["metadata"]["source"]["beatmap_id"], "443939")
        self.assertEqual(record["metadata"]["source"]["audio_sha256"], encoded_hash)

        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = write_song_record(record, Path(temp_dir) / "dawn_beginner.pt")
            self.assertTrue(record_path.is_file())
            dataset = SparseSongDataset(temp_dir, split="train", clock_mode="oracle")
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertAlmostEqual(sample["profile"][0].item(), 1.65866 / 10.0, places=5)
            self.assertEqual(sample["metadata"]["source_component_id"], comp_id)


if __name__ == "__main__":
    unittest.main()
