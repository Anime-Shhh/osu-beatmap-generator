"""CPU-only record/cache compatibility tests; no optimizer step is run."""

from __future__ import annotations

import hashlib
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
        from src.v2.data import SourceIdentity, SourceSplitAssignment
        source = SourceIdentity(
            audio_sha256=digest("encoded"),
            canonical_audio_sha256=digest("pcm"),
            audio_duration_ms=4_000.0,
        )
        assignment = SourceSplitAssignment(source, "source:" + "a" * 64, "train")
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
        self.assertEqual(len(record["clock_variants"]), 3)  # oracle + two deliberate jitter variants
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


if __name__ == "__main__":
    unittest.main()
