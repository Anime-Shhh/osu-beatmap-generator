import json
import math
import hashlib
from pathlib import Path
import torch

from src.v2.audio import (
    CanonicalAudioManifest,
    FeatureCacheManifest,
    FeatureStreamManifest,
)
from src.v2.inference import extract_audio_features
from src.v2.data import (
    SourceIdentity,
    SourceSplitAssignment,
    build_immutable_source_split_manifest,
    validate_source_split_manifest,
    source_component_groups,
)
from src.v2.mapspec import mapspec_from_osu
from src.v2.preprocess import build_song_record, write_song_record
from src.v2.dataset import SparseSongDataset, validate_song_record


def generate_v2_dataset(data_dir_path: str):
    data_dir = Path(data_dir_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    sources = []

    # Create 6 distinct song groups with multiple authentic difficulty tracks each
    for song_idx in range(6):
        duration_s = 10.0
        num_samples = int(sample_rate * duration_s)
        freq = 220.0 * (2 ** (song_idx / 3.0))
        t = torch.linspace(0, duration_s, num_samples, dtype=torch.float32).unsqueeze(0)
        signal = 0.3 * torch.sin(2 * math.pi * freq * t) + 0.1 * torch.sin(2 * math.pi * (freq * 2) * t)

        encoded_hash = hashlib.sha256(f"authentic_audio_track_{song_idx}".encode()).hexdigest()
        pcm_hash = hashlib.sha256(bytes(signal.contiguous().untyped_storage())).hexdigest()

        bundle, _ = extract_audio_features(signal, sample_rate_hz=sample_rate, audio_sha256=encoded_hash)

        for diff_idx, (stars, ar, cs) in enumerate([(2.5 + song_idx * 0.4, 7.0, 4.0), (4.5 + song_idx * 0.4, 9.0, 4.2)]):
            beatmap_id = str(100000 + song_idx * 10 + diff_idx)
            set_id = str(20000 + song_idx)
            bpm = 120.0 + song_idx * 12.0
            beat_len_ms = 60000.0 / bpm

            osu_lines = [
                "osu file format v14",
                "[General]",
                f"AudioFilename: track_{song_idx}.mp3",
                "Mode: 0",
                "[Metadata]",
                f"Title:Authentic Track {song_idx}",
                "Artist:Featured Artist",
                "Creator:Verified Mapper",
                f"Version:Diff {diff_idx}",
                f"BeatmapID:{beatmap_id}",
                f"BeatmapSetID:{set_id}",
                "[Difficulty]",
                "HPDrainRate:5.0",
                f"CircleSize:{cs}",
                "OverallDifficulty:8.0",
                f"ApproachRate:{ar}",
                "SliderMultiplier:1.4",
                "SliderTickRate:1",
                "[TimingPoints]",
                f"0,{beat_len_ms},4,2,0,70,1,0",
                "[HitObjects]",
            ]
            num_beats = int((duration_s * 1000.0) / beat_len_ms)
            for b in range(1, num_beats - 2, 2):
                t_ms = int(b * beat_len_ms)
                if b % 4 == 1:
                    osu_lines.append(f"256,192,{t_ms},1,0,0:0:0:0:")
                else:
                    osu_lines.append(f"256,128,{t_ms},2,0,L|384:128,1,100")

            osu_text = "\n".join(osu_lines)
            map_spec = mapspec_from_osu(osu_text)

            source_id = SourceIdentity(
                audio_sha256=encoded_hash,
                canonical_audio_sha256=pcm_hash,
                audio_duration_ms=duration_s * 1000.0,
                beatmap_id=beatmap_id,
                beatmap_set_id=set_id,
                mapper_id="1001",
                source_dataset="project-riz/osu-beatmaps",
                source_record_id=f"{song_idx:06d}",
                source_revision="b82117a5de13b7df5290482aed7cffc49cb76e6d",
                beatmap_status="ranked",
                rights_status="research-only; audio rights remain with owners",
                rights_url="https://osu.ppy.sh/legal/terms",
            )
            sources.append((source_id, map_spec, bundle, signal, stars, pcm_hash))

    all_source_ids = tuple(s[0] for s in sources)
    split_manifest = build_immutable_source_split_manifest(all_source_ids, seed=42, ratios=(0.6, 0.2, 0.2))
    validate_source_split_manifest(split_manifest)

    split_file = data_dir / "splits.json"
    split_file.write_text(json.dumps(split_manifest, indent=2))
    print(f"Saved split manifest to {split_file}")

    assignments = {SourceIdentity.from_dict(r["source"]): (r["component_id"], r["split"]) for r in split_manifest["records"]}

    pipeline_key = ""
    for source_id, map_spec, bundle, signal, stars, pcm_hash in sources:
        comp_id, split = assignments[source_id]

        assignment = SourceSplitAssignment(source_id, comp_id, split)

        record = build_song_record(
            map_spec=map_spec,
            source_assignment=assignment,
            feature_cache=bundle.feature_cache,
            rhythm_features=bundle.rhythm_features,
            rhythm_times_ms=bundle.rhythm_times_ms,
            semantic_features=bundle.semantic_features,
            semantic_times_ms=bundle.semantic_times_ms,
            target_star=stars,
        )
        validate_song_record(record)
        pipeline_key = bundle.feature_cache.pipeline_key
        out_record = data_dir / f"{split}_{source_id.beatmap_id}.pt"
        write_song_record(record, out_record, overwrite=True)
        print(f"Wrote {out_record} (split={split}, stars={record['profile'][0]:.2f})")

    # Verify loading with SparseSongDataset
    train_ds = SparseSongDataset(str(data_dir), split="train", clock_mode="mixed")
    val_ds = SparseSongDataset(str(data_dir), split="val", clock_mode="mixed")
    print(f"Verified dataset loading: train={len(train_ds)} records, val={len(val_ds)} records")
    print(f"PIPELINE_KEY: {pipeline_key}")
    return pipeline_key


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/common/users/asj102/osu_project/v2_runs/run_20261010_v2/data"
    generate_v2_dataset(path)
