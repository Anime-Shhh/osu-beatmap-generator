"""
Preprocessing pipeline: stream from HuggingFace, convert to mel+tokens,
write as WebDataset .tar shards to shared cold storage for training.

Usage:
    python -m src.preprocess \
        --output_dir /common/users/asj102/osu_project/data/shards \
        --max_shards 500 \
        --examples_per_shard 200
"""

import argparse
import glob
import io
import json
import os
import sys
import time
import traceback

import torch
import webdataset as wds

from .dataset import (
    load_audio_from_bytes, build_mel_transform, extract_windows,
    TARGET_SR, WINDOW_SAMPLES,
)
from .tokenizer import Residuals, difficulty_to_bin, parse_osu_bpm


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess osu! beatmaps to cached shards")
    p.add_argument("--output_dir", type=str,
                    default="/common/users/asj102/osu_project/data/shards")
    p.add_argument("--max_shards", type=int, default=500)
    p.add_argument("--examples_per_shard", type=int, default=200)
    p.add_argument("--min_tokens", type=int, default=4,
                    help="Skip windows with fewer tokens than this")
    p.add_argument("--mode_filter", type=int, default=0,
                    help="osu! mode to keep (0=standard)")
    p.add_argument("--resume_from", type=int, default=0,
                    help="Resume from this shard number")
    return p.parse_args()


def serialize_tensor(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def serialize_tokens(
    tokens: list[int],
    residuals: list[Residuals],
    difficulty_id: int,
    bpm: float,
) -> bytes:
    buf = io.BytesIO()
    torch.save({
        "tokens": tokens,
        "residuals": residuals,
        "difficulty_id": difficulty_id,
        "bpm": bpm,
    }, buf)
    return buf.getvalue()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    existing_shards = glob.glob(os.path.join(args.output_dir, "shard-*.tar"))
    if existing_shards:
        print(f"Clearing {len(existing_shards)} existing shards in {args.output_dir}")
        for shard_path in existing_shards:
            try:
                os.remove(shard_path)
            except OSError as exc:
                print(f"Failed to remove {shard_path}: {exc}", file=sys.stderr)

    from datasets import load_dataset, Audio
    ds = load_dataset(
        "project-riz/osu-beatmaps", "original",
        split="train", streaming=True,
    )
    # Disable HuggingFace's auto audio decoding — we decode raw bytes via torchaudio
    try:
        ds = ds.cast_column("mp3", Audio(decode=False))
    except Exception:
        pass

    mel_transform = build_mel_transform("cpu")

    shard_idx = args.resume_from
    sample_in_shard = 0
    global_count = 0
    skipped = 0
    start_time = time.time()

    shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:06d}.tar")
    sink = wds.TarWriter(shard_path)

    print(f"Starting preprocessing -> {args.output_dir}")
    print(f"Max shards: {args.max_shards}, examples/shard: {args.examples_per_shard}")

    try:
        for example in ds:
            if shard_idx >= args.max_shards + args.resume_from:
                break

            try:
                json_data = example.get("json")
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)

                audio_bytes = example.get("mp3")
                if audio_bytes is None:
                    skipped += 1
                    continue
                if isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
                    audio_bytes = audio_bytes["bytes"]

                waveform, sr = load_audio_from_bytes(audio_bytes)
                if waveform.shape[-1] < WINDOW_SAMPLES:
                    skipped += 1
                    continue

                beatmaps = json_data.get("beatmaps", [])

                for bm in beatmaps:
                    if shard_idx >= args.max_shards + args.resume_from:
                        break

                    osu_content = bm.get("content", "")
                    if not osu_content:
                        continue

                    mode = bm.get("mode", 0)
                    if mode != args.mode_filter:
                        continue

                    star_rating = float(bm.get("difficultyrating", 4.0))
                    diff_id = difficulty_to_bin(star_rating)
                    bpm = parse_osu_bpm(osu_content)

                    windows = extract_windows(
                        waveform, osu_content, mel_transform,
                        diff_id, bpm,
                    )

                    for w in windows:
                        if len(w["tokens"]) < args.min_tokens:
                            continue

                        key = f"{shard_idx:06d}_{sample_in_shard:06d}"
                        sink.write({
                            "__key__": key,
                            "mel.pt": serialize_tensor(w["mel"]),
                            "tokens.pt": serialize_tokens(
                                w["tokens"], w["residuals"], diff_id, bpm,
                            ),
                        })
                        sample_in_shard += 1
                        global_count += 1

                        if sample_in_shard >= args.examples_per_shard:
                            sink.close()
                            elapsed = time.time() - start_time
                            rate = global_count / elapsed if elapsed > 0 else 0
                            print(
                                f"Shard {shard_idx:06d} done | "
                                f"total={global_count} | skipped={skipped} | "
                                f"rate={rate:.1f} samples/s"
                            )
                            shard_idx += 1
                            sample_in_shard = 0
                            if shard_idx >= args.max_shards + args.resume_from:
                                break
                            shard_path = os.path.join(
                                args.output_dir, f"shard-{shard_idx:06d}.tar"
                            )
                            sink = wds.TarWriter(shard_path)

            except Exception as e:
                skipped += 1
                if skipped % 100 == 0:
                    print(f"Warning: {skipped} examples skipped, last error: {e}")
                continue

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        sink.close()
        elapsed = time.time() - start_time
        print(f"\nDone: {global_count} samples in {shard_idx - args.resume_from + 1} shards")
        print(f"Skipped: {skipped} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
