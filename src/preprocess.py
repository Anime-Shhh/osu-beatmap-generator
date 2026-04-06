"""
Preprocessing pipeline: stream from HuggingFace, convert to mel/tokens/signal,
and write WebDataset shards to shared storage.

Legacy behavior is preserved by default. Full-song samples are opt-in.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time

import torch
import webdataset as wds

from .dataset import (
    DEFAULT_FRAME_RATE,
    FULL_SONG_SAMPLE,
    WINDOW_SAMPLE,
    WINDOW_SAMPLES,
    build_mel_transform,
    extract_full_song_sample,
    extract_windows,
    infer_beatmap_status,
    is_ranked_status,
    load_audio_from_bytes,
)
from .tokenizer import Residuals, difficulty_to_bin, parse_osu_bpm


class FatalPreprocessError(RuntimeError):
    """Configuration or schema problems that should stop preprocessing."""


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess osu! beatmaps to cached shards")
    p.add_argument("--output_dir", type=str, default="/common/users/asj102/osu_project/data/shards")
    p.add_argument("--max_shards", type=int, default=500)
    p.add_argument("--examples_per_shard", type=int, default=200)
    p.add_argument("--min_tokens", type=int, default=4, help="Skip samples with fewer tokens than this")
    p.add_argument("--mode_filter", type=int, default=0, help="osu! mode to keep (0=standard)")
    p.add_argument("--resume_from", type=int, default=0, help="Resume from this shard number")
    p.add_argument("--include_full_song", action="store_true", help="Also cache full-song signal samples")
    p.add_argument("--no_legacy_windows", action="store_true", help="Skip the legacy 6-second window samples")
    p.add_argument("--quality_filter", type=str, choices=["all", "ranked_only"], default="all")
    p.add_argument("--frame_rate", type=float, default=DEFAULT_FRAME_RATE)
    return p.parse_args()


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
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


def serialize_meta(meta: dict) -> bytes:
    return json.dumps(meta, sort_keys=True).encode("utf-8")


def build_meta(sample: dict) -> dict:
    return {
        "sample_type": sample.get("sample_type", WINDOW_SAMPLE),
        "difficulty_id": int(sample.get("difficulty_id", 2)),
        "bpm": float(sample.get("bpm", 120.0)),
        "star_rating": float(sample.get("star_rating", 4.0)),
        "beatmap_status": sample.get("beatmap_status"),
        "sample_rate": 22050,
        "frame_rate": float(sample.get("frame_rate", DEFAULT_FRAME_RATE)),
        "mel_frames": int(sample["mel"].shape[-1]),
        "signal_frames": int(sample["signal"].shape[-1]) if sample.get("signal") is not None else 0,
        "token_count": int(len(sample.get("tokens", []))),
        "target_start_ms": float(sample.get("target_start_ms", 0.0)),
    }


def write_sample(sink, shard_idx: int, sample_idx: int, sample: dict) -> None:
    key = f"{shard_idx:06d}_{sample_idx:06d}_{sample.get('sample_type', 'sample')}"
    sink.write({
        "__key__": key,
        "mel.pt": serialize_tensor(sample["mel"]),
        "tokens.pt": serialize_tokens(
            sample["tokens"],
            sample["residuals"],
            int(sample["difficulty_id"]),
            float(sample["bpm"]),
        ),
        "signal.pt": serialize_tensor(sample["signal"]),
        "meta.json": serialize_meta(build_meta(sample)),
    })


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

    from datasets import Audio, load_dataset

    ds = load_dataset("project-riz/osu-beatmaps", "original", split="train", streaming=True)
    try:
        ds = ds.cast_column("mp3", Audio(decode=False))
    except Exception:
        pass

    mel_transform = build_mel_transform("cpu")

    shard_idx = args.resume_from
    sample_in_shard = 0
    global_count = 0
    skipped = 0
    streamed_examples = 0
    beatmaps_seen = 0
    beatmaps_kept = 0
    start_time = time.time()
    last_progress_log = start_time
    heartbeat_sec = 60.0

    shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:06d}.tar")
    sink = wds.TarWriter(shard_path)

    print(f"Starting preprocessing -> {args.output_dir}", flush=True)
    print(f"Max shards: {args.max_shards}, examples/shard: {args.examples_per_shard}", flush=True)
    print(
        f"Options: include_full_song={args.include_full_song} "
        f"legacy_windows={not args.no_legacy_windows} quality_filter={args.quality_filter}"
        ,
        flush=True,
    )
    print("Loading HuggingFace streaming dataset...", flush=True)

    try:
        for example in ds:
            if shard_idx >= args.max_shards + args.resume_from:
                break

            streamed_examples += 1

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

                try:
                    waveform, _ = load_audio_from_bytes(audio_bytes)
                except Exception as exc:
                    skipped += 1
                    print(
                        f"Skipping example {streamed_examples}: audio decode failed ({exc})",
                        flush=True,
                    )
                    continue
                if waveform.shape[-1] < WINDOW_SAMPLES and not args.include_full_song:
                    skipped += 1
                    continue

                beatmaps = json_data.get("beatmaps", [])
                for beatmap in beatmaps:
                    beatmaps_seen += 1
                    if shard_idx >= args.max_shards + args.resume_from:
                        break

                    osu_content = beatmap.get("content", "")
                    if not osu_content:
                        continue

                    if int(beatmap.get("mode", 0)) != args.mode_filter:
                        continue

                    beatmap_status = infer_beatmap_status(beatmap)
                    if args.quality_filter == "ranked_only":
                        if beatmap_status is None:
                            raise FatalPreprocessError(
                                "ranked_only requested, but beatmap status metadata is unavailable "
                                "during preprocessing."
                            )
                        if not is_ranked_status(beatmap_status):
                            continue

                    star_rating = float(beatmap.get("difficultyrating", 4.0))
                    difficulty_id = difficulty_to_bin(star_rating)
                    bpm = parse_osu_bpm(osu_content)
                    beatmaps_kept += 1

                    samples = []
                    if not args.no_legacy_windows and waveform.shape[-1] >= WINDOW_SAMPLES:
                        samples.extend(extract_windows(
                            waveform=waveform,
                            osu_content=osu_content,
                            mel_transform=mel_transform,
                            difficulty_id=difficulty_id,
                            bpm=bpm,
                            star_rating=star_rating,
                            beatmap_status=beatmap_status,
                            frame_rate=args.frame_rate,
                        ))
                    if args.include_full_song:
                        samples.append(extract_full_song_sample(
                            waveform=waveform,
                            osu_content=osu_content,
                            mel_transform=mel_transform,
                            difficulty_id=difficulty_id,
                            bpm=bpm,
                            star_rating=star_rating,
                            beatmap_status=beatmap_status,
                            frame_rate=args.frame_rate,
                        ))

                    for sample in samples:
                        if len(sample["tokens"]) < args.min_tokens:
                            continue
                        write_sample(sink, shard_idx, sample_in_shard, sample)
                        sample_in_shard += 1
                        global_count += 1

                        if sample_in_shard >= args.examples_per_shard:
                            sink.close()
                            elapsed = time.time() - start_time
                            rate = global_count / elapsed if elapsed > 0 else 0.0
                            print(
                                f"Shard {shard_idx:06d} done | total={global_count} | "
                                f"skipped={skipped} | rate={rate:.1f} samples/s"
                                ,
                                flush=True,
                            )
                            shard_idx += 1
                            sample_in_shard = 0
                            if shard_idx >= args.max_shards + args.resume_from:
                                break
                            shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:06d}.tar")
                            sink = wds.TarWriter(shard_path)

                now = time.time()
                if now - last_progress_log >= heartbeat_sec:
                    elapsed = now - start_time
                    rate = global_count / elapsed if elapsed > 0 else 0.0
                    print(
                        "Heartbeat | streamed_examples={examples} beatmaps_seen={seen} "
                        "beatmaps_kept={kept} current_shard={shard:06d} shard_fill={fill}/{limit} "
                        "samples_written={samples} skipped={skipped} rate={rate:.2f} samples/s".format(
                            examples=streamed_examples,
                            seen=beatmaps_seen,
                            kept=beatmaps_kept,
                            shard=shard_idx,
                            fill=sample_in_shard,
                            limit=args.examples_per_shard,
                            samples=global_count,
                            skipped=skipped,
                            rate=rate,
                        ),
                        flush=True,
                    )
                    last_progress_log = now

            except FatalPreprocessError:
                raise
            except Exception as exc:
                skipped += 1
                if skipped % 100 == 0:
                    print(f"Warning: {skipped} examples skipped, last error: {exc}", flush=True)
                continue

    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
    finally:
        sink.close()
        elapsed = time.time() - start_time
        completed_shards = shard_idx - args.resume_from + (1 if sample_in_shard > 0 else 0)
        print(f"\nDone: {global_count} samples in {completed_shards} shards", flush=True)
        print(f"Skipped: {skipped} | Time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
