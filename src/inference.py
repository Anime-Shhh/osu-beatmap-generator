"""
Inference script: Load a trained OsuMapper checkpoint and generate
a .osu beatmap file from an input .mp3 file.

Usage:
    python -m src.inference \
        --input song.mp3 \
        --output generated.osu \
        --checkpoint /common/users/asj102/osu_project/models/best.pt \
        --difficulty 4.5
"""

import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torchaudio

from .model import OsuMapper
from .dataset import (
    build_mel_transform, compute_onset_strength, TARGET_SR,
    WINDOW_SAMPLES, STRIDE_SAMPLES, WINDOW_SEC, STRIDE_SEC,
    PREDICT_START_SEC, PREDICT_SEC, N_MELS,
)
from .tokenizer import (
    detokenize_to_hitobjects, hitobjects_to_osu_lines,
    Residuals, BOS, EOS, difficulty_to_bin,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate .osu beatmap from audio")
    p.add_argument("--input", type=str, required=True, help="Path to .mp3 file")
    p.add_argument("--output", type=str, default="", help="Output .osu path (default: input_name.osu)")
    p.add_argument("--checkpoint", type=str,
                    default="/common/users/asj102/osu_project/models/best.pt")
    p.add_argument("--difficulty", type=float, default=4.0,
                    help="Target star rating for conditioning (mapped to 5-bin scheme)")
    p.add_argument("--bpm", type=float, default=0.0,
                    help="Song BPM for conditioning (0 = auto-estimate)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_tokens_per_window", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")

    # Model architecture (must match training)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--decoder_layers", type=int, default=6)
    p.add_argument("--decoder_heads", type=int, default=8)
    p.add_argument("--decoder_ff", type=int, default=2048)
    return p.parse_args()


OSU_TEMPLATE = """osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 0
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:{title}
TitleUnicode:{title}
Artist:AI Generated
ArtistUnicode:AI Generated
Creator:OsuMapper AI
Version:{difficulty_name}
Source:
Tags:ai generated osumapper
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:{od}
ApproachRate:{ar}
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,{beat_length},4,2,0,70,1,0

[HitObjects]
{hitobjects}
"""


def star_to_difficulty_name(star: float) -> str:
    if star < 2.0:
        return "Easy"
    elif star < 3.0:
        return "Normal"
    elif star < 4.5:
        return "Hard"
    elif star < 6.0:
        return "Insane"
    elif star < 7.5:
        return "Expert"
    return "Expert+"


def star_to_od_ar(star: float) -> tuple[float, float]:
    """Heuristic mapping from star rating to OD/AR."""
    od = min(10, max(2, star * 1.3))
    ar = min(10, max(3, star * 1.4))
    return round(od, 1), round(ar, 1)


def estimate_bpm(waveform: torch.Tensor, sr: int) -> float:
    """Simple onset-based BPM estimation."""
    try:
        import librosa
        y = waveform.squeeze().numpy()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            return float(tempo[0])
        return float(tempo)
    except Exception:
        return 120.0


def generate_beatmap(args) -> str:
    """Main generation pipeline."""
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Loading model from: {args.checkpoint}")

    model = OsuMapper(
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff=args.decoder_ff,
    )

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint}, using random weights")

    model = model.to(device)
    model.eval()

    # Load audio
    print(f"Loading audio: {args.input}")
    waveform, sr = torchaudio.load(args.input)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

    num_samples = waveform.shape[-1]
    duration_sec = num_samples / TARGET_SR
    print(f"Audio duration: {duration_sec:.1f}s ({num_samples} samples at {TARGET_SR}Hz)")

    if args.bpm > 0:
        bpm = args.bpm
        print(f"Using provided BPM: {bpm:.1f}")
    else:
        bpm = estimate_bpm(waveform, TARGET_SR)
        print(f"Auto-estimated BPM: {bpm:.1f}")

    diff_id = difficulty_to_bin(args.difficulty)
    diff_tensor = torch.tensor([diff_id], device=device)
    bpm_tensor = torch.tensor([[bpm]], device=device, dtype=torch.float32)
    print(f"Conditioning: difficulty_bin={diff_id}, bpm={bpm:.1f}")

    mel_transform = build_mel_transform(str(device))

    all_objects = []
    window_count = 0
    start = 0

    print(f"Generating beatmap (difficulty={args.difficulty}, temp={args.temperature})...")

    ms_per_beat = 60000.0 / bpm

    while start + WINDOW_SAMPLES <= num_samples:
        chunk = waveform[:, start : start + WINDOW_SAMPLES].to(device)
        mel = mel_transform(chunk)  # [1, N_MELS, T]
        onset = compute_onset_strength(chunk.cpu(), mel.shape[-1])  # [1, T]
        mel = torch.cat([mel, onset.unsqueeze(0).to(device)], dim=1)  # [1, N_FEATURES, T]
        mel = torch.log(mel + 1e-7)

        window_start_ms = (start / TARGET_SR) * 1000.0
        predict_start_ms = window_start_ms + PREDICT_START_SEC * 1000.0

        tokens, residuals_raw = model.generate(
            mel, diff_tensor, bpm_tensor,
            max_len=args.max_tokens_per_window, temperature=args.temperature,
        )

        residuals = [
            Residuals(time_offset_ms=r[0], x_offset_px=r[1], y_offset_px=r[2])
            for r in residuals_raw
        ]

        objects = detokenize_to_hitobjects(
            tokens, residuals, base_time_ms=predict_start_ms, ms_per_beat=ms_per_beat,
        )
        all_objects.extend(objects)
        window_count += 1

        start += STRIDE_SAMPLES

    if start < num_samples and num_samples - start > WINDOW_SAMPLES // 4:
        remaining = waveform[:, start:].to(device)
        pad_len = WINDOW_SAMPLES - remaining.shape[-1]
        remaining = torch.nn.functional.pad(remaining, (0, pad_len))
        mel = mel_transform(remaining)
        onset = compute_onset_strength(remaining.cpu(), mel.shape[-1])
        mel = torch.cat([mel, onset.unsqueeze(0).to(device)], dim=1)
        mel = torch.log(mel + 1e-7)

        window_start_ms = (start / TARGET_SR) * 1000.0
        predict_start_ms = window_start_ms

        tokens, residuals_raw = model.generate(
            mel, diff_tensor, bpm_tensor,
            max_len=args.max_tokens_per_window, temperature=args.temperature,
        )
        residuals = [
            Residuals(time_offset_ms=r[0], x_offset_px=r[1], y_offset_px=r[2])
            for r in residuals_raw
        ]
        objects = detokenize_to_hitobjects(
            tokens, residuals, base_time_ms=predict_start_ms, ms_per_beat=ms_per_beat,
        )
        end_ms = (num_samples / TARGET_SR) * 1000.0
        objects = [o for o in objects if o["time"] <= end_ms]
        all_objects.extend(objects)

    # Deduplicate overlapping windows (remove objects too close in time)
    all_objects.sort(key=lambda o: o["time"])
    deduped = []
    for obj in all_objects:
        if deduped and abs(obj["time"] - deduped[-1]["time"]) < 5:
            continue
        deduped.append(obj)

    print(f"Generated {len(deduped)} hit objects from {window_count} windows")

    # Build .osu file
    hitobject_lines = hitobjects_to_osu_lines(deduped)
    title = os.path.splitext(os.path.basename(args.input))[0]
    diff_name = star_to_difficulty_name(args.difficulty)
    od, ar = star_to_od_ar(args.difficulty)
    beat_length = 60000.0 / bpm

    osu_content = OSU_TEMPLATE.format(
        audio_filename=os.path.basename(args.input),
        title=title,
        difficulty_name=diff_name,
        od=od,
        ar=ar,
        beat_length=f"{beat_length:.12f}",
        hitobjects="\n".join(hitobject_lines),
    )

    # Output
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + ".osu"

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(osu_content)

    print(f"Saved beatmap to: {args.output}")
    return args.output


def main():
    args = parse_args()
    generate_beatmap(args)


if __name__ == "__main__":
    main()
