"""
Inference entrypoint for legacy autoregressive generation and flow generation.
"""

from __future__ import annotations

import argparse
import os

import torch
import torchaudio

from .checkpoints import load_model_state_dict
from .dataset import (
    N_MELS,
    PREDICT_START_SEC,
    STRIDE_SAMPLES,
    TARGET_SR,
    WINDOW_SAMPLES,
    build_feature_tensor,
    build_mel_transform,
    normalize_mel_batch,
)
from .flow_model import LatentFlowMatcher, integrate_flow
from .latent_ae import SignalAutoencoder
from .model import OsuMapper
from .representation import DEFAULT_FRAME_RATE, NUM_SIGNAL_CHANNELS, decode_signal_to_osu, ms_to_frame
from .tokenizer import Residuals, detokenize_to_hitobjects, difficulty_to_bin, hitobjects_to_osu_lines


def parse_args():
    p = argparse.ArgumentParser(description="Generate .osu beatmap from audio")
    p.add_argument("--input", type=str, required=True, help="Path to input audio")
    p.add_argument("--output", type=str, default="", help="Output .osu path")
    p.add_argument("--checkpoint", type=str, default="/common/users/asj102/osu_project/models/best.pt")
    p.add_argument("--ae_checkpoint", type=str, default="")
    p.add_argument("--generation_mode", type=str, choices=["auto", "legacy", "flow"], default="auto")
    p.add_argument("--difficulty", type=float, default=4.0, help="Target star rating for conditioning")
    p.add_argument("--bpm", type=float, default=0.0, help="Song BPM for conditioning (0 = auto-estimate)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_tokens_per_window", type=int, default=512)
    p.add_argument("--ode_steps", type=int, default=32)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--full_song_context", action="store_true")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--decoder_layers", type=int, default=6)
    p.add_argument("--decoder_heads", type=int, default=8)
    p.add_argument("--decoder_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=96)
    p.add_argument("--ae_hidden_dim", type=int, default=256)
    p.add_argument("--flow_hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=256)
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
    if star < 3.0:
        return "Normal"
    if star < 4.5:
        return "Hard"
    if star < 6.0:
        return "Insane"
    if star < 7.5:
        return "Expert"
    return "Expert+"


def star_to_od_ar(star: float) -> tuple[float, float]:
    od = min(10, max(2, star * 1.3))
    ar = min(10, max(3, star * 1.4))
    return round(od, 1), round(ar, 1)


def estimate_bpm(waveform: torch.Tensor, sr: int) -> float:
    try:
        import librosa

        y = waveform.squeeze().cpu().numpy()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            return float(tempo[0])
        return float(tempo)
    except Exception:
        return 120.0


def _device_from_args(args) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def _load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    return waveform


def _resolve_generation_mode(args, checkpoint: dict) -> str:
    if args.generation_mode != "auto":
        return args.generation_mode
    stage = checkpoint.get("stage", "legacy")
    return "flow" if stage == "flow" else "legacy"


def _build_normalized_mel(waveform: torch.Tensor, mel_transform, device: torch.device) -> torch.Tensor:
    feature = build_feature_tensor(waveform.cpu(), mel_transform).unsqueeze(0)  # [1,1,C,T]
    feature = normalize_mel_batch(feature)
    return feature.to(device)


def _load_legacy_model(args, checkpoint: dict, device: torch.device) -> OsuMapper:
    model = OsuMapper(
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff=args.decoder_ff,
        dropout=args.dropout,
    ).to(device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_ae_and_flow(args, checkpoint: dict, device: torch.device):
    ae = SignalAutoencoder(hidden_dim=args.ae_hidden_dim, latent_dim=args.latent_dim).to(device)
    flow = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.flow_hidden_dim,
        cond_dim=args.cond_dim,
    ).to(device)
    ae_state = checkpoint.get("ae_state_dict")
    if ae_state is None:
        if not args.ae_checkpoint:
            raise ValueError("Flow inference requires --ae_checkpoint when the flow checkpoint does not contain AE weights.")
        ae_checkpoint = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
        ae_state = ae_checkpoint.get("model_state_dict", ae_checkpoint)
    load_model_state_dict(ae, ae_state, strict=True)
    load_model_state_dict(flow, checkpoint["model_state_dict"], strict=True)
    ae.eval()
    flow.eval()
    return ae, flow


@torch.no_grad()
def generate_legacy_objects(args, checkpoint: dict, waveform: torch.Tensor, bpm: float, device: torch.device) -> list[dict]:
    model = _load_legacy_model(args, checkpoint, device)
    mel_transform = build_mel_transform("cpu")
    diff_id = difficulty_to_bin(args.difficulty)
    diff_tensor = torch.tensor([diff_id], device=device)
    bpm_tensor = torch.tensor([[bpm]], device=device, dtype=torch.float32)

    all_objects = []
    start = 0
    ms_per_beat = 60000.0 / max(bpm, 1e-6)

    while start + WINDOW_SAMPLES <= waveform.shape[-1]:
        chunk = waveform[:, start:start + WINDOW_SAMPLES]
        mel = _build_normalized_mel(chunk, mel_transform, device)
        window_start_ms = (start / TARGET_SR) * 1000.0
        predict_start_ms = window_start_ms + PREDICT_START_SEC * 1000.0
        tokens, residuals_raw = model.generate(
            mel.squeeze(0),
            diff_tensor,
            bpm_tensor,
            max_len=args.max_tokens_per_window,
            temperature=args.temperature,
        )
        residuals = [Residuals(time_offset_ms=t, x_offset_px=x, y_offset_px=y) for t, x, y in residuals_raw]
        objects = detokenize_to_hitobjects(tokens, residuals, base_time_ms=predict_start_ms, ms_per_beat=ms_per_beat)
        all_objects.extend(objects)
        start += STRIDE_SAMPLES

    if start < waveform.shape[-1] and waveform.shape[-1] - start > WINDOW_SAMPLES // 4:
        remaining = waveform[:, start:]
        if remaining.shape[-1] < WINDOW_SAMPLES:
            remaining = torch.nn.functional.pad(remaining, (0, WINDOW_SAMPLES - remaining.shape[-1]))
        mel = _build_normalized_mel(remaining, mel_transform, device)
        window_start_ms = (start / TARGET_SR) * 1000.0
        tokens, residuals_raw = model.generate(
            mel.squeeze(0),
            diff_tensor,
            bpm_tensor,
            max_len=args.max_tokens_per_window,
            temperature=args.temperature,
        )
        residuals = [Residuals(time_offset_ms=t, x_offset_px=x, y_offset_px=y) for t, x, y in residuals_raw]
        objects = detokenize_to_hitobjects(tokens, residuals, base_time_ms=window_start_ms, ms_per_beat=ms_per_beat)
        end_ms = (waveform.shape[-1] / TARGET_SR) * 1000.0
        all_objects.extend([obj for obj in objects if obj["time"] <= end_ms])

    all_objects.sort(key=lambda obj: obj["time"])
    deduped = []
    for obj in all_objects:
        if deduped and abs(obj["time"] - deduped[-1]["time"]) < 5:
            continue
        deduped.append(obj)
    return deduped


@torch.no_grad()
def generate_flow_objects(args, checkpoint: dict, waveform: torch.Tensor, bpm: float, device: torch.device) -> list[dict]:
    ae, flow = _load_ae_and_flow(args, checkpoint, device)
    mel_transform = build_mel_transform("cpu")
    mel = _build_normalized_mel(waveform, mel_transform, device)
    difficulty_id = torch.tensor([difficulty_to_bin(args.difficulty)], device=device, dtype=torch.long)
    difficulty_value = torch.tensor([[args.difficulty]], device=device, dtype=torch.float32)
    bpm_tensor = torch.tensor([[bpm]], device=device, dtype=torch.float32)

    duration_ms = waveform.shape[-1] * 1000.0 / TARGET_SR
    signal_frames = max(1, ms_to_frame(duration_ms, DEFAULT_FRAME_RATE) + 1)
    dummy_signal = torch.zeros(1, NUM_SIGNAL_CHANNELS, signal_frames, device=device)
    latent_shape = tuple(ae.encode(dummy_signal).shape)
    generated_latent = integrate_flow(
        flow_model=flow,
        latent_shape=latent_shape,
        mel=mel,
        difficulty_id=difficulty_id,
        difficulty_value=difficulty_value,
        bpm=bpm_tensor,
        steps=max(args.ode_steps, 1),
        guidance_scale=args.guidance_scale,
        device=device,
    )
    generated_signal = ae.decode(generated_latent, output_len=signal_frames)[0]
    return decode_signal_to_osu(
        generated_signal,
        bpm=bpm,
        offset_ms=0.0,
        star_rating=args.difficulty,
        frame_rate=DEFAULT_FRAME_RATE,
    )


def generate_beatmap(args) -> str:
    device = _device_from_args(args)
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generation_mode = _resolve_generation_mode(args, checkpoint)
    print(f"Resolved generation_mode={generation_mode}")

    waveform = _load_audio(args.input)
    duration_sec = waveform.shape[-1] / TARGET_SR
    print(f"Audio duration: {duration_sec:.1f}s")

    if args.bpm > 0:
        bpm = args.bpm
        print(f"Using provided BPM: {bpm:.1f}")
    else:
        bpm = estimate_bpm(waveform, TARGET_SR)
        print(f"Auto-estimated BPM: {bpm:.1f}")

    if generation_mode == "legacy":
        objects = generate_legacy_objects(args, checkpoint, waveform, bpm, device)
    else:
        objects = generate_flow_objects(args, checkpoint, waveform, bpm, device)

    hitobject_lines = hitobjects_to_osu_lines(objects)
    title = os.path.splitext(os.path.basename(args.input))[0]
    diff_name = star_to_difficulty_name(args.difficulty)
    od, ar = star_to_od_ar(args.difficulty)
    beat_length = 60000.0 / max(bpm, 1e-6)

    osu_content = OSU_TEMPLATE.format(
        audio_filename=os.path.basename(args.input),
        title=title,
        difficulty_name=diff_name,
        od=od,
        ar=ar,
        beat_length=f"{beat_length:.12f}",
        hitobjects="\n".join(hitobject_lines),
    )

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + ".osu"
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(osu_content)

    print(f"Generated {len(objects)} hit objects")
    print(f"Saved beatmap to: {args.output}")
    return args.output


def main():
    args = parse_args()
    generate_beatmap(args)


if __name__ == "__main__":
    main()
