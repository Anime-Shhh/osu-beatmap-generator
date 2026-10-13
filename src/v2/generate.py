"""Explicit v2 draft-generation CLI supporting raw audio or precomputed bundles.

This command accepts either a raw audio file (.mp3, .wav) or precomputed safe
feature bundles, computes/resolves timing hypotheses, and generates ranked-like
osu!standard .osu drafts with full provenance manifests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .clock import AudioClockResolver, ClockResolver, StaticClockResolver, TimingHypothesis
from .generation import (
    DEFAULT_AR,
    DEFAULT_CS,
    DEFAULT_HP,
    DEFAULT_OD,
    DEFAULT_SLIDER_MULTIPLIER,
    DEFAULT_SLIDER_TICK_RATE,
    GenerationRequest,
)
from .inference import (
    MAX_ABS_OFFSET_MS,
    MAX_SUPPORTED_BPM,
    MIN_SUPPORTED_BPM,
    extract_audio_features,
    generate_draft,
    load_audio,
    load_feature_bundle,
    load_v2_models,
)
from .mapspec import MapMetadata
from .evaluation import RosuDifficultyCalculator, StrictAuditPolicy


def _hypotheses(path: str | Path) -> tuple[TimingHypothesis, ...]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read clock hypotheses JSON: {exc}") from exc
    values = payload.get("hypotheses", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("clock hypotheses JSON must be a non-empty list or {hypotheses: [...]}")
    if any(not isinstance(value, Mapping) for value in values):
        raise ValueError("each clock hypothesis must be an object")
    return tuple(TimingHypothesis.from_dict(value) for value in values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one audited osu!standard v2 draft")
    parser.add_argument("--checkpoint", required=True, help="trained v2 checkpoint")

    # Audio input options (raw audio file or precomputed feature bundle)
    parser.add_argument("--audio", "--input", dest="audio", help="path to raw audio file (.mp3, .wav, etc.)")
    parser.add_argument("--feature-bundle", help="safe .pt bundle extracted from the input audio")

    # Clock / Timing options
    parser.add_argument("--clock-hypotheses", help="JSON from a pinned beat/downbeat resolver")
    parser.add_argument("--bpm", type=float, default=None, help="song BPM override (auto-estimate if omitted)")
    parser.add_argument("--offset", "--first-beat-ms", dest="offset", type=float, default=None, help="first beat offset in ms")

    # Target difficulty rating (mandatory)
    parser.add_argument(
        "--difficulty",
        "--target-star",
        dest="target_star",
        type=float,
        required=False,
        default=None,
        help="mandatory target star difficulty rating (e.g. 4.5)",
    )

    # Profile & map settings
    parser.add_argument("--ar", type=float, default=None, help="approach rate (0-10)")
    parser.add_argument("--cs", type=float, default=None, help="circle size (0-10)")
    parser.add_argument("--od", type=float, default=None, help="overall difficulty (0-10)")
    parser.add_argument("--hp", type=float, default=None, help="HP drain rate (0-10)")
    parser.add_argument("--slider-multiplier", type=float, default=None, help="slider velocity multiplier")
    parser.add_argument("--slider-tick-rate", type=float, default=None, help="slider tick rate")

    # Metadata overrides
    parser.add_argument(
        "--audio-filename",
        help="basename to reference in the output .osu (defaults to audio file basename)",
    )
    parser.add_argument("--title", default=None, help="song title metadata override")
    parser.add_argument("--artist", default=None, help="song artist metadata override")
    parser.add_argument("--creator", default=None, help="beatmap creator metadata override")
    parser.add_argument("--version", "--diff-name", dest="version", default=None, help="difficulty name / version override")

    # Output options
    parser.add_argument("--output-dir", help="directory to write packaged .osu and manifest")
    parser.add_argument("--output", "--output-osu", dest="output", help="explicit output .osu file path")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output files")

    # Generation & sampling controls
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--geometry-steps", type=int, default=32)
    parser.add_argument("--star-tolerance", type=float, default=0.5,
                        help="reject candidates farther than this from measured target stars")
    parser.add_argument("--minimum-start-gap-ms", type=float, default=10.0,
                        help="reject near-simultaneous generated starts; not a difficulty formula")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--style-cluster", default=None)

    args = parser.parse_args(argv)

    if not args.audio and not args.feature_bundle:
        parser.error("either --audio (path to raw audio) or --feature-bundle must be supplied")
    if args.audio and args.feature_bundle:
        parser.error("--audio and --feature-bundle are mutually exclusive")
    if args.feature_bundle and not args.clock_hypotheses:
        parser.error("--clock-hypotheses is required when --feature-bundle is used")
    if args.feature_bundle and not args.audio_filename:
        parser.error("--audio-filename is required when --feature-bundle is used")
    if args.target_star is None:
        parser.error("target difficulty rating is mandatory (pass --difficulty <float> or --target-star <float>)")
    if not math.isfinite(args.target_star) or args.target_star <= 0.0:
        parser.error("target difficulty rating must be finite and positive")
    if args.bpm is not None and (
        not math.isfinite(args.bpm) or not MIN_SUPPORTED_BPM <= args.bpm <= MAX_SUPPORTED_BPM
    ):
        parser.error(
            f"--bpm must be finite and within [{MIN_SUPPORTED_BPM:g}, {MAX_SUPPORTED_BPM:g}]"
        )
    if args.offset is not None and (
        not math.isfinite(args.offset) or abs(args.offset) > MAX_ABS_OFFSET_MS
    ):
        parser.error(f"--offset must be finite and within +/-{MAX_ABS_OFFSET_MS:g} ms")
    if not args.output_dir and not args.output:
        parser.error("either --output-dir or --output must be supplied")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    models = load_v2_models(args.checkpoint, device=device_name)
    calculator = RosuDifficultyCalculator()
    policy = StrictAuditPolicy(max_star_error=args.star_tolerance,
                               minimum_start_gap_ms=args.minimum_start_gap_ms)

    clock_resolver: ClockResolver
    if args.audio:
        waveform, sr = load_audio(args.audio, target_sr=24000)
        audio_name = args.audio_filename or Path(args.audio).name
        features, est_hypotheses = extract_audio_features(
            waveform,
            sr,
            models=models,
            bpm=args.bpm,
            offset_ms=args.offset,
        )
        if args.clock_hypotheses:
            clock_resolver = StaticClockResolver(_hypotheses(args.clock_hypotheses))
        else:
            clock_resolver = AudioClockResolver(est_hypotheses)
    else:
        audio_name = args.audio_filename
        features = load_feature_bundle(args.feature_bundle)
        clock_resolver = StaticClockResolver(_hypotheses(args.clock_hypotheses))

    star = float(args.target_star)
    metadata = MapMetadata(
        title=args.title or "Generated Draft",
        artist=args.artist or "Unknown Artist",
        creator=args.creator or "osu-beatmap-generator",
        version=args.version or f"v2 {star:.1f}★ Draft",
        tags="generated ranked-like draft human-review-required",
    )

    request = GenerationRequest(
        audio_filename=audio_name,
        target_star=args.target_star,
        ar=args.ar,
        cs=args.cs,
        od=args.od,
        hp=args.hp,
        slider_multiplier=args.slider_multiplier,
        slider_tick_rate=args.slider_tick_rate,
        style_cluster=args.style_cluster,
        seed=args.seed,
        candidate_count=args.candidate_count,
        feature_revision=features.feature_cache.pipeline_revision,
        model_revision=models.model_revision,
    )

    draft = generate_draft(
        models=models,
        features=features,
        clock_resolver=clock_resolver,
        request=request,
        temperature=args.temperature,
        geometry_steps=args.geometry_steps,
        metadata=metadata,
        calculator=calculator,
        audit_policy=policy,
    )

    if draft.selected_candidate is None:
        print(draft.result.to_json(), end="")
        return 2

    output_dir = args.output_dir
    osu_filename = "ranked-like-draft.osu"
    if args.output:
        out_path = Path(args.output)
        output_dir = output_dir or str(out_path.parent)
        osu_filename = out_path.name

    package = draft.package(
        output_dir,
        osu_filename=osu_filename,
        audio_source_path=args.audio,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "osu_path": str(package.osu_path),
                "manifest_path": str(package.manifest_path),
                "audio_path": str(package.audio_path) if package.audio_path is not None else None,
                "selected_candidate_id": draft.result.selected_candidate_id,
                "ranked_status": "draft_only_human_mapper_review_required",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - externally invoked, never auto-run.
    raise SystemExit(main())
