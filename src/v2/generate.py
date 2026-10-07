"""Explicit v2 draft-generation CLI using precomputed audio feature bundles.

It intentionally does not download MERT/Beat This! weights or submit any job.
Feature extraction and clock estimation are separate, pinned preprocessing
steps; this command makes that provenance visible in the generated manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .clock import StaticClockResolver, TimingHypothesis
from .generation import GenerationRequest
from .inference import generate_draft, load_feature_bundle, load_v2_models


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
    parser.add_argument("--feature-bundle", required=True, help="safe .pt bundle extracted from the input audio")
    parser.add_argument("--clock-hypotheses", required=True, help="JSON from a pinned beat/downbeat resolver")
    parser.add_argument("--audio-filename", required=True, help="basename to reference in the output .osu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-star", type=float, default=4.5)
    parser.add_argument("--ar", type=float, default=9.0)
    parser.add_argument("--cs", type=float, default=4.0)
    parser.add_argument("--od", type=float, default=8.0)
    parser.add_argument("--hp", type=float, default=6.0)
    parser.add_argument("--slider-multiplier", type=float, default=1.4)
    parser.add_argument("--slider-tick-rate", type=float, default=1.0)
    parser.add_argument("--style-cluster")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--geometry-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features = load_feature_bundle(args.feature_bundle)
    models = load_v2_models(args.checkpoint, device=args.device)
    request = GenerationRequest(
        audio_filename=args.audio_filename,
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
        clock_resolver=StaticClockResolver(_hypotheses(args.clock_hypotheses)),
        request=request,
        temperature=args.temperature,
        geometry_steps=args.geometry_steps,
    )
    if draft.selected_candidate is None:
        print(draft.result.to_json(), end="")
        return 2
    package = draft.package(args.output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "osu_path": str(package.osu_path),
                "manifest_path": str(package.manifest_path),
                "selected_candidate_id": draft.result.selected_candidate_id,
                "ranked_status": "draft_only_human_mapper_review_required",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - externally invoked, never auto-run.
    raise SystemExit(main())
