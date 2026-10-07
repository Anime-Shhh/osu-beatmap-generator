"""Public, JSON-safe contracts for audio-only v2 generation.

The generator is intentionally capable of producing many plausible maps for a
single song.  A request and its result therefore need to record more than a
single ``.osu`` path: they preserve the user-visible controls, candidate
selection decision, validation outcome, and model/cache provenance.  These
types deliberately do *not* retain an absolute input path.  A manifest should
be safe to move with a generated map without leaking workstation layout.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Mapping, Optional, Sequence


GENERATION_REQUEST_SCHEMA = "osu-beatmap-generator/v2-generation-request/1"
GENERATION_RESULT_SCHEMA = "osu-beatmap-generator/v2-generation-result/1"
DEFAULT_TARGET_STAR = 4.5
DEFAULT_AR = 9.0
DEFAULT_CS = 4.0
DEFAULT_OD = 8.0
DEFAULT_HP = 6.0
DEFAULT_SLIDER_MULTIPLIER = 1.4
DEFAULT_SLIDER_TICK_RATE = 1.0


def _json_value(value: Any, *, field_name: str) -> Any:
    """Return a recursive JSON value or raise a useful manifest error.

    Silent ``default=str`` serialization would turn a model object or a local
    path into opaque, non-reproducible metadata.  Generation manifests are a
    reproducibility boundary, so only normal JSON data belongs in them.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, field_name=field_name) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, field_name=field_name) for item in value]
    raise TypeError(f"{field_name} must be JSON-compatible, got {type(value).__name__}")


def public_audio_filename(value: str) -> str:
    """Validate the audio name stored in an output ``.osu`` and manifest.

    osu! needs a filename, not an absolute local path.  Rejecting traversal and
    nested paths here prevents an otherwise innocuous manifest field becoming
    an output-packaging path primitive.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio_filename must be a non-empty filename")
    normalized = value.replace("\\", "/")
    path = PurePath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError("audio_filename must be a basename, not a path")
    if "\x00" in value:
        raise ValueError("audio_filename must not contain NUL")
    return path.name


def _optional_rate(value: Optional[float], *, name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 10.0:
        raise ValueError(f"{name} must be finite and within [0, 10]")
    return value


def _optional_star(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("target_star must be finite and positive when supplied")
    return value


def _optional_positive_setting(value: Optional[float], *, name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.05 <= value <= 10.0:
        raise ValueError(f"{name} must be finite and within [0.05, 10]")
    return value


@dataclass(frozen=True)
class GenerationRequest:
    """An audio-only generation request with optional explicit controls.

    ``audio_filename`` is the only required audio field.  The actual input
    location is intentionally kept by the caller, while an optional decoded
    audio hash can establish provenance without exposing it. Omitting profile
    controls is valid: a deterministic moderate default is recorded in the
    request (4.5★, AR9, CS4, OD8, HP6). A learned prior must not silently alter
    bare-audio behavior between releases.
    """

    audio_filename: str
    audio_sha256: Optional[str] = None
    target_star: Optional[float] = DEFAULT_TARGET_STAR
    ar: Optional[float] = DEFAULT_AR
    cs: Optional[float] = DEFAULT_CS
    od: Optional[float] = DEFAULT_OD
    hp: Optional[float] = DEFAULT_HP
    slider_multiplier: Optional[float] = DEFAULT_SLIDER_MULTIPLIER
    slider_tick_rate: Optional[float] = DEFAULT_SLIDER_TICK_RATE
    style_cluster: Optional[str] = None
    style_seed: Optional[int] = None
    candidate_count: int = 4
    seed: Optional[int] = None
    feature_revision: Optional[str] = None
    model_revision: Optional[str] = None
    extra_controls: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GENERATION_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_filename", public_audio_filename(self.audio_filename))
        if self.audio_sha256 is not None:
            digest = self.audio_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("audio_sha256 must be a 64-character lowercase/uppercase SHA-256 hex digest")
            object.__setattr__(self, "audio_sha256", digest)
        object.__setattr__(self, "target_star", _optional_star(self.target_star))
        for name in ("ar", "cs", "od", "hp"):
            object.__setattr__(self, name, _optional_rate(getattr(self, name), name=name))
        for name in ("slider_multiplier", "slider_tick_rate"):
            object.__setattr__(
                self,
                name,
                _optional_positive_setting(getattr(self, name), name=name),
            )
        if self.candidate_count < 1 or self.candidate_count > 64:
            raise ValueError("candidate_count must be between 1 and 64")
        if self.style_cluster is not None and not self.style_cluster.strip():
            raise ValueError("style_cluster must be non-empty when supplied")
        if self.schema_version != GENERATION_REQUEST_SCHEMA:
            raise ValueError(f"unsupported request schema {self.schema_version!r}")
        _json_value(self.extra_controls, field_name="extra_controls")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audio_filename": self.audio_filename,
            "audio_sha256": self.audio_sha256,
            "target_star": self.target_star,
            "settings": {
                "ar": self.ar,
                "cs": self.cs,
                "od": self.od,
                "hp": self.hp,
                "slider_multiplier": self.slider_multiplier,
                "slider_tick_rate": self.slider_tick_rate,
            },
            "style_cluster": self.style_cluster,
            "style_seed": self.style_seed,
            "candidate_count": self.candidate_count,
            "seed": self.seed,
            "feature_revision": self.feature_revision,
            "model_revision": self.model_revision,
            "extra_controls": _json_value(self.extra_controls, field_name="extra_controls"),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationRequest":
        if payload.get("schema_version") != GENERATION_REQUEST_SCHEMA:
            raise ValueError("unsupported or missing generation request schema")
        settings = payload.get("settings", {})
        if not isinstance(settings, Mapping):
            raise ValueError("request settings must be an object")
        return cls(
            audio_filename=str(payload.get("audio_filename", "")),
            audio_sha256=payload.get("audio_sha256"),
            target_star=payload.get("target_star", DEFAULT_TARGET_STAR),
            ar=settings.get("ar", DEFAULT_AR),
            cs=settings.get("cs", DEFAULT_CS),
            od=settings.get("od", DEFAULT_OD),
            hp=settings.get("hp", DEFAULT_HP),
            slider_multiplier=settings.get("slider_multiplier", DEFAULT_SLIDER_MULTIPLIER),
            slider_tick_rate=settings.get("slider_tick_rate", DEFAULT_SLIDER_TICK_RATE),
            style_cluster=payload.get("style_cluster"),
            style_seed=payload.get("style_seed"),
            candidate_count=int(payload.get("candidate_count", 4)),
            seed=payload.get("seed"),
            feature_revision=payload.get("feature_revision"),
            model_revision=payload.get("model_revision"),
            extra_controls=payload.get("extra_controls", {}),
        )


@dataclass(frozen=True)
class CandidateManifest:
    """Serializable audit facts for one sampled candidate, not the map itself."""

    candidate_id: str
    seed: Optional[int]
    valid: bool
    selected: bool = False
    score: Optional[float] = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    difficulty: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or any(char.isspace() for char in self.candidate_id):
            raise ValueError("candidate_id must be a non-empty whitespace-free identifier")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("candidate score must be finite when supplied")
        _json_value(self.metrics, field_name="candidate metrics")
        _json_value(self.difficulty, field_name="candidate difficulty")
        _json_value(self.provenance, field_name="candidate provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "valid": self.valid,
            "selected": self.selected,
            "score": self.score,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": _json_value(self.metrics, field_name="candidate metrics"),
            "difficulty": _json_value(self.difficulty, field_name="candidate difficulty"),
            "provenance": _json_value(self.provenance, field_name="candidate provenance"),
        }


@dataclass(frozen=True)
class GenerationResult:
    """Selected candidate and audit log that accompany a generated draft map."""

    request: GenerationRequest
    architecture_id: str
    model_revision: str
    candidates: tuple[CandidateManifest, ...]
    selected_candidate_id: Optional[str]
    warnings: tuple[str, ...] = ()
    schema_version: str = GENERATION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_RESULT_SCHEMA:
            raise ValueError(f"unsupported result schema {self.schema_version!r}")
        if not self.architecture_id.strip():
            raise ValueError("architecture_id must be non-empty")
        if not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if not self.candidates:
            raise ValueError("generation results must record at least one candidate")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        selected = [candidate for candidate in self.candidates if candidate.selected]
        if self.selected_candidate_id is None:
            if selected:
                raise ValueError("no candidate ID selected but a candidate is marked selected")
        else:
            if self.selected_candidate_id not in ids:
                raise ValueError("selected_candidate_id is not present in candidates")
            if len(selected) != 1 or selected[0].candidate_id != self.selected_candidate_id:
                raise ValueError("exactly the selected candidate must have selected=True")
            if not selected[0].valid:
                raise ValueError("an invalid candidate cannot be selected for packaging")

    @property
    def selected_candidate(self) -> Optional[CandidateManifest]:
        for candidate in self.candidates:
            if candidate.candidate_id == self.selected_candidate_id:
                return candidate
        return None

    def to_dict(self, *, artifacts: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "architecture_id": self.architecture_id,
            "model_revision": self.model_revision,
            "selected_candidate_id": self.selected_candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "ranked_status": "draft_only_human_mapper_review_required",
        }
        if artifacts is not None:
            result["artifacts"] = _json_value(artifacts, field_name="artifacts")
        return result

    def to_json(self, *, artifacts: Optional[Mapping[str, Any]] = None) -> str:
        return json.dumps(self.to_dict(artifacts=artifacts), indent=2, sort_keys=True) + "\n"
