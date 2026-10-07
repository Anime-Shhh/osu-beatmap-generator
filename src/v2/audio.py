"""CPU-only contracts for canonical audio and timestamped v2 feature caches.

The v2 planner must never infer temporal alignment from tensor length alone.
Every cached stream therefore has an independently stored timestamp vector and
an immutable manifest tied to canonical decoded PCM.  This module deliberately
contains no decoder, model download, torch import, or GPU behavior: extractors
can be run elsewhere while the cache format remains auditable on a login node.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Literal, Mapping, Sequence

from .data import SourceIdentity


CANONICAL_SAMPLE_RATE_HZ = 24_000
CANONICAL_CHANNELS = 1
CACHE_SCHEMA_VERSION = 1
TIMESTAMP_REFERENCES = frozenset({"start", "center", "end", "explicit"})
OUT_OF_RANGE_POLICIES = frozenset({"clamp", "mask", "error"})


def pcm_sha256(pcm_bytes: bytes | bytearray | memoryview) -> str:
    """Hash the exact canonical PCM payload used to compute cache features."""

    return hashlib.sha256(bytes(pcm_bytes)).hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")


def _validate_relative_cache_path(value: str, field_name: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"{field_name} must be a non-empty POSIX relative path")


def _require_finite_positive(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")


@dataclass(frozen=True)
class CanonicalAudioSpec:
    """The exact decode contract used before feature extraction.

    The decoder implementation may change only with a new ``decoder_revision``
    and cache rebuild.  Keeping the PCM sample format in the identity prevents
    a byte digest from being compared across incompatible sample encodings.
    """

    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ
    channels: int = CANONICAL_CHANNELS
    sample_format: Literal["pcm_f32le", "pcm_s16le"] = "pcm_f32le"
    downmix: Literal["mean"] = "mean"
    normalization: Literal["none"] = "none"
    decoder_revision: str = "v1"

    def __post_init__(self) -> None:
        if self.sample_rate_hz != CANONICAL_SAMPLE_RATE_HZ:
            raise ValueError(
                f"v2 canonical audio must use {CANONICAL_SAMPLE_RATE_HZ} Hz"
            )
        if self.channels != CANONICAL_CHANNELS:
            raise ValueError("v2 canonical audio must be mono")
        if self.sample_format not in {"pcm_f32le", "pcm_s16le"}:
            raise ValueError("sample_format must be pcm_f32le or pcm_s16le")
        if self.downmix != "mean":
            raise ValueError("v2 canonical audio must use mean mono downmixing")
        if self.normalization != "none":
            raise ValueError("v2 canonical audio must not apply amplitude normalization")
        if not self.decoder_revision.strip():
            raise ValueError("decoder_revision must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "sample_format": self.sample_format,
            "downmix": self.downmix,
            "normalization": self.normalization,
            "decoder_revision": self.decoder_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalAudioSpec":
        return cls(
            sample_rate_hz=int(payload.get("sample_rate_hz", CANONICAL_SAMPLE_RATE_HZ)),
            channels=int(payload.get("channels", CANONICAL_CHANNELS)),
            sample_format=str(payload.get("sample_format", "pcm_f32le")),  # type: ignore[arg-type]
            downmix=str(payload.get("downmix", "mean")),  # type: ignore[arg-type]
            normalization=str(payload.get("normalization", "none")),  # type: ignore[arg-type]
            decoder_revision=str(payload.get("decoder_revision", "v1")),
        )


@dataclass(frozen=True)
class CanonicalAudioManifest:
    """Identity and duration of the decoded audio behind a feature cache."""

    source: SourceIdentity
    pcm_sha256: str
    frame_count: int
    spec: CanonicalAudioSpec = field(default_factory=CanonicalAudioSpec)

    def __post_init__(self) -> None:
        _validate_sha256(self.pcm_sha256, "pcm_sha256")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if (
            self.source.canonical_audio_sha256 is not None
            and self.source.canonical_audio_sha256 != self.pcm_sha256
        ):
            raise ValueError(
                "source canonical_audio_sha256 does not match this canonical PCM payload"
            )

    @property
    def duration_ms(self) -> float:
        return self.frame_count * 1000.0 / self.spec.sample_rate_hz

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "pcm_sha256": self.pcm_sha256,
            "frame_count": self.frame_count,
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalAudioManifest":
        source = payload.get("source")
        spec = payload.get("spec", {})
        if not isinstance(source, Mapping) or not isinstance(spec, Mapping):
            raise ValueError("canonical audio manifest requires source and spec mappings")
        return cls(
            source=SourceIdentity.from_dict(source),
            pcm_sha256=str(payload["pcm_sha256"]),
            frame_count=int(payload["frame_count"]),
            spec=CanonicalAudioSpec.from_dict(spec),
        )


@dataclass(frozen=True)
class FeatureStreamManifest:
    """Provenance for one timestamped frame-aligned feature array.

    ``timestamps_path`` is mandatory even for nominally regular streams.  Frame
    centers can shift between extractors and library versions; explicitly
    persisted timestamps prevent a later consumer from silently assuming that
    25 Hz embeddings begin at the same instant as 100 Hz mel frames.
    """

    name: str
    extractor: str
    extractor_revision: str
    feature_dim: int
    frame_count: int
    dtype: str
    feature_path: str
    feature_sha256: str
    timestamps_path: str
    timestamps_sha256: str
    timestamp_reference: str = "explicit"
    nominal_frame_rate_hz: float | None = None
    storage_format: str = "npy"

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("feature stream name must be non-empty and whitespace-free")
        if not self.extractor.strip() or not self.extractor_revision.strip():
            raise ValueError("extractor and extractor_revision must be non-empty")
        if self.feature_dim <= 0 or self.frame_count <= 0:
            raise ValueError("feature_dim and frame_count must be positive")
        if not self.dtype.strip() or not self.storage_format.strip():
            raise ValueError("dtype and storage_format must be non-empty")
        _validate_relative_cache_path(self.feature_path, "feature_path")
        _validate_relative_cache_path(self.timestamps_path, "timestamps_path")
        _validate_sha256(self.feature_sha256, "feature_sha256")
        _validate_sha256(self.timestamps_sha256, "timestamps_sha256")
        if self.timestamp_reference not in TIMESTAMP_REFERENCES:
            raise ValueError(
                f"timestamp_reference must be one of {sorted(TIMESTAMP_REFERENCES)}"
            )
        if self.nominal_frame_rate_hz is not None:
            _require_finite_positive(self.nominal_frame_rate_hz, "nominal_frame_rate_hz")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "extractor": self.extractor,
            "extractor_revision": self.extractor_revision,
            "feature_dim": self.feature_dim,
            "frame_count": self.frame_count,
            "dtype": self.dtype,
            "feature_path": self.feature_path,
            "feature_sha256": self.feature_sha256,
            "timestamps_path": self.timestamps_path,
            "timestamps_sha256": self.timestamps_sha256,
            "timestamp_reference": self.timestamp_reference,
            "nominal_frame_rate_hz": self.nominal_frame_rate_hz,
            "storage_format": self.storage_format,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureStreamManifest":
        return cls(
            name=str(payload["name"]),
            extractor=str(payload["extractor"]),
            extractor_revision=str(payload["extractor_revision"]),
            feature_dim=int(payload["feature_dim"]),
            frame_count=int(payload["frame_count"]),
            dtype=str(payload["dtype"]),
            feature_path=str(payload["feature_path"]),
            feature_sha256=str(payload["feature_sha256"]),
            timestamps_path=str(payload["timestamps_path"]),
            timestamps_sha256=str(payload["timestamps_sha256"]),
            timestamp_reference=str(payload.get("timestamp_reference", "explicit")),
            nominal_frame_rate_hz=(
                None
                if payload.get("nominal_frame_rate_hz") is None
                else float(payload["nominal_frame_rate_hz"])
            ),
            storage_format=str(payload.get("storage_format", "npy")),
        )


@dataclass(frozen=True)
class FeatureCacheManifest:
    """Portable JSON manifest for feature files belonging to one song/audio."""

    canonical_audio: CanonicalAudioManifest
    streams: tuple[FeatureStreamManifest, ...]
    pipeline_revision: str
    schema_version: int = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported cache schema {self.schema_version}; expected {CACHE_SCHEMA_VERSION}"
            )
        if not self.pipeline_revision.strip():
            raise ValueError("pipeline_revision must be non-empty")
        if not self.streams:
            raise ValueError("a feature cache must contain at least one stream")
        names = [stream.name for stream in self.streams]
        if len(names) != len(set(names)):
            raise ValueError("feature stream names must be unique")

    @property
    def cache_key(self) -> str:
        """Per-song content key independent of where cache files are stored."""

        payload = {
            "schema_version": self.schema_version,
            "pipeline_revision": self.pipeline_revision,
            "canonical_audio": {
                "pcm_sha256": self.canonical_audio.pcm_sha256,
                "frame_count": self.canonical_audio.frame_count,
                "spec": self.canonical_audio.spec.to_dict(),
            },
            "streams": [
                {
                    "name": stream.name,
                    "extractor": stream.extractor,
                    "extractor_revision": stream.extractor_revision,
                    "feature_dim": stream.feature_dim,
                    "frame_count": stream.frame_count,
                    "dtype": stream.dtype,
                    "feature_sha256": stream.feature_sha256,
                    "timestamps_sha256": stream.timestamps_sha256,
                    "timestamp_reference": stream.timestamp_reference,
                    "nominal_frame_rate_hz": stream.nominal_frame_rate_hz,
                }
                for stream in sorted(self.streams, key=lambda stream: stream.name)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"osu-v2-feature-cache\0" + encoded).hexdigest()

    @property
    def pipeline_key(self) -> str:
        """Shared extractor/projection identity suitable for checkpoints.

        ``cache_key`` deliberately includes canonical PCM and feature content,
        so it is different for every song. A checkpoint must instead bind to a
        pipeline identity: exact audio contract plus extractor/configuration,
        but not song-local frame counts, file locations, or hashes.
        """

        payload = {
            "schema_version": self.schema_version,
            "pipeline_revision": self.pipeline_revision,
            "canonical_audio_spec": self.canonical_audio.spec.to_dict(),
            "streams": [
                {
                    "name": stream.name,
                    "extractor": stream.extractor,
                    "extractor_revision": stream.extractor_revision,
                    "feature_dim": stream.feature_dim,
                    "dtype": stream.dtype,
                    "timestamp_reference": stream.timestamp_reference,
                    "nominal_frame_rate_hz": stream.nominal_frame_rate_hz,
                    "storage_format": stream.storage_format,
                }
                for stream in sorted(self.streams, key=lambda stream: stream.name)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"osu-v2-feature-pipeline\0" + encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_revision": self.pipeline_revision,
            "canonical_audio": self.canonical_audio.to_dict(),
            "streams": [stream.to_dict() for stream in self.streams],
            "cache_key": self.cache_key,
            "pipeline_key": self.pipeline_key,
        }

    def to_json(self) -> str:
        """Return canonical JSON; callers own atomic file writes."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureCacheManifest":
        canonical_audio = payload.get("canonical_audio")
        streams = payload.get("streams")
        if not isinstance(canonical_audio, Mapping) or not isinstance(streams, list):
            raise ValueError("feature cache manifest requires canonical_audio and streams")
        if any(not isinstance(stream, Mapping) for stream in streams):
            raise ValueError("every feature cache stream must be an object")
        result = cls(
            schema_version=int(payload.get("schema_version", CACHE_SCHEMA_VERSION)),
            pipeline_revision=str(payload["pipeline_revision"]),
            canonical_audio=CanonicalAudioManifest.from_dict(canonical_audio),
            streams=tuple(
                FeatureStreamManifest.from_dict(stream)
                for stream in streams
                if isinstance(stream, Mapping)
            ),
        )
        persisted_key = payload.get("cache_key")
        if persisted_key is not None and persisted_key != result.cache_key:
            raise ValueError("feature cache manifest cache_key does not match its contents")
        persisted_pipeline_key = payload.get("pipeline_key")
        if persisted_pipeline_key is not None and persisted_pipeline_key != result.pipeline_key:
            raise ValueError("feature cache manifest pipeline_key does not match its contents")
        return result

    @classmethod
    def from_json(cls, payload: str) -> "FeatureCacheManifest":
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("feature cache manifest JSON must encode an object")
        return cls.from_dict(decoded)


def validate_timestamps_ms(
    timestamps_ms: Iterable[float],
    *,
    require_nonempty: bool = True,
    strictly_increasing: bool = True,
) -> tuple[float, ...]:
    """Materialize finite timestamps, optionally requiring strict ordering."""

    values = tuple(float(value) for value in timestamps_ms)
    if require_nonempty and not values:
        raise ValueError("timestamps must be non-empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("timestamps must be finite")
    if strictly_increasing and any(
        current <= previous for previous, current in zip(values, values[1:])
    ):
        raise ValueError("timestamps must be strictly increasing")
    return values


def regular_timestamps_ms(
    frame_count: int,
    *,
    first_timestamp_ms: float,
    frame_hop_ms: float,
) -> tuple[float, ...]:
    """Make an explicit regular timestamp vector for extractor frame centers."""

    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    if not math.isfinite(first_timestamp_ms):
        raise ValueError("first_timestamp_ms must be finite")
    _require_finite_positive(frame_hop_ms, "frame_hop_ms")
    return tuple(first_timestamp_ms + index * frame_hop_ms for index in range(frame_count))


def validate_feature_stream_timestamps(
    stream: FeatureStreamManifest, timestamps_ms: Iterable[float]
) -> tuple[float, ...]:
    """Check a loaded timestamp sidecar against its immutable stream manifest.

    Byte hashes are verified by the cache reader; this lightweight helper
    catches shape/order bugs before features are passed into a trainer.
    """

    values = validate_timestamps_ms(timestamps_ms)
    if len(values) != stream.frame_count:
        raise ValueError(
            f"timestamp count for {stream.name!r} is {len(values)}, "
            f"expected {stream.frame_count}"
        )
    return values


@dataclass(frozen=True)
class TimestampAlignment:
    """Linear interpolation/gather plan from source frames to query timestamps.

    ``valid`` distinguishes a clamped value from a genuine in-range sample.
    Training callers should combine it with their chunk mask rather than making
    a padding frame look like real audio evidence.
    """

    query_timestamps_ms: tuple[float, ...]
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    right_weights: tuple[float, ...]
    valid: tuple[bool, ...]
    out_of_range: str

    def __post_init__(self) -> None:
        length = len(self.query_timestamps_ms)
        if not all(
            len(values) == length
            for values in (
                self.left_indices,
                self.right_indices,
                self.right_weights,
                self.valid,
            )
        ):
            raise ValueError("timestamp alignment fields must have equal lengths")
        if self.out_of_range not in OUT_OF_RANGE_POLICIES:
            raise ValueError(f"unsupported out_of_range policy {self.out_of_range!r}")
        if any(index < 0 for index in self.left_indices + self.right_indices):
            raise ValueError("timestamp alignment indices must be non-negative")
        if any(weight < 0.0 or weight > 1.0 for weight in self.right_weights):
            raise ValueError("timestamp alignment weights must lie in [0, 1]")


def build_timestamp_alignment(
    source_timestamps_ms: Iterable[float],
    query_timestamps_ms: Iterable[float],
    *,
    out_of_range: Literal["clamp", "mask", "error"] = "mask",
) -> TimestampAlignment:
    """Build a timestamp-aware linear interpolation plan without tensor APIs.

    For ``mask`` and ``clamp``, out-of-range queries gather the nearest real
    frame but are marked invalid.  This permits a caller to choose between a
    stable boundary value and explicit masking without rebuilding indices.
    """

    if out_of_range not in OUT_OF_RANGE_POLICIES:
        raise ValueError(f"unsupported out_of_range policy {out_of_range!r}")
    source = validate_timestamps_ms(source_timestamps_ms)
    # Query points may intentionally repeat (stacked map objects) or arrive in
    # an arbitrary candidate order.  Only source-frame timestamps must define
    # a strictly increasing interpolation axis.
    queries = validate_timestamps_ms(
        query_timestamps_ms, require_nonempty=False, strictly_increasing=False
    )
    left_indices: list[int] = []
    right_indices: list[int] = []
    right_weights: list[float] = []
    valid: list[bool] = []

    for query in queries:
        insertion = bisect_left(source, query)
        if insertion == 0:
            if query < source[0] and out_of_range == "error":
                raise ValueError(f"query timestamp {query} precedes source features")
            exact = query == source[0]
            left_indices.append(0)
            right_indices.append(0)
            right_weights.append(0.0)
            valid.append(exact)
            continue
        if insertion == len(source):
            if query > source[-1] and out_of_range == "error":
                raise ValueError(f"query timestamp {query} follows source features")
            exact = query == source[-1]
            last = len(source) - 1
            left_indices.append(last)
            right_indices.append(last)
            right_weights.append(0.0)
            valid.append(exact)
            continue
        if source[insertion] == query:
            left_indices.append(insertion)
            right_indices.append(insertion)
            right_weights.append(0.0)
            valid.append(True)
            continue
        left = insertion - 1
        right = insertion
        right_weight = (query - source[left]) / (source[right] - source[left])
        left_indices.append(left)
        right_indices.append(right)
        right_weights.append(right_weight)
        valid.append(True)

    return TimestampAlignment(
        query_timestamps_ms=queries,
        left_indices=tuple(left_indices),
        right_indices=tuple(right_indices),
        right_weights=tuple(right_weights),
        valid=tuple(valid),
        out_of_range=out_of_range,
    )


def apply_scalar_alignment(
    source_values: Sequence[float], alignment: TimestampAlignment
) -> tuple[float, ...]:
    """CPU reference implementation used for tests and preprocessing audits.

    Training code should use ``left_indices``, ``right_indices``, and
    ``right_weights`` to gather tensor rows with its native backend.
    """

    if not source_values:
        raise ValueError("source_values must be non-empty")
    maximum_index = max(alignment.left_indices + alignment.right_indices, default=-1)
    if maximum_index >= len(source_values):
        raise ValueError("alignment references a source frame outside source_values")
    return tuple(
        (1.0 - weight) * float(source_values[left]) + weight * float(source_values[right])
        for left, right, weight in zip(
            alignment.left_indices, alignment.right_indices, alignment.right_weights
        )
    )
