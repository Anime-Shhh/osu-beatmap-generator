import json
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

import src.v2.packaging as packaging
from src.v2.clock import AudioClockResolver, ClockEvidence, StaticClockResolver, TimingHypothesis
from src.v2.evaluation import (
    CallbackDifficultyCalculator,
    Candidate,
    DifficultyCalculation,
    audit_candidate,
    build_generation_result,
    rerank_candidates,
)
from src.v2.events import BeatEvent, EventKind
from src.v2.generate import main as cli_main, parse_args
from src.v2.generation import GenerationRequest, public_audio_filename
from src.v2.mapspec import compile_osu, mapspec_from_osu
from src.v2.packaging import PackagingError, write_generation_package
from src.v2.timing import TimingMap, TimingSegment

if torch is not None:
    import src.v2.inference as inference

    from src.v2.geometry import ActiveEventGeometryDiffuser, GeometryDiffusionConfig
    from src.v2.inference import (
        AudioFeatureExtractor,
        create_mel_filterbank,
        estimate_audio_clock,
        extract_audio_features,
        generate_draft,
        load_audio,
        load_v2_models,
    )
    from src.v2.sparse_model import SparseEventPlanner, SparsePlannerConfig
    from src.v2.training import (
        CHECKPOINT_SCHEMA,
        CheckpointProvenance,
        TrainingConfig,
        TrainingStage,
        checkpoint_payload,
        make_optimizer,
        save_checkpoint,
    )


VALID_OSU = """osu file format v14

[General]
AudioFilename: song.mp3
Mode: 0

[Metadata]
Title:Generated Draft
Artist:Unknown
Creator:osu-beatmap-generator
Version:Draft

[Difficulty]
HPDrainRate:6
CircleSize:4
OverallDifficulty:8
ApproachRate:9
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,2,0,70,1,0

[HitObjects]
64,96,0,1,0,0:0:0:0:
320,192,500,1,0,0:0:0:0:
"""


def _map_spec(events):
    return SimpleNamespace(
        events=tuple(events),
        timing_map=TimingMap((TimingSegment(0.0, 500.0, 0.0),)),
    )


def _compiler(_map_spec):
    return VALID_OSU


def _create_synthetic_wav(path: Path, duration_sec: float = 3.0, sample_rate: int = 24000) -> Path:
    """Create a synthetic 16-bit PCM WAV file with clear periodic pulse beats."""
    n_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        # 120 BPM = 500ms per beat = sample_rate / 2 samples per beat
        beat_interval = sample_rate // 2
        raw_bytes = bytearray(n_samples * 2)
        for i in range(n_samples):
            # Base 440 Hz carrier tone
            val = 0.3 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)
            # Add beat click pulse at every 500ms
            beat_pos = i % beat_interval
            if beat_pos < sample_rate // 50:  # 20ms pulse
                pulse_env = math.exp(-beat_pos / (sample_rate * 0.005))
                val += 0.6 * pulse_env * math.sin(2.0 * math.pi * 1000.0 * i / sample_rate)
            int_val = max(-32767, min(32767, int(val * 32767.0)))
            raw_bytes[2 * i : 2 * i + 2] = int_val.to_bytes(2, byteorder="little", signed=True)
        wf.writeframes(raw_bytes)
    return path


def _create_constant_pcm_wav(
    path: Path,
    *,
    sample_width: int,
    channels: int,
    amplitude: float = 0.25,
    frames: int = 2400,
    sample_rate: int = 24000,
) -> Path:
    """Create integer PCM with the same known amplitude in every channel."""
    if sample_width not in {2, 4}:
        raise ValueError("test fixture supports s16 and s32 PCM only")
    scale = 1 << (sample_width * 8 - 1)
    encoded = int(amplitude * scale).to_bytes(sample_width, "little", signed=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(encoded * channels * frames)
    return path


def _create_sine_pcm_wav(
    path: Path,
    *,
    frequency_hz: float,
    sample_rate: int = 48000,
    duration_sec: float = 1.0,
    amplitude: float = 0.75,
) -> Path:
    """Create a mono PCM16 sine fixture for resampler checks."""
    frame_count = int(round(sample_rate * duration_sec))
    raw_bytes = bytearray(frame_count * 2)
    for index in range(frame_count):
        value = amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
        encoded = max(-32768, min(32767, round(value * 32767.0)))
        raw_bytes[2 * index : 2 * index + 2] = encoded.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_bytes)
    return path


def _create_mock_checkpoint(path: Path) -> Path:
    """Create a small, valid v2 checkpoint for testing."""
    torch.manual_seed(42)
    planner = SparseEventPlanner(
        SparsePlannerConfig(
            rhythm_dim=101,
            semantic_dim=128,
            model_dim=32,
            local_layers=1,
            global_layers=1,
            decoder_layers=1,
            heads=4,
            ff_dim=64,
            dropout=0.0,
            max_bars=8,
            max_decoder_tokens=32,
        )
    )
    geometry = ActiveEventGeometryDiffuser(
        GeometryDiffusionConfig(
            context_dim=32,
            hidden_dim=32,
            heads=4,
            layers=1,
            dropout=0.0,
        )
    )
    training_cfg = TrainingConfig(stage=TrainingStage.PLANNER)

    # Compute pipeline key matching extract_audio_features
    from src.v2.audio import CanonicalAudioManifest, CanonicalAudioSpec, FeatureCacheManifest, FeatureStreamManifest
    from src.v2.data import SourceIdentity
    dummy_audio = CanonicalAudioManifest(
        source=SourceIdentity(audio_sha256="0" * 64, canonical_audio_sha256="0" * 64, audio_duration_ms=1000.0),
        pcm_sha256="0" * 64,
        frame_count=24000,
        spec=CanonicalAudioSpec(
            sample_rate_hz=24000,
            decoder_revision="mean-downmix-ffmpeg-resample-or-native-pcm-v3",
        ),
    )
    rhythm_stream = FeatureStreamManifest(
        name="rhythm_features",
        extractor="audio-logmel-onset",
        extractor_revision="v2-rhythm-100hz",
        feature_dim=101,
        frame_count=100,
        dtype="float32",
        feature_path="features/rhythm.npy",
        feature_sha256="0" * 64,
        timestamps_path="timestamps/rhythm.npy",
        timestamps_sha256="0" * 64,
        nominal_frame_rate_hz=100.0,
    )
    semantic_stream = FeatureStreamManifest(
        name="semantic_features",
        extractor="audio-mel-pooled",
        extractor_revision="v2-semantic-25hz-centered-v2",
        feature_dim=128,
        frame_count=25,
        dtype="float32",
        feature_path="features/semantic.npy",
        feature_sha256="0" * 64,
        timestamps_path="timestamps/semantic.npy",
        timestamps_sha256="0" * 64,
        nominal_frame_rate_hz=25.0,
    )
    cache_manifest = FeatureCacheManifest(
        canonical_audio=dummy_audio,
        streams=(rhythm_stream, semantic_stream),
        pipeline_revision="v2-multirate-audio-3",
    )

    provenance = CheckpointProvenance(
        feature_pipeline_key=cache_manifest.pipeline_key,
        split_manifest_sha256="a" * 64,
    )
    payload = checkpoint_payload(
        planner=planner,
        geometry=geometry,
        optimizer=make_optimizer(planner, geometry, training_cfg),
        scaler=None,
        epoch=1,
        global_step=1,
        training_config=training_cfg,
        provenance=provenance,
    )
    return save_checkpoint(payload, path)


class GenerationSchemaTests(unittest.TestCase):
    def test_request_round_trip_omits_local_path(self):
        request = GenerationRequest(
            audio_filename="song.mp3",
            audio_sha256="A" * 64,
            target_star=4.5,
            candidate_count=4,
            extra_controls={"density": "conservative"},
        )
        parsed = GenerationRequest.from_dict(request.to_dict())
        self.assertEqual(parsed, request)
        self.assertEqual(request.audio_sha256, "a" * 64)
        with self.assertRaises(ValueError):
            public_audio_filename("../song.mp3")
        with self.assertRaises(ValueError):
            public_audio_filename("/private/song.mp3")


class CandidateAuditTests(unittest.TestCase):
    def setUp(self):
        self.events = (
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )
        self.calculator = CallbackDifficultyCalculator(
            lambda _content: DifficultyCalculation(
                available=True,
                stars=4.5,
                calculator_id="test-pinned-calculator",
            )
        )

    def test_valid_candidate_is_audited_and_selected(self):
        candidate = Candidate("candidate-0", _map_spec(self.events), seed=7, model_quality_score=0.8)
        audit = audit_candidate(
            candidate,
            target_star=4.5,
            calculator=self.calculator,
            compiler=_compiler,
        )
        self.assertTrue(audit.valid)
        self.assertEqual(audit.difficulty.stars, 4.5)
        self.assertGreater(audit.score, 0.9)
        request = GenerationRequest(audio_filename="song.mp3", candidate_count=1)
        result = build_generation_result(
            request,
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        self.assertEqual(result.selected_candidate_id, "candidate-0")
        self.assertTrue(result.selected_candidate.valid)

    def test_same_tick_candidate_is_hard_rejected(self):
        duplicate_time_events = self.events + (
            BeatEvent(EventKind.CIRCLE, 1.0, 256.0, 192.0),
        )
        audit = audit_candidate(
            Candidate("same-tick", _map_spec(duplicate_time_events)),
            calculator=self.calculator,
            compiler=_compiler,
        )
        self.assertFalse(audit.valid)
        self.assertIsNone(audit.score)
        self.assertTrue(any("same-tick" in error for error in audit.errors))

    def test_exact_duplicate_is_demoted_but_not_made_invalid(self):
        candidates = (
            Candidate("a", _map_spec(self.events), model_quality_score=0.8),
            Candidate("b", _map_spec(self.events), model_quality_score=0.7),
        )
        audits = rerank_candidates(candidates, calculator=self.calculator, compiler=_compiler)
        self.assertEqual(audits[0].candidate.candidate_id, "a")
        duplicate = next(audit for audit in audits if audit.candidate.candidate_id == "b")
        self.assertTrue(duplicate.valid)
        self.assertTrue(any("exact duplicate" in warning for warning in duplicate.warnings))


class PackagingTests(unittest.TestCase):
    def test_packages_selected_candidate_with_manifest_last(self):
        candidate = Candidate("candidate-0", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(candidate, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="song.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            package = write_generation_package(
                result,
                candidate,
                directory,
                compiler=_compiler,
            )
            self.assertTrue(package.osu_path.is_file())
            self.assertTrue(package.manifest_path.is_file())
            manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_candidate_id"], "candidate-0")
            self.assertEqual(manifest["artifacts"]["osu_sha256"], package.osu_sha256)
            self.assertNotIn(str(Path(directory)), package.manifest_path.read_text(encoding="utf-8"))

    def test_packages_audio_before_completion_manifest(self):
        candidate = Candidate("candidate-0", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(candidate, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="renamed.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"synthetic mp3 bytes")
            destination = Path(directory) / "package"
            package = write_generation_package(
                result,
                candidate,
                destination,
                compiler=_compiler,
                audio_source_path=source,
            )
            self.assertEqual(package.audio_path, destination / "renamed.mp3")
            self.assertEqual(package.audio_path.read_bytes(), source.read_bytes())
            manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"]["audio_file"], "renamed.mp3")
            self.assertEqual(manifest["artifacts"]["audio_sha256"], package.audio_sha256)

    def test_packaging_recognizes_audio_already_in_output_directory(self):
        candidate = Candidate("candidate-0", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(candidate, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="song.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "song.mp3"
            source.write_bytes(b"preserve me")
            package = write_generation_package(
                result,
                candidate,
                directory,
                compiler=_compiler,
                audio_source_path=source,
            )
            self.assertEqual(package.audio_path, source)
            self.assertEqual(source.read_bytes(), b"preserve me")

    def test_packaging_refuses_to_replace_unrelated_audio(self):
        candidate = Candidate("candidate-0", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(candidate, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="song.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"new audio")
            destination = Path(directory) / "package"
            destination.mkdir()
            existing = destination / "song.mp3"
            existing.write_bytes(b"user audio")
            with self.assertRaises(PackagingError):
                write_generation_package(
                    result,
                    candidate,
                    destination,
                    compiler=_compiler,
                    audio_source_path=source,
                )
            self.assertEqual(existing.read_bytes(), b"user audio")
            self.assertFalse((destination / "ranked-like-draft.osu").exists())
            self.assertFalse((destination / "ranked-like-draft.manifest.json").exists())

    def test_packaging_preserves_audio_created_by_copy_race_loser(self):
        candidate = Candidate("candidate-0", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(candidate, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="song.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"source audio")
            destination = Path(directory) / "package"
            destination.mkdir()
            audio_path = destination / "song.mp3"
            real_link = packaging.os.link

            def competing_link(source_path, destination_path):
                destination_path = Path(destination_path)
                if destination_path == audio_path:
                    with destination_path.open("xb") as handle:
                        handle.write(b"concurrent user audio")
                return real_link(source_path, destination_path)

            with mock.patch.object(packaging.os, "link", side_effect=competing_link):
                with self.assertRaisesRegex(PackagingError, "refusing to overwrite"):
                    write_generation_package(
                        result,
                        candidate,
                        destination,
                        compiler=_compiler,
                        audio_source_path=source,
                    )

            self.assertEqual(audio_path.read_bytes(), b"concurrent user audio")
            self.assertFalse((destination / "ranked-like-draft.osu").exists())
            self.assertFalse((destination / "ranked-like-draft.manifest.json").exists())

    def test_packaging_preserves_replacement_after_post_copy_failure(self):
        candidate = Candidate("candidate-0", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(candidate, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="song.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"source audio")
            destination = Path(directory) / "package"
            audio_path = destination / "song.mp3"

            def fail_after_replacement(*_args, **_kwargs):
                audio_path.unlink()
                audio_path.write_bytes(b"replacement audio")
                raise PackagingError("injected map publication failure")

            with mock.patch.object(packaging, "_write_atomic", side_effect=fail_after_replacement):
                with self.assertRaisesRegex(PackagingError, "injected"):
                    write_generation_package(
                        result,
                        candidate,
                        destination,
                        compiler=_compiler,
                        audio_source_path=source,
                    )

            self.assertEqual(audio_path.read_bytes(), b"replacement audio")
            self.assertFalse((destination / "ranked-like-draft.osu").exists())
            self.assertFalse((destination / "ranked-like-draft.manifest.json").exists())

    def test_refuses_mismatched_candidate(self):
        selected = Candidate("selected", _map_spec((
            BeatEvent(EventKind.CIRCLE, 0.0, 64.0, 96.0),
            BeatEvent(EventKind.CIRCLE, 1.0, 320.0, 192.0),
        )))
        audit = audit_candidate(selected, compiler=_compiler)
        result = build_generation_result(
            GenerationRequest(audio_filename="song.mp3", candidate_count=1),
            (audit,),
            architecture_id="tempo-posterior-hierarchical-event-diffusion",
            model_revision="test-revision",
        )
        wrong = Candidate("wrong", selected.map_spec)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PackagingError):
                write_generation_package(result, wrong, directory, compiler=_compiler)


@unittest.skipIf(torch is None, "PyTorch is required for audio and model generation tests")
class AudioLoadingAndFeatureExtractionTests(unittest.TestCase):
    def test_mel_filterbank_shape_and_bounds(self):
        fb = create_mel_filterbank(24000, 1024, 96)
        self.assertEqual(fb.shape, (96, 513))
        self.assertTrue((fb >= 0.0).all())
        self.assertTrue((fb <= 1.0).all())

    def test_load_audio_wav_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "test.wav"
            _create_synthetic_wav(wav_path, duration_sec=2.0, sample_rate=24000)
            waveform, sr = load_audio(wav_path, target_sr=24000)
            self.assertEqual(sr, 24000)
            self.assertEqual(waveform.dim(), 2)
            self.assertEqual(waveform.shape[0], 1)
            self.assertEqual(waveform.shape[1], 48000)
            self.assertTrue(torch.isfinite(waveform).all())

    def test_wave_backend_mean_downmixes_s16_and_s32_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            for sample_width in (2, 4):
                for channels in (1, 2):
                    with self.subTest(sample_width=sample_width, channels=channels):
                        wav_path = Path(directory) / f"s{sample_width * 8}-{channels}ch.wav"
                        _create_constant_pcm_wav(
                            wav_path,
                            sample_width=sample_width,
                            channels=channels,
                        )
                        waveform, sr = load_audio(wav_path, target_sr=24000, backend="wave")
                        self.assertEqual(sr, 24000)
                        self.assertEqual(waveform.shape, (1, 2400))
                        self.assertAlmostEqual(float(waveform.mean().item()), 0.25, places=6)

    def test_ffmpeg_backend_uses_arithmetic_mean_for_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "stereo.wav"
            _create_constant_pcm_wav(wav_path, sample_width=2, channels=2)
            waveform, _ = load_audio(wav_path, target_sr=24000, backend="ffmpeg")
            self.assertAlmostEqual(float(waveform.mean().item()), 0.25, places=4)

    def test_ffmpeg_resampling_is_bandlimited_and_preserves_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            passband_path = _create_sine_pcm_wav(
                Path(directory) / "passband.wav", frequency_hz=1000.0
            )
            stopband_path = _create_sine_pcm_wav(
                Path(directory) / "stopband.wav", frequency_hz=18000.0
            )
            passband, _ = load_audio(passband_path, target_sr=24000, backend="ffmpeg")
            stopband, _ = load_audio(stopband_path, target_sr=24000, backend="ffmpeg")

            self.assertEqual(passband.shape, (1, 24000))
            self.assertEqual(stopband.shape, (1, 24000))
            passband_rms = float(passband[:, 100:-100].square().mean().sqrt().item())
            stopband_rms = float(stopband[:, 100:-100].square().mean().sqrt().item())
            self.assertGreater(passband_rms, 0.4)
            self.assertLess(stopband_rms, 0.01)

    def test_wave_fallback_fails_closed_for_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = _create_sine_pcm_wav(
                Path(directory) / "nonnative.wav", frequency_hz=1000.0
            )
            with self.assertRaisesRegex(RuntimeError, "does not resample"):
                load_audio(wav_path, target_sr=24000, backend="wave")

            with mock.patch.object(
                inference,
                "_load_audio_ffmpeg",
                side_effect=FileNotFoundError("injected missing FFmpeg"),
            ):
                with self.assertRaisesRegex(RuntimeError, "does not resample"):
                    load_audio(wav_path, target_sr=24000, backend="auto")

    def test_load_audio_mp3_with_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "test.wav"
            mp3_path = Path(directory) / "test.mp3"
            _create_synthetic_wav(wav_path, duration_sec=1.5, sample_rate=24000)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "192k", str(mp3_path)],
                capture_output=True,
                check=True,
            )

            waveform, sr = load_audio(mp3_path, target_sr=24000)
            self.assertEqual(sr, 24000)
            self.assertEqual(waveform.shape[0], 1)
            self.assertGreater(waveform.shape[1], 24000)
            self.assertTrue(torch.isfinite(waveform).all())

    def test_extract_audio_features_contract(self):
        waveform = torch.randn(1, 48000).clamp(-1.0, 1.0)
        bundle, hypotheses = extract_audio_features(
            waveform,
            sample_rate_hz=24000,
            rhythm_dim=101,
            semantic_dim=128,
        )
        self.assertEqual(bundle.rhythm_features.shape[1], 101)
        self.assertEqual(bundle.semantic_features.shape[1], 128)
        self.assertTrue(torch.isfinite(bundle.rhythm_features).all())
        self.assertTrue(torch.isfinite(bundle.semantic_features).all())
        self.assertTrue((bundle.rhythm_times_ms[1:] > bundle.rhythm_times_ms[:-1]).all())
        self.assertTrue((bundle.semantic_times_ms[1:] > bundle.semantic_times_ms[:-1]).all())
        self.assertGreater(len(hypotheses), 0)
        self.assertTrue(bundle.clock_evidence.beat_times_ms)
        self.assertAlmostEqual(float(bundle.semantic_times_ms[0].item()), 15.0, places=4)
        self.assertEqual(
            bundle.feature_cache.canonical_audio.spec.decoder_revision,
            "mean-downmix-ffmpeg-resample-or-native-pcm-v3",
        )
        self.assertEqual(bundle.feature_cache.pipeline_revision, "v2-multirate-audio-3")

    def test_extract_audio_features_accepts_sub_fft_audio(self):
        bundle, hypotheses = extract_audio_features(torch.zeros(1, 240), sample_rate_hz=24000)
        self.assertGreater(bundle.rhythm_features.shape[0], 0)
        self.assertGreater(len(hypotheses), 0)

    def test_audio_feature_extractor_class(self):
        extractor = AudioFeatureExtractor(sample_rate_hz=24000, rhythm_dim=101, semantic_dim=128)
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "test.wav"
            _create_synthetic_wav(wav_path, duration_sec=1.0, sample_rate=24000)
            bundle = extractor.extract(wav_path)
            self.assertEqual(bundle.rhythm_features.shape[1], 101)
            self.assertEqual(bundle.semantic_features.shape[1], 128)


@unittest.skipIf(torch is None, "PyTorch is required for clock estimation tests")
class AudioClockEstimatorTests(unittest.TestCase):
    def test_clock_estimation_finds_tempo_and_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "test_120bpm.wav"
            _create_synthetic_wav(wav_path, duration_sec=4.0, sample_rate=24000)
            waveform, sr = load_audio(wav_path, target_sr=24000)
            _, hypotheses = extract_audio_features(waveform, sr)
            self.assertGreater(len(hypotheses), 0)
            primary = hypotheses[0]
            est_beat_length = primary.timing_map.segments[0].beat_length_ms
            est_bpm = 60000.0 / est_beat_length
            # Synthetic pulse was at 120 BPM
            self.assertGreater(est_bpm, 90.0)
            self.assertLess(est_bpm, 160.0)

    def test_clock_estimation_honors_bpm_and_offset_overrides(self):
        waveform = torch.randn(1, 48000)
        _, hypotheses = extract_audio_features(
            waveform,
            sample_rate_hz=24000,
            bpm=180.0,
            offset_ms=42.0,
        )
        primary = hypotheses[0]
        segment = primary.timing_map.segments[0]
        self.assertAlmostEqual(segment.start_ms, 42.0, places=1)
        self.assertAlmostEqual(segment.beat_length_ms, 60000.0 / 180.0, places=2)

    def test_clock_estimation_rejects_invalid_or_extreme_overrides(self):
        onset = torch.ones(200)
        for bpm in (float("nan"), float("inf"), 0.0, -1.0, 19.99, 1000.01):
            with self.subTest(bpm=bpm), self.assertRaises(ValueError):
                estimate_audio_clock(onset, duration_ms=2000.0, bpm_override=bpm)
        for offset in (float("nan"), float("inf"), -float("inf"), 86_400_001.0):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                estimate_audio_clock(onset, duration_ms=2000.0, offset_override_ms=offset)
        with self.assertRaisesRegex(ValueError, "maximum"):
            estimate_audio_clock(
                onset,
                duration_ms=86_400_000.0,
                bpm_override=1000.0,
                offset_override_ms=0.0,
            )

    def test_audio_clock_resolver(self):
        evidence = ClockEvidence(
            audio_duration_ms=5000.0,
            beat_times_ms=(0.0, 500.0, 1000.0, 1500.0),
            beat_confidences=(1.0, 1.0, 1.0, 1.0),
            source="test",
            extractor_revision="v1",
        )
        resolver = AudioClockResolver()
        resolved = resolver.resolve(evidence, max_hypotheses=2)
        self.assertEqual(len(resolved), 1)
        self.assertAlmostEqual(resolved[0].timing_map.segments[0].beat_length_ms, 500.0)


@unittest.skipIf(torch is None, "PyTorch is required for end-to-end generation tests")
class EndToEndGenerationAndCLITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.wav_path = self.dir_path / "song.wav"
        _create_synthetic_wav(self.wav_path, duration_sec=2.0, sample_rate=24000)
        self.ckpt_path = self.dir_path / "checkpoint.pt"
        _create_mock_checkpoint(self.ckpt_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_draft_end_to_end_from_audio(self):
        models = load_v2_models(self.ckpt_path, device="cpu")
        waveform, sr = load_audio(self.wav_path, target_sr=24000)
        features, hypotheses = extract_audio_features(
            waveform,
            sr,
            models=models,
            bpm=120.0,
            offset_ms=0.0,
        )
        resolver = AudioClockResolver(hypotheses)
        request = GenerationRequest(
            audio_filename="song.wav",
            target_star=4.5,
            candidate_count=2,
            feature_revision=features.feature_cache.pipeline_revision,
            model_revision=models.model_revision,
        )
        draft = generate_draft(
            models=models,
            features=features,
            clock_resolver=resolver,
            request=request,
            geometry_steps=4,
        )
        self.assertIsNotNone(draft.selected_candidate)
        selected = draft.selected_candidate
        self.assertIsNotNone(selected)
        osu_text = compile_osu(selected.map_spec)
        self.assertIn("AudioFilename: song.wav", osu_text)
        self.assertIn("[TimingPoints]", osu_text)
        self.assertIn("[HitObjects]", osu_text)

    def test_cli_parse_args_validation(self):
        # Valid raw audio call with --difficulty
        args = parse_args([
            "--checkpoint", str(self.ckpt_path),
            "--audio", str(self.wav_path),
            "--difficulty", "4.5",
            "--output-dir", str(self.dir_path / "out"),
        ])
        self.assertEqual(args.target_star, 4.5)
        self.assertEqual(args.audio, str(self.wav_path))

        # Valid with --target-star alias
        args2 = parse_args([
            "--checkpoint", str(self.ckpt_path),
            "--audio", str(self.wav_path),
            "--target-star", "5.0",
            "--output", str(self.dir_path / "out" / "draft.osu"),
        ])
        self.assertEqual(args2.target_star, 5.0)

        # Missing audio & feature bundle
        with self.assertRaises(SystemExit):
            parse_args([
                "--checkpoint", str(self.ckpt_path),
                "--difficulty", "4.5",
                "--output-dir", str(self.dir_path / "out"),
            ])

        # Missing mandatory difficulty
        with self.assertRaises(SystemExit):
            parse_args([
                "--checkpoint", str(self.ckpt_path),
                "--audio", str(self.wav_path),
                "--output-dir", str(self.dir_path / "out"),
            ])

        for bpm in ("nan", "inf", "0", "-1", "19.99", "1000.01"):
            with self.subTest(bpm=bpm), self.assertRaises(SystemExit):
                parse_args([
                    "--checkpoint", str(self.ckpt_path),
                    "--audio", str(self.wav_path),
                    "--difficulty", "4.5",
                    "--bpm", bpm,
                    "--output-dir", str(self.dir_path / "out"),
                ])

        for offset in ("nan", "inf", "-inf", "86400001"):
            with self.subTest(offset=offset), self.assertRaises(SystemExit):
                parse_args([
                    "--checkpoint", str(self.ckpt_path),
                    "--audio", str(self.wav_path),
                    "--difficulty", "4.5",
                    f"--offset={offset}",
                    "--output-dir", str(self.dir_path / "out"),
                ])

    def test_cli_main_generates_valid_osu_and_manifest(self):
        out_dir = self.dir_path / "packaged_output"
        ret = cli_main([
            "--checkpoint", str(self.ckpt_path),
            "--audio", str(self.wav_path),
            "--difficulty", "4.5",
            "--bpm", "120",
            "--offset", "0",
            "--title", "E2E Test Song",
            "--artist", "Test Artist",
            "--output-dir", str(out_dir),
            "--geometry-steps", "4",
            "--candidate-count", "2",
            "--device", "cpu",
        ])
        self.assertEqual(ret, 0)
        osu_file = out_dir / "ranked-like-draft.osu"
        manifest_file = out_dir / "ranked-like-draft.manifest.json"
        self.assertTrue(osu_file.is_file())
        self.assertTrue(manifest_file.is_file())
        self.assertEqual((out_dir / "song.wav").read_bytes(), self.wav_path.read_bytes())

        osu_content = osu_file.read_text(encoding="utf-8")
        self.assertIn("Title: E2E Test Song", osu_content)
        self.assertIn("Artist: Test Artist", osu_content)
        self.assertIn("AudioFilename: song.wav", osu_content)
        self.assertIn("Version: v2 4.5★ Draft", osu_content)
        self.assertIn("[TimingPoints]", osu_content)
        self.assertIn("[HitObjects]", osu_content)

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertIn("selected_candidate_id", manifest)
        self.assertEqual(manifest["schema_version"], "osu-beatmap-generator/v2-generation-result/1")
        self.assertEqual(manifest["artifacts"]["audio_file"], "song.wav")

    def test_cli_main_generates_self_contained_mp3_package_without_timing_override(self):
        mp3_path = self.dir_path / "source.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(self.wav_path),
                "-b:a",
                "192k",
                str(mp3_path),
            ],
            capture_output=True,
            check=True,
        )
        out_dir = self.dir_path / "mp3_package"
        ret = cli_main([
            "--checkpoint", str(self.ckpt_path),
            "--input", str(mp3_path),
            "--target-star", "4.5",
            "--audio-filename", "packaged-song.mp3",
            "--output-osu", str(out_dir / "audio-only.osu"),
            "--geometry-steps", "4",
            "--candidate-count", "2",
            "--device", "cpu",
        ])
        self.assertEqual(ret, 0)
        osu_path = out_dir / "audio-only.osu"
        audio_path = out_dir / "packaged-song.mp3"
        manifest_path = out_dir / "ranked-like-draft.manifest.json"
        self.assertTrue(osu_path.is_file())
        self.assertEqual(audio_path.read_bytes(), mp3_path.read_bytes())
        self.assertTrue(manifest_path.is_file())

        parsed = mapspec_from_osu(osu_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed.general.audio_filename, "packaged-song.mp3")
        self.assertGreater(len(parsed.events), 0)
        self.assertGreater(len(parsed.timing_points), 0)


if __name__ == "__main__":
    unittest.main()
