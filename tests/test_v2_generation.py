import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.v2.evaluation import (
    CallbackDifficultyCalculator,
    Candidate,
    DifficultyCalculation,
    audit_candidate,
    build_generation_result,
    rerank_candidates,
)
from src.v2.events import BeatEvent, EventKind
from src.v2.generation import GenerationRequest, public_audio_filename
from src.v2.packaging import PackagingError, write_generation_package
from src.v2.timing import TimingMap, TimingSegment


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


if __name__ == "__main__":
    unittest.main()
