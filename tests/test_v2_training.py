"""CPU-only-with-Torch checkpoint and stage-transition regression tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight host
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingCheckpointTests(unittest.TestCase):
    def _models(self):
        from src.v2.geometry import ActiveEventGeometryDiffuser, GeometryDiffusionConfig
        from src.v2.sparse_model import SparseEventPlanner, SparsePlannerConfig

        planner = SparseEventPlanner(
            SparsePlannerConfig(
                rhythm_dim=4,
                semantic_dim=3,
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
            GeometryDiffusionConfig(context_dim=32, hidden_dim=32, heads=4, layers=1, dropout=0.0)
        )
        return planner, geometry

    def test_planner_checkpoint_does_not_load_adam_state_into_geometry_optimizer(self):
        from src.v2.training import (
            CheckpointProvenance,
            TrainingConfig,
            TrainingStage,
            checkpoint_payload,
            load_checkpoint,
            make_optimizer,
            save_checkpoint,
        )

        provenance = CheckpointProvenance(
            feature_pipeline_key="f" * 64,
            split_manifest_sha256="s" * 64,
        )
        planner, geometry = self._models()
        planner_config = TrainingConfig(stage=TrainingStage.PLANNER)
        payload = checkpoint_payload(
            planner=planner,
            geometry=geometry,
            optimizer=make_optimizer(planner, geometry, planner_config),
            scaler=None,
            epoch=2,
            global_step=10,
            training_config=planner_config,
            provenance=provenance,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(payload, Path(directory) / "planner.pt")
            restored_planner, restored_geometry = self._models()
            geometry_config = TrainingConfig(stage=TrainingStage.GEOMETRY)
            geometry_optimizer = make_optimizer(restored_planner, restored_geometry, geometry_config)
            restored = load_checkpoint(
                checkpoint,
                planner=restored_planner,
                geometry=restored_geometry,
                optimizer=geometry_optimizer,
                expected_provenance=provenance,
                optimizer_stage=TrainingStage.GEOMETRY,
            )
        self.assertFalse(restored["_optimizer_state_loaded"])
        self.assertEqual(restored["_saved_training_stage"], TrainingStage.PLANNER.value)
        self.assertFalse(geometry_optimizer.state)


if __name__ == "__main__":
    unittest.main()
