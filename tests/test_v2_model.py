import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.v2.losses import compute_planner_losses, coordinate_diffusion_loss
    from src.v2.model import BeatGridModelConfig, CoordinateDiffusionRefiner, HierarchicalBeatGridPlanner


@unittest.skipIf(torch is None, "PyTorch is not installed in this lightweight test environment")
class BeatGridModelTests(unittest.TestCase):
    def test_planner_and_geometry_refiner_shapes_and_losses(self):
        torch.manual_seed(0)
        config = BeatGridModelConfig(
            rhythm_channels=4,
            semantic_dim=12,
            hidden_dim=32,
            local_layers=1,
            global_layers=1,
            num_heads=4,
            events_per_slot=2,
            section_slots=8,
            dropout=0.0,
        )
        planner = HierarchicalBeatGridPlanner(config)
        batch, slots, lanes = 2, 16, config.events_per_slot
        phase = torch.zeros(batch, slots, 2)
        phase[..., 1] = 1.0
        slot_mask = torch.ones(batch, slots, dtype=torch.bool)
        slot_mask[1, -3:] = False
        outputs = planner(
            torch.randn(batch, config.rhythm_channels, 40),
            torch.tensor([3.5, 5.2]),
            num_slots=slots,
            semantic_features=torch.randn(batch, 12, config.semantic_dim),
            slot_mask=slot_mask,
            beat_phase=phase,
            map_settings=torch.randn(batch, config.map_setting_dim),
        )
        self.assertEqual(tuple(outputs["occupancy_logits"].shape), (batch, slots, lanes))
        self.assertEqual(tuple(outputs["kind_logits"].shape), (batch, slots, lanes, 3))
        self.assertEqual(tuple(outputs["base_coordinates"].shape), (batch, slots, lanes, 2))

        targets = {
            "slot_mask": slot_mask,
            "occupancy": torch.zeros(batch, slots, lanes),
            "kind": torch.full((batch, slots, lanes), -100, dtype=torch.long),
            "time_residual_beats": torch.zeros(batch, slots, lanes),
            "coordinates": torch.zeros(batch, slots, lanes, 2),
            "end_beat_delta": torch.zeros(batch, slots, lanes),
            "slider_length": torch.zeros(batch, slots, lanes),
            "slider_repeats": torch.zeros(batch, slots, lanes, dtype=torch.long),
        }
        targets["occupancy"][0, 2, 0] = 1
        targets["kind"][0, 2, 0] = 0
        targets["coordinates"][0, 2, 0] = torch.tensor([0.2, 0.6])
        targets["occupancy"][1, 4, 1] = 1
        targets["kind"][1, 4, 1] = 1
        targets["coordinates"][1, 4, 1] = torch.tensor([0.8, 0.4])
        targets["slider_length"][1, 4, 1] = 160
        targets["slider_repeats"][1, 4, 1] = 2

        losses = compute_planner_losses(
            outputs, targets, subdivisions_per_beat=config.subdivisions_per_beat
        )
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()

        refiner = CoordinateDiffusionRefiner(config, layers=1)
        active_mask = targets["occupancy"].bool() & targets["slot_mask"].unsqueeze(-1)
        diffusion = coordinate_diffusion_loss(
            refiner,
            outputs["context"].detach(),
            targets["coordinates"],
            active_mask,
            noise_level=torch.tensor([0.2, 0.8]),
        )
        self.assertTrue(torch.isfinite(diffusion))
        diffusion.backward()


if __name__ == "__main__":
    unittest.main()
