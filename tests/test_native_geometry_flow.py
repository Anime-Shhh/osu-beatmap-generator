import unittest
import torch
from src.beatmap_flow.geometry import NativeGeometryFlow, GeometryConfig, geometry_loss, sample_geometry_flow, event_condition


class GeometryFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = NativeGeometryFlow(GeometryConfig(width=24, heads=4, context_layers=1, flow_layers=1)).eval()
        n = 5
        self.batch = {'condition': torch.randn(2, n, 214), 'starts': torch.arange(n)[None].expand(2, -1)*200.,
            'mask': torch.tensor([[True]*n, [False]*n]), 'geometry': torch.rand(2, n, 33, 2),
            'kinds': torch.tensor([[0, 1, 0, 1, 2]]*2), 'counts': torch.tensor([[0, 2, 0, 1, 0]]*2),
            'curves': torch.tensor([[0, 2, 0, 0, 0]]*2), 'repeats': torch.ones(2, n, dtype=torch.long),
            'lengths': torch.tensor([[0., 150., 0., 100., 0.]]*2), 'features': torch.rand(2, n, 4),
            'positions': torch.arange(n)[None].expand(2, -1), 'combo': torch.zeros(2, n)}

    def test_real_time_condition(self):
        audio = torch.randn(80, 101)
        result = event_condition(audio, torch.tensor([100., 300., 600.]),
            torch.tensor([100., 500., 600.]), torch.tensor([0, 1, 0]), torch.rand(6))
        self.assertEqual(result.shape, (3, 214))
        torch.testing.assert_close(result[:, :101], audio[[10, 30, 60]])

    def test_empty_row_loss_and_backward(self):
        self.model.train()
        loss = geometry_loss(self.model, self.batch)
        self.assertTrue(all(torch.isfinite(v) for v in loss.values()))
        loss['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_inpainting_and_inactive_points(self):
        known_mask = torch.zeros(2, 5, dtype=torch.bool)
        known_mask[0, :2] = True
        result = sample_geometry_flow(self.model, self.batch, steps=3,
                                      known=self.batch['geometry'], known_mask=known_mask)
        torch.testing.assert_close(result[0, :2, 0], self.batch['geometry'][0, :2, 0])
        torch.testing.assert_close(result[0, 1, 1:3], self.batch['geometry'][0, 1, 1:3])
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(float(result[1].abs().sum()), 0.)
        self.assertEqual(float(result[0, 0, 1:].abs().sum()), 0.)


if __name__ == '__main__': unittest.main()
