import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from src.beatmap_flow.data import OnsetWindows
from src.beatmap_flow.rhythm import OnsetFlow, RhythmConfig, decode_events, rhythm_loss, sample_rhythm
from src.rhythm_flow.data import targets_for_times


class OnsetFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.model = OnsetFlow(RhythmConfig(width=16, audio_layers=2, flow_layers=2,
                                           window_frames=64, overlap_frames=16)).eval()

    def test_padding_invariance(self):
        audio = torch.randn(1, 101, 31)
        profile = torch.randn(1, 6)
        x = torch.randn(1, 1, 31)
        onset = torch.rand_like(x)
        with torch.no_grad():
            self.model.velocity.weight.normal_()
            out = self.model(audio, profile, torch.ones(1, 31, dtype=torch.bool), x, torch.tensor([.4]), onset)
            padded = self.model(F.pad(audio, (0, 25)), profile,
                torch.arange(56)[None] < 31, F.pad(x, (0, 25)), torch.tensor([.4]), F.pad(onset, (0, 25)))
        for a, b in zip(out, padded):
            torch.testing.assert_close(a, b[..., :31], atol=2e-5, rtol=2e-5)

    def test_empty_and_partial_windows_backward(self):
        self.model.train()
        b = {'audio': torch.randn(2, 101, 64), 'profile': torch.rand(2, 6),
             'mask': torch.arange(64)[None] < torch.tensor([0, 39])[:, None],
             'onset': torch.zeros(2, 1, 64), 'kind': torch.full((2, 64), -1),
             'duration': torch.zeros(2, 64)}
        result = rhythm_loss(self.model, b)
        self.assertTrue(all(torch.isfinite(v) for v in result.values()))
        result['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_full_length_and_seed(self):
        audio, profile = torch.randn(1, 101, 147), torch.rand(1, 6)
        a = sample_rhythm(self.model, audio, profile, steps=3, generator=torch.Generator().manual_seed(2))
        b = sample_rhythm(self.model, audio, profile, steps=3, generator=torch.Generator().manual_seed(2))
        self.assertEqual(a[0].shape, (1, 1, 147))
        self.assertEqual(a[1].shape, (1, 4, 147))
        for x, y in zip(a, b): torch.testing.assert_close(x, y)

    def test_phrase_attention_padding_and_empty_rows(self):
        model = OnsetFlow(RhythmConfig(width=16, audio_layers=2, flow_layers=2,
            window_frames=64, overlap_frames=16, context_layers=2, density_auxiliary=True)).eval()
        with torch.no_grad():
            model.phrase_context.out.weight.normal_(std=.1)
            audio, profile = torch.randn(2, 101, 31), torch.rand(2, 6)
            mask = torch.arange(31)[None] < torch.tensor([31, 0])[:, None]
            a, _ = model.encode_audio(audio, profile, mask)
            b, _ = model.encode_audio(F.pad(audio, (0, 25)), profile, F.pad(mask, (0, 25)))
        self.assertTrue(torch.isfinite(b).all())
        torch.testing.assert_close(a, b[..., :31], atol=2e-5, rtol=2e-5)
        self.assertEqual(float(b[1].abs().sum()), 0.)
        model.train()
        batch = {'audio': audio, 'profile': profile, 'mask': mask,
                 'onset': torch.zeros(2, 1, 31), 'onset_count': torch.zeros(2, 1, 31),
                 'kind': torch.full((2, 31), -1), 'duration': torch.zeros(2, 31)}
        batch['onset_count'][0, 0, 10] = 1
        result = rhythm_loss(model, batch)
        result['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(result['phrase_density_huber']), 0.)

    def test_type_duration_independent_of_peak_height(self):
        onset, marks = torch.zeros(100), torch.zeros(4, 100)
        onset[10], onset[50] = .8, .6
        marks[1, 10], marks[0, 50] = 3, 3
        marks[3, 10] = torch.log1p(torch.tensor(2.))
        events = decode_events(onset, marks, threshold=.4)
        self.assertEqual([e['kind'] for e in events], [1, 0])
        self.assertAlmostEqual(events[0]['end_ms'], 300., places=3)
        self.assertEqual(events[1]['end_ms'], 500.)

    def test_dataset_labels_match_native_times_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'audio.pt'
            torch.save({'audio': torch.randn(51, 101).half()}, path)
            events = [[14., 14., 0], [116., 310., 1], [403., 493., 2]]
            row = {'audio_key': 'key', 'audio_file': path.name,
                   'audio_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'source_component': 's', 'split': 'train', 'profile': [.5]*6,
                   'events': events}
            (Path(folder)/'manifest.json').write_text(json.dumps({
                'schema': 'osu-rhythm-flow-cache/1', 'frame_ms': 10,
                'feature_pipeline_key': 'fixture', 'charts': [row]}))
            data = OnsetWindows(folder, 'train', window=64, samples_per_source=1)
            sample = data[0]
            expected = targets_for_times(torch.arange(64)*10., events)[:3].amax(0)*sample['mask']
            torch.testing.assert_close(sample['onset'][0], expected)
            self.assertEqual(int(sample['kind'][12]), 1)
            self.assertAlmostEqual(float(sample['duration'][12].expm1()*100), 194., places=3)
            self.assertEqual(int(sample['mask'].sum()), 51)


if __name__ == '__main__':
    unittest.main()
