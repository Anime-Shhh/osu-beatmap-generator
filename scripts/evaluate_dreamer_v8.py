"""Run pinned upstream osu-dreamer v8, then evaluate without reference-driven selection.

Upstream source is provided through PYTHONPATH; no upstream code is vendored here.
The published checkpoint's training membership is unknown, so this is a development
comparison, not evidence of held-out generalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import zipfile

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--original-osz', type=Path, required=True)
    p.add_argument('--profile-guide', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidates', type=int, default=16)
    p.add_argument('--steps', type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from osu_dreamer.models.inference.artifact import load_inference
    from osu_dreamer.data.load_audio import load_wave, make_spec
    from osu_dreamer.data.beatmap.decode import decode_beatmap, Metadata
    from src.beatmap_flow.generate import profile_for_request
    from src.v2.evaluation import RosuDifficultyCalculator
    from src.v2.mapspec import mapspec_from_osu
    from scripts.compare_native_rhythm import compare, events_of
    from scripts.summarize_kaguya_comparison import spatial_pairs

    if not torch.cuda.is_available():
        raise RuntimeError('Run through a GPU Slurm allocation')
    torch.set_num_threads(1)
    model = load_inference(a.checkpoint).cuda().eval()
    print(json.dumps({'loaded': True, 'torch': torch.__version__,
        'parameters': sum(x.numel() for x in model.parameters())}), flush=True)
    settings, _ = profile_for_request(4.5, json.loads(a.profile_guide.read_text()))
    requested = [4.5, settings.approach_rate, settings.overall_difficulty,
                 settings.circle_size, settings.hp_drain_rate]
    labels = torch.tensor([requested], dtype=torch.float32, device='cuda')
    audio = torch.tensor(make_spec(load_wave(a.audio)), dtype=torch.float32, device='cuda')
    calc = RosuDifficultyCalculator()
    candidates = []
    # Finish inference before loading any reference chart.
    for i in range(a.candidates):
        seed = 1731 + i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        start = time.monotonic()
        with torch.inference_mode():
            signals, predicted_labels = model.sample(audio, labels, a.steps)
        signals = signals.cpu().numpy()[0]
        predicted_labels = predicted_labels.cpu().numpy()[0]
        if not np.isfinite(signals).all() or not np.isfinite(predicted_labels).all():
            raise ValueError('Nonfinite model output')
        text = decode_beatmap(Metadata('audio.mp3', 'DADDY ! DADDY ! DO ! (TV Size)',
            'Suzuki Masayuki feat. Suzuki Airi', f'osu-dreamer v8 seed {seed}'), predicted_labels, signals)
        path = a.output/f'candidate-{i}.osu'
        path.write_text(text)
        stars = calc.calculate(text).stars
        spec = mapspec_from_osu(text)
        events = events_of(spec)
        invalid_heads = sum(not (0 <= e['x'] <= 512 and 0 <= e['y'] <= 384) for e in events)
        candidates.append({'candidate': i, 'seed': seed, 'stars': stars,
            'predicted_labels': predicted_labels.tolist(), 'seconds': time.monotonic()-start,
            'objects': len(events), 'out_of_bounds_heads': invalid_heads, 'osu_sha256': sha(path)})
        print(json.dumps(candidates[-1]), flush=True)
    # Select by measured difficulty only, never by matching the human reference.
    selected = min(candidates, key=lambda c: (c['out_of_bounds_heads'] > 0, abs(c['stars']-4.5)))
    selected_path = a.output/f"candidate-{selected['candidate']}.osu"
    with zipfile.ZipFile(a.output/'Kaguya-osu-dreamer-v8.osz', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(a.audio, 'audio.mp3')
        archive.write(selected_path, f"Kaguya - osu-dreamer v8 ({selected['stars']:.2f} stars).osu")
    references = []
    with zipfile.ZipFile(a.original_osz) as archive:
        for name in archive.namelist():
            if not name.endswith('.osu'): continue
            text = archive.read(name).decode('utf-8-sig')
            spec = mapspec_from_osu(text)
            if spec.general.mode != 0: continue
            if hashlib.sha256(archive.read(str(Path(name).parent/spec.general.audio_filename))).hexdigest() != sha(a.audio):
                raise ValueError('Original audio mismatch')
            references.append((abs(calc.calculate(text).stars-4.5), spec))
    _, reference = min(references, key=lambda r:r[0])
    truth = events_of(reference)
    beat_length = reference.timing_map.beat_length_at_time(truth[0]['start_ms'])
    for c in candidates:
        spec = mapspec_from_osu((a.output/f"candidate-{c['candidate']}.osu").read_text())
        events = events_of(spec)
        c.update(compare(events, truth, reference.timing_map))
        c['spatial_pairs'] = spatial_pairs(events, beat_length)
    summary = {}
    for name, values in {
        'f1_50ms': [c['onsets']['50']['f1'] for c in candidates],
        'f1_10ms': [c['onsets']['10']['f1'] for c in candidates],
        'objects': [c['objects'] for c in candidates],
        'stars': [c['stars'] for c in candidates],
        'short_circle_pairs_recovered': [c['short_circle_pairs']['both_as_circles_50ms'] for c in candidates],
    }.items():
        summary[name] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
    report = {'upstream_commit': '0ec0b8e72fb1642fc3ecb365a61df4ec36e060dc',
        'checkpoint_url': 'https://huggingface.co/jaswon/osu-dreamer/resolve/main/v8.pt',
        'checkpoint_sha256': sha(a.checkpoint), 'audio_sha256': sha(a.audio),
        'torch': torch.__version__, 'parameters': sum(x.numel() for x in model.parameters()),
        'requested_labels': requested, 'steps': a.steps, 'reference_version': reference.metadata.version,
        'reference_spatial_pairs': spatial_pairs(truth, beat_length),
        'training_overlap': 'unknown for published weights; not a held-out benchmark',
        'selection': 'prefer in-bounds heads, then closest measured stars; no reference matching',
        'star_tolerance_passes': sum(abs(c['stars']-4.5) <= .5 for c in candidates),
        'selected': selected['candidate'], 'summary': summary, 'candidates': candidates}
    (a.output/'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'candidates'}), flush=True)


if __name__ == '__main__':
    main()
