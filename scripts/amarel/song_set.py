"""Generate a multi-difficulty Dreamer playtest set from a supplied MP3.

Select by measured difficulty among valid exports; never use a reference chart.
"""
import argparse
import hashlib
import math
import random
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

from src.dreamer_adapter.contracts import digest, write_json
from src.dreamer_adapter.evaluate import decode
from src.v2.evaluation import RosuDifficultyCalculator
from src.v2.mapspec import mapspec_from_osu, validate_map_spec, ValidationPolicy
from src.v2.slider_path import event_path


def metadata(text, title, artist, version):
    values = dict(Title=title, TitleUnicode=title, Artist=artist,
                  ArtistUnicode=artist, Creator='osu!dreamer', Version=version,
                  BeatmapID='0', BeatmapSetID='-1')
    start = text.index('[Metadata]') + len('[Metadata]')
    end = text.index('[', start)
    section = text[start:end]
    for key, value in values.items():
        if '\n' in value or '\r' in value:
            raise ValueError('metadata must be single-line')
        pattern = r'(?m)^' + key + r'\s*:.*$'
        if re.search(pattern, section):
            section = re.sub(pattern, lambda _: key + ': ' + value, section)
        else:
            section = section.rstrip() + '\n' + key + ': ' + value + '\n\n'
    return text[:start] + section + text[end:]


def inspect(text, duration_ms):
    spec = mapspec_from_osu(text)
    report = validate_map_spec(spec, ValidationPolicy(require_beat_snap=False,
                                                     allow_custom_samples=True))
    errors = list(report.errors)
    if not spec.events:
        errors.append('empty map')
    for event in spec.events:
        path = event_path(event)
        if not np.isfinite(path).all() or np.any(path < 0) or np.any(path[:, 0] > 512) or np.any(path[:, 1] > 384):
            errors.append('rendered body outside playfield')
    # Check serialized starts independently of internal event time units.
    objects = text.split('[HitObjects]', 1)[1].strip().splitlines()
    starts = [float(line.split(',')[2]) for line in objects if line.strip()]
    if any(t < 0 or t > duration_ms for t in starts):
        errors.append('object start outside audio')
    difficulty = RosuDifficultyCalculator().calculate(text)
    if not difficulty.available or difficulty.stars is None or not math.isfinite(difficulty.stars):
        errors.append('difficulty unavailable')
    return dict(stars=difficulty.stars, calculator=difficulty.calculator_id,
                invalid=bool(errors), errors=sorted(set(errors)),
                objects=len(spec.events), first_object_ms=min(starts, default=None),
                last_object_ms=max(starts, default=None))


@torch.inference_mode()
def run(audio, checkpoint, output, title, artist, targets=(2., 3., 4.),
        candidates=16, seed=1731, device='cuda', threads=4, source_url=None):
    from osu_dreamer.data.load_audio import make_spec, load_wave
    from osu_dreamer.models.inference.artifact import load_inference
    import soundfile as sf

    if not 1 <= candidates <= 64 or not targets or len(set(targets)) != len(targets) or any(not 0 < t <= 15 for t in targets):
        raise ValueError('invalid candidate count or target difficulties')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    audio = Path(audio)
    if audio.suffix.lower() != '.mp3':
        raise ValueError('supply an MP3; format is preserved in the archive')
    duration_ms = sf.info(audio).duration * 1000
    torch.set_num_threads(threads)
    features = torch.tensor(make_spec(load_wave(audio)), device=device, dtype=torch.float32)
    model = load_inference(Path(checkpoint)).to(device).eval()
    audit = dict(title=title, artist=artist, audio_sha256=digest(audio),
                 audio_source=source_url, duration_ms=duration_ms,
                 checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                 sampler_steps=8, target_tolerance=.2, device=device,
                 selection='closest measured stars among valid candidates',
                 timing='upstream native export; no reference timing or post-hoc snapping',
                 results=[])
    selected_texts = {}
    for target in targets:
        # Explicit simple playtest profiles, with CS fixed across the set.
        ar, od, cs, hp = min(9., target + 3), min(8., target + 2), 4., min(7., target + 2)
        labels = torch.tensor([[target, ar, od, cs, hp]], device=device)
        rows, valid = [], []
        for candidate_seed in range(seed, seed + candidates):
            torch.manual_seed(candidate_seed)
            np.random.seed(candidate_seed)
            random.seed(candidate_seed)
            signals, settings = model.sample(features, labels, 8)
            try:
                text = decode(signals[0].cpu().numpy(), settings[0].cpu().numpy())
                result = inspect(text, duration_ms)
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as exc:
                result = dict(stars=None, invalid=True, errors=[repr(exc)])
                text = ''
            row = dict(seed=candidate_seed, **result)
            rows.append(row)
            if not row['invalid']:
                valid.append((abs(row['stars'] - target), row, text))
            print(dict(target=target, **row), flush=True)
        entry = dict(requested_stars=target, conditioning=[target, ar, od, cs, hp], candidates=rows, selected=None)
        if valid:
            error, chosen, text = min(valid, key=lambda item: item[0])
            stars = chosen['stars']
            version = f'Fine-tuned Dreamer - {target:g} requested - {stars:.2f} stars'
            text = metadata(text, title, artist, version)
            final = inspect(text, duration_ms)
            assert not final['invalid'] and abs(final['stars'] - stars) < 1e-6
            filename = f'dreamer-{target:g}.osu'
            (output / filename).write_text(text)
            selected_texts[filename] = text
            entry['selected'] = dict(**chosen, target_pass=error <= .2, filename=filename,
                                     sha256=hashlib.sha256(text.encode()).hexdigest())
        audit['results'].append(entry)
        write_json(output / 'generation.json', audit)
    if len(selected_texts) != len(targets):
        raise RuntimeError('at least one target has no valid candidate; see generation.json')
    package = output / 'dreamer-set.osz'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(audio, 'audio.mp3')
        for filename, text in selected_texts.items():
            archive.writestr(filename, text)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert hashlib.sha256(archive.read('audio.mp3')).hexdigest() == audit['audio_sha256']
        for filename, text in selected_texts.items():
            assert archive.read(filename) == text.encode()
            assert mapspec_from_osu(text).general.audio_filename == 'audio.mp3'
    audit.update(archive_sha256=digest(package), archive_verified=True)
    write_json(output / 'generation.json', audit)
    print('COMPLETE', package, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('audio', 'checkpoint', 'output', 'title', 'artist'):
        p.add_argument('--' + arg, required=True)
    p.add_argument('--targets', type=float, nargs='+', default=[2., 3., 4.])
    p.add_argument('--candidates', type=int, default=16)
    p.add_argument('--seed', type=int, default=1731)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--source-url')
    run(**vars(p.parse_args()))
