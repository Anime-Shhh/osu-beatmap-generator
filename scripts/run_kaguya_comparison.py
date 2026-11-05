"""Fresh same-audio multi-seed inference and native-reference rhythm comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

from scripts.compare_native_rhythm import compare, events_of
from src.v2.mapspec import mapspec_from_osu
from src.v2.evaluation import RosuDifficultyCalculator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    decoder = p.add_mutually_exclusive_group(required=True)
    decoder.add_argument('--calibration', type=Path)
    decoder.add_argument('--threshold', type=float, help='fixed-threshold diagnostic, not recalibration')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-manifest', type=Path, required=True)
    p.add_argument('--candidates', type=int, default=16)
    a = p.parse_args()
    root = Path('/common/users/asj102/osu_project')
    old = root/'flow_runs/run_20261025'
    archive_path = next(root.glob('1147361*.osz'))
    audio_path = old/'kaguya-original-input/audio.mp3'
    # Star tolerance is widened for a diagnostic comparison, explicitly recorded.
    # No reference events or geometry enter inference or candidate selection.
    command = [sys.executable, '-m', 'src.beatmap_flow.generate', str(audio_path),
        '--stars', '4.5', '--rhythm-checkpoint', str(a.checkpoint),
        '--geometry-checkpoint', str(old/'geometry/best.pt'),
        '--profile-guide', str(old/'profile-guide.json'),
        '--candidates', str(a.candidates), '--steps', '48', '--seed', '1731',
        '--star-tolerance', '2', '--output-dir', str(a.output)]
    command += ['--calibration', str(a.calibration)] if a.calibration else ['--threshold', str(a.threshold)]
    run = subprocess.run(command)
    if run.returncode not in (0, 2): raise RuntimeError('inference failed')
    audit = json.loads((a.output/'audit.json').read_text())
    calculator = RosuDifficultyCalculator()
    with zipfile.ZipFile(archive_path) as archive:
        originals = []
        for name in archive.namelist():
            if not name.endswith('.osu'): continue
            text = archive.read(name).decode('utf-8-sig')
            spec = mapspec_from_osu(text)
            if spec.general.mode != 0: continue
            if hashlib.sha256(archive.read(spec.general.audio_filename)).hexdigest() != audit['hashes']['audio']:
                raise ValueError('original audio mismatch')
            originals.append((abs(calculator.calculate(text).stars-4.5), spec, calculator.calculate(text).stars))
    _, reference, stars = min(originals, key=lambda x: x[0])
    truth = events_of(reference)
    scores = []
    for c in audit['candidates']:
        path = a.output/f"candidate-{c['candidate']}-events.json"
        if path.exists():
            events = json.loads(path.read_text())
            scores.append({'candidate': c['candidate'], 'seed': c['seed'],
                'stars': c.get('difficulty', {}).get('stars'), **compare(events, truth, reference.timing_map)})
    summaries = {}
    values = {
        'onset_f1_50ms': [r['onsets']['50']['f1'] for r in scores],
        'onset_recall_50ms': [r['onsets']['50']['recall'] for r in scores],
        'onset_precision_50ms': [r['onsets']['50']['precision'] for r in scores],
        'onset_f1_10ms': [r['onsets']['10']['f1'] for r in scores],
        'objects': [r['generated']['objects'] for r in scores],
        'short_circle_pairs_recovered': [r['short_circle_pairs']['both_as_circles_50ms'] for r in scores],
    }
    for k, v in values.items():
        summaries[k] = {'mean': statistics.mean(v), 'stdev': statistics.stdev(v) if len(v)>1 else 0,
                        'min': min(v), 'max': max(v)}
    report = {'reference_version': reference.metadata.version, 'reference_stars': stars,
        'role': 'development comparison, reference unavailable during inference',
        'selection': 'closest measured stars among candidates passing diagnostic gates; no reference matching selection',
        'star_tolerance': 2, 'default_tolerance_passes': sum(
            c.get('difficulty', {}).get('stars') is not None and abs(c['difficulty']['stars']-4.5) <= .5
            and not [e for e in c.get('errors', []) if not e.startswith('realized difficulty')]
            for c in audit['candidates']),
        'checkpoint_sha256': audit['hashes']['rhythm'], 'threshold': audit['threshold'],
        'selected': audit['selected'], 'summary': summaries, 'candidates': scores}
    (a.output/'rhythm-comparison.json').write_text(json.dumps(report, indent=2))
    if run.returncode == 0:
        subprocess.run([sys.executable, '-m', 'scripts.compare_native_rhythm',
            '--original-osz', str(archive_path), '--generated-osu', str(a.output/'generated.osu'),
            '--generated-audio', str(a.output/'audio.mp3'), '--cache-manifest', str(a.cache_manifest),
            '--output-dir', str(a.output/'selected-comparison')], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({k:v for k,v in report.items() if k != 'candidates'}), flush=True)


if __name__ == '__main__': main()
