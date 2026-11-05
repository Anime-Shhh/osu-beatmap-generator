"""Label a selected diagnostic map with its measured stars and package original audio."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from src.v2.evaluation import RosuDifficultyCalculator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.run/'comparison-summary.json').read_text())
    name = report['recommended_model']
    source = a.run/f'kaguya-{name}'
    parent = (source/'generated.osu').read_text()
    stars = report['models'][name]['selected_stars']
    replacements = {'Title': 'DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size)',
                    'Artist': 'Suzuki Masayuki', 'Creator': 'Audio Native Flow',
                    'Version': f'Experimental {name} {stars:.2f} stars'}
    metadata = False
    lines = []
    for line in parent.splitlines():
        if line.startswith('['): metadata = line == '[Metadata]'
        if metadata and ':' in line:
            key = line.split(':', 1)[0]
            if key in replacements: line = key+':'+replacements[key]
        lines.append(line)
    text = '\n'.join(lines)+'\n'
    # Gameplay sections remain byte-identical; the rewrite only labels the draft.
    if text.split('[TimingPoints]', 1)[1] != parent.split('[TimingPoints]', 1)[1]:
        raise ValueError('packaging changed gameplay')
    measured = RosuDifficultyCalculator().calculate(text).stars
    if abs(measured-stars) > 1e-8: raise ValueError('difficulty changed during labeling')
    output = a.run/'playtest'
    output.mkdir(exist_ok=False)
    osu = output/'Kaguya - Audio Native Flow.osu'
    osu.write_text(text)
    osz = output/f'Kaguya-{name}-{stars:.2f}-stars.osz'
    with zipfile.ZipFile(osz, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(osu, osu.name)
        archive.write(source/'audio.mp3', 'audio.mp3')
    audit = {'selected_model': name, 'measured_stars': stars, 'requested_stars': 4.5,
             'selection_basis': report['selection_basis'], 'quality': 'experimental draft; human playtest pending',
             'parent_osu_sha256': hashlib.sha256((source/'generated.osu').read_bytes()).hexdigest(),
             'osu_sha256': hashlib.sha256(osu.read_bytes()).hexdigest(),
             'osz_sha256': hashlib.sha256(osz.read_bytes()).hexdigest()}
    (output/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'osz': str(osz), **audit}), flush=True)


if __name__ == '__main__': main()
