"""Reproducible structural comparison; head-to-head speeds are not osu! aim strain."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

from src.v2.evaluation import RosuDifficultyCalculator
from src.v2.events import EventKind
from src.v2.mapspec import mapspec_from_osu


def summarize(text, calculator):
    spec = mapspec_from_osu(text)
    events = spec.events
    heads = [(e.x, e.y, spec.timing_map.beat_to_time(e.beat_position)) for e in events]
    gaps = [b[2]-a[2] for a,b in zip(heads, heads[1:])]
    distances = [math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(heads, heads[1:])]
    speeds = [d*1000/g for d,g in zip(distances,gaps) if g > 0]
    sliders = [e for e in events if e.kind == EventKind.SLIDER]
    return {
        'title': spec.metadata.title, 'version': spec.metadata.version,
        'difficulty': calculator.calculate(text).to_dict(),
        'object_count':len(events), 'slider_count':len(sliders),
        'curve_types':dict(Counter(e.curve_type for e in sliders)),
        'corner_heads':sum(x in (0,512) and y in (0,384) for x,y,_ in heads),
        'minimum_start_gap_ms': min(gaps,default=0),
        'mean_head_distance_px':mean(distances) if distances else 0,
        'mean_head_speed_px_per_second':mean(speeds) if speeds else 0,
        'max_head_speed_px_per_second':max(speeds,default=0),
        'max_slider_length_px':max((e.slider_length for e in sliders),default=0),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',type=Path,nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    calculator=RosuDifficultyCalculator(); results={}
    for path in args.paths:
        if path.suffix.lower()=='.osz':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith('.osu'):
                        results[f'{path.name}:{name}']=summarize(archive.read(name).decode('utf-8-sig'),calculator)
        else:
            results[path.name]=summarize(path.read_text(encoding='utf-8-sig'),calculator)
    content=json.dumps(results,indent=2)+'\n'
    if args.output:
        args.output.write_text(content)
    else:
        print(content,end='')


if __name__=='__main__':
    main()
