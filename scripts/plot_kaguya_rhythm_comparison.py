"""Render an auditable native/generated rhythm comparison as PNG and SVG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.comparison_dir/'comparison.json').read_text())
    timeline = json.loads((a.comparison_dir/'timelines.json').read_text())
    ref = next(r for r in timeline['references'] if r['version'] == report['closest_original'])
    metrics = next(r for r in report['maps'] if r['version'] == report['closest_original'])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1.25, 1, 1]})
    colors = ['#176b9c', '#ce6134']
    segments = metrics['segments']
    xs = [(r['start_ms']+r['end_ms'])/2000 for r in segments]
    axes[0].plot(xs, [r['reference'] for r in segments], '-o', color=colors[0], label='Original: 324 objects / 4.49 stars')
    axes[0].plot(xs, [r['generated'] for r in segments], '-o', color=colors[1], label='Generated: 242 objects / 5.01 stars')
    axes[0].set(xlabel='Song time (seconds)', ylabel='Objects per 8 seconds', xlim=(0, 96), ylim=(0, 38))
    axes[0].legend(loc='lower left', ncol=2, fontsize=10)
    axes[0].grid(alpha=.2)
    for ax, limits in zip(axes[1:], [(40, 48), (64, 72)]):
        for level, events, color in [(1, ref['events'], colors[0]), (0, timeline['generated'], colors[1])]:
            for e in events:
                start, end = e['start_ms']/1000, e['end_ms']/1000
                if not limits[0] <= start < limits[1]: continue
                if e['kind']:
                    ax.plot([start, min(end, limits[1])], [level, level], color=color, lw=3, alpha=.7)
                ax.scatter([start], [level], marker='o' if e['kind'] == 0 else '|',
                           s=30 if e['kind'] == 0 else 130, color=color, zorder=3)
        ax.set(yticks=[0, 1], yticklabels=['Generated', 'Original'], ylim=(-.5, 1.5),
               xlim=limits, xlabel='Song time (seconds)')
        ax.set_title(f'{limits[0]}–{limits[1]} seconds: note starts and slider durations', fontsize=11, loc='left')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Kaguya: missing rhythmic detail, despite higher measured difficulty', x=.08, ha='left', fontsize=17)
    fig.text(.08, .925, 'Same original MP3 • 195/324 native starts matched within 50 ms • 6/36 short circle pairs reproduced', fontsize=11)
    fig.text(.08, .018, 'Circles show circle starts; vertical marks and lines show slider starts and holds. One human chart is a reference, not the only valid interpretation.', fontsize=9)
    fig.tight_layout(rect=(.01, .035, 1, .90), h_pad=1.6)
    for suffix in ('png', 'svg'):
        path = a.output_dir/f'kaguya-rhythm-comparison.{suffix}'
        fig.savefig(path, dpi=160)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__': main()
