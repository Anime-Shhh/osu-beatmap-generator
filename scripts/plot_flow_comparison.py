"""Plot the completed Kaguya comparison without treating sampling ranges as CIs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.run/'comparison-summary.json').read_text())
    names = ['original', 'control', 'phrase']
    labels = ['Previous model', 'More data', 'More data + phrase context']
    colors = ['#d67632', '#23886b', '#8659b3']
    timelines = {name: json.loads((a.run/f'kaguya-{name}/selected-comparison/timelines.json').read_text()) for name in names}
    version = report['models']['original']['selected_kaguya']['version']
    truth = next(x['events'] for x in timelines['original']['references'] if x['version'] == version)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [1, 1.4, 1.4]})
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        value = report['models'][name]['kaguya_across_seeds']['onset_f1_50ms']
        axes[0].barh(i, value['mean']*100, color=color, height=.55)
        axes[0].errorbar(value['mean']*100, i, xerr=[[100*(value['mean']-value['min'])],
            [100*(value['max']-value['mean'])]], fmt='none', color='#222222', capsize=5)
        axes[0].text(value['max']*100+1, i, f"{value['mean']*100:.1f}%", va='center')
    axes[0].set(yticks=range(3), yticklabels=labels, xlim=(0, 100),
                xlabel='Onset F1 against the original map (50 ms tolerance)')
    axes[0].invert_yaxis(); axes[0].grid(axis='x', alpha=.2)
    for ax, limits in zip(axes[1:], [(40, 48), (64, 72)]):
        rows = [('Human reference', truth, '#176b9c')]+[(label, timelines[name]['generated'], color)
            for name, label, color in zip(names, labels, colors)]
        for i, (_, events, color) in enumerate(rows):
            for event in events:
                start, end = event['start_ms']/1000, event['end_ms']/1000
                if not limits[0] <= start < limits[1]: continue
                if event['kind']: ax.plot([start, min(end, limits[1])], [i, i], lw=2.5, color=color, alpha=.65)
                ax.scatter(start, i, marker='|' if event['kind'] else 'o',
                           s=100 if event['kind'] else 23, color=color, zorder=3)
        ax.set(yticks=range(4), yticklabels=[r[0] for r in rows], ylim=(3.6, -.6), xlim=limits,
               xlabel='Song time (seconds)')
        ax.set_title(f'{limits[0]}–{limits[1]} seconds: selected maps, note starts and slider holds', loc='left', fontsize=11)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Kaguya: previous model vs. retrained rhythm models', x=.04, ha='left', fontsize=18)
    fig.text(.04, .94, 'Exact original MP3 • Same 4.5-star request and placement model • 16 matched generation seeds', fontsize=11)
    fig.text(.04, .025, 'Bars: mean and min–max over sampling seeds, not confidence intervals. Timelines: candidates selected by measured difficulty, not reference similarity.', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .92), h_pad=2)
    fig.savefig(a.run/'kaguya-comparison.png', dpi=160)


if __name__ == '__main__': main()
