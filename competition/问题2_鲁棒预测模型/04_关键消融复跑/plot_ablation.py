"""Recreate the two evidence-based ablation figures in the paper."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
views = ('clean', 'text_0.3_middle', 'text_0.5_end',
         'audio_0.3_middle', 'vision_0.3_middle',
         'all_modalities_0.3_middle')


def load(name):
    with (root / name / 'compact_valid_calibrated.csv').open(encoding='utf-8-sig', newline='') as f:
        return {r['scenario']: r for r in csv.DictReader(f)}


def mean_six(rows, key):
    return np.mean([float(rows[v][key]) for v in views])


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
names = ('full_s42', 'no_mask_s42', 'no_aux_s42')
labels = ('Full', 'No training mask', 'No auxiliary loss')
colors = ('#2864a4', '#d98932', '#548c79')
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.0), layout='constrained')
for ax, (metric, ylabel) in zip(axs, (('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'))):
    x = np.arange(2)
    for j, (name, label, color) in enumerate(zip(names, labels, colors)):
        rows = load(name)
        values = [float(rows[v][metric]) for v in ('clean', 'text_0.5_end')]
        bars = ax.bar(x + (j-1)*0.25, values, width=0.23, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.008, f'{value:.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, ('Clean input', 'Text end 50% masked'))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.35, 0.75)
    ax.grid(axis='y', alpha=0.18)
axs[0].legend(loc='upper right', frameon=False, fontsize=8)
fig.savefig(root / 'figure10_seed42.png', dpi=240, bbox_inches='tight')
plt.close(fig)

seeds = (42, 43, 44)
specs = (('accuracy', 1, 'Accuracy'), ('macro_f1', 1, 'Macro F1'),
         ('mae', -1, 'MAE'), ('pearson', 1, 'Pearson'))
fig, axs = plt.subplots(2, 2, figsize=(10.4, 5.8), layout='constrained')
for ax, (metric, sign, title) in zip(axs.flat, specs):
    ax.axhline(0, color='#777777', linewidth=1, linestyle='--')
    all_delta = []
    for variant, label, color in (('no_mask', 'No training mask', '#d98932'),
                                  ('no_aux', 'No auxiliary loss', '#548c79')):
        delta = []
        for seed in seeds:
            full = mean_six(load(f'full_s{seed}'), metric)
            ablated = mean_six(load(f'{variant}_s{seed}'), metric)
            delta.append(sign * (full - ablated))
        all_delta.extend(delta)
        ax.plot(seeds, delta, marker='o', color=color, linewidth=2, label=label)
        for seed, value in zip(seeds, delta):
            offset = (0, 9) if variant == 'no_mask' else (0, -16)
            ax.annotate(f'{value:+.3f}', (seed, value), xytext=offset,
                        textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(seeds)
    ax.set_xlabel('Training seed')
    ax.set_ylabel('Full minus ablation, oriented')
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.18)
    margin = max(0.010, max(abs(v) for v in all_delta)*0.35)
    ax.set_ylim(min(0, min(all_delta))-margin, max(0, max(all_delta))+margin)
handles, legend_labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.07),
           ncol=2, frameon=False, fontsize=9)
fig.savefig(root / 'figure11_three_seeds.png', dpi=240, bbox_inches='tight')
plt.close(fig)
print('Wrote figure10_seed42.png and figure11_three_seeds.png')
