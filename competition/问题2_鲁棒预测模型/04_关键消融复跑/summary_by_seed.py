"""Rebuild the paired three-seed ablation summary from valid 32-view CSVs."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
views = ('clean', 'text_0.3_middle', 'text_0.5_end',
         'audio_0.3_middle', 'vision_0.3_middle',
         'all_modalities_0.3_middle')
metrics = ('accuracy', 'macro_f1', 'mae', 'pearson')
rows = []
for seed in (42, 43, 44):
    for variant in ('full', 'no_mask', 'no_aux'):
        with (root / f'{variant}_s{seed}' / 'compact_valid_calibrated.csv').open(
                encoding='utf-8-sig', newline='') as handle:
            cases = {row['scenario']: row for row in csv.DictReader(handle)}
        row = {'seed': seed, 'variant': variant}
        for metric in metrics:
            row[f'clean_{metric}'] = float(cases['clean'][metric])
            row[f'text_0.5_end_{metric}'] = float(cases['text_0.5_end'][metric])
            row[f'six_mean_{metric}'] = sum(float(cases[v][metric]) for v in views) / 6
        rows.append(row)
with (root / 'comparison_by_seed.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print('Wrote', len(rows), 'rows to comparison_by_seed.csv')
for variant in ('no_mask', 'no_aux'):
    print(variant)
    for metric in metrics:
        sign = -1 if metric == 'mae' else 1
        diff = []
        for seed in (42, 43, 44):
            full = next(r for r in rows if r['seed'] == seed and r['variant'] == 'full')
            other = next(r for r in rows if r['seed'] == seed and r['variant'] == variant)
            diff.append(sign * (full[f'six_mean_{metric}'] - other[f'six_mean_{metric}']))
        print(metric, [round(x, 4) for x in diff], 'mean', round(sum(diff)/3, 4))
