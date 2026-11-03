"""Summarize the three same-environment seed-42 valid runs."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
views = ('clean', 'text_0.3_middle', 'text_0.5_end',
         'audio_0.3_middle', 'vision_0.3_middle',
         'all_modalities_0.3_middle')
metrics = ('accuracy', 'macro_f1', 'mae', 'pearson')
rows = []
for name in ('full_s42', 'no_mask_s42', 'no_aux_s42'):
    path = root / name / 'compact_valid_calibrated.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        cases = {row['scenario']: row for row in csv.DictReader(handle)}
    row = {'variant': name}
    for metric in metrics:
        row[f'clean_{metric}'] = float(cases['clean'][metric])
        row[f'text_0.5_end_{metric}'] = float(cases['text_0.5_end'][metric])
        row[f'six_mean_{metric}'] = sum(float(cases[v][metric]) for v in views) / 6
    rows.append(row)
with (root / 'comparison_valid.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('Wrote comparison_valid.csv from three 32-scenario valid files')
