"""Recompute the paired video-level interval in paper Table 13."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


here = Path(__file__).resolve().parent
data = list(csv.DictReader((here / 'valid_sample_evidence.csv').open(encoding='utf-8-sig', newline='')))
groups = defaultdict(list)
for i, row in enumerate(data):
    groups[row['video_id']].append(i)
groups = list(groups.values())
y = np.array([int(r['true_class']) for r in data])
z = np.array([float(r['true_intensity']) for r in data])


def metrics(indices, name):
    truth = y[indices]
    pred = np.array([int(data[i][f'{name}_class']) for i in indices])
    reg = np.array([float(data[i][f'{name}_intensity']) for i in indices])
    f1 = []
    for label in (0, 1, 2):
        tp = np.sum((truth == label) & (pred == label))
        fp = np.sum((truth != label) & (pred == label))
        fn = np.sum((truth == label) & (pred != label))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return np.array([np.mean(truth == pred), np.mean(f1),
                     np.mean(np.abs(z[indices] - reg)),
                     np.corrcoef(z[indices], reg)[0, 1]])


def main():
    names = ('Accuracy', 'macroF1', 'MAE', 'Pearson')
    all_indices = np.arange(len(data))
    point = metrics(all_indices, 'text_0.5_end') - metrics(all_indices, 'clean')
    rng = np.random.default_rng(20260925)
    boot = np.empty((2000, 4))
    for b in range(len(boot)):
        choice = rng.integers(0, len(groups), size=len(groups))
        indices = np.fromiter((i for j in choice for i in groups[j]), dtype=int)
        boot[b] = metrics(indices, 'text_0.5_end') - metrics(indices, 'clean')
    limits = np.quantile(boot, [0.025, 0.975], axis=0)
    path = here / 'valid_video_bootstrap.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('metric', 'change', 'lower_95', 'upper_95', 'unit', 'seed', 'repetitions'))
        for j, name in enumerate(names):
            writer.writerow((name, point[j], limits[0, j], limits[1, j],
                             'video_id', 20260925, len(boot)))
    print(f'{len(data)} samples, {len(groups)} video groups -> {path}')


if __name__ == '__main__':
    main()
