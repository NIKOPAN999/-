"""Export paired per-sample predictions from a frozen checkpoint and calibration."""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from calibration import apply_calibration
from data import AlignedDataset, load_aligned
from run import load_checkpoint, predict_raw, read_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--split', choices=('valid', 'test'), required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, _ = load_checkpoint(args.checkpoint, device)
    params = read_calibration(args.calibration, args.checkpoint)
    dataset = AlignedDataset(load_aligned(args.data)[args.split], stats)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    conditions = {'clean': None, 'text_0.5_end': ((0,), 0.5, 'end')}
    results = {}
    for name, condition in conditions.items():
        labels, logits, scores = predict_raw(model, loader, device, condition)
        classes, values = apply_calibration(logits, scores, params)
        results[name] = (labels, classes, values)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ('sample_id', 'video_id', 'true_class', 'true_intensity',
                  'clean_class', 'clean_intensity',
                  'text_0.5_end_class', 'text_0.5_end_intensity')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(dataset.sample_ids):
            row = {'sample_id': sample_id,
                   'video_id': sample_id.split('$_$')[0],
                   'true_class': int(results['clean'][0]['cls'][i]),
                   'true_intensity': float(results['clean'][0]['reg'][i])}
            for name in conditions:
                _, classes, values = results[name]
                row[f'{name}_class'] = int(classes[i])
                row[f'{name}_intensity'] = float(values[i])
            writer.writerow(row)
    print(f'Wrote {len(dataset)} {args.split} samples to {args.output_csv}')


if __name__ == '__main__':
    main()
