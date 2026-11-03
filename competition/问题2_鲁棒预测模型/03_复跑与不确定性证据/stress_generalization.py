"""Evaluate fixed model on generic two-block and asynchronous valid masks."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

core = Path(__file__).resolve().parent.parent / '01_问题二模型代码与参数'
sys.path.insert(0, str(core))
from calibration import apply_calibration
from data import AlignedDataset, block_mask, load_aligned
from run import calculate_metrics, load_checkpoint, read_calibration, to_device


def two_text_blocks(batch):
    """Mask the same rounded 50% content count as block_mask, split by one kept slot."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    for i in range(len(out['ids'])):
        slots = torch.where(out['content'][i])[0]
        n = len(slots)
        if n <= 1:
            continue
        length = min(n-1, max(1, round(0.5*n)))
        first = length // 2
        second = length - first
        start = (n - length - 1) // 2
        if first == 0:
            indices = slots[(n-length)//2:(n-length)//2+length]
        else:
            indices = torch.cat((slots[start:start+first],
                                 slots[start+first+1:start+first+1+second]))
        assert len(indices) == length
        out['ids'][i, indices] = 103
        out['observed'][i, indices, 0] = False
    return out


def transform(name, batch):
    if name == 'clean':
        return batch
    if name == 'text_middle_0.5_one':
        return block_mask(batch, (0,), 0.5, 'middle')
    if name == 'text_middle_0.5_two':
        return two_text_blocks(batch)
    if name == 'triple_middle_0.3_sync':
        return block_mask(batch, (0, 1, 2), 0.3, 'middle')
    if name == 'triple_0.3_async':
        out = block_mask(batch, (0,), 0.3, 'front')
        out = block_mask(out, (1,), 0.3, 'middle')
        return block_mask(out, (2,), 0.3, 'end')
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, _ = load_checkpoint(args.checkpoint, device)
    params = read_calibration(args.calibration, args.checkpoint)
    data = AlignedDataset(load_aligned(args.data)['valid'], stats)
    loader = DataLoader(data, batch_size=16, shuffle=False)
    two_possible = sum(min(int(n)-1, max(1, round(0.5*int(n)))) >= 2
                       for n in data.content.sum(axis=1))
    names = ('clean', 'text_middle_0.5_one', 'text_middle_0.5_two',
             'triple_middle_0.3_sync', 'triple_0.3_async')
    predictions = {}
    for name in names:
        logits, scores = [], []
        with torch.no_grad():
            model.eval()
            for clean in loader:
                masked = transform(name, clean)
                result = model(to_device(masked, device))
                logits.append(result['logits'].cpu().numpy())
                scores.append(result['reg'].cpu().numpy())
        classes, values = apply_calibration(np.concatenate(logits),
                                            np.concatenate(scores), params)
        predictions[name] = (classes, values)
    labels = {'cls': data.cls, 'reg': data.reg}
    results = [{'scenario': name, 'n': len(data),
                **calculate_metrics(labels, *predictions[name])} for name in names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'valid_generalization_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    fields = ['sample_id', 'true_class', 'true_intensity']
    for name in names:
        fields.extend((name+'_class', name+'_intensity'))
    with (args.output_dir / 'valid_generalization_samples.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, sid in enumerate(data.sample_ids):
            row = {'sample_id': sid, 'true_class': int(data.cls[i]),
                   'true_intensity': float(data.reg[i])}
            for name in names:
                classes, values = predictions[name]
                row[name+'_class'] = int(classes[i])
                row[name+'_intensity'] = float(values[i])
            writer.writerow(row)
    for row in results:
        print(row['scenario'], *(f'{row[k]:.4f}' for k in ('accuracy','macro_f1','mae','pearson')),
              flush=True)
    print(f'two-block split feasible in {two_possible}/{len(data)} samples; '
          'others use one block with the same nominal masked count', flush=True)


if __name__ == '__main__':
    main()
