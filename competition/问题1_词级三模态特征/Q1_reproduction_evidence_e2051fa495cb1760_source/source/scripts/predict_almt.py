"""ALMT per-sample inference adapter for the shared MMSA predictor.

Installs the missing return_sample_results behavior in this process only.
The shared MMSA repository and the common predictor are unchanged.
"""
from pathlib import Path
import sys

import torch

B = Path('/root/autodl-tmp/team/shared/baselines')
sys.path.insert(0, str(B / 'repos/MMSA/src'))
sys.path.insert(0, str(B / 'configs/mmsa'))

from MMSA.trains.singleTask.ALMT import ALMT
import predict_mmsa


def do_test_with_samples(self, model, dataloader, mode='VAL', return_sample_results=False):
    model.eval()
    preds, labels, ids = [], [], []
    loss = 0.0
    with torch.inference_mode():
        for batch in dataloader:
            vision = batch['vision'].to(self.args.device)
            audio = batch['audio'].to(self.args.device)
            text = batch['text'].to(self.args.device)
            true = batch['labels']['M'].to(self.args.device).view(-1, 1)
            pred = model(text, audio, vision)
            loss += self.criterion(pred, true).item()
            preds.append(pred.detach().cpu())
            labels.append(true.detach().cpu())
            ids.extend(str(x) for x in batch['id'])
    pred = torch.cat(preds)
    true = torch.cat(labels)
    results = dict(self.metrics(pred, true))
    results['Loss'] = round(loss / len(dataloader), 4)
    if return_sample_results:
        results.update(Ids=ids, SResults=pred.numpy().reshape(-1).tolist(),
                       Labels=true.numpy().reshape(-1).tolist())
    return results


ALMT.do_test = do_test_with_samples

if __name__ == '__main__':
    if '--model' not in sys.argv or sys.argv[sys.argv.index('--model') + 1] != 'almt':
        raise SystemExit('This adapter only supports --model almt')
    predict_mmsa.main()
