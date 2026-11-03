"""One real-data forward pass plus per-sample prediction contract check."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

B = Path('/root/autodl-tmp/team/shared/baselines')
sys.path.insert(0, str(B / 'repos/MMSA/src'))
sys.path.insert(0, str(B / 'configs/mmsa'))
from MMSA.config import get_config_regression
from MMSA.data_loader import MMDataLoader
from MMSA.models import AMIO
from run_mmsa import materialize_config
from predict_almt import do_test_with_samples

args = get_config_regression('almt', 'mosei', str(materialize_config()))
args.update(device=torch.device('cuda:0'), train_mode='regression',
            custom_feature=str(B / 'data/mosei_att2_aligned50.pkl'),
            feature_T=None, feature_A=None, feature_V=None,
            use_bert=True, use_finetune=False, need_data_aligned=True)
loaders = MMDataLoader(args, 0)
batch = next(iter(loaders['valid']))
model = AMIO(args).to(args.device).eval()
with torch.inference_mode():
    result = model(batch['text'].to(args.device),batch['audio'].to(args.device),batch['vision'].to(args.device))
assert result.shape == (len(batch['id']), 1), result.shape
trainer = SimpleNamespace(args=args, criterion=torch.nn.MSELoss(), metrics=lambda p,y:{'MAE':float((p-y).abs().mean())})
out = do_test_with_samples(trainer, model, [batch], return_sample_results=True)
assert len(out['Ids']) == len(out['SResults']) == len(out['Labels']) == len(batch['id'])
print(dict(batch=len(batch['id']),shape=list(result.shape),keys=list(out),valid_ids=out['Ids'][:3]))
