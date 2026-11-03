"""训练轻量版 Self-MM（BERT 冻结），并导出不含 BERT 的紧凑权重。

用法：python code/train_light.py --seed 1111 --pool mean
训练、选模与原版 MMSA Self-MM 完全相同：只用附件2训练集训练，按验证集 Loss 选检查点并早停（耐心 8）。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import light_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--pool', choices=list(C.POOLS), default='mean')
a = ap.parse_args()

C.apply_pool_patch(a.pool)
from MMSA import MMSA_run  # noqa: E402

out = C.EXP / f'outputs/{a.pool}/seed{a.seed}'
out.mkdir(parents=True, exist_ok=True)
(out / 'override.json').write_text(json.dumps({'seed': a.seed, 'pool': a.pool, 'override': C.override_for(a.pool)}, indent=2), encoding='utf-8')
MMSA_run('self_mm', 'mosei', config_file=str(C.materialize_config()), config=C.override_for(a.pool), seeds=[a.seed], gpu_ids=[0],
         num_workers=2, custom_feature=str(C.DATA_ATT2), model_save_dir=str(out / 'models'),
         res_save_dir=str(out / 'results'), log_dir=str(out / 'logs'))

# 导出紧凑权重：去掉 BERT，记录公开 BERT 的指纹，并核对训练后 BERT 未被改动
full = torch.load(out / 'models/self_mm-mosei.pth', map_location='cpu', weights_only=True)
bert, rest = C.split_state(full)
from transformers import BertModel  # noqa: E402
ref = BertModel.from_pretrained(str(C.BERT_DIR)).state_dict()
fp_trained, fp_ref = C.bert_fingerprint({f'model.{k}': v for k, v in ref.items()}), C.bert_fingerprint(bert)
same = all(torch.equal(bert[f'model.{k}'], v) for k, v in ref.items() if f'model.{k}' in bert)
torch.save({'state': rest, 'bert_fingerprint': fp_ref, 'pool': a.pool, 'seed': a.seed, 'override': C.override_for(a.pool)},
           out / 'compact.pt')
n_rest = sum(v.numel() for v in rest.values())
C.save_json({'seed': a.seed, 'pool': a.pool, 'bert_unchanged_vs_pretrained': bool(same), 'non_bert_params': int(n_rest),
             'compact_bytes': (out / 'compact.pt').stat().st_size}, out / 'compact_info.json')
print('compact saved', out / 'compact.pt', 'non-BERT params', n_rest, 'BERT unchanged', same, fp_trained == fp_ref)
