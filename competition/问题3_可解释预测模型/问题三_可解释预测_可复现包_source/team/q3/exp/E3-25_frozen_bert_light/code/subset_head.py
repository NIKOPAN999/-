"""轻量版 Self-MM：训练子集融合头，并输出各数据划分的 8 子集预测。

协议与正式版（q3/code/train_subset_head.py）相同：
- 编码器冻结；文本缺失 = 正文词元换成 [UNK]（id 100，保留 [CLS]/[SEP] 与注意力掩码），语音、视觉缺失 = 零向量；
- 新头由原融合头初始化，损失 = 完整输入 MAE + λ·七种缺失组合平均 MAE，λ ∈ {0.1, 0.25, 0.5}；
- AdamW(lr 2e-4, wd 1e-4)，最多 60 轮，耐心 12；验证集完整 MAE 不得高于原头 0.005，按“完整 MAE + 0.25×缺失 MAE”选取。
另外输出一组“替换遮挡基线”的预测（文本 [MASK]、语音视觉用训练集均值），只用于归因的基线敏感性检验，不参与训练和选择。

子集编号：bit0 = 文本，bit1 = 语音，bit2 = 视觉；7 = 完整输入，0 = 全部遮挡。
用法：python code/subset_head.py --seed 1111 --pool attn4
     只用已保存的半精度融合头重算预测（不训练）：加 --infer_only
"""
import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import light_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--pool', choices=list(C.POOLS), default='mean')
ap.add_argument('--weights', type=float, nargs='+', default=[0.1, 0.25, 0.5])
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--infer_only', action='store_true', help='不训练，读取 subset_head_fp16.pt 重算各划分的 8 子集预测')
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

run = C.EXP / f'outputs/{a.pool}/seed{a.seed}'
model, args, pack = C.load_light_model(run / 'compact.pt', a.pool)
from MMSA.data_loader import MMDataset  # noqa: E402

UNK, MASK = 100, 103


class Head(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.drop = copy.deepcopy(base.post_fusion_dropout)
        self.l1 = copy.deepcopy(base.post_fusion_layer_1)
        self.l2 = copy.deepcopy(base.post_fusion_layer_2)
        self.l3 = copy.deepcopy(base.post_fusion_layer_3)

    def forward(self, x):
        x = torch.relu(self.l1(self.drop(x)))
        x = torch.relu(self.l2(x))
        return self.l3(x).squeeze(-1)


def mask_text(t, token_id):
    tm = t.clone()
    ids = tm[:, 0, :]
    body = (ids != 0) & (ids != 101) & (ids != 102)
    ids[body] = token_id
    return tm


def loader(feature, mode):
    args.update(custom_feature=str(feature))
    return DataLoader(MMDataset(args, mode=mode), batch_size=32, num_workers=2, shuffle=False)


def audio_video_means():
    """训练集有效位置（与文本正文位置一致）上的语音、视觉均值，作替换基线用。"""
    sa, sv, n = 0, 0, 0
    for b in loader(C.DATA_ATT2, 'train'):
        m = b['text'][:, 1, :].float().clone()
        m[:, 0] = 0
        idx = m.sum(1).long()
        m[torch.arange(len(m)), idx] = 0      # 去掉 [SEP]
        sa = sa + (b['audio'] * m[..., None]).sum((0, 1))
        sv = sv + (b['vision'] * m[..., None]).sum((0, 1))
        n += m.sum()
    return (sa / n).float(), (sv / n).float()


def states(feature, mode, a_mean, v_mean):
    cache = C.EXP / f'work/{a.pool}_seed{a.seed}_{Path(feature).stem}_{mode}.pt'
    if cache.is_file():
        return torch.load(cache, weights_only=False)
    keys = ('T0', 'T1', 'A0', 'A1', 'V0', 'V1', 'Tm', 'Am', 'Vm')
    feats, ids, ys = {k: [] for k in keys}, [], []
    M = model.Model
    with torch.inference_mode():
        for b in loader(feature, mode):
            t, aud, vis = b['text'].to(args.device), b['audio'].to(args.device), b['vision'].to(args.device)
            L = t[:, 1, :].sum(1).int().cpu()
            am = torch.zeros_like(aud) + a_mean.to(args.device)
            vm = torch.zeros_like(vis) + v_mean.to(args.device)
            out = {'T0': M.text_model(mask_text(t, UNK))[:, 0, :], 'T1': M.text_model(t)[:, 0, :],
                   'Tm': M.text_model(mask_text(t, MASK))[:, 0, :],
                   'A0': M.audio_model(torch.zeros_like(aud), L), 'A1': M.audio_model(aud, L), 'Am': M.audio_model(am, L),
                   'V0': M.video_model(torch.zeros_like(vis), L), 'V1': M.video_model(vis, L), 'Vm': M.video_model(vm, L)}
            for k in keys:
                feats[k].append(out[k].float().cpu())
            ids.extend(map(str, b['id']))
            ys.append(b['labels']['M'].reshape(-1).cpu())
    data = {k: torch.cat(v) for k, v in feats.items()}
    data['y'], data['ids'] = torch.cat(ys).float(), ids
    assert len(ids) == len(set(ids))
    torch.save(data, cache)
    return data


def matrix(d, t0='T0', a0='A0', v0='V0'):
    xs = [torch.cat([d['T1'] if m & 1 else d[t0], d['A1'] if m & 2 else d[a0], d['V1'] if m & 4 else d[v0]], 1) for m in range(8)]
    return torch.stack(xs, 1)


a_mean, v_mean = audio_video_means()
if a.infer_only:
    D = {'valid': states(C.DATA_ATT2, 'valid', a_mean, v_mean), 'test': states(C.DATA_ATT2, 'test', a_mean, v_mean),
         'att4': states(C.DATA_ATT4, 'test', a_mean, v_mean)}
    dev = args.device
    head = Head(model.Model).to(dev).eval()
    best_state = {k: v.float() for k, v in torch.load(run / 'subset_head_fp16.pt', map_location='cpu', weights_only=True).items()}
    head.load_state_dict(best_state)
    best, cands, base_full, base_missing = json.loads((run / 'subset_head.json').read_text(encoding='utf-8'))['selected'], [], None, None
else:
    D = {s: states(C.DATA_ATT2, s, a_mean, v_mean) for s in ('train', 'valid', 'test')}
    D['att4'] = states(C.DATA_ATT4, 'test', a_mean, v_mean)
    dev = args.device
    Xtr, ytr = matrix(D['train']).to(dev), D['train']['y'].to(dev)
    Xva, yva = matrix(D['valid']).to(dev), D['valid']['y'].to(dev)

    head = Head(model.Model).to(dev).eval()
    with torch.inference_mode():
        pv = head(Xva.reshape(-1, Xva.shape[-1])).reshape(-1, 8)
    base_full = float((pv[:, 7] - yva).abs().mean())
    base_missing = float((pv[:, :7] - yva[:, None]).abs().mean())
    print('original head: valid full MAE %.4f, missing MAE %.4f' % (base_full, base_missing), flush=True)

    best, cands, best_state = None, [], None
    for w in a.weights:
        h = copy.deepcopy(head)
        opt = torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=1e-4)
        cur, stale = None, 0
        for ep in range(1, a.epochs + 1):
            h.train()
            for idx in torch.randperm(len(Xtr), device=dev).split(128):
                p = h(Xtr[idx].reshape(-1, Xtr.shape[-1])).reshape(-1, 8)
                loss = (p[:, 7] - ytr[idx]).abs().mean() + w * (p[:, :7] - ytr[idx, None]).abs().mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(h.parameters(), 2.0); opt.step()
            h.eval()
            with torch.inference_mode():
                p = h(Xva.reshape(-1, Xva.shape[-1])).reshape(-1, 8)
                vf, vm = float((p[:, 7] - yva).abs().mean()), float((p[:, :7] - yva[:, None]).abs().mean())
            score = vf + 0.25 * vm
            if vf <= base_full + 0.005 and (cur is None or score < cur['score'] - 1e-5):
                cur = dict(weight=w, epoch=ep, valid_full_MAE=vf, valid_missing_MAE=vm, score=score)
                cur_state = {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 12:
                break
        if cur is not None:
            cands.append(cur)
            if best is None or cur['score'] < best['score']:
                best, best_state = cur, cur_state
        print('candidate', json.dumps(cur), flush=True)
    assert best is not None
    head.load_state_dict(best_state)
    head.eval()
    torch.save({k: v.half() for k, v in best_state.items()}, run / 'subset_head_fp16.pt')
    torch.save(best_state, run / 'subset_head.pt')


# 半精度头与全精度头的预测差（紧凑提交只放半精度）
h16 = copy.deepcopy(head)
h16.load_state_dict({k: v.half().float() for k, v in best_state.items()})
h16.eval()

outdir = run / ('predictions_fp16' if a.infer_only else 'predictions')
outdir.mkdir(exist_ok=True)
fp16_err = 0.0
with torch.inference_mode():
    for split, d in D.items():
        X = matrix(d).to(dev)
        p = head(X.reshape(-1, X.shape[-1])).reshape(-1, 8).cpu().numpy()
        p16 = h16(X.reshape(-1, X.shape[-1])).reshape(-1, 8).cpu().numpy()
        fp16_err = max(fp16_err, float(np.abs(p - p16).max()))
        rows = [p]
        if split in ('valid', 'test', 'att4'):
            for tag, (t0, a0, v0) in {'maskT': ('Tm', 'A0', 'V0'), 'meanAV': ('T0', 'Am', 'Vm')}.items():
                Xs = matrix(d, t0, a0, v0).to(dev)
                rows.append(head(Xs.reshape(-1, Xs.shape[-1])).reshape(-1, 8).cpu().numpy())
        with (outdir / f'{split}.csv').open('w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            cols = ['id', 'y_true'] + [f'p{m}' for m in range(8)]
            if len(rows) > 1:
                cols += [f'maskT_p{m}' for m in range(8)] + [f'meanAV_p{m}' for m in range(8)]
            wr.writerow(cols)
            for j, i in enumerate(d['ids']):
                wr.writerow([i, float(d['y'][j])] + [f'{v:.6f}' for r in rows for v in r[j]])

if a.infer_only:
    print('infer_only: predictions rewritten with fp16 head', flush=True)
    raise SystemExit
C.save_json({'seed': a.seed, 'pool': a.pool, 'selected': best, 'candidates': cands,
             'original_head_valid_full_MAE': base_full, 'original_head_valid_missing_MAE': base_missing,
             'fp16_head_max_abs_pred_diff': fp16_err}, run / 'subset_head.json')
print('selected', json.dumps(best), 'fp16 max diff', fp16_err, flush=True)
