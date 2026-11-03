"""轻量版三种子集成：位置级积分梯度、证据片段、删除检验与附件4 声画时间定位。

对集成模型 f(x) = 三个种子子集融合头输出的平均：
- 文本：在词嵌入层从 [UNK] 基线积分（正文词元换成 [UNK]，其余不变），128 步中点黎曼和；
- 语音、视觉：从零向量积分，其余两个模态保持原值；
- 位置得分 = 该位置各维积分梯度之和，乘以定向符号 s = sign(f(全) − f(空))，正值表示把预测推向当前方向。
证据片段：每个模态取定向得分之和最大的连续 W 个正文位置（W = min(3, 正文长度)）。
删除检验：遮挡证据片段（文本换 [UNK]，声画置零），定向降幅 s·(f(全) − f(删)) 与所有其他同长度窗口的平均降幅比较；
另做前 k 个最高分位置（k=1,3,5）与 20 组随机 k 位置的对照。
附件4 的位置 → 单词 → 秒：附件4 对齐版第 p 行就是第 p 个 BERT 词元的特征（同一单词的子词特征完全相同），
单词时间取自问题一的附件4 强制对齐结果 att4_word_times.csv。

用法：python code/attribution.py --pool mean4 --seeds 1111 1112 1113
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import light_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--pool', choices=list(C.POOLS), required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 1112, 1113])
ap.add_argument('--steps', type=int, default=128)
ap.add_argument('--n_valid', type=int, default=100)
a = ap.parse_args()
torch.backends.cudnn.enabled = False          # cuDNN 的 LSTM 在 eval 模式下不支持反向传播
UNK = 100
WORD_TIMES = C.TEAM / 'q1/exp/Q1-3_att4_word_times/tables/att4_word_times.csv'
FPS = 30.0
OUT = C.EXP / f'tables/{a.pool}'
OUT.mkdir(parents=True, exist_ok=True)

models, heads = [], []
sys.path.insert(0, str(C.EXP / 'code'))
for s in a.seeds:
    run = C.EXP / f'outputs/{a.pool}/seed{s}'
    m, args, _ = C.load_light_model(run / 'compact.pt', a.pool)
    hp = run / 'subset_head.pt'
    st = torch.load(hp if hp.is_file() else run / 'subset_head_fp16.pt', map_location='cpu', weights_only=True)
    st = {k: v.float() for k, v in st.items()}
    l1 = torch.nn.Linear(*st['l1.weight'].shape[::-1]); l2 = torch.nn.Linear(*st['l2.weight'].shape[::-1]); l3 = torch.nn.Linear(*st['l3.weight'].shape[::-1])
    for lin, n in ((l1, 'l1'), (l2, 'l2'), (l3, 'l3')):
        lin.weight.data, lin.bias.data = st[n + '.weight'], st[n + '.bias']
    head = torch.nn.Sequential(l1, torch.nn.ReLU(), l2, torch.nn.ReLU(), l3).to(args.device).eval()
    models.append(m); heads.append(head)
dev = args.device
bert = models[0].Model.text_model.model          # 三个种子共用同一个冻结 BERT
emb = bert.embeddings.word_embeddings


SCORERS = [getattr(m.Model.text_model, 'e325_scorer', None) for m in models]


def text_vec(ids, attn, seg, inputs_embeds=None):
    """BERT 只算一次，再按各种子自己的池化得到文本表示列表。"""
    out = C.bert_forward(bert, ids, attn, seg, a.pool, inputs_embeds)
    return [C.pool_hidden(out, attn, a.pool, sc) for sc in SCORERS]


def ens(tv, aud, vis, L):
    """tv: 各种子的文本表示列表 [(B, D)]；aud/vis: (B, 50, d)；返回集成输出 (B,)。"""
    outs = []
    for t, m, h in zip(tv, models, heads):
        av = m.Model.audio_model(aud, L)
        vv = m.Model.video_model(vis, L)
        outs.append(h(torch.cat([t, av, vv], -1)).squeeze(-1))
    return torch.stack(outs).mean(0)


def body_positions(attn_row):
    n = int(attn_row.sum())
    return list(range(1, n - 1))


def predict(ids, attn, seg, aud, vis):
    L = attn.sum(1).int().cpu()
    with torch.no_grad():
        return ens(text_vec(ids, attn, seg), aud, vis, L)


def unk(ids, pos):
    ids = ids.clone()
    ids[:, pos] = UNK
    return ids


def integrated_gradients(ids, attn, seg, aud, vis, sign):
    """返回 {'T','A','V': 每个位置的定向积分梯度 (50,)}，以及完整性检查用的端点差。"""
    L = attn.sum(1).int().cpu()
    body = body_positions(attn[0])
    alphas = (torch.arange(a.steps, device=dev, dtype=torch.float32) + 0.5) / a.steps
    res, checks = {}, {}
    # 文本
    x = emb(ids).detach()
    x0 = emb(unk(ids, body)).detach()
    tv_list, grads = [], torch.zeros_like(x)
    for chunk in alphas.split(32):
        xe = (x0 + chunk[:, None, None] * (x - x0)).requires_grad_(True)
        k = len(chunk)
        tv = text_vec(ids.expand(k, -1), attn.expand(k, -1), seg.expand(k, -1), inputs_embeds=xe)
        out = ens(tv, aud.expand(k, -1, -1), vis.expand(k, -1, -1), L.expand(k))
        g, = torch.autograd.grad(out.sum(), xe)
        grads += g.sum(0, keepdim=True)
    attr = ((x - x0) * grads / a.steps).sum(-1)[0]
    res['T'] = (sign * attr).detach().cpu().numpy()
    checks['T'] = (float(attr.sum()), float(predict(ids, attn, seg, aud, vis) - predict(unk(ids, body), attn, seg, aud, vis)))
    # 语音、视觉
    tv_fixed = [v.detach() for v in text_vec(ids, attn, seg)]
    for key, xin in (('A', aud), ('V', vis)):
        grads = torch.zeros_like(xin)
        for chunk in alphas.split(32):
            k = len(chunk)
            xp = (chunk[:, None, None] * xin).requires_grad_(True)
            if key == 'A':
                out = ens([v.expand(k, -1) for v in tv_fixed], xp, vis.expand(k, -1, -1), L.expand(k))
            else:
                out = ens([v.expand(k, -1) for v in tv_fixed], aud.expand(k, -1, -1), xp, L.expand(k))
            g, = torch.autograd.grad(out.sum(), xp)
            grads += g.sum(0, keepdim=True)
        attr = (xin * grads / a.steps).sum(-1)[0]
        res[key] = (sign * attr).detach().cpu().numpy()
        zero = torch.zeros_like(xin)
        full = predict(ids, attn, seg, aud, vis)
        base = predict(ids, attn, seg, zero if key == 'A' else aud, zero if key == 'V' else vis)
        checks[key] = (float(attr.sum()), float(full - base))
    return res, checks


def delete(ids, attn, seg, aud, vis, key, pos):
    if key == 'T':
        return predict(unk(ids, pos), attn, seg, aud, vis)
    x = (aud if key == 'A' else vis).clone()
    x[:, pos] = 0
    return predict(ids, attn, seg, x, vis) if key == 'A' else predict(ids, attn, seg, aud, x)


def best_window(score, body, W):
    W = min(W, len(body))
    starts = range(body[0], body[-1] - W + 2)
    sums = {st: float(score[st:st + W].sum()) for st in starts}
    st = max(sums, key=sums.get)
    return list(range(st, st + W)), [list(range(s0, s0 + W)) for s0 in starts if s0 != st]


def load_split(feature, mode):
    args.update(custom_feature=str(feature))
    from MMSA.data_loader import MMDataset
    ds = MMDataset(args, mode=mode)
    return ds


def run_sample(item, rng):
    ids_t = torch.as_tensor(np.asarray(item['text']), device=dev)[None]      # (1,3,50)
    ids, attn, seg = ids_t[:, 0].long(), ids_t[:, 1].float(), ids_t[:, 2].long()
    aud = torch.as_tensor(np.asarray(item['audio']), device=dev, dtype=torch.float32)[None]
    vis = torch.as_tensor(np.asarray(item['vision']), device=dev, dtype=torch.float32)[None]
    body = body_positions(attn[0])
    full = float(predict(ids, attn, seg, aud, vis))
    empty = float(predict(unk(ids, body), attn, seg, torch.zeros_like(aud), torch.zeros_like(vis)))
    sign = 1.0 if full - empty >= 0 else -1.0
    scores, checks = integrated_gradients(ids, attn, seg, aud, vis, sign)
    out = {'full': full, 'empty': empty, 'sign': sign, 'scores': scores, 'checks': checks, 'body': body, 'spans': {}, 'topk': {}}
    for key in 'TAV':
        span, others = best_window(scores[key], body, 3)
        d_span = sign * (full - float(delete(ids, attn, seg, aud, vis, key, span)))
        d_rand = [sign * (full - float(delete(ids, attn, seg, aud, vis, key, o))) for o in others] or [0.0]
        out['spans'][key] = {'positions': span, 'drop': d_span, 'random_mean': float(np.mean(d_rand)),
                             'random_q90': float(np.quantile(d_rand, 0.9)), 'n_random': len(d_rand)}
        order = sorted(body, key=lambda p: -scores[key][p])
        tk = {}
        for k in (1, 3, 5):
            kk = min(k, len(body))
            top = sign * (full - float(delete(ids, attn, seg, aud, vis, key, order[:kk])))
            rnd = [sign * (full - float(delete(ids, attn, seg, aud, vis, key, list(rng.choice(body, kk, replace=False))))) for _ in range(20)]
            tk[k] = (top, float(np.mean(rnd)))
        out['topk'][key] = tk
    return out


def main_modality(pred8):
    """由 8 子集集成预测算 Shapley，返回主模态（|φ| 最大者）。"""
    sys.path.insert(0, str(C.BASE / 'common'))
    from model_api import venn_and_shapley
    F = frozenset
    game = {F(m for bit, m in ((1, 'T'), (2, 'A'), (4, 'V')) if mask & bit): np.array([pred8[mask]]) for mask in range(8)}
    r = venn_and_shapley(game)
    phi = np.array([r['phi'][m][0] for m in 'TAV'])
    return 'TAV'[int(np.abs(phi).argmax())], phi


def ens_pred8(split):
    per = []
    for s in a.seeds:
        rows = list(csv.DictReader((C.EXP / f'outputs/{a.pool}/seed{s}/predictions/{split}.csv').open(encoding='utf-8')))
        per.append({r['id']: [float(r[f'p{m}']) for m in range(8)] for r in rows})
    return {i: np.mean([p[i] for p in per], 0) for i in per[0]}


rng = np.random.default_rng(1111)
# ---------------- 附件4 ----------------
wt = {}
for r in csv.DictReader(WORD_TIMES.open(encoding='utf-8-sig')):
    wt[(r['sample_id'], int(r['token_pos']))] = r
ds4 = load_split(C.DATA_ATT4, 'test')
pred4 = ens_pred8('att4')
ev_rows, pos_rows, summary = [], [], {'att4': [], 'valid': []}


def span_time(sid, positions):
    """位置 → 单词 → 秒。标点等非词位置与其前面的单词共用特征，归入前一个已对齐词。"""
    words = []
    for p in positions:
        q = p
        while q > 0 and wt.get((sid, q), {}).get('status') != 'aligned':
            q -= 1
        r = wt.get((sid, q))
        if r and r['status'] == 'aligned':
            words.append((int(r['word_idx']), r['word'], float(r['start_sec']), float(r['end_sec'])))
    if not words:
        return None
    words = sorted(set(words))
    return {'words': ' '.join(w[1] for w in words), 'start': min(w[2] for w in words), 'end': max(w[3] for w in words)}


for j in range(len(ds4)):
    item = ds4[j]
    sid = str(item['id'])
    o = run_sample(item, rng)
    mm, phi = main_modality(pred4[sid])
    effect = o['full'] - o['empty']
    for key in 'TAV':
        sp = o['spans'][key]
        tm = span_time(sid, sp['positions'])
        peak = max(sp['positions'], key=lambda p: o['scores'][key][p])
        tpk = span_time(sid, [peak])
        key_t = (tpk['start'] + tpk['end']) / 2 if tpk else None
        zero_modality = key == 'V' and float(np.abs(np.asarray(item['vision'])).sum()) == 0 or key == 'A' and float(np.abs(np.asarray(item['audio'])).sum()) == 0
        ev_rows.append({'sample_id': sid, 'modality': key, 'is_main': key == mm, 'full_pred': o['full'], 'empty_pred': o['empty'],
                        'phi': float(phi['TAV'.index(key)]), 'span_pos': f"{sp['positions'][0]}-{sp['positions'][-1]}",
                        'span_words': tm['words'] if tm else '', 'start_sec': tm['start'] if tm else '', 'end_sec': tm['end'] if tm else '',
                        'peak_pos': peak, 'keyframe_sec': round(key_t, 3) if key_t is not None else '',
                        'keyframe_frame': int(round(key_t * FPS)) if key_t is not None else '',
                        'span_drop': sp['drop'], 'random_mean_drop': sp['random_mean'], 'random_q90_drop': sp['random_q90'],
                        'feature_all_zero': zero_modality})
        for p in o['body']:
            r = wt.get((sid, p), {})
            pos_rows.append({'sample_id': sid, 'modality': key, 'position': p, 'wordpiece': r.get('wordpiece', ''), 'word': r.get('word', ''),
                             'start_sec': r.get('start_sec', ''), 'end_sec': r.get('end_sec', ''), 'ig_directed': float(o['scores'][key][p])})
    summary['att4'].append({'id': sid, 'main': mm, 'effect': effect, 'checks': o['checks'], 'topk': o['topk'][mm], 'span': o['spans'][mm]})
    print('att4', sid, mm, 'effect %.3f' % effect, {k: round(o['spans'][k]['drop'], 3) for k in 'TAV'}, flush=True)


def write(rows, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


write(ev_rows, OUT / 'att4_evidence.csv')
write(pos_rows, OUT / 'att4_ig_positions.csv')

# ---------------- 验证集抽样 ----------------
dsv = load_split(C.DATA_ATT2, 'valid')
predv = ens_pred8('valid')
pick = sorted(np.random.default_rng(1111).choice(len(dsv), a.n_valid, replace=False).tolist())
vrows = []
for j in pick:
    item = dsv[j]
    sid = str(item['id'])
    o = run_sample(item, rng)
    mm, phi = main_modality(predv[sid])
    rec = {'id': sid, 'main': mm, 'effect': o['full'] - o['empty'], 'span_drop': o['spans'][mm]['drop'],
           'span_random_mean': o['spans'][mm]['random_mean']}
    for k in (1, 3, 5):
        rec[f'top{k}_drop'], rec[f'rand{k}_drop'] = o['topk'][mm][k]
    rec['ig_completeness_err_T'] = abs(o['checks']['T'][0] - o['checks']['T'][1])
    vrows.append(rec)
write(vrows, OUT / 'valid_deletion_100.csv')

v = {k: np.array([r[k] for r in vrows]) for k in vrows[0] if k not in ('id', 'main')}
boot = lambda x: np.quantile([np.random.default_rng(b).choice(x, len(x)).mean() for b in range(2000)], [0.025, 0.975]).tolist()
adv = np.mean([v[f'top{k}_drop'] - v[f'rand{k}_drop'] for k in (1, 3, 5)], 0)
res = {'pool': a.pool, 'seeds': a.seeds, 'n_valid': len(vrows),
       'valid_topk_minus_random': {k: float(np.mean(v[f'top{k}_drop'] - v[f'rand{k}_drop'])) for k in (1, 3, 5)},
       'valid_mean_advantage': float(adv.mean()), 'valid_mean_advantage_ci95': boot(adv), 'valid_positive_fraction': float((adv > 0).mean()),
       'valid_span_drop_mean': float(v['span_drop'].mean()), 'valid_span_random_mean': float(v['span_random_mean'].mean()),
       'valid_ig_completeness_err_T_median': float(np.median(v['ig_completeness_err_T']))}
a4 = np.array([np.mean([s['topk'][k][0] - s['topk'][k][1] for k in (1, 3, 5)]) for s in summary['att4']])
res.update({'att4_mean_advantage': float(a4.mean()), 'att4_mean_advantage_ci95': boot(a4), 'att4_positive': int((a4 > 0).sum()),
            'att4_ig_checks': {s['id']: s['checks'] for s in summary['att4']}})
C.save_json(res, OUT / 'attribution_summary.json')
print(json.dumps({k: res[k] for k in res if k != 'att4_ig_checks'}, indent=1))
