# -*- coding: utf-8 -*-
"""
统一的缺失注入协议（问题二的所有训练增强与规律分析都用这一份，保证各模型的缺失完全一致）。

输入输出均为 MMSA 格式的单个划分 dict（text_bert: (N,3,50)；audio: (N,L,74)；vision: (N,L,35)；
audio_lengths / vision_lengths: (N,)）。对齐版中 L=50 且与 BERT 词元逐位对应。

缺失表示与赛题附件3一致：文本词元 -> [UNK](id=100)，语音/视觉整行 -> 0；[CLS] 与 [SEP] 永不遮挡。

三种模式：
  bernoulli_sync   每个位置以概率 rate 独立遮挡，三模态共用同一掩码（= 附件3 对齐版的生成机制）
  bernoulli_async  三模态各自独立抽掩码
  span             在指定模态上遮挡一段连续区间，起点由相对位置 pos∈[0,1] 给出，长度由占有效长度的比例 frac 给出

bernoulli_sync(rate=0.2, seed=1111) 与 MMSA generate_m 在 missing_seed=[1111,1111,1111] 时的抽样方式相同
（先 np.random.seed(seed)，再对整个 (N,L) 矩阵抽 uniform），可用于复现“附件3同分布”的验证集。
"""
import numpy as np

UNK_ID = 100
MODALITIES = ("T", "A", "V")


def _body_mask(lengths, L):
    """有效词元位置（去掉 [CLS] 与 [SEP]）为 True。lengths 含 [CLS]/[SEP]。"""
    idx = np.arange(L)[None, :]
    return (idx >= 1) & (idx < (lengths[:, None] - 1))


def _apply(split, drop):
    """drop: dict 模态 -> (N,L) 布尔矩阵，True 表示该位置缺失。返回新的 split（不修改原数据）。"""
    out = dict(split)
    if "T" in drop:
        tb = split["text_bert"].copy()
        ids = tb[:, 0, :]
        ids[drop["T"]] = UNK_ID
        tb[:, 0, :] = ids
        out["text_bert"] = tb
    if "A" in drop:
        a = split["audio"].copy()
        a[drop["A"]] = 0
        out["audio"] = a
    if "V" in drop:
        v = split["vision"].copy()
        v[drop["V"]] = 0
        out["vision"] = v
    out["missing_mask"] = {m: drop[m] for m in drop}
    return out


def bernoulli_sync(split, rate, seed=1111, modalities=MODALITIES):
    lengths = np.asarray(split["audio_lengths"])
    N, L = split["text_bert"].shape[0], split["text_bert"].shape[2]
    rng = np.random.RandomState(seed)
    m = (rng.uniform(size=(N, L)) < rate) & _body_mask(lengths, L)
    return _apply(split, {k: m for k in modalities})


def bernoulli_async(split, rate, seed=1111, modalities=MODALITIES):
    lengths = np.asarray(split["audio_lengths"])
    N, L = split["text_bert"].shape[0], split["text_bert"].shape[2]
    body = _body_mask(lengths, L)
    drop = {}
    for i, k in enumerate(modalities):
        rng = np.random.RandomState(seed + 1000 * (i + 1))
        drop[k] = (rng.uniform(size=(N, L)) < rate) & body
    return _apply(split, drop)


def span(split, pos, frac, modalities=MODALITIES):
    """在每条样本的有效区间内遮挡一段连续位置：长度 = max(1, round(frac * 有效长度))，
    起点使区间中心尽量落在相对位置 pos 处（pos=0 句首，0.5 句中，1 句尾），并截断在有效区间内。"""
    lengths = np.asarray(split["audio_lengths"])
    N, L = split["text_bert"].shape[0], split["text_bert"].shape[2]
    m = np.zeros((N, L), bool)
    for i, Li in enumerate(lengths):
        n_body = int(Li) - 2
        if n_body <= 0:
            continue
        k = max(1, int(round(frac * n_body)))
        k = min(k, n_body)
        center = pos * (n_body - 1)
        start = int(round(center - (k - 1) / 2))
        start = min(max(start, 0), n_body - k)
        m[i, 1 + start: 1 + start + k] = True
    return _apply(split, {k_: m for k_ in modalities})


def observed_missing_rate(split):
    """从数据本身读出每条样本各模态的缺失率（测试时附件3就用它）：文本看 [UNK]，A/V 看全零行。"""
    lengths = np.asarray(split["audio_lengths"])
    N, L = split["text_bert"].shape[0], split["text_bert"].shape[2]
    body = _body_mask(lengths, L)
    t = (split["text_bert"][:, 0, :] == UNK_ID) & body
    a = np.all(split["audio"] == 0, axis=2) & body
    v = np.all(split["vision"] == 0, axis=2) & body
    denom = np.maximum(body.sum(1), 1)
    return {"T": t.sum(1) / denom, "A": a.sum(1) / denom, "V": v.sum(1) / denom}
