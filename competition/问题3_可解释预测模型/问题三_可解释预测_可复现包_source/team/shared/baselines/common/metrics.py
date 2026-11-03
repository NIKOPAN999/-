# -*- coding: utf-8 -*-
"""
统一评测指标（全队所有模型、所有实验只用这一份，避免各仓库自带 metric 口径不一致）。

赛题口径：情感极性（三分类：[-3,0) 负、0 中、(0,3] 正）用 Accuracy、F1；情感强度（回归）用 MAE、Pearson。
为便于和文献对照，同时给出 MMSA 常用的 Acc-2（两种口径）、Acc-5、Acc-7。

极性由强度预测得到：|ŷ| <= tau 判为中性。tau 默认 0.1（与 MMSA 的 Acc-3 分箱一致）；
正式结果应在验证集上选 tau（select_tau），再原样用于测试集和附件3/4。
"""
import numpy as np


def polarity(y, tau=0.0):
    """0=负 1=中 2=正（与附件2 classification_labels 一致，已核实 100% 吻合）。"""
    y = np.asarray(y, dtype=float)
    return np.where(y < -tau, 0, np.where(y > tau, 2, 1))


def _f1(y_true, y_pred, labels, average):
    f1s, supports = [], []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        supports.append(np.sum(y_true == c))
    f1s, supports = np.array(f1s), np.array(supports)
    if average == "macro":
        return float(f1s.mean())
    return float((f1s * supports).sum() / max(supports.sum(), 1))  # weighted


def evaluate(y_true, y_pred, tau=0.1):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out = {}
    # 三分类（赛题主口径）
    t3, p3 = polarity(y_true, 0.0), polarity(y_pred, tau)
    out["Acc3"] = float((t3 == p3).mean())
    out["F1_3_weighted"] = _f1(t3, p3, [0, 1, 2], "weighted")
    out["F1_3_macro"] = _f1(t3, p3, [0, 1, 2], "macro")
    # Acc-2 两种文献口径
    t2, p2 = (y_true >= 0).astype(int), (y_pred >= 0).astype(int)                 # 负 vs 非负
    out["Has0_Acc2"] = float((t2 == p2).mean())
    out["Has0_F1"] = _f1(t2, p2, [0, 1], "weighted")
    nz = y_true != 0                                                               # 负 vs 正（去掉真值为 0 的样本）
    t2n, p2n = (y_true[nz] > 0).astype(int), (y_pred[nz] > 0).astype(int)
    out["Non0_Acc2"] = float((t2n == p2n).mean()) if nz.any() else float("nan")
    out["Non0_F1"] = _f1(t2n, p2n, [0, 1], "weighted") if nz.any() else float("nan")
    # 多档强度
    out["Acc5"] = float((np.round(np.clip(y_true, -2, 2)) == np.round(np.clip(y_pred, -2, 2))).mean())
    out["Acc7"] = float((np.round(np.clip(y_true, -3, 3)) == np.round(np.clip(y_pred, -3, 3))).mean())
    # 回归
    out["MAE"] = float(np.mean(np.abs(y_true - y_pred)))
    out["Pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    out["tau"] = tau
    return out


def select_tau(y_true_valid, y_pred_valid, grid=None, key="F1_3_weighted"):
    """在验证集上选中性死区阈值 tau，返回 (最优 tau, 对应指标)。"""
    grid = np.round(np.arange(0.0, 0.81, 0.02), 2) if grid is None else grid
    best = max(((evaluate(y_true_valid, y_pred_valid, t)[key], t) for t in grid))
    return float(best[1]), best[0]
