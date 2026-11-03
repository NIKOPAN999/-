# -*- coding: utf-8 -*-
"""
全队约定的模型接口：任何基线或我们自己的模型，只要包一层实现下面两个方法，
问题二的评测脚本和问题三的解释流程（子集博弈 / Shapley / 韦恩分解 / IG）就能直接接入，无需关心模型内部结构。

约定（输入均为 MMSA 格式的 split dict，见 convert_to_mmsa.py；输出均为 numpy）：

    class Wrapper:
        name: str                                  # 如 "lnln_seed1111"
        def predict(self, split) -> np.ndarray:    # 形状 (N,)，情感强度预测，范围约 [-3, 3]
        def predict_subset(self, split, keep) -> np.ndarray:
            # keep ⊆ {"T","A","V"}；不在 keep 中的模态按统一缺失协议整体遮挡：
            #   文本除 [CLS]/[SEP] 外全部置为 [UNK]=100，语音/视觉有效位置全部置 0
            # 默认实现即：predict(mask_modalities(split, drop=set("TAV") - keep))

问题三的 8 个子集预测 = [predict_subset(split, S) for S in 所有子集]，其中 S=∅ 为全遮挡基线。
"""
import itertools

import numpy as np

from missing import UNK_ID, _body_mask

ALL = ("T", "A", "V")
SUBSETS = [frozenset(c) for r in range(4) for c in itertools.combinations(ALL, r)]   # 8 个，含空集


def mask_modalities(split, drop):
    """把 drop 中的模态在所有有效位置上整体遮挡（与问题二缺失表示一致）。"""
    out = dict(split)
    lengths = np.asarray(split["audio_lengths"])
    N, L = split["text_bert"].shape[0], split["text_bert"].shape[2]
    body = _body_mask(lengths, L)
    if "T" in drop:
        tb = split["text_bert"].copy()
        tb[:, 0, :][body] = UNK_ID
        out["text_bert"] = tb
    if "A" in drop:
        a = split["audio"].copy(); a[body] = 0; out["audio"] = a
    if "V" in drop:
        v = split["vision"].copy(); v[body] = 0; out["vision"] = v
    return out


class BaseWrapper:
    name = "base"

    def predict(self, split):
        raise NotImplementedError

    def predict_subset(self, split, keep):
        return self.predict(mask_modalities(split, set(ALL) - set(keep)))


def subset_game(wrapper, split):
    """返回 {frozenset(S): (N,) 预测}，共 8 次前向。问题三的 Shapley / 韦恩分解都从这里出发。"""
    return {S: np.asarray(wrapper.predict_subset(split, S), dtype=float) for S in SUBSETS}


def venn_and_shapley(game):
    """game: subset_game 的输出。按 03 文档 3.1–3.3 计算定向博弈的 7 个韦恩区域与 Shapley 值（逐样本）。"""
    F = frozenset
    full, empty = game[F(ALL)], game[F()]
    s = np.sign(full - empty); s[s == 0] = 1
    v = {S: s * (game[S] - empty) for S in game}                       # 定向博弈
    U = {m: v[F(ALL)] - v[F(ALL) - {m}] for m in ALL}
    P = {}
    for i, j in itertools.combinations(ALL, 2):
        k = (set(ALL) - {i, j}).pop()
        P[i + j] = v[F({i, k})] + v[F({j, k})] - v[F({k})] - v[F(ALL)]
    C = sum(v[F({m})] for m in ALL) - sum(v[F(p)] for p in itertools.combinations(ALL, 2)) + v[F(ALL)]
    phi_oriented = {m: U[m] + 0.5 * sum(P[p] for p in P if m in p) + C / 3 for m in ALL}
    phi = {m: s * phi_oriented[m] for m in ALL}                        # 还原为带方向的 Shapley 值
    I_hat = (v[F(ALL)] - sum(v[F({m})] for m in ALL)) / np.maximum(np.abs(v[F(ALL)]), 1e-8)
    return {"U": U, "P": P, "C": C, "phi": phi, "interaction_index": I_hat, "sign": s}
