"""问题三：三种模态贡献定义的对比（Shapley / 留一法 / 单模态法），只用已保存的 8 个子集预测，不重新推理。

输入：可复现包内 team/q3/exp/E3-25_frozen_bert_light/outputs/attn4/seed*/predictions/valid.csv（p0..p7，bit0=T bit1=A bit2=V）
输出：当前目录下的 q3_compare_table.md、q3_compare_numbers.json
用法：python q3_compare_贡献定义对比.py <attn4 目录>
"""
import csv
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("team/q3/exp/E3-25_frozen_bert_light/outputs/attn4")
SEEDS = (1111, 1112, 1113)
BIT = {"T": 1, "A": 2, "V": 4}


def load(seed):
    rows = list(csv.DictReader(open(SRC / f"seed{seed}/predictions/valid.csv", encoding="utf-8-sig")))
    return [r["id"] for r in rows], np.array([[float(r[f"p{m}"]) for m in range(8)] for r in rows])


def contributions(P):
    """返回三种定义下的贡献 (N,3) 与总效应。"""
    full, empty = P[:, 7], P[:, 0]
    sh = []
    for m in "TAV":
        b = BIT[m]
        others = [x for x in "TAV" if x != m]
        j, k = BIT[others[0]], BIT[others[1]]
        sh.append((P[:, b] - P[:, 0]) / 3 + (P[:, b | j] - P[:, j]) / 6 + (P[:, b | k] - P[:, k]) / 6 + (P[:, 7] - P[:, 7 - b]) / 3)
    loo = [full - P[:, 7 - BIT[m]] for m in "TAV"]
    single = [P[:, BIT[m]] - empty for m in "TAV"]
    return {"Shapley": np.stack(sh, 1), "留一法": np.stack(loo, 1), "单模态法": np.stack(single, 1)}, full - empty


ids, _ = load(SEEDS[0])
Ps = [load(s)[1] for s in SEEDS]
P = np.mean(Ps, 0)
C, eff = contributions(P)
zeta = np.where(eff >= 0, 1.0, -1.0)
cert = np.abs(eff) >= 0.1
n = int(cert.sum())
main = {k: np.abs(v).argmax(1) for k, v in C.items()}
per_seed = [contributions(p) for p in Ps]
out = {"n_valid": len(ids), "n_certain": n}
share = {k: np.abs(v) / np.maximum(np.abs(v).sum(1, keepdims=True), 1e-12) for k, v in C.items()}
# 共有区域（定向）：正=冗余（任一模态在场即可），负=协同（须同时在场）
v = {m: zeta * (P[:, m] - P[:, 0]) for m in range(8)}
R = {"TA": v[5] + v[6] - v[4] - v[7], "TV": v[3] + v[6] - v[2] - v[7], "AV": v[3] + v[5] - v[1] - v[7]}
Rmax = np.max(np.stack(list(R.values()), 1), 1); Rmin = np.min(np.stack(list(R.values()), 1), 1)
red = cert & (Rmax > 0.1 * np.abs(eff)) & (Rmax > 0.05)
syn = cert & (Rmin < -0.1 * np.abs(eff)) & (Rmin < -0.05) & ~red
out["redundant_n"], out["synergy_n"] = int(red.sum()), int(syn.sum())
rows = []
for k in ("Shapley", "留一法", "单模态法"):
    c, e = C[k][cert], eff[cert]
    gap = np.abs(c.sum(1) - e) / np.abs(e)
    mk = main[k][cert]
    seeds_main = np.stack([np.abs(ps[0][k]).argmax(1)[cert] for ps in per_seed], 1)
    ratio = zeta * C[k].sum(1) / np.abs(eff)
    r = {"sum_gap_median": float(np.median(gap)), "sum_gap_gt20": float((gap > 0.2).mean()),
         "agree_shapley": float((mk == main["Shapley"][cert]).mean()),
         "share_tv": float((np.abs(share[k] - share["Shapley"]).sum(1) / 2)[cert].mean()),
         "seed_stable": float((seeds_main == seeds_main[:, :1]).all(1).mean()), "non_text_main": int((mk != 0).sum()),
         "ratio_red": float(np.median(ratio[red])), "ratio_syn": float(np.median(ratio[syn]))}
    out[k] = r
    rows.append((k, r))

pct = lambda x: f"{100 * x:.1f}%"
t = ["::: table {#tab:q3cmp} 三种模态贡献定义在验证集上的对比（三模态合计效应不小于0.1的样本） {widths=1.5,2.4,1.6,1.6,1.6,1.6,1.6 align=l,c,c,c,c,c,c}",
     "定义 | 加总缺口中位数（>20%的比例） | 冗余样本加总/总效应 | 协同样本加总/总效应 | 主模态与Shapley一致 | 份额与Shapley的差异 | 三种子主模态一致", "---"]
for k, r in rows:
    t.append(f"{k} | {pct(r['sum_gap_median'])}（{pct(r['sum_gap_gt20'])}） | {r['ratio_red']:.2f} | {r['ratio_syn']:.2f} | "
             f"{pct(r['agree_shapley'])} | {pct(r['share_tv'])} | {pct(r['seed_stable'])}")
t.append(":::")
Path("q3_compare_table.md").write_text("\n".join(t) + "\n", encoding="utf-8")
json.dump(out, open(Path("q3_compare_numbers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print("\n".join(t))
