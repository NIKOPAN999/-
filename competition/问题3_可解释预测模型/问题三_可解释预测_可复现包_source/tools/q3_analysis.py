"""问题三（轻量版 Self-MM）结果汇总：指标、Shapley 与区域分解、交互类型、基线敏感性、错误归因、附件4汇总与提交 CSV、论文用图。

输入：服务器 q3/exp/E3-25_frozen_bert_light 下载到本地的
  outputs/<pool>/seed*/predictions/{valid,test,att4}.csv、outputs/<pool>/seed*/{subset_head.json,compact_info.json}
  tables/<pool>/{att4_evidence.csv, att4_ig_positions.csv, valid_deletion_100.csv, attribution_summary.json}
另读：正式版（全参数微调）三种子集成的逐样本结果（问题三队员子包 03_结果/selfmm_ensemble_{valid,test}.csv），作性能对照。
输出：figures/q3_*.png、figures/q3_numbers.json、paper_src/generated/q3_*.md、问题三提交 CSV。
"""
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
POOL = os.environ.get("Q3_POOL", "attn4")
SRC = ROOT / "问题三队员工作" / "E3-25_轻量版结果"
PKG = ROOT / "问题三队员工作" / "问题三_组队提交子包_20260926"
ATT4_VIDEO = ROOT / "E题数据" / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本" / "videos"
FIG = PAPER / "figures"
GEN = PAPER / "paper_src" / "generated"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
SEEDS = (1111, 1112, 1113)

INK, SLATE, GRID, MUTED = "#263238", "#546E7A", "#E3E7EA", "#90A4AE"
BLUE, ORANGE, VERMIL = "#0072B2", "#E69F00", "#D55E00"
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": SLATE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150, "axes.unicode_minus": False,
})
W_FULL = 6.3
MOD_CN = {"T": "文本", "A": "语音", "V": "视觉"}
MOD_COLOR = {"T": BLUE, "A": ORANGE, "V": MUTED}
NUM = {}

sys.path.insert(0, str(PKG / "team/shared/baselines/common"))
from metrics import evaluate, select_tau  # noqa: E402  团队统一指标


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("saved", name)


def rd(x, n=4):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- 读入
def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_seed(seed, split):
    rows = read_csv(SRC / f"outputs/{POOL}/seed{seed}/predictions/{split}.csv")
    ids = [r["id"] for r in rows]
    y = np.array([float(r["y_true"]) for r in rows])
    P = {"orig": np.array([[float(r[f"p{m}"]) for m in range(8)] for r in rows])}
    if f"maskT_p0" in rows[0]:
        P["maskT"] = np.array([[float(r[f"maskT_p{m}"]) for m in range(8)] for r in rows])
        P["meanAV"] = np.array([[float(r[f"meanAV_p{m}"]) for m in range(8)] for r in rows])
    return ids, y, P


D = {}
for split in ("valid", "test", "att4"):
    per = [load_seed(s, split) for s in SEEDS]
    ids = per[0][0]
    assert all(p[0] == ids for p in per)
    D[split] = {"ids": ids, "y": per[0][1], "seed": [p[2]["orig"] for p in per],
                **{k: np.mean([p[2][k] for p in per], 0) for k in per[0][2]}}


# ---------------------------------------------------------------- Shapley 与区域分解（与 shared/baselines/common/model_api.py 相同）
def venn_and_shapley(P):
    """P: (N, 8) 子集预测，bit0=T bit1=A bit2=V。返回 phi(N,3)、区域 dict、交互指数、定向符号。"""
    v8 = {m: P[:, m] for m in range(8)}
    full, empty = v8[7], v8[0]
    s = np.sign(full - empty); s[s == 0] = 1
    v = {m: s * (v8[m] - empty) for m in range(8)}
    bit = {"T": 1, "A": 2, "V": 4}
    ALL = 7
    U = {m: v[ALL] - v[ALL - bit[m]] for m in "TAV"}
    Pr = {}
    for i, j in itertools.combinations("TAV", 2):
        k = ({"T", "A", "V"} - {i, j}).pop()
        Pr[i + j] = v[bit[i] | bit[k]] + v[bit[j] | bit[k]] - v[bit[k]] - v[ALL]
    Cc = v[1] + v[2] + v[4] - v[3] - v[5] - v[6] + v[ALL]
    phi_o = {m: U[m] + 0.5 * sum(Pr[p] for p in Pr if m in p) + Cc / 3 for m in "TAV"}
    phi = np.stack([s * phi_o[m] for m in "TAV"], 1)
    I_hat = (v[ALL] - v[1] - v[2] - v[4]) / np.maximum(np.abs(v[ALL]), 1e-8)
    return phi, {"U_T": U["T"], "U_A": U["A"], "U_V": U["V"], "P_TA": Pr["TA"], "P_TV": Pr["TV"], "P_AV": Pr["AV"], "C": Cc}, I_hat, s


def interaction_type(P, I_hat, s, thr=0.1):
    out = []
    for j in range(len(P)):
        eff = P[j, 7] - P[j, 0]
        if abs(eff) < thr:
            out.append("indeterminate"); continue
        singles = [s[j] * (P[j, b] - P[j, 0]) for b in (1, 2, 4)]
        if min(singles) < -0.2: out.append("conflict")
        elif I_hat[j] > 0.2: out.append("synergy")
        elif I_hat[j] < -0.2: out.append("redundant")
        else: out.append("additive_or_unique")
    return np.array(out)


for split in D:
    phi, R, I_hat, s = venn_and_shapley(D[split]["orig"])
    D[split].update(phi=phi, R=R, I=I_hat, sign=s, types=interaction_type(D[split]["orig"], I_hat, s))
    share = np.abs(phi) / np.maximum(np.abs(phi).sum(1, keepdims=True), 1e-12)
    eff = D[split]["orig"][:, 7] - D[split]["orig"][:, 0]
    main = np.array(["TAV"[k] for k in np.abs(phi).argmax(1)])
    D[split].update(share=share, main=main, effect=eff, uncertain=np.abs(eff) < 0.1)
    # 两种替换基线下的主模态（基线敏感性）
    for alt in ("maskT", "meanAV"):
        if alt in D[split]:
            pa, *_ = venn_and_shapley(D[split][alt])
            D[split][f"main_{alt}"] = np.array(["TAV"[k] for k in np.abs(pa).argmax(1)])
    # 三个种子各自的主模态
    D[split]["main_seed"] = np.stack([np.array(["TAV"[k] for k in np.abs(venn_and_shapley(p)[0]).argmax(1)]) for p in D[split]["seed"]])

# ---------------------------------------------------------------- 性能
yv, yt = D["valid"]["y"], D["test"]["y"]
pv, pt = D["valid"]["orig"][:, 7], D["test"]["orig"][:, 7]
tau, _ = select_tau(yv, pv)
NUM["tau"] = tau


def consistent(pred, t):
    """提交口径：判中性则强度为 0。"""
    return np.where(np.abs(pred) <= t, 0.0, pred)


def summ(y, p, t):
    e = evaluate(y, p, t)
    e0 = evaluate(y, consistent(p, t), t)
    return {"Acc3": rd(e["Acc3"]), "F1_macro": rd(e["F1_3_macro"]), "F1_weighted": rd(e["F1_3_weighted"]),
            "MAE": rd(e0["MAE"]), "MAE_raw": rd(e["MAE"]), "Pearson": rd(e0["Pearson"]), "Acc2_non0": rd(e["Non0_Acc2"])}


def crossfit(y, p, ids, reps=50):
    vid = np.array([i.split("$_$")[0] for i in ids]); uv = np.unique(vid)
    accs, f1s, maes = [], [], []
    for r in range(reps):
        perm = np.random.default_rng(1000 + r).permutation(uv); f = np.isin(vid, perm[: len(uv) // 2])
        pc = np.empty(len(y)); tt = np.empty(len(y))
        for tr, te in ((f, ~f), (~f, f)):
            t, _ = select_tau(y[tr], p[tr]); tt[te] = t; pc[te] = p[te]
        from metrics import polarity
        t3 = polarity(y); p3 = np.array([polarity([pc[j]], tt[j])[0] for j in range(len(y))])
        accs.append((t3 == p3).mean())
        from metrics import _f1
        f1s.append(_f1(t3, p3, [0, 1, 2], "macro"))
        maes.append(np.mean(np.abs(y - np.where(np.abs(pc) <= tt, 0, pc))))
    return {"Acc3": rd(np.mean(accs)), "Acc3_sd": rd(np.std(accs)), "F1_macro": rd(np.mean(f1s)), "MAE": rd(np.mean(maes))}


NUM["light"] = {"valid": summ(yv, pv, tau), "test": summ(yt, pt, tau), "valid_crossfit": crossfit(yv, pv, D["valid"]["ids"]),
                "valid_missing_MAE": rd(np.abs(D["valid"]["orig"][:, :7] - yv[:, None]).mean()),
                "test_missing_MAE": rd(np.abs(D["test"]["orig"][:, :7] - yt[:, None]).mean())}
NUM["light_seeds"] = {}
for k, s in enumerate(SEEDS):
    NUM["light_seeds"][s] = {"valid_MAE_raw": rd(np.abs(D["valid"]["seed"][k][:, 7] - yv).mean()),
                             "test_MAE_raw": rd(np.abs(D["test"]["seed"][k][:, 7] - yt).mean())}
    info = json.loads((SRC / f"outputs/{POOL}/seed{s}/subset_head.json").read_text(encoding="utf-8"))
    NUM["light_seeds"][s]["head"] = info["selected"]
    NUM["light_seeds"][s]["compact_bytes"] = json.loads((SRC / f"outputs/{POOL}/seed{s}/compact_info.json").read_text(encoding="utf-8"))["compact_bytes"]
sv = [NUM["light_seeds"][s]["valid_MAE_raw"] for s in SEEDS]
NUM["light_seed_valid_MAE_mean_sd"] = [rd(np.mean(sv)), rd(np.std(sv, ddof=1))]

# 正式版（全参数微调 BERT）的同口径对照
ref = {}
for split in ("valid", "test"):
    rows = read_csv(PKG / f"03_结果/selfmm_ensemble_{split}.csv")
    ref[split] = (np.array([float(r["y_true"]) for r in rows]), np.array([float(r["pred_intensity"]) for r in rows]))
tau_ref, _ = select_tau(*ref["valid"])
NUM["finetuned"] = {"tau": tau_ref, "valid": summ(*ref["valid"], tau_ref), "test": summ(*ref["test"], tau_ref),
                    "valid_crossfit": crossfit(*ref["valid"], [r["id"] for r in read_csv(PKG / "03_结果/selfmm_ensemble_valid.csv")])}

# ---------------------------------------------------------------- 模态作用
V = D["valid"]
NUM["modality_valid"] = {
    "main_counts": {m: int((V["main"] == m).sum()) for m in "TAV"},
    "uncertain": int(V["uncertain"].sum()),
    "main_counts_certain": {m: int(((V["main"] == m) & ~V["uncertain"]).sum()) for m in "TAV"},
    "empty_pred_mean": rd(V["orig"][:, 0].mean()), "empty_pred_sd": rd(V["orig"][:, 0].std()),
    "train_label_mean_note": "见 NUM['valid_label_mean']",
    "mean_share": {m: rd(V["share"][:, k].mean()) for k, m in enumerate("TAV")},
    "pooled_share": {m: rd(np.abs(V["phi"][:, k]).sum() / np.abs(V["phi"]).sum()) for k, m in enumerate("TAV")},
    "seed_agreement_all3": rd((V["main_seed"] == V["main_seed"][0]).all(0).mean()),
    "agree_maskT": rd((V["main_maskT"] == V["main"]).mean()), "agree_meanAV": rd((V["main_meanAV"] == V["main"]).mean()),
    "types": {t: int((V["types"] == t).sum()) for t in np.unique(V["types"])},
    "regions_mean": {k: rd(v.mean(), 5) for k, v in V["R"].items()},
    "regions_mean_abs": {k: rd(np.abs(v).mean(), 5) for k, v in V["R"].items()},
    "shapley_identity_max_err": float(np.abs(V["phi"].sum(1) - (V["orig"][:, 7] - V["orig"][:, 0])).max()),
}
# 按真实极性分组的平均份额
pol = np.where(yv < 0, "负向", np.where(yv > 0, "正向", "中性"))
NUM["valid_label_mean"] = rd(yv.mean())
NUM["modality_valid"]["share_by_polarity"] = {g: {m: rd(V["share"][pol == g, k].mean()) for k, m in enumerate("TAV")} for g in ("负向", "中性", "正向")}
# 视觉主导样本
vd = V["main"] == "V"
NUM["modality_valid"]["V_main"] = {"n": int(vd.sum()), "mean_abs_y": rd(np.abs(yv[vd]).mean()) if vd.any() else None,
                                   "mean_abs_effect": rd(np.abs(V["effect"][vd]).mean()) if vd.any() else None,
                                   "mean_abs_effect_T_main": rd(np.abs(V["effect"][~vd]).mean())}

# ---------------------------------------------------------------- 错误归因（验证集）
pc = consistent(pv, tau)
err = np.abs(pc - yv)
bins = [(-3.01, -2), (-2, -1), (-1, -1e-9), (0, 0), (1e-9, 1), (1, 2), (2, 3.01)]
labels = ["[−3,−2)", "[−2,−1)", "[−1,0)", "0", "(0,1]", "(1,2]", "(2,3]"]
rows = []
for (lo, hi), lab in zip(bins, labels):
    msk = (yv == 0) if lab == "0" else ((yv >= lo) & (yv < hi) if hi <= 0 else (yv > lo) & (yv <= hi))
    rows.append({"bin": lab, "n": int(msk.sum()), "MAE": rd(err[msk].mean(), 3), "mean_y": rd(yv[msk].mean(), 2), "mean_pred": rd(pv[msk].mean(), 2)})
NUM["error_bins"] = rows
from metrics import polarity  # noqa: E402
t3, p3 = polarity(yv), polarity(pv, tau)
cm = np.array([[int(((t3 == i) & (p3 == j)).sum()) for j in range(3)] for i in range(3)])
NUM["confusion_valid"] = cm.tolist()
q = np.quantile(V["share"][:, 0], [0.25, 0.5, 0.75])
qi = np.digitize(V["share"][:, 0], q)
NUM["mae_by_text_share_quartile"] = [rd(err[qi == k].mean(), 3) for k in range(4)]
NUM["mae_by_text_share_quartile_mean_abs_y"] = [rd(np.abs(yv[qi == k]).mean(), 3) for k in range(4)]
NUM["mae_by_main"] = {m: [int((V["main"] == m).sum()), rd(err[V["main"] == m].mean(), 3)] for m in "TAV" if (V["main"] == m).any()}
# 句长（正文词元数）与误差
NUM["worst"] = [{"id": V["ids"][j], "y": rd(yv[j], 2), "pred": rd(pv[j], 2), "phi": [rd(x, 3) for x in V["phi"][j]]}
                for j in np.argsort(-err)[:5]]

# ---------------------------------------------------------------- 附件4
A4 = D["att4"]
ev = read_csv(SRC / f"tables/{POOL}/att4_evidence.csv")
EV = {(r["sample_id"], r["modality"]): r for r in ev}
att_summary = json.loads((SRC / f"tables/{POOL}/attribution_summary.json").read_text(encoding="utf-8"))
NUM["faithfulness"] = {k: att_summary[k] for k in att_summary if k != "att4_ig_checks"}


def level(r):
    d, rm = float(r["span_drop"]), float(r["random_mean_drop"])
    q9 = float(r["random_q90_drop"])
    if d >= 0.3 and d > q9:
        return "强"
    if d >= 0.05 and d > rm:
        return "中"
    return "弱"


att4_rows = []
for j, sid in enumerate(A4["ids"]):
    p = A4["orig"][j, 7]
    pol_code = int(polarity([p], tau)[0])
    main = A4["main"][j]
    main_show = main if not A4["uncertain"][j] else "uncertain"
    e = {m: EV[(sid, m)] for m in "TAV"}
    vzero = e["V"]["feature_all_zero"] == "True"
    rec = {
        "sample_id": sid,
        "pred_polarity_code": pol_code,
        "pred_polarity": ["negative", "neutral", "positive"][pol_code],
        "pred_intensity": round(float(consistent(np.array([p]), tau)[0]), 6),
        "pred_score_raw": round(float(p), 6),
        "phi_T": round(float(A4["phi"][j, 0]), 6), "phi_A": round(float(A4["phi"][j, 1]), 6), "phi_V": round(float(A4["phi"][j, 2]), 6),
        "share_T": round(float(A4["share"][j, 0]), 4), "share_A": round(float(A4["share"][j, 1]), 4), "share_V": round(float(A4["share"][j, 2]), 4),
        "main_modality": main_show, "max_share_modality": main,
        "full_minus_empty": round(float(A4["effect"][j]), 4),
        "interaction_type": A4["types"][j],
        **{k: round(float(A4["R"][k][j]), 6) for k in A4["R"]},
        "text_evidence": e["T"]["span_words"], "text_token_span": e["T"]["span_pos"],
        "text_evidence_sec": f"{float(e['T']['start_sec']):.2f}-{float(e['T']['end_sec']):.2f}" if e["T"]["start_sec"] else "",
        "text_evidence_level": level(e["T"]),
        "audio_evidence_sec": f"{float(e['A']['start_sec']):.2f}-{float(e['A']['end_sec']):.2f}" if e["A"]["start_sec"] else "",
        "audio_evidence_words": e["A"]["span_words"], "audio_evidence_level": level(e["A"]),
        "vision_keyframe_sec": "" if vzero else e["V"]["keyframe_sec"],
        "vision_keyframe_frame": "" if vzero else e["V"]["keyframe_frame"],
        "vision_evidence_sec": "" if vzero else f"{float(e['V']['start_sec']):.2f}-{float(e['V']['end_sec']):.2f}",
        "vision_evidence_level": "无（视觉特征全零）" if vzero else level(e["V"]),
        "main_evidence_drop": round(float(e[main]["span_drop"]), 4), "main_evidence_random_drop": round(float(e[main]["random_mean_drop"]), 4),
        "vision_evidence_words": "" if vzero else e["V"]["span_words"],
        **{f"{nm}_evidence_drop": round(float(e[m]["span_drop"]), 4) for m, nm in (("T", "text"), ("A", "audio"), ("V", "vision"))},
        **{f"{nm}_evidence_random_drop": round(float(e[m]["random_mean_drop"]), 4) for m, nm in (("T", "text"), ("A", "audio"), ("V", "vision"))},
    }
    att4_rows.append(rec)

sub = ROOT / "问题三队员工作" / "问题三_整改后提交文件"
sub.mkdir(parents=True, exist_ok=True)
with open(sub / "attachment4_predictions_explanations.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(att4_rows[0])); w.writeheader(); w.writerows(att4_rows)
NUM["att4"] = {"polarity_counts": {k: sum(r["pred_polarity"] == k for r in att4_rows) for k in ("negative", "neutral", "positive")},
               "main_counts": {k: sum(r["main_modality"] == k for r in att4_rows) for k in ("T", "A", "V", "uncertain")},
               "types": {t: int((A4["types"] == t).sum()) for t in np.unique(A4["types"])},
               "mean_share": {m: rd(A4["share"][:, k].mean()) for k, m in enumerate("TAV")},
               "levels_main": {lv: sum(r[{"T": "text", "A": "audio", "V": "vision"}[r["max_share_modality"]] + "_evidence_level"] == lv for r in att4_rows) for lv in ("强", "中", "弱")},
               "text_levels": {lv: sum(r["text_evidence_level"] == lv for r in att4_rows) for lv in ("强", "中", "弱")}}

# 与正式版附件4 输出的一致性
old = {int(r["sample_id"]): r for r in read_csv(PKG / "03_结果/Q3_附件4_预测与解释结果.csv")}   # 原文件有两列 sample_id，按整数对齐
agree_pol = sum(int(old[int(r["sample_id"])]["pred_polarity_code"]) == r["pred_polarity_code"] for r in att4_rows)
agree_main = sum(old[int(r["sample_id"])]["main_modality"] == r["max_share_modality"] for r in att4_rows)
NUM["att4"]["agree_with_finetuned"] = {"polarity": agree_pol, "main_modality": agree_main}

# ---------------------------------------------------------------- 图
# 图A：验证集性能与错误结构
fig, ax = plt.subplots(1, 3, figsize=(W_FULL, 2.2), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.25, 1.1]))
a0 = ax[0]
cmn = cm / cm.sum(1, keepdims=True)
a0.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#FFFFFF", BLUE]), vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        a0.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.55 else INK)
a0.set_xticks(range(3), ["负", "中", "正"]); a0.set_yticks(range(3), ["负", "中", "正"])
a0.set_xlabel("预测极性"); a0.set_ylabel("真实极性"); a0.set_title("(a) 验证集混淆矩阵", loc="left")
a0.spines[:].set_visible(False)
a1 = ax[1]
x = np.arange(len(rows))
a1.plot(x, [r["mean_y"] for r in rows], "o-", color=MUTED, ms=3, lw=1, label="平均真值")
a1.plot(x, [r["mean_pred"] for r in rows], "s-", color=BLUE, ms=3, lw=1, label="平均预测")
a1.axhspan(-tau, tau, color=GRID, lw=0)
a1.set_xticks(x, [r["bin"] for r in rows], rotation=45, fontsize=7)
a1.set_ylabel("情感强度"); a1.set_title("(b) 按真实强度分箱", loc="left"); a1.legend(loc="upper left", handlelength=1.5)
a1.axhline(0, color=SLATE, lw=0.5)
a2 = ax[2]
a2.bar(range(4), NUM["mae_by_text_share_quartile"], color=BLUE, width=0.6)
for k, v in enumerate(NUM["mae_by_text_share_quartile"]):
    a2.text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
a2.set_xticks(range(4), ["Q1\n(低)", "Q2", "Q3", "Q4\n(高)"]); a2.set_xlabel("文本份额四分位"); a2.set_ylabel("MAE")
a2.set_title("(c) 误差与文本依赖度", loc="left"); a2.set_ylim(0, max(NUM["mae_by_text_share_quartile"]) * 1.25)
savefig(fig, "fig_q3_valid_errors.png")

# 图B：三模态作用差异
fig, ax = plt.subplots(1, 3, figsize=(W_FULL, 2.2), gridspec_kw=dict(wspace=0.5, width_ratios=[1.3, 1.2, 1]))
b0 = ax[0]
groups = ["负向", "中性", "正向"]
bottom = np.zeros(3)
for m in "TAV":
    vals = np.array([NUM["modality_valid"]["share_by_polarity"][g][m] for g in groups])
    b0.bar(range(3), vals, bottom=bottom, color=MOD_COLOR[m], width=0.6, label=MOD_CN[m])
    bottom += vals
b0.set_xticks(range(3), [f"{g}\n(n={int((pol == g).sum())})" for g in groups]); b0.set_ylim(0, 1)
b0.set_ylabel("平均绝对 Shapley 份额"); b0.set_title("(a) 按真实极性分组", loc="left")
b0.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28), handlelength=1)
b1 = ax[1]
keys = ["U_T", "U_A", "U_V", "P_TA", "P_TV", "P_AV", "C"]
absm = [np.abs(V["R"][k]).mean() for k in keys]
sgn = [V["R"][k].mean() for k in keys]
cols = [BLUE, ORANGE, MUTED, SLATE, SLATE, SLATE, INK]
b1.bar(range(7), absm, color=cols, width=0.65)
b1.set_yscale("log"); b1.set_ylim(min(absm) / 3, max(absm) * 6)
for k, (h, sm) in enumerate(zip(absm, sgn)):
    b1.text(k, h * 1.25, f"{sm:+.4f}", ha="center", va="bottom", fontsize=6, rotation=90, color=SLATE)
b1.set_xticks(range(7), ["$R_T$", "$R_A$", "$R_V$", "$R_{TA}$", "$R_{TV}$", "$R_{AV}$", "$R_{TAV}$"], fontsize=7)
b1.set_ylabel("平均绝对贡献（对数轴）"); b1.set_title("(b) 七个区域", loc="left")
b2 = ax[2]
tn = {"additive_or_unique": "可加/独有", "indeterminate": "效应过小", "synergy": "协同", "redundant": "冗余", "conflict": "冲突"}
tc = NUM["modality_valid"]["types"]
order = [k for k in ("additive_or_unique", "indeterminate", "synergy", "redundant", "conflict") if k in tc]
b2.barh(range(len(order)), [tc[k] for k in order], color=BLUE, height=0.6)
for k, t in enumerate(order):
    b2.text(tc[t] + 8, k, str(tc[t]), va="center", fontsize=7)
b2.set_yticks(range(len(order)), [tn[k] for k in order]); b2.invert_yaxis(); b2.set_xlabel("样本数")
b2.set_title("(c) 交互类型", loc="left"); b2.set_xlim(0, max(tc.values()) * 1.25)
savefig(fig, "fig_q3_modality_roles.png")

# 图C：删除检验
fs = NUM["faithfulness"]
fig, ax = plt.subplots(1, 2, figsize=(W_FULL * 0.72, 2.0), gridspec_kw=dict(wspace=0.45))
vd_rows = read_csv(SRC / f"tables/{POOL}/valid_deletion_100.csv")
for a_, (name, rws) in zip(ax, (("验证集抽样 100 条", vd_rows),)):
    top = [np.mean([float(r[f"top{k}_drop"]) for r in rws]) for k in (1, 3, 5)]
    rnd = [np.mean([float(r[f"rand{k}_drop"]) for r in rws]) for k in (1, 3, 5)]
    xx = np.arange(3)
    a_.bar(xx - 0.18, top, 0.34, color=BLUE, label="删除得分最高的 k 个位置")
    a_.bar(xx + 0.18, rnd, 0.34, color=MUTED, label="随机删除 k 个位置")
    a_.set_xticks(xx, ["k=1", "k=3", "k=5"]); a_.set_ylabel("定向预测降幅"); a_.set_title(f"(a) {name}", loc="left")
    a_.legend(loc="upper left", fontsize=7)
    a_.set_ylim(0, max(top) * 1.55)
ev_main = [(r, EV[(r["sample_id"], r["max_share_modality"])]) for r in att4_rows]
d_sp = [float(e["span_drop"]) for _, e in ev_main]; d_rd = [float(e["random_mean_drop"]) for _, e in ev_main]
ax[1].scatter(d_rd, d_sp, s=12, color=BLUE, zorder=3)
lim = max(max(d_sp), max(d_rd)) * 1.1
ax[1].plot([0, lim], [0, lim], color=MUTED, lw=0.6, ls="--")
ax[1].set_xlabel("同长度随机片段平均降幅"); ax[1].set_ylabel("证据片段降幅"); ax[1].set_title("(b) 附件4 主要模态证据片段", loc="left")
savefig(fig, "fig_q3_deletion.png")

# 图D：典型样本的时间轴重要性（文本 / 语音 / 视觉）与关键帧
pos = read_csv(SRC / f"tables/{POOL}/att4_ig_positions.csv")


def timeline(sid, ax_list, title):
    rs = [r for r in pos if r["sample_id"] == sid]
    for a_, m in zip(ax_list, "TAV"):
        pts = [(float(r["start_sec"]), float(r["end_sec"]), float(r["ig_directed"]), r["word"]) for r in rs if r["modality"] == m and r["start_sec"] not in ("", "-1")]
        mx = max([abs(p[2]) for p in pts] + [1e-9])
        e = EV[(sid, m)]
        if e["start_sec"]:
            a_.axvspan(float(e["start_sec"]), float(e["end_sec"]), color=ORANGE, alpha=0.25, lw=0)
        for s0, s1, v, w in pts:
            a_.bar((s0 + s1) / 2, v, width=max(s1 - s0, 0.04) * 0.9, color=BLUE if v >= 0 else VERMIL, lw=0)
        a_.axhline(0, color=SLATE, lw=0.4)
        a_.set_ylabel(f"{MOD_CN[m]}\n{float(A4['share'][A4['ids'].index(sid), 'TAV'.index(m)]):.0%}", rotation=0, ha="right", va="center")
        a_.set_yticks([]); a_.spines["left"].set_visible(False)
        if m == "V" and e["feature_all_zero"] == "True":
            a_.text(0.5, 0.5, "视觉特征全零", transform=a_.transAxes, ha="center", fontsize=7, color=SLATE)
    ax_list[0].set_title(title, loc="left")
    ax_list[-1].set_xlabel("原视频时间 / s")


def keyframe(sid, t, path):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(ATT4_VIDEO / f"{sid}.mp4"), "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "3", str(path)], check=True)


TYP = os.environ.get("Q3_TYPICAL", "10,14").split(",")
for sid in TYP:
    j = A4["ids"].index(sid)
    fig = plt.figure(figsize=(W_FULL, 2.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], hspace=0.15, wspace=0.08)
    axs = [fig.add_subplot(gs[k, 0]) for k in range(3)]
    for a_ in axs[1:]:
        a_.sharex(axs[0]); a_.sharey(axs[0])          # 三行同一纵轴刻度，高度可直接比较
    for a_ in axs[:-1]:
        plt.setp(a_.get_xticklabels(), visible=False)
    timeline(sid, axs, f"(a) 样本 {sid} 三模态位置重要性（三行同一刻度；浅橙底为证据片段）")
    e = EV[(sid, "V")]
    axk = fig.add_subplot(gs[:, 1])
    if e["feature_all_zero"] != "True" and e["keyframe_sec"]:
        png = FIG / f"_q3_keyframe_{sid}.jpg"
        keyframe(sid, float(e["keyframe_sec"]), png)
        axk.imshow(plt.imread(png)); png.unlink()
        axk.set_title(f"(b) 视觉关键帧 {float(e['keyframe_sec']):.2f} s（第 {e['keyframe_frame']} 帧）", loc="left", fontsize=8)
    axk.axis("off")
    savefig(fig, f"fig_q3_case_{sid}.png")

# ---------------------------------------------------------------- 论文表格
GEN.mkdir(parents=True, exist_ok=True)
L, R_ = NUM["light"], NUM["finetuned"]
pct = lambda x: f"{100 * x:.2f}%"
perf = ["::: table {#tab:q3perf} 问题三预测性能（中性输出强度为0；阈值由验证集选定） {widths=3.2,1.6,1.2,1.3,1.1,1.2,1.7 align=l,c,c,c,c,c,c}",
        "模型 | 数据 | Acc3 | macro-F1 | MAE | Pearson | 参数文件/种子", "---",
        f"轻量版（冻结BERT） | 验证集 | {pct(L['valid']['Acc3'])} | {L['valid']['F1_macro']:.4f} | {L['valid']['MAE']:.4f} | {L['valid']['Pearson']:.4f} | {NUM['light_seeds'][1111]['compact_bytes'] / 1e6:.2f} MB",
        f"轻量版，阈值按原视频交叉拟合 | 验证集 | {pct(L['valid_crossfit']['Acc3'])} | {L['valid_crossfit']['F1_macro']:.4f} | {L['valid_crossfit']['MAE']:.4f} | — | —",
        f"轻量版（冻结BERT） | 测试集 | {pct(L['test']['Acc3'])} | {L['test']['F1_macro']:.4f} | {L['test']['MAE']:.4f} | {L['test']['Pearson']:.4f} | —",
        f"全参数微调版（对照） | 验证集 | {pct(R_['valid']['Acc3'])} | {R_['valid']['F1_macro']:.4f} | {R_['valid']['MAE']:.4f} | {R_['valid']['Pearson']:.4f} | 439 MB",
        f"全参数微调版（对照） | 测试集 | {pct(R_['test']['Acc3'])} | {R_['test']['F1_macro']:.4f} | {R_['test']['MAE']:.4f} | {R_['test']['Pearson']:.4f} | —", ":::"]
(GEN / "q3_perf_table.md").write_text("\n".join(perf) + "\n", encoding="utf-8")

POLCN = {"negative": "负向", "neutral": "中性", "positive": "正向"}


def main_evidence(r):
    m = r["main_modality"] if r["main_modality"] != "uncertain" else r["max_share_modality"]
    if m == "T":
        txt, lv = f"“{r['text_evidence']}” {r['text_evidence_sec'].replace('-', '–')} s", r["text_evidence_level"]
    elif m == "A":
        txt, lv = f"语音 {r['audio_evidence_sec'].replace('-', '–')} s", r["audio_evidence_level"]
    else:
        txt, lv = f"关键帧 {float(r['vision_keyframe_sec']):.2f} s（第{r['vision_keyframe_frame']}帧）", r["vision_evidence_level"]
    return txt, lv


t4 = ["::: table {#tab:q3att4} 附件4全部样本的预测与解释结果 {widths=0.6,0.7,1.0,1.0,1.0,1.0,1.5,4.9,0.7 align=c,c,c,c,c,c,c,l,c size=8.5 margin=35}",
      r"编号 | 极性 | 强度 | $\phi_T$ | $\phi_A$ | $\phi_V$ | 主模态 | 主要证据（片段或关键帧、原视频时间） | 等级", "---"]
for r in att4_rows:
    ev_txt, lv = main_evidence(r)
    mm = {"T": "文本", "A": "语音", "V": "视觉", "uncertain": f"不确定（{MOD_CN[r['max_share_modality']]}）"}[r["main_modality"]]
    t4.append(f"{r['sample_id']} | {POLCN[r['pred_polarity']]} | {r['pred_intensity']:.3f} | {r['phi_T']:.3f} | {r['phi_A']:.3f} | {r['phi_V']:.3f} | {mm} | {ev_txt} | {lv}")
t4.append(":::")
(GEN / "q3_att4_table.md").write_text("\n".join(t4) + "\n", encoding="utf-8")
NUM["att4_rows"] = att4_rows

(FIG / "q3_numbers.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
print(json.dumps({k: NUM[k] for k in ("tau", "light", "finetuned", "light_seed_valid_MAE_mean_sd")}, ensure_ascii=False, indent=1))
print(json.dumps(NUM["modality_valid"], ensure_ascii=False)[:1500])
print(json.dumps(NUM["att4"], ensure_ascii=False))
