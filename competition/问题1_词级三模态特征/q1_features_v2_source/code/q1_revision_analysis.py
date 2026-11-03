"""问题一论文中引用的全部统计量。

只读取已保存的结果（研究归档、v2 聚合结果、服务器实验表格、附件2 重叠样本抽取），
输出 ../figures/q1_revision_numbers.json，供论文正文与作图脚本引用。
用法：python q1_revision_analysis.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
Q1 = ROOT / "mosei_q1" / "results"
V2 = Q1 / "q1_v2"
ZIP = Q1 / "q1_features_e2051fa495cb1760_research.zip"
RAW = ROOT / "E题数据" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
sys.path.insert(0, str(ROOT / "mosei_q1" / "scripts"))
from aggregate_v2 import SUPPORT, overlap  # noqa: E402

NUM = {}
Z = zipfile.ZipFile(ZIP)
META = {}
for n in Z.namelist():
    if n.endswith("metadata.json"):
        m = json.loads(Z.read(n))
        META[m["sample"]["sample_id"]] = m
IDS = sorted(META)
MAN = {r["sample_id"]: r for r in json.loads((Q1 / "q1/input/manifest.json").read_text(encoding="utf-8"))["samples"]}


def native(sid):
    return np.load(io.BytesIO(Z.read(f"samples/{META[sid]['sample']['key']}/native.npz")))


def aligned_v1(sid):
    return np.load(io.BytesIO(Z.read(f"samples/{META[sid]['sample']['key']}/aligned.npz")))


def aligned_v2(sid):
    return np.load(V2 / "samples" / sid / "aligned_v2.npz")


def f(x):
    return float(x) if x not in ("", "None", None) else np.nan


def rd(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ------------------------------------------------------------------ A2：清音帧问题的规模
def a2_voicing():
    names = META[IDS[0]]["audio_feature_names"]
    f0 = names.index("F0semitoneFrom27.5Hz_sma3nz")
    A = np.concatenate([native(s)["audio"] for s in IDS])
    unv = A[:, f0] == 0
    nz = [i for i, n in enumerate(names) if n.endswith("_sma3nz")]
    amp = [i for i, n in enumerate(names) if "amplitudeLogRelF0" in n]
    silent = (A[:, names.index("Loudness_sma3")] <= 0.0011) & (A[:, names.index("alphaRatio_sma3")] == 0) & (A[:, names.index("mfcc1_sma3")] == 0)
    # 词级 F0：v1（全部帧平均）与 v2（只用浊音帧）的差异
    d = []
    for s in IDS:
        v1, v2 = aligned_v1(s), aligned_v2(s)
        m = v2["audio_voiced_mask"] & (v2["alignment_state"] > 0)
        d += list(v2["audio"][m, f0] - v1["audio"][m, f0])
    d = np.array(d)
    NUM["a2"] = dict(frames=int(len(A)), unvoiced_frac=rd(unv.mean(), 4), silent_frac=rd(silent.mean(), 4),
                     nz_features=len(nz), zero_on_unvoiced=int(sum(np.all(A[unv][:, i] == 0) for i in nz)),
                     amp_minus201_frac=rd(np.mean(A[unv][:, amp] == -201), 3),
                     word_f0_bias_st_median=rd(np.median(d), 2), word_f0_bias_st_p90=rd(np.percentile(d, 90), 2),
                     voiced_ratio_word_median=rd(np.median(np.concatenate([aligned_v2(s)["voiced_ratio"][aligned_v2(s)["alignment_state"] > 0] for s in IDS])), 3))


# ------------------------------------------------------------------ A3/A4：全量 WhisperX 与 MFA 对照
def a3_mfa():
    R = list(csv.DictReader(open(V2 / "server_tables/Q1-4_whisperx_vs_mfa_words.csv", encoding="utf-8-sig")))
    for r in R:
        for k in ("wx_start", "wx_end", "wx_score", "mfa_start", "mfa_end"):
            r[k] = f(r[k])
        r["mfa_valid"] = r["mfa_valid"] == "True"
    failed = sorted({r["sample_id"] for r in R} - {r["sample_id"] for r in R if r["mfa_valid"]})
    B = [r for r in R if r["mfa_valid"]]
    ds = np.array([r["wx_start"] - r["mfa_start"] for r in B]) * 1000
    de = np.array([r["wx_end"] - r["mfa_end"] for r in B]) * 1000
    inter = np.array([max(0, min(r["wx_end"], r["mfa_end"]) - max(r["wx_start"], r["mfa_start"])) for r in B])
    union = np.array([max(r["wx_end"], r["mfa_end"]) - min(r["wx_start"], r["mfa_start"]) for r in B])
    iou = inter / union
    sc = np.array([r["wx_score"] for r in B])
    dur = np.array([r["wx_end"] - r["wx_start"] for r in B])
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    calib = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (sc >= a) & (sc < b)
        calib.append(dict(bin=f"[{a:.1f},{min(b, 1.0):.1f}{']' if b > 1 else ')'}", n=int(m.sum()),
                          median_abs_ms=rd(np.median(np.abs(np.r_[ds[m], de[m]])), 1), disagree=rd(np.mean(iou[m] < 0.5), 3)))
    by = defaultdict(list)
    for r, a in zip(B, ds):
        by[r["sample_id"]].append(a)
    per_sample = sorted(((np.median(np.abs(v)), k, float(np.median(v)), len(v)) for k, v in by.items()), reverse=True)
    pos = np.array(["first" if int(r["word_pos"]) == 0 else "last" if int(r["word_pos"]) == int(r["n_words"]) - 1 else "mid" for r in B])
    NUM["a3"] = dict(words=len(R), compared=len(B), mfa_failed_samples=failed, mfa_failed_words=len(R) - len(B),
                     abs_median_ms=rd(np.median(np.abs(np.r_[ds, de])), 1), abs_mean_ms=rd(np.mean(np.abs(np.r_[ds, de])), 1),
                     start_within={t: rd(np.mean(np.abs(ds) <= t), 3) for t in (20, 50, 100)},
                     end_within={t: rd(np.mean(np.abs(de) <= t), 3) for t in (20, 50, 100)},
                     signed_start_median_ms=rd(np.median(ds), 1), signed_end_median_ms=rd(np.median(de), 1),
                     iou_median=rd(np.median(iou), 3), iou_ge_05=rd(np.mean(iou >= 0.5), 3), iou_zero=rd(np.mean(iou == 0), 3),
                     by_position={g: dict(n=int((pos == g).sum()), median_abs_ms=rd(np.median(np.abs(np.r_[ds[pos == g], de[pos == g]])), 1),
                                          disagree=rd(np.mean(iou[pos == g] < 0.5), 3)) for g in ("first", "mid", "last")},
                     calibration=calib, one_frame_words=int((dur <= 0.0201).sum()), one_frame_disagree=rd(np.mean(iou[dur <= 0.0201] < 0.5), 3),
                     samples_median_gt200=int(sum(p[0] > 200 for p in per_sample)), samples_median_gt100=int(sum(p[0] > 100 for p in per_sample)),
                     worst=[dict(sample_id=k, median_signed_ms=rd(med, 0), words=n) for a, k, med, n in per_sample[:6]])
    NUM["a3"]["_arrays"] = dict(ds=ds.tolist(), de=de.tolist(), iou=iou.tolist(), score=sc.tolist())


# ------------------------------------------------------------------ A5：无约束识别检验转写一致性（Q1-5 完成后）
def a5_asr():
    p = V2 / "server_tables/Q1-5_asr_wer_per_sample.csv"
    if not p.exists():
        return
    R = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in R:
        for k in ("ref_words", "hyp_words", "hits", "subs", "dels", "ins"):
            r[k] = int(r[k])
        r["wer"] = float(r["wer"])
    tot = {k: sum(r[k] for r in R) for k in ("ref_words", "subs", "dels", "ins")}
    w = np.array([r["wer"] for r in R])
    NUM["a5_asr"] = dict(corpus_wer=rd((tot["subs"] + tot["dels"] + tot["ins"]) / tot["ref_words"], 4), **tot,
                         wer_median=rd(np.median(w), 3), wer_le_10=int((w <= 0.10).sum()), wer_gt_30=int((w > 0.30).sum()),
                         samples_with_insertions=int(sum(r["ins"] > 0 for r in R)),
                         high_wer=[dict(sample_id=r["sample_id"], wer=r["wer"], ins=r["ins"], dels=r["dels"], inserted=r["inserted"][:80])
                                   for r in sorted(R, key=lambda r: -r["wer"])[:10]],
                         leading_insertions=[dict(sample_id=r["sample_id"], inserted=r["inserted"][:60], spans=r["inserted_spans"][:60])
                                             for r in R if r["ins"] > 0 and r["inserted_spans"] and float(r["inserted_spans"].split("-")[0]) < 1.0][:12])
    hit = np.array([r["hits"] / max(1, r["ref_words"]) for r in R])
    NUM["a5_asr"].update(consistent=int((hit >= 0.7).sum()), partial=int(((hit >= 0.3) & (hit < 0.7)).sum()), mismatch=int((hit < 0.3).sum()),
                         mismatch_samples=[r["sample_id"] for r, h in zip(R, hit) if h < 0.3],
                         mismatch_videos=sorted({r["sample_id"].split("$_$")[0] for r, h in zip(R, hit) if h < 0.3}),
                         lead_extra_speech=int(sum(1 for r in R if r["ins"] > 0 and r["inserted_spans"] and float(r["inserted_spans"].split("-")[0]) < 1.0)))
    # 第三方参照：识别命中的词上，比较 WhisperX / MFA 与 Whisper 的词边界（先扣除 Whisper 的系统偏移）
    W = [r for r in csv.DictReader(open(V2 / "server_tables/Q1-5_asr_word_times.csv", encoding="utf-8-sig")) if r["op"] == "H"]
    M = {(r["sample_id"], int(r["word_pos"])): r for r in csv.DictReader(open(V2 / "server_tables/Q1-4_whisperx_vs_mfa_words.csv", encoding="utf-8-sig"))}
    cols = dict(sx=[], sm=[], ex=[], em=[])
    per = defaultdict(lambda: [[], []])
    for w in W:
        m = M[(w["sample_id"], int(w["word_pos"]))]
        a0, a1 = f(w["asr_start"]), f(w["asr_end"])
        cols["sx"].append((f(m["wx_start"]) - a0) * 1000); cols["ex"].append((f(m["wx_end"]) - a1) * 1000)
        if m["mfa_valid"] == "True":
            cols["sm"].append((f(m["mfa_start"]) - a0) * 1000); cols["em"].append((f(m["mfa_end"]) - a1) * 1000)
            per[w["sample_id"]][0].append(abs(f(m["wx_start"]) - a0)); per[w["sample_id"]][1].append(abs(f(m["mfa_start"]) - a0))
    third = {}
    for k, v in cols.items():
        v = np.array(v); b = np.median(v); c = np.abs(v - b)
        third[k] = dict(n=len(v), bias_ms=rd(b, 1), dev_median_ms=rd(np.median(c), 1), within50=rd(np.mean(c <= 50), 3), within100=rd(np.mean(c <= 100), 3))
    big = [d["sample_id"] for d in NUM["a3"]["worst"]]
    adj = []
    for sid, (x, y) in per.items():
        if np.median(np.abs(np.array(x) - np.array(y))) > 1.0:            # 两工具在该样本上相差超过 1 s
            adj.append(dict(sample_id=sid, n=len(x), wx_ms=rd(1000 * np.median(x), 0), mfa_ms=rd(1000 * np.median(y), 0)))
    NUM["a3_third"] = dict(hits=len(W), **third, gross_disagreements=adj, whisperx_closer=int(sum(a["wx_ms"] < a["mfa_ms"] for a in adj)))


# ------------------------------------------------------------------ A5：内容层面的音画同步（嘴部动作与响度包络）
def a5_sync():
    G, LAGS = 0.01, np.arange(-50, 51)

    def series(sid):
        m, n = META[sid], native(sid)
        T = m["media"]["clip_duration"]
        t = np.arange(0, T, G) + G / 2
        names = m["audio_feature_names"]
        loud = np.interp(t, n["audio_intervals"][:, 0] + 0.010, n["audio"][:, names.index("Loudness_sma3")])
        vn = m["vision_feature_names"]
        idx = np.clip(np.searchsorted(n["vision_intervals"][:, 0], t, side="right") - 1, 0, len(n["vision_intervals"]) - 1)
        mouth = (n["vision"][:, vn.index("AU25_r")] + n["vision"][:, vn.index("AU26_r")])[idx]
        return loud, mouth, n["vision_mask"][idx]

    def xcorr(loud, mouth, valid):
        r = np.full(len(LAGS), np.nan)
        for k, L in enumerate(LAGS):
            if L >= 0:
                a, b, v = loud[:len(loud) - L], mouth[L:], valid[L:]
            else:
                a, b, v = loud[-L:], mouth[:len(mouth) + L], valid[:len(valid) + L]
            a, b = a[v], b[v]
            if len(a) > 100 and a.std() > 0 and b.std() > 0:
                r[k] = np.corrcoef(a, b)[0, 1]
        return r

    S = [series(s) for s in IDS]
    S = [s for s in S if s[2].sum() >= 200]
    R = np.array([xcorr(*s) for s in S])
    rng = np.random.default_rng(0)
    N = []
    for _ in range(300):
        i, j = rng.choice(len(S), 2, replace=False)
        n = min(len(S[i][0]), len(S[j][1]))
        N.append(xcorr(S[i][0][:n], S[j][1][:n], S[j][2][:n]))
    N = np.array(N)
    mean, nmean = np.nanmean(R, 0), np.nanmean(N, 0)
    k = int(np.nanargmax(mean))
    boot = [LAGS[np.nanargmax(np.nanmean(R[rng.integers(0, len(S), len(S))], 0))] * 10 for _ in range(2000)]
    NUM["a5_sync"] = dict(samples=len(S), peak_lag_ms=int(LAGS[k] * 10), peak_r=rd(mean[k]), r_at0=rd(mean[LAGS == 0][0]),
                          null_at_peak=rd(nmean[k]), ci95_ms=[int(x) for x in np.percentile(boot, [2.5, 97.5])])
    NUM["a5_sync"]["_curves"] = dict(lags=(LAGS * 10).tolist(), mean=np.nan_to_num(mean).tolist(), null=np.nan_to_num(nmean).tolist(),
                                     p10=np.nanpercentile(R, 25, 0).tolist(), p90=np.nanpercentile(R, 75, 0).tolist())


# ------------------------------------------------------------------ A5：说话人身份抽帧核查
def a5_speaker():
    rv = json.loads((V2 / "speaker_review.json").read_text(encoding="utf-8"))
    NUM["a5_speaker"] = dict(method=rv["method"], counts=dict(Counter(v["category"] for v in rv["samples"].values())),
                             flagged={k: v for k, v in rv["samples"].items() if v["category"] != "single_visible_person"})


# ------------------------------------------------------------------ A6：与附件2的外部核对（18 条重叠样本）
def a6_external():
    X = np.load(V2 / "external/att2_overlap_18.npz")
    hits = json.loads((V2 / "external/att2_overlap.json").read_text())
    cos_all, cos_trunc, pos_ok, n_clean = [], [], 0, 0
    au_ratio, v_ratio = [], []
    P1, P2, VM, A = [], [], [], []
    for sid in hits:
        m = META[sid]
        ids, wid = m["token_map"]["input_ids"], m["token_map"]["word_ids"]
        T2 = X[f"al|{sid}|text"]
        v1 = aligned_v1(sid)
        clean = len(ids) <= 50
        for i in range(len(m["words"])):
            pos = [p for p, w in enumerate(wid) if w == i and p < 50]
            if not pos:
                continue
            a, b = v1["text"][i], T2[pos].mean(0)
            c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            (cos_all if clean else cos_trunc).append(c)
        au_ratio.append(int(str(X[f"un|{sid}|audio_lengths"])) / m["media"]["audio_duration"])
        v_ratio.append(int(str(X[f"un|{sid}|vision_lengths"])) / (m["media"]["video_frames"] / 2))
        if clean:
            n_clean += 1
            nzp = set(np.where(np.abs(X[f"al|{sid}|audio"]).sum(1) > 0)[0])
            pos_ok += int(nzp == set(range(1, len(ids) - 1)))
            v2 = aligned_v2(sid)
            for p in range(1, len(ids) - 1):
                w = wid[p]
                if w is None:
                    continue
                P1.append(v1["audio"][w]); P2.append(v2["audio"][w]); VM.append(v2["audio_voiced_mask"][w]); A.append(X[f"al|{sid}|audio"][p])
    P1, P2, VM, A = map(np.array, (P1, P2, VM, A))
    names = META[IDS[0]]["audio_feature_names"]
    f0 = names.index("F0semitoneFrom27.5Hz_sma3nz")

    def best(x, msk):
        r = [abs(pearsonr(x[msk], A[msk, j])[0]) if A[msk, j].std() > 0 else 0 for j in range(A.shape[1])]
        return rd(max(r)), int(np.argmax(r))
    ar = np.array(au_ratio)
    # 我们的逐样本有效人脸比例与附件2 视觉有效长度比例（15 Hz）
    ours_face = [native(s)["vision_mask"].mean() for s in hits]
    NUM["a6"] = dict(overlap=len(hits), splits=dict(Counter(v[0] for v in hits.values())), clean_samples=n_clean,
                     text_cos_mean=rd(np.mean(cos_all), 4), text_cos_min=rd(np.min(cos_all), 4), text_words=len(cos_all),
                     text_cos_trunc_mean=rd(np.mean(cos_trunc), 3), token_positions_match=f"{pos_ok}/{n_clean}",
                     audio_len_per_s=dict(median=rd(np.median(ar), 2), min=rd(np.min(ar), 2), max=rd(np.max(ar), 2),
                                          within_19p8_20p1=int(np.sum((ar > 19.8) & (ar < 20.1)))),
                     audio_len_ratio_all=[rd(x, 2) for x in ar],
                     f0_best_v1=best(P1[:, f0], np.ones(len(P1), bool)), f0_best_v2=best(P2[:, f0], VM),
                     face_ratio_corr=rd(pearsonr(ours_face, v_ratio)[0]), token_pairs=int(len(A)))


# ------------------------------------------------------------------ B1：句级特征与情感标签（描述性，不参与任何设计选择）
def b1_labels():
    rows = []
    for s in IDS:
        a = aligned_v2(s)
        m = META[s]
        names, vn = m["audio_feature_names"], m["vision_feature_names"]
        ok = a["alignment_state"] > 0

        def wm(x, msk):
            msk = msk & ok
            return float(x[msk].mean()) if msk.any() else np.nan
        rows.append(dict(sid=s, video=s.split("$_$")[0], label=float(MAN[s]["label"]), cls=MAN[s]["annotation"],
                         AU12=wm(a["vision"][:, vn.index("AU12_r")], a["vision_mask"]),
                         AU06=wm(a["vision"][:, vn.index("AU06_r")], a["vision_mask"]),
                         AU04=wm(a["vision"][:, vn.index("AU04_r")], a["vision_mask"]),
                         AU15=wm(a["vision"][:, vn.index("AU15_r")], a["vision_mask"]),
                         F0=wm(a["audio"][:, names.index("F0semitoneFrom27.5Hz_sma3nz")], a["audio_voiced_mask"]),
                         F0_std=wm(a["prosody"][:, 0], a["audio_voiced_mask"]),
                         loud=wm(a["audio"][:, names.index("Loudness_sma3")], a["audio_energy_mask"]),
                         loud_std=wm(a["prosody"][:, 2], a["audio_energy_mask"]),
                         speech_rate=float(a["utt_rates"][0]),
                         F0_slope_abs=wm(np.abs(a["prosody"][:, 1]), a["audio_voiced_mask"]),
                         loud_slope_abs=wm(np.abs(a["prosody"][:, 3]), a["audio_energy_mask"]),
                         text=np.mean(a["text"][ok], 0)))
    labels = np.array([r["label"] for r in rows])
    vids = np.array([r["video"] for r in rows])
    multi = [v for v in set(vids) if (vids == v).sum() >= 2]
    out = {}
    for k in ("AU12", "AU06", "AU04", "AU15", "F0", "F0_std", "loud", "loud_std", "speech_rate"):
        x = np.array([r[k] for r in rows], float)
        msk = np.isfinite(x)
        rho, p = spearmanr(x[msk], labels[msk])
        xc, lc = np.full_like(x, np.nan), np.full_like(labels, np.nan)
        for v in multi:
            sel = (vids == v) & msk
            if sel.sum() >= 2:
                xc[sel], lc[sel] = x[sel] - x[sel].mean(), labels[sel] - labels[sel].mean()
        mc = np.isfinite(xc) & np.isfinite(lc)
        rw, pw = spearmanr(xc[mc], lc[mc])
        out[k] = dict(n=int(msk.sum()), rho=rd(rho), p=rd(p, 4), n_within=int(mc.sum()), rho_within=rd(rw), p_within=rd(pw, 4))
    # 按原视频留一交叉验证的岭回归探针（固定正则系数，只作描述）
    def probe(Xf):
        Xf = np.asarray(Xf, float)
        pred = np.zeros(len(labels))
        for v in set(vids):
            te, tr = vids == v, vids != v
            mu, sd = np.nanmean(Xf[tr], 0), np.nanstd(Xf[tr], 0) + 1e-6
            Xtr, Xte = np.nan_to_num((Xf[tr] - mu) / sd), np.nan_to_num((Xf[te] - mu) / sd)
            lam = 10.0 * Xtr.shape[1]
            w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (labels[tr] - labels[tr].mean()))
            pred[te] = Xte @ w + labels[tr].mean()
        return dict(r=rd(pearsonr(pred, labels)[0]), mae=rd(np.mean(np.abs(pred - labels))))
    feats = {"text": np.array([r["text"] for r in rows]),
             "audio": np.array([[r[k] for k in ("F0", "F0_std", "loud", "loud_std", "speech_rate")] for r in rows]),
             "audio_mean": np.array([[r[k] for k in ("F0", "loud", "speech_rate")] for r in rows]),
             "audio_mean_dyn": np.array([[r[k] for k in ("F0", "loud", "speech_rate", "F0_std", "loud_std", "F0_slope_abs", "loud_slope_abs")] for r in rows]),
             "vision": np.array([[r[k] for k in ("AU12", "AU06", "AU04", "AU15")] for r in rows])}
    NUM["b1"] = dict(corr=out, videos=len(set(vids)), multi_clip_videos=len(multi),
                     probe={k: probe(v) for k, v in feats.items()}, baseline_mae=rd(np.mean(np.abs(labels - labels.mean()))),
                     cls_counts=dict(Counter(r["cls"] for r in rows)))
    NUM["b1"]["_rows"] = [dict(cls=r["cls"], AU12=r["AU12"], F0=r["F0"], label=r["label"], video=r["video"]) for r in rows]


# ------------------------------------------------------------------ B2：词覆盖时长与非词片段
def b2_coverage():
    S = json.loads((V2 / "summary_v2.json").read_text(encoding="utf-8"))
    cov = np.array([s["word_time"] / s["clip_duration"] for s in S])
    kinds, dur = Counter(), defaultdict(float)
    for s in IDS:
        a = aligned_v2(s)
        for (x, y), t in zip(a["gap_intervals"], a["gap_type"]):
            kinds[str(t)] += 1
            dur[str(t)] += y - x
    NUM["b2"] = dict(word_time=rd(sum(s["word_time"] for s in S), 2), clip_time=rd(sum(s["clip_duration"] for s in S), 2),
                     coverage_total=rd(sum(s["word_time"] for s in S) / sum(s["clip_duration"] for s in S), 3),
                     coverage_median=rd(np.median(cov), 3), coverage_min=rd(cov.min(), 3), coverage_max=rd(cov.max(), 3),
                     gaps=dict(kinds), gap_time={k: rd(v, 1) for k, v in dur.items()},
                     states=dict(zip(("invalid", "doubtful", "reliable"), np.array([s["states"] for s in S]).sum(0).tolist())))
    NUM["b2"]["_coverage"] = cov.tolist()


# ------------------------------------------------------------------ B6：PTS 与标称帧率网格
def b6_pts():
    cache = V2 / "external/fps.json"
    if cache.exists():
        fps = json.loads(cache.read_text())
    else:
        fps = {}
        for s in IDS:
            out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                                  "-of", "json", str(RAW / META[s]["sample"]["relative_path"])], capture_output=True).stdout
            fps[s] = json.loads(out)["streams"][0]["r_frame_rate"]
        cache.write_text(json.dumps(fps, indent=1))
    irr_samples, irr_last, wrong, maxdev, drift_int = 0, 0, 0, [], []
    tot = 0
    for s in IDS:
        rate = float(Fraction(fps[s]))
        pts = np.array(META[s]["media"]["source_video_pts"])
        k = np.arange(len(pts))
        dev = np.abs(pts - pts[0] - k / rate)
        d = np.diff(pts)
        irr = np.where(np.abs(d - 1 / rate) > 0.002)[0]
        if len(irr):
            irr_samples += 1
            irr_last += int(np.all(irr >= len(pts) - 3))
        wrong += int((dev > 0.5 / rate).sum()); tot += len(pts)
        maxdev.append(dev.max() * 1000)
        if fps[s] in ("30000/1001", "24000/1001"):
            drift_int.append(np.abs(pts - pts[0] - k / round(rate)).max() * 1000)
    NUM["b6"] = dict(fps=dict(Counter(fps.values())), irregular_samples=irr_samples, irregular_only_last_frames=irr_last,
                     wrong_slot_frames=wrong, frames=tot, wrong_frac=rd(wrong / tot, 4), max_dev_ms=rd(max(maxdev), 1),
                     noninteger_fps_samples=len(drift_int), int_fps_drift_max_ms=rd(max(drift_int), 1), t0_all_zero=True)


# ------------------------------------------------------------------ B7：词区间平移的特征敏感性（基线）
def b7_shift():
    shifts = [-500, -300, -200, -100, -50, 50, 100, 200, 300, 500]
    ref = {}
    feats = {}
    for s in IDS:
        n, a = native(s), aligned_v2(s)
        names = META[s]["audio_feature_names"]
        nz = np.array([x.endswith("_sma3nz") for x in names])
        feats[s] = (n, a, nz)
    # 逐维标准化统计
    A = np.concatenate([feats[s][0]["audio"][feats[s][0]["audio_mask"]] for s in IDS])
    V = np.concatenate([feats[s][0]["vision"][feats[s][0]["vision_mask"]] for s in IDS])
    amu, asd, vmu, vsd = A.mean(0), A.std(0) + 1e-6, V.mean(0), V.std(0) + 1e-6

    def pooled(n, nz, s0, e0):
        u = n["audio_intervals"][:, 0]
        sup = np.column_stack([u + SUPPORT["20ms"][0], u + SUPPORT["20ms"][1]])
        za = (n["audio"] - amu) / asd
        w = overlap(sup, s0, e0) * n["audio_mask"]
        xa = (w @ za[:, ~nz]) / w.sum() if w.sum() > 0 else None
        wv = overlap(n["vision_intervals"], s0, e0) * n["vision_mask"]
        xv = (wv @ ((n["vision"] - vmu) / vsd)) / wv.sum() if wv.sum() > 0 else None
        return xa, xv

    def cos(p, q):
        return float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-12))
    res = {d: ([], []) for d in shifts}
    base = {}
    for s in IDS:
        n, a, nz = feats[s]
        T = META[s]["media"]["clip_duration"]
        for i, ((s0, e0), st) in enumerate(zip(a["word_intervals"], a["alignment_state"])):
            if st != 2:
                continue
            xa, xv = pooled(n, nz, s0, e0)
            base[(s, i)] = (xa, xv)
            for d in shifts:
                s1, e1 = s0 + d / 1000, e0 + d / 1000
                if s1 < 0 or e1 > T:
                    continue
                ya, yv = pooled(n, nz, s1, e1)
                if xa is not None and ya is not None:
                    res[d][0].append(cos(xa, ya))
                if xv is not None and yv is not None:
                    res[d][1].append(cos(xv, yv))
    curve = {d: dict(audio=rd(np.mean(v[0])), vision=rd(np.mean(v[1])), n=len(v[0])) for d, v in res.items()}
    # 观测：WhisperX 与 MFA 两种区间下的相似度（按 IoU 分组）
    R = list(csv.DictReader(open(V2 / "server_tables/Q1-4_whisperx_vs_mfa_words.csv", encoding="utf-8-sig")))
    obs = defaultdict(lambda: ([], []))
    for r in R:
        if r["mfa_valid"] != "True":
            continue
        k = (r["sample_id"], int(r["word_pos"]))
        if k not in base:
            continue
        n, a, nz = feats[k[0]]
        ya, yv = pooled(n, nz, f(r["mfa_start"]), f(r["mfa_end"]))
        s0, e0 = f(r["wx_start"]), f(r["wx_end"])
        inter = max(0, min(e0, f(r["mfa_end"])) - max(s0, f(r["mfa_start"])))
        iou = inter / (max(e0, f(r["mfa_end"])) - min(s0, f(r["mfa_start"])))
        g = "IoU≥0.5" if iou >= 0.5 else ("0<IoU<0.5" if iou > 0 else "IoU=0")
        xa, xv = base[k]
        if xa is not None and ya is not None:
            obs[g][0].append(cos(xa, ya))
        if xv is not None and yv is not None:
            obs[g][1].append(cos(xv, yv))
    NUM["b7"] = dict(curve={str(k): v for k, v in curve.items()},
                     observed={g: dict(audio=rd(np.mean(v[0])), vision=rd(np.mean(v[1])), n=len(v[0])) for g, v in obs.items()},
                     reliable_words=len(base))


# ------------------------------------------------------------------ B8/B10：视觉帧质量、人脸缺失与运动能量
def b8_vision():
    J = list(csv.DictReader(open(V2 / "server_tables/Q1-7_au_jitter_by_confidence.csv", encoding="utf-8-sig")))
    P = list(csv.DictReader(open(V2 / "server_tables/Q1-7_face_presence_per_sample.csv", encoding="utf-8-sig")))
    tot = {k: sum(int(r[k]) for r in P) for k in ("frames", "no_face", "low_conf", "valid")}
    # 运动能量：无人脸样本与有人脸样本
    me = {r["sample_id"]: float(r["motion_mean"]) for r in P}
    NUM["b8"] = dict(jitter=J, **tot, no_face_frac=rd(tot["no_face"] / tot["frames"], 4), low_conf_frac=rd(tot["low_conf"] / tot["frames"], 4),
                     samples_with_no_face_frames=int(sum(int(r["no_face"]) > 0 for r in P)),
                     motion_median=rd(np.median(list(me.values())), 4))


# ------------------------------------------------------------------ A7：复现实验（Q1-6 完成后）
def a7_repro():
    p = V2 / "server_tables/Q1-6_reproduction_summary.json"
    if p.exists():
        NUM["a7"] = json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ B14：含数字的词
def b14_digits():
    import re
    d = [(s, w["word"], w["confidence"]) for s in IDS for w in META[s]["words"] if re.search(r"\d", w["word"])]
    NUM["b14"] = dict(count=len(d), samples=len({x[0] for x in d}), words=[x[1] for x in d],
                      median_score=rd(np.median([x[2] for x in d]), 3))


if __name__ == "__main__":
    for fn in (a2_voicing, a3_mfa, a5_asr, a5_sync, a5_speaker, a6_external, b1_labels, b2_coverage, b6_pts, b7_shift, b8_vision, a7_repro, b14_digits):
        fn()
        print("done", fn.__name__, flush=True)
    out = PAPER / "figures" / "q1_revision_numbers.json"
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    brief = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v for k, v in NUM.items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:9000])
