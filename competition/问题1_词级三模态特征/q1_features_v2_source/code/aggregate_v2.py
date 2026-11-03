"""问题一特征 v2：由已保存的原生帧特征（native.npz）重新聚合，无需重新提取。

相对于直接按帧平均的改进：
- A2：eGeMAPSv02 中带 `_sma3nz` 后缀的 15 个参数只在浊音帧有定义（清音帧输出 0 或 −201），
      改为只在浊音帧（F0>0）上聚合，并输出词内浊音比例；能量/频谱组剔除全零的静默帧。
- B4：按 openSMILE 配置的真实分析窗确定每帧的时间支撑：20 ms Hamming 分支与 60 ms Gauss 分支
      均为左对齐、10 ms 步长，再经 3 帧滑动平均（sma3），有效支撑分别为 [u-10, u+30) ms 与 [u-10, u+70) ms。
- B3：F0 与响度在词内额外输出标准差与斜率（韵律动态），以及词时长。
- A4：词区间改为三态：2 可靠 / 1 存疑 / 0 无效（只占 1 个 CTC 帧判无效；分数 < --rho-min 判存疑）。
- B10：区分“无人脸”（success=0）与“有人脸但置信度低”；增加不依赖人脸的画面运动能量。
- B2：输出非词片段（句首、句尾、词间停顿）及其三模态均值，以及句级全局特征。
- C8：输出目录直接用样本编号命名。
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

HOP = 0.010
SUPPORT = {"20ms": (-0.010, 0.030), "60ms": (-0.010, 0.070)}  # 相对 openSMILE 报告的帧起点
CTC_FRAME = 0.020
MIN_PAUSE = 0.100  # 词间间隔短于 100 ms 视为相邻词之间的分隔帧，不计为停顿


# ------------------------------------------------------------------ 数据读取（研究归档 zip 或运行目录）
class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.suffix == ".zip" else None

    def keys(self):
        if self.zip:
            return sorted({n.split("/")[1] for n in self.zip.namelist() if n.startswith("samples/") and n.count("/") >= 2})
        return sorted(p.name for p in (self.path / "samples").iterdir() if (p / "complete.json").exists())

    def read(self, key, name):
        if self.zip:
            data = self.zip.read(f"samples/{key}/{name}")
        else:
            data = (self.path / "samples" / key / name).read_bytes()
        return np.load(io.BytesIO(data), allow_pickle=False) if name.endswith(".npz") else json.loads(data)


# ------------------------------------------------------------------ 聚合工具
def overlap(intervals, s, e):
    return np.maximum(0.0, np.minimum(e, intervals[:, 1]) - np.maximum(s, intervals[:, 0]))


def weighted_stats(t, x, w):
    """加权均值、加权标准差与加权最小二乘斜率（单位：特征单位/秒）。"""
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan, np.nan
    mu = float((w * x).sum() / sw)
    sd = float(np.sqrt((w * (x - mu) ** 2).sum() / sw))
    tm = (w * t).sum() / sw
    var_t = (w * (t - tm) ** 2).sum()
    slope = float((w * (t - tm) * (x - mu)).sum() / var_t) if var_t > 1e-12 and (w > 0).sum() >= 2 else np.nan
    return mu, sd, slope


def pool(intervals, feats, mask, s, e):
    w = overlap(intervals, s, e) * mask
    if w.sum() <= 0:
        return np.zeros(feats.shape[1], np.float32), False, w
    return (w @ feats / w.sum()).astype(np.float32), True, w


# ------------------------------------------------------------------ 单条样本
def load_evidence(mfa_csv=None, asr_csv=None, asr_words_csv=None):
    """全量 MFA 对照与无约束识别的逐词证据（均可缺省）。"""
    import csv
    ev = dict(mfa={}, asr_hit=set(), consistency={})
    if mfa_csv:
        for r in csv.DictReader(open(mfa_csv, encoding="utf-8-sig")):
            if r["mfa_valid"] == "True":
                a0, a1, b0, b1 = (float(r[k]) for k in ("wx_start", "wx_end", "mfa_start", "mfa_end"))
                iou = max(0.0, min(a1, b1) - max(a0, b0)) / (max(a1, b1) - min(a0, b0))
                ev["mfa"][(r["sample_id"], int(r["word_pos"]))] = iou
    if asr_words_csv:
        for r in csv.DictReader(open(asr_words_csv, encoding="utf-8-sig")):
            if r["op"] == "H":
                ev["asr_hit"].add((r["sample_id"], int(r["word_pos"])))
    if asr_csv:
        for r in csv.DictReader(open(asr_csv, encoding="utf-8-sig")):
            h = int(r["hits"]) / max(1, int(r["ref_words"]))
            ev["consistency"][r["sample_id"]] = (h, "consistent" if h >= 0.7 else "partial" if h >= 0.3 else "mismatch")
    return ev


def aggregate(src, key, frames_dir=None, rho_min=0.5, min_frames=2, review=None, evidence=None):
    meta = src.read(key, "metadata.json")
    nat = src.read(key, "native.npz")
    old = src.read(key, "aligned.npz")
    names = meta["audio_feature_names"]
    nz = np.array([n.endswith("_sma3nz") for n in names])
    long_win = np.array([n.startswith(("F0", "jitter", "shimmer", "HNR", "logRelF0")) or "amplitudeLogRelF0" in n for n in names])
    audio, a_int, a_ok = nat["audio"].astype(np.float64), nat["audio_intervals"], nat["audio_mask"]
    u = a_int[:, 0]
    sup = {k: np.column_stack([u + lo, u + hi]) for k, (lo, hi) in SUPPORT.items()}
    f0 = audio[:, names.index("F0semitoneFrom27.5Hz_sma3nz")]
    loud = audio[:, names.index("Loudness_sma3")]
    # 数字静音帧：响度处于下限 0.001，alpha 比、Hammarberg 指数与 MFCC1 恰为 0（两个频谱斜率为固定常数）
    silent = (loud <= 0.0011) & (audio[:, names.index("alphaRatio_sma3")] == 0) & (audio[:, names.index("mfcc1_sma3")] == 0)
    voiced = a_ok & (f0 > 0)
    energy_ok = a_ok & ~silent
    # 浊音比例按 10 ms 单元计：以 60 ms 窗中心为单元中心
    cell = np.column_stack([u + 0.030 - HOP / 2, u + 0.030 + HOP / 2])

    vis, v_int, v_ok = nat["vision"].astype(np.float64), nat["vision_intervals"], nat["vision_mask"]
    conf = nat["vision_confidence"]
    if frames_dir is not None and (Path(frames_dir) / f"{key}.npz").exists():
        fr = np.load(Path(frames_dir) / f"{key}.npz")
        success, motion = fr["success"].astype(bool), fr["motion"].astype(np.float64)
    else:
        success, motion = conf > 0, None

    words = meta["words"]
    L = len(words)
    text = old["text"]
    intervals = old["word_intervals"].copy()
    out_audio = np.zeros((L, 25), np.float32)
    m_energy = np.zeros(L, bool); m_voiced = np.zeros(L, bool)
    voiced_ratio = np.zeros(L, np.float32)
    prosody = np.full((L, 6), np.nan, np.float32)  # F0 标准差、F0 斜率、响度标准差、响度斜率、词时长、浊音比例
    out_vis = np.zeros((L, 22), np.float32); m_vis = np.zeros(L, bool)
    face_absent = np.zeros(L, np.float32); low_conf = np.zeros(L, np.float32)
    out_motion = np.zeros(L, np.float32); m_motion = np.zeros(L, bool)
    state = np.zeros(L, np.int8)
    for i, w in enumerate(words):
        s, e = intervals[i]
        if not w["valid"] or s < 0:
            continue
        # 三态判定：只占 1 个 CTC 帧（20 ms）的词在全量 MFA 对照中全部不一致，判为无效；
        # 分数低于 rho_min 的词判为存疑（阈值由 MFA 一致性代理校准，见论文 6.1 节）
        short = (e - s) < min_frames * CTC_FRAME - 1e-6
        low = w["confidence"] is None or w["confidence"] < rho_min
        state[i] = 0 if short else (1 if low else 2)
        if state[i] == 0:
            intervals[i] = (-1, -1)
            continue
        # 能量/频谱组：20 ms 分支，剔除静音帧
        grp_e = (~nz) & (~long_win)
        val, ok, _ = pool(sup["20ms"], audio[:, grp_e], energy_ok, s, e)
        out_audio[i, grp_e] = val; m_energy[i] = ok
        # nz 组：只用浊音帧；按分支取支撑
        any_v = False
        for win, grp in (("20ms", nz & ~long_win), ("60ms", nz & long_win)):
            val, ok, _ = pool(sup[win], audio[:, grp], voiced, s, e)
            out_audio[i, grp] = val; any_v |= ok
        m_voiced[i] = any_v
        cov = overlap(cell, s, e)
        voiced_ratio[i] = float((cov * voiced).sum() / max(e - s, 1e-9))
        # 韵律动态：以窗中心为采样时刻
        w60 = overlap(sup["60ms"], s, e) * voiced
        w20 = overlap(sup["20ms"], s, e) * energy_ok
        _, f0_sd, f0_sl = weighted_stats(u + 0.030, f0, w60)
        _, ld_sd, ld_sl = weighted_stats(u + 0.010, loud, w20)
        prosody[i] = [f0_sd, f0_sl, ld_sd, ld_sl, e - s, voiced_ratio[i]]
        # 视觉：与 v1 相同的有效帧定义；另给出无人脸与低置信度的时长占比
        val, ok, _ = pool(v_int, vis, v_ok, s, e)
        out_vis[i] = val; m_vis[i] = ok
        ov = overlap(v_int, s, e)
        face_absent[i] = float((ov * ~success).sum() / max(ov.sum(), 1e-9))
        low_conf[i] = float((ov * (success & ~v_ok)).sum() / max(ov.sum(), 1e-9))
        if motion is not None:
            val, ok, _ = pool(v_int, motion[:, None], np.ones(len(motion), bool), s, e)
            out_motion[i] = val[0]; m_motion[i] = ok
    prosody = np.nan_to_num(prosody, nan=0.0)
    sid = meta["sample"]["sample_id"]
    ev = evidence or dict(mfa={}, asr_hit=set(), consistency={})
    mfa_iou = np.array([ev["mfa"].get((sid, i), -1.0) for i in range(L)], np.float32)   # -1：MFA 未给出结果
    asr_hit = np.array([(sid, i) in ev["asr_hit"] for i in range(L)])
    hit_ratio, consistency = ev["consistency"].get(sid, (np.nan, "unknown"))
    if consistency == "mismatch":
        # 音轨中几乎识别不到给定转写：词时间缺乏声学支撑，全部降为“存疑”，保留特征供追溯
        state = np.minimum(state, 1).astype(np.int8)

    # 非词片段：句首、词间停顿、句尾（只取有效或存疑词的区间）
    T = meta["media"]["clip_duration"]
    ok_iv = sorted((float(a), float(b)) for (a, b), st in zip(intervals, state) if st > 0)
    gaps, prev = [], 0.0
    for k, (a, b) in enumerate(ok_iv):
        if (k == 0 and a - prev > 1e-3) or (k > 0 and a - prev >= MIN_PAUSE):
            gaps.append((prev, a, "lead" if k == 0 else "pause"))
        prev = max(prev, b)
    if T - prev > 1e-3:
        gaps.append((prev, T, "trail"))
    g_feat = np.zeros((len(gaps), 25 + 22 + 1), np.float32)
    g_mask = np.zeros((len(gaps), 3), bool)
    for k, (a, b, _) in enumerate(gaps):
        va, oka, _ = pool(sup["20ms"], audio[:, ~nz], energy_ok, a, b)
        g_feat[k, :25][~nz] = va
        vv, okv, _ = pool(v_int, vis, v_ok, a, b)
        g_feat[k, 25:47] = vv
        if motion is not None:
            vm, okm, _ = pool(v_int, motion[:, None], np.ones(len(motion), bool), a, b)
            g_feat[k, 47] = vm[0]
        else:
            okm = False
        g_mask[k] = (oka, okv, okm)

    # 句级全局特征：浊音帧上的 nz 组、能量组的均值与标准差，视觉均值与标准差，运动能量，语速与停顿占比
    def ms(x, m):
        return (x[m].mean(0), x[m].std(0)) if m.any() else (np.zeros(x.shape[1]), np.zeros(x.shape[1]))
    ae_mu, ae_sd = ms(audio[:, ~nz], energy_ok)
    av_mu, av_sd = ms(audio[:, nz], voiced)
    vv_mu, vv_sd = ms(vis, v_ok)
    speech = sum(b - a for a, b in ok_iv)
    utt = dict(audio_energy_mean=ae_mu, audio_energy_std=ae_sd, audio_voiced_mean=av_mu, audio_voiced_std=av_sd,
               vision_mean=vv_mu, vision_std=vv_sd,
               motion=np.array([motion[1:].mean() if motion is not None and len(motion) > 1 else 0.0,
                                motion[1:].std() if motion is not None and len(motion) > 1 else 0.0]),
               rates=np.array([len(ok_iv) / max(T, 1e-9), speech / max(T, 1e-9), voiced.mean(), v_ok.mean(), (~success).mean()]))

    arrays = dict(text=text, audio=out_audio, vision=out_vis, prosody=prosody, motion=out_motion,
                  word_intervals=intervals, alignment_state=state, mfa_iou=mfa_iou, asr_hit=asr_hit,
                  alignment_score=np.array([w["confidence"] if w["confidence"] is not None else -1 for w in words], np.float32),
                  text_mask=np.ones(L, bool), audio_energy_mask=m_energy, audio_voiced_mask=m_voiced,
                  voiced_ratio=voiced_ratio, vision_mask=m_vis, face_absent_ratio=face_absent, low_conf_ratio=low_conf,
                  motion_mask=m_motion, padding_mask=np.ones(L, bool), valid_length=np.int32(L),
                  gap_intervals=np.array([(a, b) for a, b, _ in gaps], np.float64).reshape(-1, 2),
                  gap_type=np.array([t for _, _, t in gaps], dtype="<U5"), gap_features=g_feat, gap_masks=g_mask,
                  **{f"utt_{k}": np.asarray(v, np.float32) for k, v in utt.items()})
    info = dict(sample_id=meta["sample"]["sample_id"], key=key, words=L, clip_duration=T,
                speaker_review=(review or {}).get(meta["sample"]["sample_id"]),
                transcript_consistency=consistency, asr_hit_ratio=None if not np.isfinite(hit_ratio) else round(float(hit_ratio), 3),
                word_time=float(sum(b - a for a, b in ok_iv)), states=[int((state == k).sum()) for k in (0, 1, 2)],
                voiced_frames=float(voiced.mean()), silent_frames=float(silent.mean()),
                audio_names=names, nz_features=[n for n, z in zip(names, nz) if z],
                long_window_features=[n for n, z in zip(names, long_win) if z])
    return arrays, info, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="研究归档 zip 或运行目录")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", default=None, help="Q1-7 输出的逐帧 success/motion 目录")
    p.add_argument("--rho-min", type=float, default=0.5)
    p.add_argument("--review", default=None, help="抽帧核查结果 JSON（说话人身份）")
    p.add_argument("--mfa", default=None, help="Q1-4 全量 WhisperX/MFA 逐词对照表")
    p.add_argument("--asr", default=None, help="Q1-5 逐样本识别结果表")
    p.add_argument("--asr-words", default=None, help="Q1-5 逐词识别匹配表")
    p.add_argument("--min-frames", type=int, default=2)
    args = p.parse_args()
    src = Source(args.source)
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))["samples"] if args.review else None
    evidence = load_evidence(args.mfa, args.asr, args.asr_words)
    out = Path(args.out)
    summary = []
    for key in src.keys():
        arrays, info, meta = aggregate(src, key, args.frames, args.rho_min, args.min_frames, review, evidence)
        d = out / "samples" / info["sample_id"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "aligned_v2.npz", "wb") as f:
            np.savez_compressed(f, **arrays)
        (d / "metadata_v2.json").write_text(json.dumps(dict(
            info=info, schema="q1-v2", source_key=key,
            parameters=dict(rho_min=args.rho_min, min_frames=args.min_frames, support=SUPPORT, hop=HOP),
            conventions=dict(alignment_state="2=reliable,1=doubtful,0=invalid(interval set to [-1,-1]); all words doubtful if transcript_consistency=mismatch",
                             mfa_iou="IoU between WhisperX and MFA intervals; -1 if MFA gave no result", asr_hit="word recognised by unconstrained ASR",
                             masks="True=available", prosody="F0_std,F0_slope(semitone/s),loud_std,loud_slope(/s),duration_s,voiced_ratio",
                             gap_features="25 audio (energy group only) + 22 vision + 1 motion")), ensure_ascii=False, indent=1), encoding="utf-8")
        summary.append(info)
    (out / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    st = np.array([s["states"] for s in summary]).sum(0)
    print(json.dumps(dict(samples=len(summary), invalid=int(st[0]), doubtful=int(st[1]), reliable=int(st[2]),
                          word_time=round(sum(s["word_time"] for s in summary), 2),
                          clip_time=round(sum(s["clip_duration"] for s in summary), 2)), ensure_ascii=False))


if __name__ == "__main__":
    main()
