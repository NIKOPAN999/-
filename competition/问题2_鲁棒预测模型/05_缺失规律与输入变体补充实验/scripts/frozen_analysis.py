"""问题二补充分析：用队员定型的压缩权重与匹配校准，在附件2 valid 上做只推理的分析（不训练、不选模）。

1. 复核：32 场景指标与队员附件 compact_valid_calibrated.csv 比较；
2. 剂量—反应网格：模态 {文本, 语音, 视觉, 三模态同步} × 比例 0.1–0.6 × 位置 {前, 中, 末}，保存逐样本预测；
3. 缺失形态：总比例 {0.1, 0.2, 0.3} 下，“多段短缺失”（每段 1–3 个内容词元，段间不相邻）与“单段连续缺失”
   （随机起点）对比，模态 {文本, 三模态同步}，各 5 次独立抽样，两种形态共用同一随机种子序列；
4. 原始输出：clean 下的三类 logits、未校准强度与校准后输出，用于符号投影与强度收缩诊断。
输出：outputs/frozen/*.csv、*.npz。
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "work/code_base"
sys.path.insert(0, str(BASE))
os.environ.setdefault("SELFMM_BERT_BASE", "/root/autodl-tmp/team/shared/baselines/pretrained/bert-base-uncased")
from calibration import apply_calibration  # noqa: E402
from data import AlignedDataset, block_mask, load_aligned  # noqa: E402
import run as R  # noqa: E402

DATA = Path("/root/autodl-tmp/team/shared/baselines/data/mosei_att2_aligned50.pkl")
OUT = ROOT / "outputs/frozen"
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
ckpt = BASE / "student_compact.pt"
model, stats, state = R.load_checkpoint(ckpt, dev)
model.eval()
cal = R.read_calibration(BASE / "student_compact_calibration.json", ckpt)
raw = load_aligned(DATA)
ds = AlignedDataset(raw["valid"], stats)
loader = DataLoader(ds, batch_size=64)
y_cls, y_reg = ds.cls, ds.reg
ids = ds.sample_ids


@torch.no_grad()
def run_masked(mask_fn):
    """mask_fn(batch, batch_start) -> masked batch；返回 logits、原始强度、校准类别与强度、实际新增遮挡比例。"""
    L, S, fr = [], [], []
    for b0, clean in enumerate(loader):
        batch = mask_fn(clean, b0 * 64) if mask_fn else clean
        o = model(R.to_device(batch, dev))
        L.append(o["logits"].cpu().numpy()); S.append(o["reg"].cpu().numpy())
        c = clean["content"].sum(1).clamp_min(1).numpy()
        removed = (clean["observed"].numpy() & ~batch["observed"].numpy()).sum(1)   # (B,3) 新增删除的已观测位置
        fr.append(removed / c[:, None])
    logits, score = np.concatenate(L), np.concatenate(S)
    cls, val = apply_calibration(logits, score, cal)
    return logits, score, cls, val, np.concatenate(fr)


def metrics(cls, val):
    return R.calculate_metrics({"cls": y_cls, "reg": y_reg}, cls, val)


# ---------- 1. 复核 32 场景 ----------
ref = {r["scenario"]: r for r in csv.DictReader(open(BASE / "compact_valid_calibrated.csv", encoding="utf-8-sig"))}
maxdiff = 0.0
for name, cond in R.scenarios():
    _, _, cls, val, _ = run_masked((lambda b, s, c=cond: block_mask(b, *c)) if cond else None)
    m = metrics(cls, val)
    for k in ("accuracy", "macro_f1", "mae", "pearson"):
        maxdiff = max(maxdiff, abs(m[k] - float(ref[name][k])))
print("reproduction max abs diff over 32 scenarios x 4 metrics:", maxdiff, flush=True)

# ---------- 4. clean 原始输出 ----------
logits, score, cls, val, _ = run_masked(None)
clean_cls, clean_val = cls.copy(), val.copy()
with open(OUT / "clean_raw_outputs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "y_cls", "y_reg", "logit_neg", "logit_neu", "logit_pos", "raw_reg", "pred_cls", "pred_intensity",
                "content_len", "text_cov", "audio_cov", "vision_cov"])
    cov = ds.observed.sum(1) / np.maximum(ds.content.sum(1), 1)[:, None]
    for i in range(len(ids)):
        w.writerow([ids[i], int(y_cls[i]), float(y_reg[i]), *map(float, logits[i]), float(score[i]), int(cls[i]), float(val[i]),
                    int(ds.content[i].sum()), *map(float, cov[i])])

# ---------- 2. 剂量—反应网格 ----------
MODES = {"text": (0,), "audio": (1,), "vision": (2,), "sync3": (0, 1, 2)}
rows, per_sample = [], {}
for mname, modes in MODES.items():
    for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        for pos in ("front", "middle", "end"):
            _, _, cls, val, fr = run_masked(lambda b, s, m=modes, r=ratio, p=pos: block_mask(b, m, r, p))
            m = metrics(cls, val)
            rows.append({"modality": mname, "ratio": ratio, "position": pos, **{k: m[k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
                         "removed_frac_mean": float(fr[:, list(modes)].mean())})
            per_sample[f"{mname}_{ratio}_{pos}"] = np.stack([cls, val])
    print("dose", mname, flush=True)
with open(OUT / "dose_response.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
np.savez_compressed(OUT / "dose_response_per_sample.npz", y_cls=y_cls, y_reg=y_reg, ids=np.array(ids),
                    clean=np.stack([clean_cls, clean_val]),
                    **per_sample)


# ---------- 3. 缺失形态：多段短缺失 vs 单段连续缺失 ----------
def pattern_mask(batch, start, modes, ratio, kind, draw):
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    for i in range(len(out["ids"])):
        g = np.random.default_rng([draw, start + i])
        slots = torch.where(out["content"][i])[0].numpy()
        n = len(slots)
        if n <= 1:
            continue
        K = min(n - 1, max(1, round(ratio * n)))
        chosen = np.zeros(n, bool)
        if kind == "block":
            st = g.integers(0, n - K + 1)
            chosen[st:st + K] = True
        else:
            left, tries = K, 0
            while left > 0 and tries < 200:
                tries += 1
                ln = int(min(left, g.integers(1, 4)))
                st = int(g.integers(0, n - ln + 1))
                lo, hi = max(0, st - 1), min(n, st + ln + 1)
                if chosen[lo:hi].any() and tries < 150:      # 段间至少隔一个位置，尽量不相邻
                    continue
                newly = ~chosen[st:st + ln]
                chosen[st:st + ln] = True
                left -= int(newly.sum())
        ix = torch.as_tensor(slots[chosen])
        for mode in modes:
            if mode == 0:
                out["ids"][i, ix] = 103
            elif mode == 1:
                out["audio"][i, ix] = 0
            else:
                out["vision"][i, ix] = 0
            out["observed"][i, ix, mode] = False
    return out


prow = []
for mname, modes in (("text", (0,)), ("sync3", (0, 1, 2))):
    for ratio in (0.1, 0.2, 0.3):
        for kind in ("scatter", "block"):
            for draw in range(5):
                _, _, cls, val, fr = run_masked(lambda b, s, m=modes, r=ratio, k=kind, d=draw: pattern_mask(b, s, m, r, k, d))
                m = metrics(cls, val)
                prow.append({"modality": mname, "ratio": ratio, "kind": kind, "draw": draw,
                             **{k: m[k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
                             "removed_frac_mean": float(fr[:, list(modes)].mean())})
                per_sample[f"pat_{mname}_{ratio}_{kind}_{draw}"] = np.stack([cls, val])
    print("pattern", mname, flush=True)
with open(OUT / "pattern_scatter_vs_block.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
np.savez_compressed(OUT / "pattern_per_sample.npz", **{k: v for k, v in per_sample.items() if k.startswith("pat_")})
json.dump({"reproduction_max_abs_diff": maxdiff, "calibration": cal, "n_valid": len(ids)}, open(OUT / "summary.json", "w"), indent=1)
print("FROZEN_DONE", flush=True)
