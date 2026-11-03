# -*- coding: utf-8 -*-
"""E2-14：论文问题二的定型模型（队员提交的 student_compact.pt）在 E2-9、E2-11 的同一组 valid 缺失场景上导出逐样本输出。

场景与 E2-9、E2-11 完全相同：clean、sync_{0.1,0.2,0.3,0.5}（三模态同步，种子 1111，与附件3机制一致）、async_0.2。
缺失由 bernoulli_split 生成（逐字照抄 q2/code/code2/data.py 中的同名函数，避免与队员 data.py 重名冲突）：
文本写成 [UNK]（ID 100）、语音与视觉置零，随后走队员代码处理附件3的同一路径（[UNK]→[MASK]、观测标记置 0）。
输出未校准的回归值与三类 logits，由 E2-9 的 eval_views.py 统一做交叉拟合校准。只读 valid，不训练、不选模。
用法：python predict_team_views.py --out outputs/team_views.npz
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path("/root/autodl-tmp/team/q2/exp/E2-13_review_supplement/work/code_base")
sys.path.insert(0, str(BASE))
os.environ.setdefault("SELFMM_BERT_BASE", "/root/autodl-tmp/team/shared/baselines/pretrained/bert-base-uncased")
from data import AlignedDataset  # noqa: E402
import run as R  # noqa: E402

DATA = Path("/root/autodl-tmp/team/shared/baselines/data/mosei_att2_aligned50.pkl")
VIEWS = [("clean", 0.0, True)] + [(f"sync_{r}", r, True) for r in (0.1, 0.2, 0.3, 0.5)] + [("async_0.2", 0.2, False)]


def bernoulli_split(split: dict, rate: float, seed: int = 1111, sync: bool = True) -> dict:
    ids = np.rint(np.asarray(split["text_bert"])[:, 0]).astype(np.int64)
    attn = np.asarray(split["text_bert"])[:, 1].astype(bool)
    content = attn & (ids != 101) & (ids != 102)
    masks = []
    for m in range(1 if sync else 3):
        rng = np.random.RandomState(seed + 1000 * m)
        masks.append((rng.uniform(size=ids.shape) < rate) & content)
    if sync:
        masks = masks * 3
    out = dict(split)
    text_bert = np.array(split["text_bert"], copy=True)
    text_bert[:, 0][masks[0]] = 100
    out["text_bert"] = text_bert
    for m, name in ((1, "audio"), (2, "vision")):
        arr = np.array(split[name], copy=True)
        arr[masks[m]] = 0
        out[name] = arr
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.device("cuda")
model, stats, _ = R.load_checkpoint(BASE / "student_compact.pt", dev)
model.eval()
valid = pickle.load(open(DATA, "rb"))["valid"]
y = np.asarray(valid["regression_labels"], dtype=float).ravel()
out = {}
with torch.no_grad():
    for name, rate, sync in VIEWS:
        split = valid if rate == 0 else bernoulli_split(valid, rate, 1111, sync)
        ds = AlignedDataset(split, stats)
        L, S = [], []
        for batch in DataLoader(ds, batch_size=64):
            o = model(R.to_device(batch, dev))
            L.append(o["logits"].float().cpu().numpy()); S.append(o["reg"].float().cpu().numpy())
        logits, reg = np.concatenate(L), np.concatenate(S).ravel()
        out[f"{name}__reg"], out[f"{name}__logits"] = reg, np.log(np.clip(
            np.exp(logits - logits.max(1, keepdims=True)) / np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True), 1e-12, 1))
        print(name, "raw MAE", round(float(np.abs(reg - y).mean()), 4), "argmax acc",
              round(float((logits.argmax(1) == np.where(y < 0, 0, np.where(y > 0, 2, 1))).mean()), 4), flush=True)
np.savez_compressed(a.out, y=y, ids=np.asarray([str(i) for i in valid["id"]]), **out)
print("saved", a.out)
