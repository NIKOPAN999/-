# -*- coding: utf-8 -*-
"""E2-14：用 E1 训练好的 MMSA 基线（MISA / MulT / ALMT / Self-MM）在与 E2-9、E2-11 完全相同的 valid 缺失场景上导出逐样本预测。
由 E2-11 的 predict_selfmm_views.py 改写，只增加 --model 参数，其余不变。

场景由 code2 的 data.bernoulli_split 生成（与 code2 的 p0_eval.py 相同）：clean、sync_{0.1,0.2,0.3,0.5}
（三模态同步，种子 1111 = 附件3机制，文本→[UNK]，A/V→0）、async_0.2。Self-MM 原样接收 [UNK]（与附件3格式一致）。
加载方式照搬 shared/baselines/configs/mmsa/predict_mmsa.py（读取训练时的 override.json，配置完全一致）；
MMSA 的 DataLoader 会打乱顺序，因此按样本 id 对齐回 pkl 顺序。只读 valid。
用法：python predict_mmsa_views.py --model misa --seed 1111 --out <npz>
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

B = Path("/root/autodl-tmp/team/shared/baselines")
sys.path.insert(0, str(B / "repos" / "MMSA" / "src"))
sys.path.insert(0, str(B / "configs" / "mmsa"))
sys.path.insert(0, "/root/autodl-tmp/team/q2/code/code2")
from data import bernoulli_split  # noqa: E402
from run_mmsa import materialize_config  # noqa: E402

VIEWS = [("clean", 0.0, True)] + [(f"sync_{r}", r, True) for r in (0.1, 0.2, 0.3, 0.5)] + [("async_0.2", 0.2, False)]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["misa", "mult", "almt", "self_mm"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from MMSA.config import get_config_regression  # noqa: E402
from MMSA.data_loader import MMDataLoader  # noqa: E402
from MMSA.models import AMIO  # noqa: E402
from MMSA.trains import ATIO  # noqa: E402

E1 = Path("/root/autodl-tmp/team/exp_common/E1_complete_baselines") / f"mmsa_{a.model}" / "outputs"
run_dir = E1 / "mmsa" / a.model / "e1" / f"seed{a.seed}"
override = json.loads((run_dir / "override.json").read_text(encoding="utf-8"))["override"]
ckpt = run_dir / "models" / f"{a.model}-mosei.pth"
cfg = materialize_config()
device = torch.device("cuda:0")
valid = pickle.load(open(B / "data" / "mosei_att2_aligned50.pkl", "rb"))["valid"]
order = {str(i): k for k, i in enumerate(valid["id"])}
y = np.asarray(valid["regression_labels"], dtype=float).ravel()

out = {}
for name, rate, sync in VIEWS:
    split = valid if rate == 0 else bernoulli_split(valid, rate, 1111, sync)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, dir="/root/autodl-tmp/team/q2/exp/E2-14_baseline_compare/outputs") as f:
        pickle.dump({"train": split, "valid": split, "test": split}, f)
        tmp = f.name
    try:
        args = get_config_regression(a.model, "mosei", str(cfg))
        args.update(model_save_path=str(ckpt), device=device, train_mode="regression", cur_seed=1,
                    custom_feature=tmp, feature_T=None, feature_A=None, feature_V=None)
        args.update(override)
        loaders = MMDataLoader(args, 2)
        model = AMIO(args).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        if a.model == "almt":
            # ALMT 训练器的 do_test 不支持逐样本返回，这里照抄其前向过程：model(text, audio, vision)
            ids_, preds_ = [], []
            with torch.no_grad():
                for bd in loaders["valid"]:
                    o = model(bd["text"].to(device), bd["audio"].to(device), bd["vision"].to(device))
                    preds_.append(o.detach().cpu().numpy().reshape(-1)); ids_.extend(list(bd["id"]))
            r = {"Ids": ids_, "SResults": np.concatenate(preds_)}
        else:
            r = ATIO().getTrain(args).do_test(model, loaders["valid"], mode="VALID", return_sample_results=True)
    finally:
        os.remove(tmp)
    pred = np.full(len(y), np.nan)
    for i, p in zip(r["Ids"], np.asarray(r["SResults"], dtype=float).reshape(-1)):
        pred[order[str(i)]] = p
    assert not np.isnan(pred).any(), name
    out[f"{name}__reg"] = pred
    print(name, "MAE", round(float(np.abs(pred - y).mean()), 4), flush=True)
np.savez_compressed(a.out, y=y, ids=np.asarray([str(i) for i in valid["id"]]), **out)
print("saved", a.out)
