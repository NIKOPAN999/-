# -*- coding: utf-8 -*-
"""
把赛题附件2/3/4统一转换为 MMSA 标准格式的 pkl，供 MMSA / EMT-DLFR / LNLN / EMOE 等基线直接读取。

MMSA 标准格式（这些仓库的数据加载器所读字段的并集）：
    {split: {raw_text, id, text, text_bert, audio, vision,
             audio_lengths, vision_lengths, regression_labels, classification_labels}}

子命令（对齐版；未对齐版见文末说明）：
    att2          附件2 aligned_50.pkl -> 补 audio_lengths / vision_lengths          [numpy 1.x 或 2.x 均可]
    att3          附件3 对齐版 30 个单文件 -> 推理用 pkl（标签为占位值）            [numpy 1.x 或 2.x 均可]
    att4-export   附件4 对齐版 20 个单文件 -> 中间 npz（原文件为 numpy 2 格式）      [必须 numpy >= 2]
    att4-pack     中间 npz -> 推理用 pkl                                              [建议 numpy 1.x，保证新旧 numpy 都能读]

输出统一写到 --out 目录（默认 ../data），并生成同名 .json 清单记录样本数、字段、形状和校验结果。

设计约定（与 01_数据核查与建模影响.md 一致）：
  * 对齐版的 audio/vision 与 BERT 词元逐位对应，因此 audio_lengths = vision_lengths = text_bert 注意力掩码之和；
  * 附件3 中的 [UNK]=100 与零行就是题目给定的缺失，转换时原样保留，下游推理必须把缺失注入率设为 0；
  * 附件3 没有 `text` 连续特征，填全零占位；下游必须 use_bert=True（从 text_bert 过 BERT），不得使用该占位；
  * 推理用 pkl 的 train/valid/test 三个划分内容相同（部分加载器会同时构建三个划分），标签全为 0 占位，
    只能用于导出预测，不能用于计算指标。
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("Q3_DATA_ROOT", "./data_original")
ATT2_ALIGNED = f"{ROOT}/附件2-数据集特征文件/aligned_50.pkl"
ATT3_ALIGNED_DIR = f"{ROOT}/附件3-模态缺失特征样本/对齐版本"
ATT4_ALIGNED_DIR = f"{ROOT}/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
NP_MAJOR = int(np.__version__.split(".")[0])


def _lengths_from_mask(text_bert):
    return text_bert[:, 1, :].sum(axis=1).astype(np.int64)


def _check_split(name, s):
    """返回该划分的形状与一致性信息，用于清单。"""
    n = len(s["regression_labels"])
    info = {k: list(np.asarray(v).shape) for k, v in s.items() if k not in ("raw_text", "id")}
    info["n"] = n
    assert all(len(s[k]) == n for k in s), f"{name}: 字段长度不一致"
    return info


def _dump(obj, manifest, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, stem + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    manifest["numpy_used_for_writing"] = np.__version__
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"写出 {pkl_path}  ({os.path.getsize(pkl_path)/2**20:.1f} MB)")


def cmd_att2(out_dir):
    with open(ATT2_ALIGNED, "rb") as f:
        data = pickle.load(f)
    manifest = {"source": ATT2_ALIGNED, "version": "aligned_50", "splits": {}}
    for split, s in data.items():
        tb = np.asarray(s["text_bert"]).astype(np.float32)
        lengths = _lengths_from_mask(tb)
        y = np.asarray(s["regression_labels"]).astype(np.float32)
        c = np.asarray(s["classification_labels"]).astype(np.float32)
        s["text_bert"] = tb
        s["text"] = np.asarray(s["text"]).astype(np.float32)
        s["audio"] = np.asarray(s["audio"]).astype(np.float32)
        s["vision"] = np.asarray(s["vision"]).astype(np.float32)
        s["audio_lengths"] = lengths.copy()
        s["vision_lengths"] = lengths.copy()
        s["regression_labels"] = y
        s["classification_labels"] = c
        # 校验三分类标签语义：期望 0=负(y<0) 1=中(y==0) 2=正(y>0)
        expect = np.where(y < 0, 0, np.where(y > 0, 2, 1))
        info = _check_split(split, s)
        info["classification_matches_sign(0neg_1neu_2pos)"] = float((expect == c).mean())
        info["label_counts"] = {str(int(k)): int(v) for k, v in zip(*np.unique(c, return_counts=True))}
        info["length_min_max"] = [int(lengths.min()), int(lengths.max())]
        manifest["splits"][split] = info
    _dump(data, manifest, out_dir, "mosei_att2_aligned50")


def _stack_samples(samples, ids):
    """samples: list of dict(text_bert(3,50), audio(50,74), vision(50,35), [text(50,768)], [raw_text])"""
    n = len(samples)
    tb = np.stack([np.asarray(x["text_bert"]).reshape(3, 50) for x in samples]).astype(np.float32)
    audio = np.stack([np.asarray(x["audio"]).reshape(50, 74) for x in samples]).astype(np.float32)
    vision = np.stack([np.asarray(x["vision"]).reshape(50, 35) for x in samples]).astype(np.float32)
    has_text = all("text" in x for x in samples)
    text = (np.stack([np.asarray(x["text"]).reshape(50, 768) for x in samples]).astype(np.float32)
            if has_text else np.zeros((n, 50, 768), np.float32))
    raw = np.array([str(np.asarray(x.get("raw_text", "")).ravel()[0]) if np.asarray(x.get("raw_text", "")).size else ""
                    for x in samples], dtype=object)
    lengths = _lengths_from_mask(tb)
    split = {
        "raw_text": raw, "id": np.array(ids, dtype=object),
        "text": text, "text_bert": tb, "audio": audio, "vision": vision,
        "audio_lengths": lengths.copy(), "vision_lengths": lengths.copy(),
        "regression_labels": np.zeros(n, np.float32),       # 占位，无标签
        "classification_labels": np.zeros(n, np.float32),   # 占位，无标签
    }
    return split, has_text


def cmd_att3(out_dir):
    files = sorted(glob.glob(ATT3_ALIGNED_DIR + "/*.pkl"))
    samples, ids = [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        t = d.get("test", d)
        samples.append({k: np.asarray(v)[0] for k, v in t.items()})   # 每个文件是 batch=1
        ids.append(os.path.splitext(os.path.basename(f))[0])
    split, has_text = _stack_samples(samples, ids)
    body_unk = [(split["text_bert"][i, 0, 1:L-1] == 100).mean() for i, L in enumerate(split["audio_lengths"])]
    manifest = {
        "source": ATT3_ALIGNED_DIR, "version": "aligned_50", "n": len(ids), "ids": ids,
        "has_continuous_text": has_text,
        "note": "text 字段为全零占位（附件3无连续文本特征），下游必须 use_bert=True；标签为占位；缺失已在输入中，推理时缺失注入率必须设为 0",
        "mean_unk_rate_in_body": float(np.mean(body_unk)),
        "shapes": _check_split("att3", split),
    }
    _dump({"train": split, "valid": split, "test": split}, manifest, out_dir, "mosei_att3_aligned50_infer")


def cmd_att4_export(out_dir):
    if NP_MAJOR < 2:
        sys.exit("att4-export 需要 numpy>=2（附件4 pkl 由 numpy 2 保存）。请用 Python 3.13 环境运行。")
    files = sorted(glob.glob(ATT4_ALIGNED_DIR + "/*.pkl"))
    arrs = {k: [] for k in ("text_bert", "audio", "vision", "text")}
    ids, raws, src_ids = [], [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for k in arrs:
            arrs[k].append(np.asarray(d[k], dtype=np.float32))
        ids.append(os.path.splitext(os.path.basename(f))[0])
        raws.append(str(np.asarray(d["raw_text"]).ravel()[0]))
        src_ids.append(str(np.asarray(d["id"]).ravel()[0]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "_att4_aligned50_intermediate.npz")
    np.savez(path, ids=np.array(ids), raw_text=np.array(raws), source_id=np.array(src_ids),
             **{k: np.stack(v) for k, v in arrs.items()})
    print(f"写出中间文件 {path}；接着用 numpy 1.x 环境运行 att4-pack")


def cmd_att4_pack(out_dir):
    path = os.path.join(out_dir, "_att4_aligned50_intermediate.npz")
    z = np.load(path, allow_pickle=False)
    samples = [{"text_bert": z["text_bert"][i], "audio": z["audio"][i], "vision": z["vision"][i],
                "text": z["text"][i], "raw_text": np.array([z["raw_text"][i]])} for i in range(len(z["ids"]))]
    split, has_text = _stack_samples(samples, [str(x) for x in z["ids"]])
    manifest = {
        "source": ATT4_ALIGNED_DIR, "version": "aligned_50", "n": int(len(z["ids"])),
        "ids": [str(x) for x in z["ids"]], "source_ids": [str(x) for x in z["source_id"]],
        "has_continuous_text": has_text,
        "note": "三模态完整、无标签（占位 0）；视频见附件4 对齐版本/videos/{id}.mp4",
        "shapes": _check_split("att4", split),
    }
    _dump({"train": split, "valid": split, "test": split}, manifest, out_dir, "mosei_att4_aligned50_infer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["att2", "att3", "att4-export", "att4-pack"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    a = ap.parse_args()
    {"att2": cmd_att2, "att3": cmd_att3, "att4-export": cmd_att4_export, "att4-pack": cmd_att4_pack}[a.cmd](
        os.path.normpath(a.out))

# 未对齐版说明：附件2 unaligned_50.pkl 自带 audio_lengths / vision_lengths，可被各基线直接读取，无需转换；
# 附件3 未对齐版只有 raw_text，需要在装有 transformers 的运行环境中用 bert-base-uncased 分词器
# （max_length=50，padding='max_length'）生成 text_bert，并先在附件2上核对重新分词结果与原 text_bert 是否一致。
