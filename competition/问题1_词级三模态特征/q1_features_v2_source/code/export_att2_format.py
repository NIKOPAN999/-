"""把问题一 v2 特征导出为与附件2 对齐版（aligned_50.pkl）相同的组织方式。

附件2 对齐版按 BERT 词元位置组织（经 18 条重叠样本核验）：位置 0 为 [CLS]，1…P-2 为正文词元，
P-1 为 [SEP]，其后补零到 50；语音、视觉在正文词元位置上取值。本脚本按同一规则导出：
- 每个正文词元取其所属单词的词级特征（同一单词的多个子词重复同一向量）；
- 标点词元、[CLS]、[SEP] 与补齐位置的特征为 0，并在 *_mask 中标为 False；
- 词元数超过 50 时，默认按附件2 的方式截断（保留前 48 个正文词元）；--split 时改为滑动窗口
  （每窗 48 个正文词元、步长 24），各窗口作为独立条目，编号追加 #k，避免丢失句尾信息。
维度与附件2 不同（768/25/22 对 768/74/35），不能直接替换附件2 输入已训练模型。
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

T = 50


def windows(n_body, split, size=T - 2, step=(T - 2) // 2):
    if n_body <= size or not split:
        return [(0, min(n_body, size))]
    out, a = [], 0
    while True:
        out.append((a, min(a + size, n_body)))
        if a + size >= n_body:
            return out
        a += step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", required=True, help="e2051 研究归档 zip（提供 WordPiece 映射）")
    p.add_argument("--v2", required=True, help="aggregate_v2.py 的输出目录")
    p.add_argument("--out", required=True)
    p.add_argument("--split", action="store_true")
    args = p.parse_args()
    z = zipfile.ZipFile(args.research)
    metas = {}
    for n in z.namelist():
        if n.endswith("metadata.json"):
            m = json.loads(z.read(n))
            metas[m["sample"]["sample_id"]] = m
    keys = ["id", "raw_text", "text", "audio", "vision", "text_bert", "text_mask", "audio_mask", "vision_mask",
            "alignment_state", "token_time", "regression_labels", "annotations", "source_word"]
    D = {k: [] for k in keys}
    for sid in sorted(metas):
        m = metas[sid]
        a = np.load(Path(args.v2) / "samples" / sid / "aligned_v2.npz")
        ids, att, wid = m["token_map"]["input_ids"], m["token_map"]["attention_mask"], m["token_map"]["word_ids"]
        body = list(range(1, len(ids) - 1))                       # 正文词元位置（去掉 [CLS]、[SEP]）
        for k, (s, e) in enumerate(windows(len(body), args.split)):
            pos = body[s:e]
            tb = np.zeros((3, T), np.int64)
            tb[0, 0], tb[0, len(pos) + 1] = ids[0], ids[-1]
            tb[0, 1:len(pos) + 1] = [ids[q] for q in pos]
            tb[1, :len(pos) + 2] = 1
            feats = {name: np.zeros((T, a[name].shape[1]), np.float32) for name in ("text", "audio", "vision")}
            masks = {name: np.zeros(T, bool) for name in ("text", "audio", "vision")}
            state = np.full(T, -1, np.int8)
            ttime = np.full((T, 2), -1.0, np.float32)
            src = np.full(T, -1, np.int32)
            for j, q in enumerate(pos, start=1):
                w = wid[q]
                if w is None:                                    # 标点词元
                    continue
                st = a["alignment_state"][w]
                feats["text"][j], masks["text"][j] = a["text"][w], True
                feats["audio"][j], masks["audio"][j] = a["audio"][w], bool(a["audio_energy_mask"][w]) and st > 0
                feats["vision"][j], masks["vision"][j] = a["vision"][w], bool(a["vision_mask"][w]) and st > 0
                state[j], ttime[j], src[j] = st, a["word_intervals"][w], w
            D["id"].append(sid if k == 0 and not args.split else f"{sid}#{k}")
            D["raw_text"].append(m["sample"]["raw_text"])
            for name in ("text", "audio", "vision"):
                D[name].append(feats[name]); D[f"{name}_mask"].append(masks[name])
            D["text_bert"].append(tb); D["alignment_state"].append(state); D["token_time"].append(ttime); D["source_word"].append(src)
            D["regression_labels"].append(np.float32(m["sample"]["label"])); D["annotations"].append(m["sample"]["annotation"])
    out = {"q1": {k: (np.stack(v) if k not in ("id", "raw_text", "annotations") else np.array(v)) for k, v in D.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    q = out["q1"]
    print(json.dumps(dict(entries=len(q["id"]), text=list(q["text"].shape), audio=list(q["audio"].shape), vision=list(q["vision"].shape),
                          text_bert=list(q["text_bert"].shape), split=args.split), ensure_ascii=False))


if __name__ == "__main__":
    main()
