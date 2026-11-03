"""Q1-6：比较两次独立运行的问题一输出。

区分两类核验：输入原件按哈希校验；输出特征按数值容差比较（GPU 浮点推理跨硬件不保证逐位一致）。
逐条报告：词边界是否逐一相同、各数组的最大绝对误差、四个输出文件的哈希是否一致。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q1.core import read_json, sha256  # noqa: E402

FILES = ["aligned.npz", "native.npz", "metadata.json", "alignment_whisperx.json"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="original run directory")
    p.add_argument("--b", required=True, help="reproduction run directory")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    a_root, b_root = Path(args.a) / "samples", Path(args.b) / "samples"
    rows = []
    for d in sorted(a_root.iterdir()):
        e = b_root / d.name
        if not (d / "complete.json").exists() or not (e / "complete.json").exists():
            rows.append(dict(key=d.name, status="missing"))
            continue
        r = dict(key=d.name, sample_id=read_json(d / "metadata.json")["sample"]["sample_id"], status="ok")
        for f in FILES:
            r[f"hash_equal_{f}"] = sha256(d / f) == sha256(e / f)
        wa = read_json(d / "alignment_whisperx.json")["words"]
        wb = read_json(e / "alignment_whisperx.json")["words"]
        r["words"] = len(wa)
        r["boundaries_identical"] = all(x["start"] == y["start"] and x["end"] == y["end"] and x["valid"] == y["valid"]
                                        for x, y in zip(wa, wb)) and len(wa) == len(wb)
        r["max_boundary_diff_s"] = max((max(abs(x["start"] - y["start"]), abs(x["end"] - y["end"]))
                                        for x, y in zip(wa, wb) if x["valid"] and y["valid"]), default=0.0)
        r["max_score_diff"] = max(abs((x["confidence"] or 0) - (y["confidence"] or 0)) for x, y in zip(wa, wb))
        for name in FILES[:2]:
            A, B = np.load(d / name), np.load(e / name)
            for k in A.files:
                x, y = A[k], B[k]
                if x.shape != y.shape:
                    r[f"{name}:{k}"] = "shape_mismatch"
                elif x.dtype == bool or np.issubdtype(x.dtype, np.integer):
                    r[f"{name}:{k}"] = int((x != y).sum())            # 不一致元素个数
                else:
                    r[f"{name}:{k}"] = float(np.max(np.abs(x - y))) if x.size else 0.0  # 最大绝对误差
        rows.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("key", "sample_id", "status"), k))
    with open(out / "Q1-6_reproduction_per_sample.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = dict(samples=len(rows), compared=len(ok),
                   boundaries_identical=sum(r["boundaries_identical"] for r in ok),
                   hash_equal={f: sum(r[f"hash_equal_{f}"] for r in ok) for f in FILES})
    for k in [c for c in fields if ":" in c]:
        vals = [r[k] for r in ok if isinstance(r.get(k), (int, float))]
        summary[k] = dict(max=max(vals) if vals else None, nonzero_samples=sum(v != 0 for v in vals))
    (out / "Q1-6_reproduction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
