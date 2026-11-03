"""Q1-7：OpenFace 帧质量与画面运动能量。

1. 按置信度分段统计相邻帧 AU 强度差分（时序抖动），检验低置信度帧是否确实不可信；
2. 区分“未检测到人脸（success=0）”与“有人脸但置信度低”，逐帧导出供重新聚合；
3. 对每个视频按原解码顺序解码为 160×90 灰度图，计算相邻帧平均绝对差作为画面运动能量，
   作为不依赖人脸的视觉兜底特征（无人脸画面同样可用）。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q1.core import read_json  # noqa: E402

AU = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
      "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
BINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]


def motion_energy(ffmpeg, video, frames, w=160, h=90):
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    buf = subprocess.run(cmd, check=True, capture_output=True).stdout
    x = np.frombuffer(buf, np.uint8).reshape(-1, h, w).astype(np.float32) / 255.0
    if len(x) != frames:
        raise ValueError(f"decoded {len(x)} frames, expected {frames}: {video}")
    e = np.zeros(len(x), np.float32)
    e[1:] = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2))
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "work/runs/e2051fa495cb1760"))
    p.add_argument("--out", required=True)
    p.add_argument("--ffmpeg", default="/root/autodl-tmp/envs/q1-mfa/bin/ffmpeg")
    p.add_argument("--data-root", default=str(ROOT / "data/raw100"))
    args = p.parse_args()
    config = read_json(ROOT / "configs/local.json")
    manifest = read_json(ROOT / "work/manifest.json")
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    jit = {b: [] for b in BINS}
    pose_jit = {b: [] for b in BINS}
    counts = dict(frames=0, no_face=0, low_conf=0, valid=0)
    per_sample = []
    for row in manifest["samples"]:
        d = Path(args.run) / "samples" / row["key"]
        with open(d / "openface/features.csv", newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            items = [{k.strip(): v.strip() for k, v in r.items()} for r in rd]
        succ = np.array([float(r["success"]) == 1 for r in items])
        conf = np.array([float(r["confidence"]) for r in items], dtype=np.float32)
        aus = np.array([[float(r[c]) for c in AU] for r in items], dtype=np.float32)
        pose = np.array([[float(r[c]) for c in ("pose_Rx", "pose_Ry", "pose_Rz")] for r in items], dtype=np.float32)
        both = succ[1:] & succ[:-1]
        c = np.minimum(conf[1:], conf[:-1])
        dau = np.abs(np.diff(aus, axis=0)).mean(axis=1)
        dpo = np.abs(np.diff(pose, axis=0)).mean(axis=1)
        for b in BINS:
            sel = both & (c >= b[0]) & (c < b[1])
            jit[b] += dau[sel].tolist()
            pose_jit[b] += dpo[sel].tolist()
        media = read_json(d / "media.json")
        video = Path(args.data_root) / row["relative_path"]
        me = motion_energy(args.ffmpeg, video, media["video_frames"])
        np.savez_compressed(out / "frames" / f"{row['key']}.npz", success=succ, confidence=conf, motion=me)
        valid = succ & (conf >= config["vision_confidence_min"])
        counts["frames"] += len(succ); counts["no_face"] += int((~succ).sum())
        counts["low_conf"] += int((succ & ~valid).sum()); counts["valid"] += int(valid.sum())
        per_sample.append(dict(sample_id=row["sample_id"], frames=len(succ), no_face=int((~succ).sum()),
                               low_conf=int((succ & ~valid).sum()), valid=int(valid.sum()),
                               motion_mean=round(float(me[1:].mean()), 5)))
        print(row["sample_id"], per_sample[-1], flush=True)
    table = []
    for b in BINS:
        v, q = np.array(jit[b]), np.array(pose_jit[b])
        table.append(dict(conf_bin=f"[{b[0]:.2f},{min(b[1], 1.0):.2f}{']' if b[1] > 1 else ')'}", pairs=len(v),
                          au_jitter_mean=round(float(v.mean()), 4) if len(v) else None,
                          au_jitter_median=round(float(np.median(v)), 4) if len(v) else None,
                          pose_jitter_mean=round(float(q.mean()), 5) if len(q) else None))
    (out / "tables").mkdir(exist_ok=True)
    for name, data in [("Q1-7_au_jitter_by_confidence.csv", table), ("Q1-7_face_presence_per_sample.csv", per_sample)]:
        with open(out / "tables" / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(json.dumps(dict(counts=counts, jitter=table), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
