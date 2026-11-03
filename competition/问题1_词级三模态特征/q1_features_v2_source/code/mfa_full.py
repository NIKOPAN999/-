"""Q1-4：用 MFA 对 100 条样本全量做强制对齐，与 WhisperX 结果逐词对照。

复用 e2051 批次已准备好的 16 kHz audio.wav 与同一份规范化转写；按原视频编号组织语料
（同一视频的片段视为同一说话人，便于 MFA 做说话人自适应）。输出每条样本的 MFA 词区间
（经 q1.core.map_alignment 映射回原文单词），以及两工具逐词对照表。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q1.core import map_alignment, parse_textgrid, read_json, words_from_text, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "work/runs/e2051fa495cb1760"))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=16)
    args = p.parse_args()
    run, out = Path(args.run), Path(args.out)
    config = read_json(ROOT / "configs/local.json")
    manifest = read_json(ROOT / "work/manifest.json")
    corpus = out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rows = {}
    for row in manifest["samples"]:
        d = run / "samples" / row["key"]
        spk = corpus / f"v_{row['video_id']}"
        spk.mkdir(exist_ok=True)
        wav = spk / f"{row['key']}.wav"
        if not wav.exists():
            os.symlink(d / "audio.wav", wav)
        words = words_from_text(row["raw_text"])
        (spk / f"{row['key']}.lab").write_text(" ".join(w["normalized"] for w in words), encoding="utf-8")
        rows[row["key"]] = (row, words, read_json(d / "media.json"))
    aligned = out / "textgrids"
    cmd = [config["mfa_bin"], "align", str(corpus), str(ROOT / config["mfa_dictionary"]), str(ROOT / config["mfa_acoustic"]),
           str(aligned), "--output_format", "long_textgrid", "--clean", "-j", str(args.jobs),
           "--temporary_directory", str(out / "mfa_temp")]
    print("COMMAND", json.dumps(cmd), flush=True)
    subprocess.run(cmd, check=True)

    table = []
    for key, (row, words, media) in rows.items():
        grid = aligned / f"v_{row['video_id']}" / f"{key}.TextGrid"
        result = dict(backend="mfa", sample_id=row["sample_id"], coordinate="seconds from prepared audio.wav start")
        if grid.exists():
            emitted = parse_textgrid(grid)
            mapped = map_alignment(words, emitted, media["audio_duration"])
            result.update(words=mapped, emitted=emitted)
        else:
            result.update(words=[], error="MFA produced no TextGrid (likely failed/OOV)")
            mapped = [dict(index=w["index"], word=w["word"], start=None, end=None, valid=False, reason="no_textgrid") for w in words]
        write_json(out / "alignments" / f"{key}.json", result)
        wx = read_json(run / "samples" / key / "alignment_whisperx.json")["words"]
        for a, b in zip(wx, mapped):
            table.append(dict(sample_id=row["sample_id"], key=key, word_pos=a["index"], word=a["word"], n_words=len(words),
                              wx_start=a["start"], wx_end=a["end"], wx_score=a["confidence"], wx_valid=a["valid"],
                              mfa_start=b["start"], mfa_end=b["end"], mfa_valid=b["valid"], mfa_reason=b["reason"]))
    (out / "tables").mkdir(exist_ok=True)
    with open(out / "tables/Q1-4_whisperx_vs_mfa_words.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    ok = sum(r["mfa_valid"] for r in table)
    print(json.dumps(dict(samples=len(rows), words=len(table), mfa_valid_words=ok), ensure_ascii=False))


if __name__ == "__main__":
    main()
