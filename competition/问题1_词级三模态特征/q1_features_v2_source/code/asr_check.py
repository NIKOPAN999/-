"""Q1-5：无约束语音识别检验“转写与语音一致”假设，并提供第三方词时间参照。

对 e2051 批次的 16 kHz audio.wav 用 faster-whisper（large-v3）做不受转写约束的识别，
逐条与官方转写做词级编辑距离对齐，统计替换/删除/插入，并定位转写之外的语音（插入）与
转写中未被识别的词（删除）所在的时段。识别结果只用于核查，不替换官方转写。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q1.core import read_json, words_from_text, write_json  # noqa: E402


def norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'")).strip("'")


def edit_align(ref, hyp):
    """词级 Levenshtein 对齐，返回 (op, i_ref, j_hyp) 序列；op ∈ {H, S, D, I}。"""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("H" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", i - 1, None)); i -= 1
        else:
            ops.append(("I", None, j - 1)); j -= 1
    return ops[::-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "work/runs/e2051fa495cb1760"))
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--model_dir", default="/root/autodl-tmp/model_download/faster-whisper")
    args = p.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cuda", compute_type="float16", download_root=args.model_dir)
    manifest = read_json(ROOT / "work/manifest.json")
    out = Path(args.out)
    per_sample, per_word = [], []
    for row in manifest["samples"]:
        wav = Path(args.run) / "samples" / row["key"] / "audio.wav"
        segments, info = model.transcribe(str(wav), language="en", beam_size=5, word_timestamps=True,
                                          condition_on_previous_text=False, vad_filter=False, temperature=0.0)
        hyp_words = [dict(word=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3), prob=round(w.probability, 3))
                     for s in segments for w in (s.words or [])]
        ref_words = words_from_text(row["raw_text"])
        ref = [norm(w["word"]) for w in ref_words]
        hyp = [norm(w["word"]) for w in hyp_words]
        keep = [k for k, h in enumerate(hyp) if h]
        hyp_words = [hyp_words[k] for k in keep]
        hyp = [hyp[k] for k in keep]
        ops = edit_align(ref, hyp)
        cnt = {k: sum(o[0] == k for o in ops) for k in "HSDI"}
        ins = [hyp_words[j] for o, i, j in ops if o == "I"]
        dels = [ref_words[i]["word"] for o, i, j in ops if o == "D"]
        per_sample.append(dict(sample_id=row["sample_id"], ref_words=len(ref), hyp_words=len(hyp),
                               hits=cnt["H"], subs=cnt["S"], dels=cnt["D"], ins=cnt["I"],
                               wer=round((cnt["S"] + cnt["D"] + cnt["I"]) / max(1, len(ref)), 4),
                               inserted=" ".join(w["word"] for w in ins),
                               inserted_spans=";".join(f"{w['start']:.2f}-{w['end']:.2f}" for w in ins),
                               deleted=" ".join(dels), hyp_text=" ".join(w["word"] for w in hyp_words)))
        for o, i, j in ops:
            if o in "HS":
                per_word.append(dict(sample_id=row["sample_id"], word_pos=i, word=ref_words[i]["word"], op=o,
                                     asr_word=hyp_words[j]["word"], asr_start=hyp_words[j]["start"],
                                     asr_end=hyp_words[j]["end"], asr_prob=hyp_words[j]["prob"]))
        write_json(out / "asr" / f"{row['key']}.json", dict(sample_id=row["sample_id"], words=hyp_words, ops=ops))
        print(row["sample_id"], per_sample[-1]["wer"], flush=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, data in [("Q1-5_asr_wer_per_sample.csv", per_sample), ("Q1-5_asr_word_times.csv", per_word)]:
        with open(out / "tables" / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    tot = {k: sum(r[k] for r in per_sample) for k in ["ref_words", "subs", "dels", "ins"]}
    print(json.dumps(dict(model=args.model, corpus_wer=round((tot["subs"] + tot["dels"] + tot["ins"]) / tot["ref_words"], 4), **tot)))


if __name__ == "__main__":
    main()
