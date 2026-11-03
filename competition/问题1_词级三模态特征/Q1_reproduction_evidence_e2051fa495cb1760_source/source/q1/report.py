from __future__ import annotations

import csv
import html
import wave
from pathlib import Path

import numpy as np

from .core import write_csv, write_json


def compare(directory, row, results):
    directory = Path(directory)
    a, b = results["whisperx"], results["mfa"]
    records = []
    for wa, wb in zip(a["words"], b["words"], strict=True):
        both = wa["valid"] and wb["valid"]
        records.append(dict(index=wa["index"], word=wa["word"],
                            whisperx_start=wa["start"], whisperx_end=wa["end"],
                            whisperx_valid=wa["valid"], whisperx_reason=wa["reason"],
                            mfa_start=wb["start"], mfa_end=wb["end"], mfa_valid=wb["valid"],
                            mfa_reason=wb["reason"],
                            start_disagreement_ms=abs(wa["start"] - wb["start"]) * 1000 if both else None,
                            end_disagreement_ms=abs(wa["end"] - wb["end"]) * 1000 if both else None))
    write_csv(directory / "comparison.csv", records)
    diffs = [r[key] for r in records for key in ["start_disagreement_ms", "end_disagreement_ms"] if r[key] is not None]
    summary = dict(sample_id=row["sample_id"], words=len(records),
                   whisperx_coverage=a["valid_words"] / len(records), mfa_coverage=b["valid_words"] / len(records),
                   mean_boundary_disagreement_ms=float(np.mean(diffs)) if diffs else None,
                   p95_boundary_disagreement_ms=float(np.percentile(diffs, 95)) if diffs else None,
                   accuracy_measured=False, interpretation="Between-tool disagreement is NOT ground-truth error")
    write_json(directory / "comparison.json", summary)
    template = directory / "manual_reference.csv"
    if not template.exists():
        write_csv(template, [dict(index=r["index"], word=r["word"], start="", end="", note="") for r in records])
    alignment_html(directory, row, results)
    return summary


def alignment_html(directory, row, results):
    """Local playback and selectable time segments; no network or third-party JS."""
    directory = Path(directory)
    with wave.open(str(directory / "audio.wav")) as audio:
        sr = audio.getframerate()
        signal = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(float)
    duration = len(signal) / sr
    width = 1100
    bins = np.array_split(signal, min(width, len(signal)))
    scale = max(float(np.abs(signal).max()), 1)
    waveform = "".join(f'<line x1="{i}" x2="{i}" y1="{45 - np.max(np.abs(v))/scale*40:.2f}" '
                       f'y2="{45 + np.max(np.abs(v))/scale*40:.2f}" stroke="#64748b"/>'
                       for i, v in enumerate(bins))
    tracks, tables = [], []
    for index, (name, result) in enumerate(results.items()):
        y = 105 + index * 50
        tracks.append(f'<text x="0" y="{y-4}">{html.escape(name)}</text>')
        trs = []
        for w in result["words"]:
            label = html.escape(w["word"])
            if w["valid"]:
                x, length = w["start"] / duration * width, (w["end"] - w["start"]) / duration * width
                tracks.append(f'<rect x="{x:.2f}" y="{y}" width="{max(length, .1):.2f}" height="22" '
                              f'fill="{["#2563eb", "#16a34a"][index % 2]}" stroke="white" '
                              f'onclick="playAt({w["start"]},{w["end"]})"><title>{label} '
                              f'{w["start"]:.3f}–{w["end"]:.3f}s</title></rect>')
                times = f'<button onclick="playAt({w["start"]},{w["end"]})">{w["start"]:.3f}–{w["end"]:.3f}</button>'
            else:
                times = html.escape(w["reason"])
            trs.append(f'<tr><td>{w["index"]}</td><td>{label}</td><td>{times}</td></tr>')
        tables.append(f'<section><h2>{html.escape(name)}</h2><table><tr><th>Index</th><th>Word</th>'
                      f'<th>WAV seconds / issue</th></tr>{"".join(trs)}</table></section>')
    source = '<!doctype html><meta charset="utf-8"><title>Alignment review</title>'
    source += '<style>body{font:16px system-ui;margin:32px;color:#172033}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}main{display:flex;gap:48px}svg{width:100%;max-width:1200px}button,rect{cursor:pointer}</style>'
    source += f'<h1>{html.escape(row["sample_id"])}</h1><p>{html.escape(row["raw_text"])}</p>'
    source += '<p>Click a word interval to listen. Disagreement is not accuracy; annotate manual_reference.csv independently.</p>'
    source += '<audio id="audio" controls src="audio.wav"></audio>'
    source += f'<svg viewBox="0 0 {width} {120 + 50*len(results)}">{waveform}{"".join(tracks)}</svg>'
    source += f'<main>{"".join(tables)}</main>'
    source += '<script>const a=document.getElementById("audio");let stop=Infinity;function playAt(s,e){stop=e;a.currentTime=s;a.play()}a.addEventListener("timeupdate",()=>{if(a.currentTime>=stop){a.pause();stop=Infinity}});</script>'
    (directory / "alignment_review.html").write_text(source, encoding="utf-8")


def score_reference(directory, reference, results):
    with Path(reference).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    truth, seen = [], set()
    for row in records:
        if not row["start"] or not row["end"]:
            continue
        i, start, end = int(row["index"]), float(row["start"]), float(row["end"])
        if i in seen or not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError("Duplicate/invalid manual reference")
        seen.add(i)
        truth.append((i, row["word"], start, end))
    if not truth:
        raise ValueError("No manually annotated boundaries; accuracy cannot be evaluated")
    scores = {}
    for backend, result in results.items():
        errors = []
        for i, word, start, end in truth:
            if i < 0 or i >= len(result["words"]) or result["words"][i]["word"] != word:
                raise ValueError("Manual reference word/index mismatch")
            w = result["words"][i]
            if w["valid"]:
                errors.extend([abs(start - w["start"]) * 1000, abs(end - w["end"]) * 1000])
        scores[backend] = dict(annotated_words=len(truth), evaluated_words=len(errors) // 2,
                               boundary_mae_ms=float(np.mean(errors)) if errors else None,
                               boundary_p95_ms=float(np.percentile(errors, 95)) if errors else None,
                               within_100ms=float(np.mean(np.asarray(errors) <= 100)) if errors else None)
    write_json(Path(directory) / "manual_scores.json", scores)
    return scores
