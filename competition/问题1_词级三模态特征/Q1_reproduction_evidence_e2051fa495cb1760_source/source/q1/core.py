from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def words_from_text(text):
    """Keep original character offsets; normalization never overwrites the supplied text."""
    return [dict(index=i, word=m.group(), normalized=m.group().lower().replace("’", "'"),
                 char_start=m.start(), char_end=m.end())
            for i, m in enumerate(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", text))]


def letters(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def map_alignment(reference, emitted, duration):
    """Monotonic token DP with exact 1-to-up-to-4 contraction matching.

    Missing tokens remain missing. A reference word never borrows a substring
    of an emitted token, and one emitted word never supplies two word times.
    """
    ref = [letters(w["normalized"]) for w in reference]
    out = [letters(w.get("word", "")) for w in emitted]
    nr, no = len(ref), len(out)
    costs = np.zeros((nr + 1, no + 1), dtype=int)
    costs[:, no] = np.arange(nr, -1, -1)
    costs[nr, :] = np.arange(no, -1, -1)
    action = {}
    for i in range(nr - 1, -1, -1):
        for j in range(no - 1, -1, -1):
            best, step = costs[i + 1, j] + 1, (1, 0)
            if costs[i, j + 1] + 1 < best:
                best, step = costs[i, j + 1] + 1, (0, 1)
            for k in range(1, min(4, no - j) + 1):
                if all(out[j:j+k]) and "".join(out[j:j+k]) == ref[i] and costs[i + 1, j + k] <= best:
                    best, step = costs[i + 1, j + k], (1, k)
            costs[i, j], action[i, j] = best, step
    mapping, i, j = {}, 0, 0
    while i < nr and j < no:
        di, dj = action[i, j]
        if di and dj:
            mapping[i] = list(range(j, j + dj))
        i, j = i + di, j + dj
    rows = []
    for word in reference:
        row = {**word, "start": None, "end": None, "confidence": None,
               "valid": False, "reason": "unmatched_text", "emitted_indices": []}
        selected = mapping.get(word["index"])
        if selected:
            row["emitted_indices"] = selected
            entries = [emitted[i] for i in selected]
            times = [(e.get("start"), e.get("end")) for e in entries]
            if all(isinstance(t, (int, float)) and math.isfinite(t) for pair in times for t in pair):
                start, end = min(t[0] for t in times), max(t[1] for t in times)
                ordered = all(a[1] <= b[0] + 1e-4 for a, b in zip(times, times[1:]))
                if 0 <= start < end <= duration + 1e-3 and all(a < b for a, b in times) and ordered:
                    scores = [e.get("score") for e in entries]
                    scores = [s for s in scores if isinstance(s, (int, float)) and math.isfinite(s)]
                    row.update(start=float(start), end=float(min(end, duration)), valid=True,
                               reason="ok", confidence=float(np.mean(scores)) if scores else None)
                else:
                    row["reason"] = "invalid_time_bounds"
            else:
                row["reason"] = "missing_time"
        rows.append(row)
    previous_end = -1.0
    for row in rows:
        if row["valid"]:
            if row["start"] < previous_end - 1e-4:
                row.update(valid=False, reason="overlapping_or_nonmonotonic")
            else:
                previous_end = row["end"]
    return rows


def parse_textgrid(path):
    """Read MFA long TextGrid word tiers, preserving OOV as mismatches."""
    source = Path(path).read_text(encoding="utf-8-sig")
    tiers = re.split(r"\bitem\s*\[\d+\]\s*:", source)[1:]
    number = r"([-+\d.eE]+)"
    pattern = re.compile(r"intervals\s*\[\d+\]\s*:\s*xmin\s*=\s*" + number
                         + r"\s*xmax\s*=\s*" + number
                         + r'\s*text\s*=\s*"((?:[^"]|"")*)"')
    found = []
    for tier in tiers:
        name = re.search(r'name\s*=\s*"([^"]*)"', tier)
        if name and (name.group(1) == "words" or name.group(1).endswith(" - words")):
            found.append([dict(word=t.replace('""', '"'), start=float(a), end=float(b))
                          for a, b, t in pattern.findall(tier) if t.strip()])
    if len(found) != 1:
        raise ValueError(f"Expected one MFA word tier, got {len(found)}: {path}")
    return found[0]


def overlap_pool(targets, frame_intervals, features, valid, quality=None):
    """Duration-weighted mean. Empty support is zero + mask=False, never NaN."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    intervals = np.asarray(frame_intervals, dtype=np.float64).reshape(-1, 2)
    features = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if features.ndim != 2 or len(intervals) != len(features) or valid.shape != (len(features),):
        raise ValueError("Frame intervals/features/valid mask have inconsistent shapes")
    quality = np.ones(len(features)) if quality is None else np.asarray(quality, dtype=float)
    if quality.shape != valid.shape or np.any(~np.isfinite(quality)) or np.any(quality < 0):
        raise ValueError("Invalid quality weights")
    good = valid & np.isfinite(features).all(axis=1) & np.isfinite(intervals).all(axis=1)
    good &= intervals[:, 1] > intervals[:, 0]
    safe = np.where(good[:, None], features, 0)
    result = np.zeros((len(targets), features.shape[1]), dtype=np.float32)
    masks = np.zeros(len(targets), dtype=bool)
    counts = np.zeros(len(targets), dtype=np.int32)
    for i, (start, end) in enumerate(targets):
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            continue
        weights = np.maximum(0, np.minimum(end, intervals[:, 1]) - np.maximum(start, intervals[:, 0]))
        weights = np.where(good, weights, 0) * quality
        counts[i] = np.count_nonzero(weights)
        if weights.sum() > 0:
            result[i] = weights @ safe / weights.sum()
            masks[i] = True
    return result, masks, counts


def make_manifest(data_root, expected=100):
    from openpyxl import load_workbook
    root = Path(data_root).resolve()
    labels = list(root.rglob("label-100.xlsx"))
    if len(labels) != 1:
        raise ValueError("data_root must contain exactly one label-100.xlsx")
    workbook = load_workbook(labels[0], read_only=True, data_only=True)
    values = iter(workbook.active.values)
    header = list(next(values))
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    if not required.issubset(header):
        raise ValueError(f"Missing label columns: {required - set(header)}")
    rows, seen = [], set()
    for values_row in values:
        if all(v is None for v in values_row):
            continue
        item = dict(zip(header, values_row))
        video_id = str(item["video_id"])
        clip = item["clip_id"]
        clip_id = str(int(clip)) if isinstance(clip, (float, int)) and float(clip).is_integer() else str(clip)
        source = (labels[0].parent / video_id / f"{clip_id}.mp4").resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing/unsafe video path for {video_id}/{clip_id}")
        sample_id = f"{video_id}$_${clip_id}"
        if sample_id in seen:
            raise ValueError(f"Duplicate id: {sample_id}")
        seen.add(sample_id)
        text = item["text"]
        if not isinstance(text, str) or not words_from_text(text):
            raise ValueError(f"Empty/invalid transcript: {sample_id}")
        rows.append(dict(sample_id=sample_id, key=fingerprint(sample_id)[:16], video_id=video_id,
                         clip_id=clip_id, relative_path=source.relative_to(root).as_posix(),
                         raw_text=text, label=str(item["label"]), annotation=str(item["annotation"]),
                         sha256=sha256(source), whitespace_words=len(text.split()),
                         words=len(words_from_text(text))))
    workbook.close()
    videos = {v.resolve() for v in root.rglob("*.mp4")}
    referenced = {(root / r["relative_path"]).resolve() for r in rows}
    if len(rows) != expected or videos != referenced:
        raise ValueError(f"Coverage mismatch: labels={len(rows)}, videos={len(videos)}, expected={expected}")
    return dict(schema=1, expected_count=expected, label_sha256=sha256(labels[0]), samples=rows)
