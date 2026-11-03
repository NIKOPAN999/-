"""Audit and adapt aligned MMSA-style datasets without silently realigning them."""
from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def video_id(sample_id):
    if "$_$" in sample_id:
        return sample_id.split("$_$", 1)[0]
    match = re.fullmatch(r"(.+)\[\d+\]", sample_id)
    return match.group(1) if match else None


def canonical_id(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = list(value)
        if len(parts) == 2:
            return str(parts[0]) + "$_$" + str(parts[1])
        raise ValueError(f"Unrecognized sample ID: {value!r}")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_dataset(path):
    # These are the team's original contest data / author-released feature files.
    with open(path, "rb") as f:
        source = pickle.load(f)
    result, report = {}, {"data_sha256": sha256(path), "splits": {},
                          "test_policy": "metadata and IDs only; no test performance or label summaries"}
    sample_sets, video_sets = {}, {}
    dims = None
    for name in ("train", "valid", "test"):
        row = source[name]
        text = np.asarray(row["text_bert"])
        audio, vision = np.asarray(row["audio"]), np.asarray(row["vision"])
        if text.ndim != 3 or text.shape[1] != 3:
            raise ValueError(f"{name}: expected text_bert (N,3,L), got {text.shape}")
        n, _, length = text.shape
        if n == 0 or length < 3:
            raise ValueError(f"{name}: empty split or sequence too short")
        if audio.ndim != 3 or vision.ndim != 3 or audio.shape[:2] != (n,length) or vision.shape[:2] != (n,length):
            raise ValueError(f"{name}: requires genuinely aligned text/audio/vision; no automatic resampling")
        shape = (length, audio.shape[2], vision.shape[2])
        if dims is not None and shape != dims:
            raise ValueError("Feature dimensions or sequence lengths differ across splits")
        dims = shape
        ids = [canonical_id(i) for i in row["id"]]
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"{name}: ID count mismatch or duplicate IDs")
        sample_sets[name] = set(ids)
        videos = [video_id(i) for i in ids]
        video_sets[name] = set(videos) if all(v is not None for v in videos) else None
        info = {"n_samples": n, "text_shape": list(text.shape),
                "audio_shape": list(audio.shape), "vision_shape": list(vision.shape),
                "n_video_groups": len(video_sets[name]) if video_sets[name] is not None else None,
                "grouping_verified": video_sets[name] is not None}
        if name == "test":
            report["splits"][name] = info
            continue
        if not all(np.isfinite(x).all() for x in (text, audio, vision)):
            raise ValueError(f"{name}: nonfinite features; inspect and document preprocessing before training")
        token = np.rint(text[:, 0]).astype(np.int64)
        if not np.allclose(text[:, 0], token) or token.min() < 0:
            raise ValueError(f"{name}: text_bert token IDs are not nonnegative integers")
        attn = text[:, 1].astype(bool)
        if not np.isin(text[:,1], (0,1)).all() or np.any(np.diff(attn.astype(int), axis=1) > 0):
            raise ValueError(f"{name}: attention mask must be contiguous prefix ones then padding zeros")
        lengths = attn.sum(1)
        if np.any(lengths < 2):
            raise ValueError(f"{name}: fewer than two special-token positions")
        if np.any(token[:,0] != 101) or np.any(token[np.arange(n), lengths-1] != 102):
            raise ValueError(f"{name}: expected BERT [CLS]/[SEP] at sequence boundaries")
        y = np.asarray(row["regression_labels"], dtype=np.float32).reshape(-1)
        if y.shape != (n,) or not np.isfinite(y).all() or np.any(np.abs(y) > 3.00001):
            raise ValueError(f"{name}: invalid sentiment labels")
        classes = (np.sign(y) + 1).astype(np.int64)
        if "classification_labels" in row:
            provided = np.asarray(row["classification_labels"]).reshape(-1)
            if not np.array_equal(provided, classes):
                raise ValueError(f"{name}: classification labels disagree with sign(regression)+1")
        body = attn & (token != 101) & (token != 102)
        for mode, arr in (("audio",audio), ("vision",vision)):
            valid = body & (np.abs(arr).sum(-1) > 0)
            if name == "train" and not valid.any():
                raise ValueError(f"No observed {mode} features in train")
            info[mode + "_observed_fraction"] = float(valid.sum() / max(1, body.sum()))
            info[mode + "_nonzero_outside_body"] = int(((np.abs(arr).sum(-1)>0) & ~body).sum())
        info["content_length_min_max"] = [int(body.sum(1).min()), int(body.sum(1).max())]
        info["class_counts"] = {str(i): int((classes==i).sum()) for i in range(3)}
        info["zero_audio_but_nonzero_vision"] = int((body & (np.abs(audio).sum(-1)==0) & (np.abs(vision).sum(-1)>0)).sum())
        info["special_token_steps_are_in_rnn_length"] = True
        result[name] = {"text_bert": text, "audio": audio, "vision": vision,
                        "regression_labels": y, "classification_labels": classes, "id": ids}
        report["splits"][name] = info
    for a,b in (("train","valid"),("train","test"),("valid","test")):
        if sample_sets[a] & sample_sets[b]:
            raise ValueError(f"Sample overlap: {a}/{b}")
        if video_sets[a] is not None and video_sets[b] is not None and video_sets[a] & video_sets[b]:
            raise ValueError(f"Original-video overlap: {a}/{b}")
    report["schema"] = {"length": dims[0], "audio_dim": dims[1], "vision_dim": dims[2]}
    report["video_disjointness_verified"] = all(v is not None for v in video_sets.values())
    return result, report
