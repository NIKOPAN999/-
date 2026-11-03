from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .core import overlap_pool, write_json
from .media import decode_frames, execute


class Extractors:
    def __init__(self, config):
        self.config = config
        self.bert = None
        self.smile = None

    def text(self, words, raw_text):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self.bert is None:
            path = self.config["bert_model_path"]
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            model = AutoModel.from_pretrained(path, local_files_only=True).to(self.config["device"]).eval()
            self.bert = tokenizer, model
        tokenizer, model = self.bert
        # Keep punctuation in BERT's context while mapping spoken words by the
        # original character offsets. Special tokens have empty offsets.
        encoded = tokenizer(raw_text, return_offsets_mapping=True,
                            return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > model.config.max_position_embeddings:
            raise ValueError("Transcript exceeds BERT context; must implement documented chunking, not truncation")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        mapping = []
        for start, end in offsets:
            owners = [w["index"] for w in words
                      if start < end and min(end, w["char_end"]) > max(start, w["char_start"])]
            if len(owners) > 1:
                raise ValueError("A tokenizer span crossed reference word boundaries")
            mapping.append(owners[0] if owners else None)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        with torch.inference_mode():
            hidden = model(**{k: v.to(self.config["device"]) for k, v in encoded.items()}).last_hidden_state[0]
        hidden = hidden.float().cpu().numpy()
        features = np.zeros((len(words), hidden.shape[1]), dtype=np.float32)
        for i in range(len(words)):
            indices = [j for j, word_id in enumerate(mapping) if word_id == i]
            if not indices:
                raise ValueError(f"BERT has no WordPiece for reference word {i}")
            features[i] = hidden[indices].mean(axis=0)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite BERT features")
        return features, dict(tokens=tokens, word_ids=mapping, character_offsets=offsets,
                              input_ids=encoded["input_ids"][0].tolist(),
                              attention_mask=encoded["attention_mask"][0].tolist(),
                              special_token_time_policy="not included in word timeline")

    def audio(self, directory, media):
        import opensmile
        if self.smile is None:
            self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                        feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        frame = self.smile.process_file(str(Path(directory) / "audio.wav"))
        starts = frame.index.get_level_values("start").total_seconds().to_numpy()
        ends = frame.index.get_level_values("end").total_seconds().to_numpy()
        intervals = np.column_stack([starts, ends]) + media["audio_offset_seconds"]
        values = frame.to_numpy(dtype=np.float32)
        if values.shape[1] != 25:
            raise ValueError(f"Expected 25 eGeMAPSv02 LLDs, got {values.shape[1]}")
        valid = np.isfinite(values).all(axis=1) & np.isfinite(intervals).all(axis=1)
        valid &= (ends > starts) & (starts >= 0) & (ends <= media["audio_duration"] + 0.03)
        return np.nan_to_num(values, nan=0, posinf=0, neginf=0), intervals, valid, list(frame.columns)

    def vision(self, row, directory, media):
        directory = Path(directory)
        frames = decode_frames(self.config, row, directory, media)
        out = directory / "openface"
        out.mkdir(exist_ok=True)
        marker = out / "complete.json"
        if not marker.exists():
            execute([self.config["openface_bin"], "-fdir", frames, "-out_dir", out,
                     "-of", "features.csv", "-aus", "-pose", "-gaze"],
                    directory / "openface.log", timeout=3600)
            if not (out / "features.csv").is_file():
                raise FileNotFoundError("OpenFace did not produce the explicitly named features.csv")
            write_json(marker, dict(source_sha256=row["sha256"]))
        with (out / "features.csv").open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = [c.strip() for c in reader.fieldnames]
            items = [{k.strip(): v.strip() for k, v in r.items()} for r in reader]
        selected = self.config["vision_columns"]
        if not set(selected + ["frame", "confidence", "success"]).issubset(columns):
            raise ValueError("OpenFace CSV lacks configured columns; check executable/model version")
        n = media["video_frames"]
        values = np.zeros((n, len(selected)), dtype=np.float32)
        confidence = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        seen = set()
        for item in items:
            i = int(item["frame"]) - 1
            if not 0 <= i < n or i in seen:
                raise ValueError("OpenFace frame ids do not match decoded frames")
            seen.add(i)
            vector = np.asarray([float(item[c]) for c in selected], dtype=np.float32)
            conf = float(item["confidence"])
            confidence[i] = conf if np.isfinite(conf) else 0
            good = float(item["success"]) == 1 and conf >= self.config["vision_confidence_min"]
            valid[i] = good and np.isfinite(vector).all()
            if valid[i]:
                values[i] = vector
        if len(seen) != n:
            raise ValueError("OpenFace skipped CSV rows; frame mapping needs inspection")
        return values, np.asarray(media["video_intervals"]), valid, confidence, selected


def extract(config, extractors, row, directory, media, alignment):
    directory = Path(directory)
    words = alignment["words"]
    intervals = np.array([[w["start"] + media["audio_offset_seconds"],
                           w["end"] + media["audio_offset_seconds"]] if w["valid"] else [-1, -1]
                          for w in words], dtype=np.float64)
    text, token_map = extractors.text(words, row["raw_text"])
    audio, audio_intervals, audio_valid, audio_names = extractors.audio(directory, media)
    vision, vision_intervals, vision_valid, confidence, vision_names = extractors.vision(row, directory, media)
    amean, amask, acount = overlap_pool(intervals, audio_intervals, audio, audio_valid)
    vmean, vmask, vcount = overlap_pool(intervals, vision_intervals, vision, vision_valid)
    aligned = dict(text=text, audio=amean, vision=vmean, word_intervals=intervals,
                   alignment_mask=np.array([w["valid"] for w in words]),
                   text_mask=np.ones(len(words), dtype=bool), audio_mask=amask, vision_mask=vmask,
                   padding_mask=np.ones(len(words), dtype=bool), valid_length=np.int32(len(words)),
                   audio_frame_counts=acount, vision_frame_counts=vcount)
    # The output is variable-length. Padding mask uses True=real position, not True=padding.
    for key, value in aligned.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Non-finite output: {key}")
    target = directory / "aligned.npz"
    with target.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **aligned)
    target.with_suffix(".npz.tmp").replace(target)
    with (directory / "native.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, audio=audio, audio_intervals=audio_intervals, audio_mask=audio_valid,
                            vision=vision, vision_intervals=vision_intervals, vision_mask=vision_valid,
                            vision_confidence=confidence,
                            video_frame_numbers=np.arange(1, media["video_frames"] + 1, dtype=np.int32),
                            video_pts=np.asarray(media["video_pts"]))
    (directory / "native.npz.tmp").replace(directory / "native.npz")
    quality = dict(sample_id=row["sample_id"], words=len(words), alignment_coverage=alignment["valid_words"] / len(words),
                   audio_word_coverage=float(amask.mean()), vision_word_coverage=float(vmask.mean()),
                   vision_frame_success_rate=float(vision_valid.mean()),
                   text_dim=text.shape[1], audio_dim=amean.shape[1], vision_dim=vmean.shape[1],
                   duration=media["clip_duration"], backend=alignment["backend"])
    warnings = []
    if quality["alignment_coverage"] < config["alignment_coverage_min"]:
        warnings.append("alignment_coverage_below_threshold")
    if quality["vision_word_coverage"] < config["vision_word_coverage_min"]:
        warnings.append("vision_word_coverage_below_threshold")
    review = config.get("face_reviews", {}).get(row["sample_id"])
    if review != "single_visible_speaker":
        warnings.append("speaker_identity_not_reviewed")
    metadata = dict(schema=1, sample=row, words=words, token_map=token_map, media=media,
                    audio_feature_names=audio_names, vision_feature_names=vision_names,
                    text_feature_definition="bert-base-uncased last hidden state averaged by reference word",
                    mask_convention="True=available; padding_mask=True is a real word; -1 intervals are invalid",
                    aggregation="overlap duration weighted mean over valid native frames",
                    openface_policy="single-person tracking; multi-person clips require explicit handling",
                    quality=quality, warnings=warnings, config=config)
    write_json(directory / "metadata.json", metadata)
    return quality, warnings
