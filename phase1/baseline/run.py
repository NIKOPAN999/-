"""Independent Self-MM adaptation for contest problem 2.

Train parameters and ULGM pseudo-labels use attachment-2 train only. Valid
selects the epoch and calibration. The held-out test is evaluated once after
selection, while attachment 3 is reserved for final unlabeled inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader

from calibration import apply_calibration, fit_calibration
from data import (AlignedDataset, Statistics, block_mask, fit_statistics,
                  load_aligned, load_attachment3, random_block_mask)
from model import SelfMMContest
from settings import ATTACHMENT3, BERT_NAME, CONFIG, DATA_FILE, OUTPUT_DIR
from ulgm import UnimodalLabelBank


METRICS = ("accuracy", "macro_f1", "mae", "pearson")
MODES = ("text", "audio", "vision")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def scenarios():
    yield "clean", None
    for mode, name in enumerate(MODES):
        for ratio in (0.1, 0.3, 0.5):
            for position in ("front", "middle", "end"):
                yield f"{name}_{ratio}_{position}", ((mode,), ratio, position)
    for modes, name in [((0, 1), "text_audio"), ((0, 2), "text_vision"),
                        ((1, 2), "audio_vision")]:
        yield f"{name}_0.3_middle", (modes, 0.3, "middle")
    yield "all_modalities_0.3_middle", ((0, 1, 2), 0.3, "middle")


SELECTION_VIEWS = {"clean", "text_0.3_middle", "text_0.5_end",
                   "audio_0.3_middle", "vision_0.3_middle",
                   "all_modalities_0.3_middle"}


def calculate_metrics(labels, classes, intensities):
    truth_cls = np.asarray(labels["cls"], dtype=np.int64)
    truth_reg = np.asarray(labels["reg"], dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64)
    intensities = np.asarray(intensities, dtype=np.float64)
    corr = (float(np.corrcoef(truth_reg, intensities)[0, 1])
            if truth_reg.std() > 0 and intensities.std() > 0 else None)
    return {"accuracy": float(accuracy_score(truth_cls, classes)),
            "macro_f1": float(f1_score(truth_cls, classes, labels=[0, 1, 2],
                                        average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(truth_reg, intensities)),
            "pearson": corr,
            "f1_negative": float(f1_score(truth_cls == 0, classes == 0,
                                           zero_division=0)),
            "f1_neutral": float(f1_score(truth_cls == 1, classes == 1,
                                          zero_division=0)),
            "f1_positive": float(f1_score(truth_cls == 2, classes == 2,
                                           zero_division=0))}


@torch.no_grad()
def predict_raw(model, loader, device, scenario=None):
    model.eval()
    labels = {"cls": [], "reg": []}
    logits, scores = [], []
    for clean in loader:
        batch = block_mask(clean, *scenario) if scenario else clean
        if "cls" in clean:
            labels["cls"].extend(clean["cls"].tolist())
            labels["reg"].extend(clean["reg"].tolist())
        output = model(to_device(batch, device))
        logits.append(output["logits"].cpu().numpy())
        scores.append(output["reg"].cpu().numpy())
    return labels, np.concatenate(logits), np.concatenate(scores)


def collect_cases(model, loader, device, names=None):
    return {name: predict_raw(model, loader, device, condition)
            for name, condition in scenarios() if names is None or name in names}


def evaluate_cases(cases, calibration=None):
    rows = []
    for name, (labels, logits, scores) in cases.items():
        classes, values = apply_calibration(logits, scores, calibration)
        rows.append({"scenario": name, **calculate_metrics(labels, classes, values)})
    return rows


def evaluate_grid(model, loader, device, calibration=None):
    return evaluate_cases(collect_cases(model, loader, device), calibration)


def mean_four(rows):
    return {key: (float(np.mean([row[key] for row in rows]))
                  if all(row[key] is not None for row in rows) else None)
            for key in METRICS}


def summarize(rows):
    text = [r for r in rows if r["scenario"].startswith(
        ("text_0.1_", "text_0.3_", "text_0.5_"))]
    if len(rows) != 32 or len(text) != 9:
        raise RuntimeError("Expected 32 evaluation scenarios and 9 text-only scenarios")
    return [rows[0], {"scenario": "all_31_mean", **mean_four(rows[:31])},
            {"scenario": "text_missing_9_mean", **mean_four(text)},
            next(r for r in rows if r["scenario"] == "text_0.5_end"), rows[-1]]


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def print_table(rows, title):
    print("\n" + title, flush=True)
    print(f"{'view':<26} {'Accuracy up':>11} {'macro F1 up':>11} "
          f"{'MAE down':>11} {'Pearson up':>11}", flush=True)
    for row in rows:
        values = ["n/a" if row[key] is None else f"{row[key]:.4f}" for key in METRICS]
        print(f"{row['scenario']:<26} " + " ".join(f"{value:>11}" for value in values),
              flush=True)


def selection_score(rows):
    chosen = [row for row in rows if row["scenario"] in SELECTION_VIEWS]
    if len(chosen) != len(SELECTION_VIEWS):
        raise RuntimeError("Missing a validation selection scenario")
    return float(np.mean([
        0.55 * row["accuracy"] + 0.20 * row["macro_f1"] +
        0.125 * (1 - row["mae"] / 3) +
        0.125 * (((row["pearson"] if row["pearson"] is not None else -1) + 1) / 2)
        for row in chosen]))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_name(name, count):
    return (name.startswith("bert.embeddings.") or name.startswith("bert.pooler.") or
            any(name.startswith(f"bert.encoder.layer.{i}.") for i in range(count)))


def base_fingerprint(state, freeze):
    digest = hashlib.sha256()
    for name in sorted(state):
        if frozen_name(name, freeze):
            digest.update(name.encode("utf-8"))
            digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, stats, config, epoch, score):
    torch.save({"state": model.state_dict(), "statistics": stats.state_dict(),
                "config": config, "bert_config": model.bert.config.to_dict(),
                "epoch": epoch, "selection_score": score}, path)


def export_compact(source, destination):
    source_state = torch.load(source, map_location="cpu", weights_only=False)
    freeze = source_state["config"]["freeze_bert_layers"]
    source_state["base_fingerprint"] = base_fingerprint(source_state["state"], freeze)
    retained, scales = {}, {}
    for name, tensor in source_state["state"].items():
        if frozen_name(name, freeze):
            continue
        value = tensor.detach().cpu()
        if not value.is_floating_point():
            retained[name] = value
            continue
        value = value.float()
        if name.startswith("bert.") and value.ndim > 1:
            axes = tuple(range(1, value.ndim))
            scale = (value.abs().amax(dim=axes, keepdim=True) / 127).clamp_min(1e-12)
            retained[name] = (value / scale).round().clamp(-127, 127).to(torch.int8)
            scales[name] = scale
        else:
            retained[name] = value.half()
    source_state["state"] = retained
    source_state["quantization_scales"] = scales
    source_state["compact_format"] = "frozen_bert_base_plus_channel_int8_v1"
    torch.save(source_state, destination)
    return Path(destination).stat().st_size


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    compact = state.get("compact_format") == "frozen_bert_base_plus_channel_int8_v1"
    source = (os.environ.get("SELFMM_BERT_BASE", config["bert_name"])
              if compact else config["bert_name"])
    model = SelfMMContest(source, config["hidden"], config["dropout"],
                          config["freeze_bert_layers"],
                          None if compact else state["bert_config"])
    if compact:
        for field in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"):
            if getattr(model.bert.config, field) != state["bert_config"][field]:
                raise ValueError(f"Pretrained BERT mismatch: {field}")
        if base_fingerprint(model.state_dict(), config["freeze_bert_layers"]) != state["base_fingerprint"]:
            raise ValueError("The compact checkpoint requires the original pretrained BERT base")
        restored = {name: (value.float() * state["quantization_scales"][name]
                           if name in state["quantization_scales"] else
                           value.float() if value.is_floating_point() else value)
                    for name, value in state["state"].items()}
        missing, unexpected = model.load_state_dict(restored, strict=False)
        if unexpected or any(not frozen_name(name, config["freeze_bert_layers"])
                             for name in missing):
            raise RuntimeError(f"Invalid compact checkpoint keys: {missing}, {unexpected}")
    else:
        model.load_state_dict(state["state"])
    return model.to(device), Statistics.from_state_dict(state["statistics"]), state


def write_calibration(path, calibration, checkpoint):
    calibration["checkpoint_sha256"] = file_sha256(checkpoint)
    Path(path).write_text(json.dumps(calibration, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def read_calibration(path, checkpoint):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("split") != "valid" or record.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("Calibration must belong to this checkpoint and attachment-2 valid")
    return record["parameters"]


def package(output, checkpoint, calibration):
    archive_path = output / "submission_code6.zip"
    source = Path(__file__).resolve().parent
    names = ("run.py", "model.py", "ulgm.py", "data.py", "calibration.py",
             "settings.py", "smoke.py", "export_sample_evidence.py",
             "requirements.txt", "README.md")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(source / name, f"code6/{name}")
        for name in ("training_conditions.json", "environment.json"):
            archive.write(output / name, f"code6/{name}")
        archive.write(checkpoint, "code6/student_compact.pt")
        archive.write(calibration, "code6/student_compact_calibration.json")
    size = archive_path.stat().st_size
    print(f"Submission candidate: {archive_path} ({size / 1e6:.2f} MB)", flush=True)
    if size > 50_000_000:
        print("WARNING: the candidate exceeds the 50 MB attachment limit", flush=True)
    return archive_path


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "student_best.pt"
    compact_path = out / "student_compact.pt"
    if full_path.exists() or compact_path.exists():
        raise FileExistsError(f"Existing checkpoint in {out}; choose a new --output")
    config = {**CONFIG, "bert_name": args.bert_name}
    seed_all(config["seed"])
    raw = load_aligned(args.data)
    stats = fit_statistics(raw["train"])
    train_set = AlignedDataset(raw["train"], stats)
    valid_set = AlignedDataset(raw["valid"], stats)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config["batch_size"] * 2)
    device = torch.device("cuda")
    model = SelfMMContest(args.bert_name, config["hidden"], config["dropout"],
                          config["freeze_bert_layers"]).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bert.")],
         "lr": config["bert_lr"]},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("bert.")],
         "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])
    bank = UnimodalLabelBank(torch.from_numpy(train_set.reg), config["hidden"])
    random_generator = torch.Generator().manual_seed(config["seed"] + 17)
    conditions = {
        "paper": "Self-MM, AAAI 2021, arXiv:2102.04830",
        "adaptation": "aligned feature interface, local gap augmentation, three-class head, valid-only calibration",
        "train_samples": len(train_set), "valid_samples": len(valid_set),
        "test_samples": len(raw["test"]["classification_labels"]),
        "training_split": "attachment-2 train only",
        "normalization": "audio/vision observed train positions only",
        "pseudo_labels": "ULGM feature centers and momentum on train only; neutral excluded from pos/neg centers",
        "checkpoint_and_calibration": "attachment-2 valid only",
        "test": "one final evaluation after compact checkpoint and calibration fixed",
        "attachment3": "aligned version, unlabeled final inference only",
        "metrics": list(METRICS), "config": config,
    }
    (out / "training_conditions.json").write_text(
        json.dumps(conditions, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({
        "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "data": str(args.data),
        "data_bytes": args.data.stat().st_size,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAINING CONDITIONS=" + json.dumps(conditions, ensure_ascii=False), flush=True)
    best_score, stale = -math.inf, 0
    epoch_records = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        bank.begin_epoch()
        losses = []
        for clean in train_loader:
            damaged = random_block_mask(clean, random_generator,
                                        config["mask_probability"], config["text_bias"],
                                        config["max_mask_ratio"], config["triple_probability"])
            batch = to_device(damaged, device)
            result = model(batch)
            truth_reg = clean["reg"].to(device)
            truth_cls = clean["cls"].to(device)
            pseudo = bank.targets(clean["index"], device)
            ce = F.cross_entropy(result["logits"], truth_cls, reduction="none")
            ce_weight = 1.0 + (config["neutral_class_weight"] - 1.0) * (truth_cls == 1).float()
            ce = (ce * ce_weight).sum() / ce_weight.sum()
            aux_losses = []
            for mode_index, mode in enumerate(MODES):
                difference_weight = torch.tanh((pseudo[mode] - truth_reg).abs())
                available = (result["availability"][:, mode_index] > 0).float()
                aux_losses.append(((result["aux"][mode] - pseudo[mode]).abs() *
                                   difference_weight * available).sum() /
                                  available.sum().clamp_min(1))
            loss = (config["regression_weight"] * F.l1_loss(result["reg"], truth_reg) +
                    config["classification_weight"] * ce +
                    config["auxiliary_weight"] * sum(aux_losses) / 3)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bank.record(clean["index"], result["features"])
            losses.append(float(loss.detach()))
        pseudo_update = bank.update(epoch)
        valid_rows = evaluate_cases(collect_cases(model, valid_loader, device,
                                                   SELECTION_VIEWS))
        score = selection_score(valid_rows)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "validation_score": score, "pseudo_label_update": pseudo_update,
                  "validation": valid_rows}
        epoch_records.append(record)
        with (out / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print_table([valid_rows[0],
                     {"scenario": "selected_mean", **mean_four(valid_rows)},
                     next(row for row in valid_rows if row["scenario"] == "text_0.5_end")],
                    f"VALID epoch {epoch}/{config['epochs']} | loss={record['train_loss']:.4f} "
                    f"| selection_score={score:.4f} | pseudo_shift={pseudo_update['mean_shift']:.4f}")
        if score > best_score + 1e-5:
            best_score, stale = score, 0
            save_checkpoint(full_path, model, stats, config, epoch, score)
        else:
            stale += 1
            if stale >= config["patience"]:
                break

    write_csv(out / "epoch_metrics.csv",
              [{"epoch": record["epoch"], "train_loss": record["train_loss"],
                "validation_score": record["validation_score"],
                "scenario": row["scenario"], **row}
               for record in epoch_records for row in record["validation"]],
              ("epoch", "scenario", "train_loss", "validation_score", *METRICS,
               "f1_neutral"))
    del model, optimizer
    torch.cuda.empty_cache()
    selected, _, selected_state = load_checkpoint(full_path, device)
    print(f"Selected epoch {selected_state['epoch']} on attachment-2 VALID", flush=True)
    cases = collect_cases(selected, valid_loader, device, SELECTION_VIEWS)
    calibration = fit_calibration(cases, "accuracy", 0.01)
    full_calibration_path = out / "student_best_calibration.json"
    write_calibration(full_calibration_path, calibration, full_path)
    full_rows = evaluate_grid(selected, valid_loader, device, calibration["parameters"])
    write_csv(out / "valid_calibrated.csv", full_rows,
              ("scenario", *METRICS, "f1_negative", "f1_neutral", "f1_positive"))
    write_csv(out / "final_metrics.csv", summarize(full_rows), ("scenario", *METRICS))
    print_table(summarize(full_rows),
                f"FULL attachment-2 VALID (n={len(valid_set)}) | epoch {selected_state['epoch']}")
    del selected
    torch.cuda.empty_cache()

    size = export_compact(full_path, compact_path)
    print(f"Compact channel_int8 checkpoint: {compact_path} ({size / 1e6:.2f} MB)", flush=True)
    compact, _, _ = load_checkpoint(compact_path, device)
    compact_cases = collect_cases(compact, valid_loader, device, SELECTION_VIEWS)
    compact_calibration = fit_calibration(compact_cases, "accuracy", 0.01)
    compact_calibration_path = out / "student_compact_calibration.json"
    write_calibration(compact_calibration_path, compact_calibration, compact_path)
    compact_rows = evaluate_grid(compact, valid_loader, device,
                                 compact_calibration["parameters"])
    compact_summary = summarize(compact_rows)
    write_csv(out / "compact_valid_calibrated.csv", compact_rows,
              ("scenario", *METRICS, "f1_negative", "f1_neutral", "f1_positive"))
    write_csv(out / "compact_final_metrics.csv", compact_summary, ("scenario", *METRICS))
    print_table(compact_summary, f"FINAL COMPACT attachment-2 VALID (n={len(valid_set)}) "
                                 f"| epoch {selected_state['epoch']}")
    package(out, compact_path, compact_calibration_path)

    # Final hold-out evaluation: no parameter, threshold, or checkpoint updates below.
    test_set = AlignedDataset(raw["test"], stats)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"] * 2)
    test_rows = evaluate_grid(compact, test_loader, device,
                              compact_calibration["parameters"])
    test_summary = summarize(test_rows)
    write_csv(out / "compact_test_calibrated.csv", test_rows,
              ("scenario", *METRICS, "f1_negative", "f1_neutral", "f1_positive"))
    write_csv(out / "compact_test_final_metrics.csv", test_summary, ("scenario", *METRICS))
    print_table(test_summary, f"FINAL COMPACT attachment-2 TEST (n={len(test_set)}) "
                              "| calibration fitted on VALID")
    print(f"Neutral F1 diagnostic: VALID clean={compact_rows[0]['f1_neutral']:.4f} | "
          f"TEST clean={test_rows[0]['f1_neutral']:.4f}", flush=True)
    write_csv(out / "compact_valid_test_metrics.csv",
              [{"split": split, "n_samples": size, "checkpoint": compact_path.name,
                **row} for split, size, summary in
               (("valid", len(valid_set), compact_summary),
                ("test", len(test_set), test_summary)) for row in summary],
              ("split", "n_samples", "checkpoint", "scenario", *METRICS))
    print(f"Saved paired metrics: {out / 'compact_valid_test_metrics.csv'}", flush=True)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, _ = load_checkpoint(args.checkpoint, device)
    raw = load_aligned(args.data)
    dataset = AlignedDataset(raw[args.split], stats)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"] * 2)
    calibration = (read_calibration(args.calibration, args.checkpoint)
                   if args.calibration else None)
    rows = evaluate_grid(model, loader, device, calibration)
    write_csv(args.output_csv, rows,
              ("scenario", *METRICS, "f1_negative", "f1_neutral", "f1_positive"))
    print_table(summarize(rows), f"attachment-2 {args.split.upper()} (n={len(dataset)})")


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, _ = load_checkpoint(args.checkpoint, device)
    calibration = read_calibration(args.calibration, args.checkpoint)
    dataset = load_attachment3(args.attachment3, stats)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"] * 2)
    _, logits, scores = predict_raw(model, loader, device)
    classes, values = apply_calibration(logits, scores, calibration)
    names = ("Negative", "Neutral", "Positive")
    rows = [{"sample_id": Path(sample).stem, "polarity_code": int(label),
             "polarity": names[int(label)], "intensity": round(float(value), 6)}
            for sample, label, value in zip(dataset.sample_ids, classes, values)]
    if len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30:
        raise RuntimeError("Attachment 3 needs 30 distinct predictions")
    for row in rows:
        value, code = row["intensity"], row["polarity_code"]
        if not -3 <= value <= 3 or not ((code == 0 and value < 0) or
                                        (code == 1 and value == 0) or
                                        (code == 2 and value > 0)):
            raise RuntimeError(f"Inconsistent polarity and intensity: {row}")
    write_csv(args.output_csv, rows, ("sample_id", "polarity_code", "polarity", "intensity"))
    metadata = {"source": "attachment-3 aligned unlabeled", "sample_count": len(rows),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "calibration": str(args.calibration),
                "calibration_sha256": file_sha256(args.calibration),
                "prediction_csv": str(args.output_csv)}
    args.output_csv.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} attachment-3 predictions to {args.output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=DATA_FILE)
    train_parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    train_parser.add_argument("--bert-name", default=BERT_NAME)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--data", type=Path, default=DATA_FILE)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--calibration", type=Path)
    eval_parser.add_argument("--split", choices=("valid", "test"), default="valid")
    eval_parser.add_argument("--output-csv", type=Path, required=True)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--calibration", type=Path, required=True)
    infer_parser.add_argument("--attachment3", type=Path, default=ATTACHMENT3)
    infer_parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "infer": infer}[args.command](args)


if __name__ == "__main__":
    main()
