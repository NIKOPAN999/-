"""Small, validation-only decision calibration for the four contest metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def align_intensity_to_polarity(classes, scores, minimum_magnitude=1e-6):
    """Project intensity onto the interval required by the predicted polarity.

    The contest defines negative as [-3, 0), neutral as exactly 0, and
    positive as (0, 3]. For non-neutral predictions this is the smallest
    change to the calibrated score that also survives six-decimal CSV output.
    """
    classes = np.asarray(classes)
    scores = np.asarray(scores, dtype=np.float64)
    if classes.shape != scores.shape or not np.isin(classes, (0, 1, 2)).all():
        raise ValueError("Polarity codes and intensity scores must align")
    if not 0 < minimum_magnitude <= 3:
        raise ValueError("minimum_magnitude must lie in (0, 3]")
    return np.where(classes == 1, 0.0,
                    np.where(classes == 0, np.minimum(scores, -minimum_magnitude),
                             np.maximum(scores, minimum_magnitude)))


def apply_calibration(logits, scores, parameters=None):
    logits = np.asarray(logits, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    parameters = parameters or {}
    biases = np.asarray(parameters.get("class_bias", [0.0, 0.0, 0.0]), dtype=np.float64)
    if biases.shape != (3,):
        raise ValueError("class_bias must contain three values")
    scale = float(parameters.get("reg_scale", 1.0))
    offset = float(parameters.get("reg_offset", 0.0))
    if scale <= 0:
        raise ValueError("reg_scale must be positive")
    calibrated_scores = np.clip(scale * scores + offset, -3.0, 3.0)
    decision_logits = logits + biases
    slope = float(parameters.get("score_neutral_slope", 0.0))
    pivot = float(parameters.get("score_neutral_pivot", 0.0))
    if slope < 0 or not 0 <= pivot <= 3:
        raise ValueError("Invalid score-guided neutral calibration")
    if slope:
        decision_logits = decision_logits.copy()
        decision_logits[:, 1] += slope * (pivot - np.abs(calibrated_scores))
    classes = decision_logits.argmax(axis=-1)
    calibrated_scores = align_intensity_to_polarity(classes, calibrated_scores)
    return classes, calibrated_scores


def four_metrics(y_cls, y_reg, pred_cls, pred_reg):
    y_cls = np.asarray(y_cls)
    y_reg = np.asarray(y_reg)
    pred_reg = np.asarray(pred_reg)
    pearson = (float(np.corrcoef(y_reg, pred_reg)[0, 1])
               if np.std(y_reg) > 0 and np.std(pred_reg) > 0 else None)
    return {"accuracy": float(accuracy_score(y_cls, pred_cls)),
            "macro_f1": float(f1_score(y_cls, pred_cls, labels=[0, 1, 2],
                                       average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(y_reg, pred_reg)),
            "pearson": pearson}


def fit_calibration(cases, class_objective="f1", max_f1_drop=0.01):
    """Fit bounded class offsets and affine regression on labelled valid views.

    cases maps scenario name to (labels_dict, logits[N,3], scores[N]). Neither
    attachment 3 nor the held-out attachment-2 test split is used here.
    """
    if "clean" not in cases:
        raise ValueError("A clean validation view is required")
    if class_objective not in ("f1", "accuracy") or not 0 <= max_f1_drop <= 0.1:
        raise ValueError("Invalid class objective or F1 tolerance")
    for labels, logits, scores in cases.values():
        if len(labels["cls"]) != len(logits) or len(logits) != len(scores):
            raise ValueError("Validation labels and predictions are misaligned")

    names = list(cases)
    base = {}
    for name, (labels, logits, scores) in cases.items():
        cls, reg = apply_calibration(logits, scores)
        base[name] = four_metrics(labels["cls"], labels["reg"], cls, reg)
    base_mean_f1 = np.mean([base[name]["macro_f1"] for name in names])
    base_mean_acc = np.mean([base[name]["accuracy"] for name in names])

    best_bias = [0.0, 0.0, 0.0]
    best_class = (base_mean_f1, base_mean_acc, 0.0)
    best_class_clean_f1 = base["clean"]["macro_f1"]
    class_candidates = [(base_mean_f1, base_mean_acc,
                         base["clean"]["macro_f1"], base["clean"]["accuracy"],
                         0.0, best_bias)]
    for negative in np.arange(-0.4, 0.401, 0.05):
        for neutral in np.arange(-0.4, 0.401, 0.05):
            bias = [float(negative), float(neutral), 0.0]
            view_f1, view_acc = [], []
            for labels, logits, _ in cases.values():
                pred = (np.asarray(logits) + bias).argmax(-1)
                view_f1.append(f1_score(labels["cls"], pred, labels=[0, 1, 2],
                                        average="macro", zero_division=0))
                view_acc.append(accuracy_score(labels["cls"], pred))
            if view_f1[names.index("clean")] < base["clean"]["macro_f1"] - 0.005:
                continue
            if np.mean(view_acc) < base_mean_acc - 0.01:
                continue
            candidate = (float(np.mean(view_f1)), float(np.mean(view_acc)),
                         -float(abs(negative) + abs(neutral)))
            class_candidates.append((candidate[0], candidate[1],
                                     float(view_f1[names.index("clean")]),
                                     float(view_acc[names.index("clean")]),
                                     -candidate[2], bias))
            if candidate > best_class and candidate[0] > base_mean_f1 + 0.001:
                best_class, best_bias = candidate, bias
                best_class_clean_f1 = float(view_f1[names.index("clean")])

    if class_objective == "accuracy":
        eligible = [row for row in class_candidates
                    if row[0] >= best_class[0] - max_f1_drop and
                    row[2] >= best_class_clean_f1 - max_f1_drop]
        chosen = max(eligible, key=lambda row: (row[1], row[3], row[0], -row[4]))
        if chosen[1] > base_mean_acc + 0.001:
            best_bias = chosen[5]

    best_scale, best_offset = 1.0, 0.0
    chosen_classes = {name: (np.asarray(logits) + best_bias).argmax(-1)
                      for name, (_, logits, _) in cases.items()}
    regression_baseline = {}
    for name, (labels, _, scores) in cases.items():
        initial = align_intensity_to_polarity(
            chosen_classes[name], np.clip(np.asarray(scores), -3, 3))
        regression_baseline[name] = four_metrics(
            labels["cls"], labels["reg"], chosen_classes[name], initial)
    baseline_mae = float(np.mean([row["mae"] for row in regression_baseline.values()]))
    baseline_pearson = float(np.mean([
        row["pearson"] if row["pearson"] is not None else -1.0
        for row in regression_baseline.values()]))
    best_reg = (-baseline_mae, baseline_pearson, 0.0)
    # Search regression scale and offset on validation predictions only.
    for scale in np.arange(0.5, 1.201, 0.025):
        for offset in np.arange(-0.25, 0.251, 0.025):
            maes, correlations = [], []
            for name, (labels, _, scores) in cases.items():
                predicted = align_intensity_to_polarity(
                    chosen_classes[name],
                    np.clip(float(scale) * np.asarray(scores) + float(offset), -3, 3))
                maes.append(mean_absolute_error(labels["reg"], predicted))
                correlations.append(float(np.corrcoef(labels["reg"], predicted)[0, 1])
                                    if np.std(labels["reg"]) > 0 and np.std(predicted) > 0
                                    else -1.0)
            clean_pearson = regression_baseline["clean"]["pearson"]
            if correlations[names.index("clean")] < (clean_pearson if clean_pearson is not None else -1) - 0.005:
                continue
            if np.mean(correlations) < baseline_pearson - 0.005:
                continue
            candidate = (-float(np.mean(maes)), float(np.mean(correlations)),
                         -float(abs(scale - 1) + abs(offset)))
            if candidate > best_reg and -candidate[0] < baseline_mae - 0.001:
                best_reg, best_scale, best_offset = candidate, float(scale), float(offset)

    parameters = {"class_bias": best_bias, "reg_scale": best_scale,
                  "reg_offset": best_offset}
    after = {}
    for name, (labels, logits, scores) in cases.items():
        cls, reg = apply_calibration(logits, scores, parameters)
        after[name] = four_metrics(labels["cls"], labels["reg"], cls, reg)
    return {"split": "valid", "scenarios": names,
            "intensity_rule": "sign_consistent_v1",
            "class_objective": class_objective, "max_f1_drop": max_f1_drop,
            "parameters": parameters,
            "before": base, "after": after}
