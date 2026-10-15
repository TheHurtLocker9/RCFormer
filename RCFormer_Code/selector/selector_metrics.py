"""Metrics used by SC-Efficient and final evaluation."""

from __future__ import annotations

import math
import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def class_balanced_nll(log_probs, labels, indices):
    losses = []
    selected_labels = labels[indices]
    for class_id in torch.unique(selected_labels):
        class_indices = indices[selected_labels == class_id]
        if class_indices.numel():
            losses.append(
                -log_probs[class_indices, labels[class_indices]].mean()
            )
    if not losses:
        return log_probs.new_tensor(float("nan"))
    return torch.stack(losses).mean()


def classification_metrics(log_probs, labels, indices):
    idx = torch.as_tensor(indices, dtype=torch.long, device=log_probs.device)
    y_true = labels[idx].detach().cpu().numpy()
    probs = log_probs[idx].exp().detach().cpu().numpy()
    y_pred = np.argmax(probs, axis=1)
    result = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC-AUC": None,
    }
    try:
        if probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
            result["ROC-AUC"] = float(roc_auc_score(y_true, probs[:, 1]))
        elif probs.shape[1] > 2 and len(np.unique(y_true)) == probs.shape[1]:
            result["ROC-AUC"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        result["ROC-AUC"] = None
    return result


def task_metric_with_fallback(metrics, primary_metric):
    requested = normalize_metric_name(primary_metric)
    value = metrics.get(requested)
    if value is not None and math.isfinite(float(value)):
        return float(value), requested, ""
    acc = metrics.get("ACC")
    if acc is None or not math.isfinite(float(acc)):
        return None, requested, f"{requested}_undefined_and_acc_unavailable"
    return float(acc), "ACC", f"{requested}_undefined_fallback_ACC"


def normalize_metric_name(name):
    normalized = str(name).strip().lower().replace("_", "-")
    if normalized in {"acc", "accuracy"}:
        return "ACC"
    if normalized in {"macro-f1", "macrof1", "f1"}:
        return "Macro-F1"
    if normalized in {"roc-auc", "rocauc", "auc"}:
        return "ROC-AUC"
    raise ValueError(f"unsupported metric: {name}")


def paired_interval(values, z=1.645):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"n": 0, "mean": None, "std": None, "lower": None, "upper": None}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = float(z) * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }
