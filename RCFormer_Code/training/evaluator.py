"""Model evaluation for the outer splits."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from selector.selector_metrics import classification_metrics, task_metric_with_fallback


@torch.no_grad()
def evaluate_split(model, data, indices, primary_metric):
    model.eval()
    log_probs = model(data.x, data.edge_index)
    index = torch.as_tensor(indices, dtype=torch.long, device=log_probs.device)
    metrics = classification_metrics(log_probs, data.y, index)
    metrics["NLL"] = float(F.nll_loss(log_probs[index], data.y[index]).item())
    primary, effective, fallback = task_metric_with_fallback(metrics, primary_metric)
    metrics.update(
        {
            "primary_metric": primary,
            "primary_metric_name": effective,
            "primary_metric_fallback": fallback,
        }
    )
    return metrics


def evaluate_all(model, data, split, primary_metric):
    return {
        key: evaluate_split(model, data, split[key], primary_metric)
        for key in ("train", "valid", "test")
    }
