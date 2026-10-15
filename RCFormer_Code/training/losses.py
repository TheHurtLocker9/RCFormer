"""Final-training loss assembly."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from regularization.gaussian_consistency import (
    consistency_is_active,
    consistency_loss,
    consistency_weight,
    gaussian_feature_noise,
    sample_consistency_nodes,
)


def supervised_nll(log_probs, labels, train_indices):
    return F.nll_loss(log_probs[train_indices], labels[train_indices])


def final_training_loss(model, data, train_indices, epoch, consistency_cfg, device, seed):
    clean_log_probs = model(data.x, data.edge_index)
    supervised = supervised_nll(clean_log_probs, data.y, train_indices)
    total = supervised
    consistency_value = clean_log_probs.new_tensor(0.0)
    if consistency_is_active(epoch, consistency_cfg):
        noisy_features = gaussian_feature_noise(
            data.x, float(consistency_cfg.get("sigma", 0.1))
        )
        noisy_log_probs = model(noisy_features, data.edge_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 104729 * (int(epoch) + 1))
        nodes = sample_consistency_nodes(
            train_indices,
            float(consistency_cfg.get("sample_ratio", 0.5)),
            device,
            generator=generator,
        )
        consistency_value = consistency_loss(
            clean_log_probs,
            noisy_log_probs,
            nodes,
            temperature=float(consistency_cfg.get("temperature", 1.0)),
            kind=str(consistency_cfg.get("kind", "kl_stopgrad_clean")),
        )
        total = total + consistency_weight(epoch, consistency_cfg) * consistency_value
    return total, {
        "supervised_nll": float(supervised.detach().item()),
        "consistency_loss": float(consistency_value.detach().item()),
    }
