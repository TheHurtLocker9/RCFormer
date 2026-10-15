"""Training-only Gaussian feature-noise consistency."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def gaussian_feature_noise(features, sigma=0.1):
    return features + float(sigma) * torch.randn_like(features)


def consistency_is_active(epoch, cfg):
    if not bool(cfg.get("enabled", False)):
        return False
    every = max(1, int(cfg.get("every", 1)))
    return int(epoch) % every == 0


def consistency_weight(epoch, cfg):
    target = float(cfg.get("weight", 0.0))
    warmup = max(1, int(cfg.get("warmup_epochs", 20)))
    return target * min(1.0, float(epoch) / warmup)


def sample_consistency_nodes(train_indices, ratio, device, generator=None):
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    count = max(1, int(indices.numel() * float(ratio)))
    permutation = torch.randperm(
        indices.numel(), device=device, generator=generator
    )[:count]
    return indices[permutation]


def consistency_loss(clean_log_probs, noisy_log_probs, indices, temperature=1.0, kind="kl_stopgrad_clean"):
    temperature = float(temperature)
    clean = clean_log_probs[indices]
    noisy = noisy_log_probs[indices]
    if kind != "kl_stopgrad_clean":
        raise ValueError(f"unsupported consistency loss: {kind}")
    return F.kl_div(
        noisy,
        clean.detach().exp(),
        reduction="batchmean",
    ) * (temperature**2)
