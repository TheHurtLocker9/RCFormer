"""Train-only structural evidence used by RC-v2."""

from __future__ import annotations

import math
from collections import Counter

import torch
from data.preprocessing import unique_undirected_edges


def compute_train_only_statistics(data, train_indices, num_classes, selector_cfg):
    train_idx = torch.as_tensor(train_indices, dtype=torch.long)
    labels = data.y.view(-1).detach().cpu()
    edge_index = getattr(data, 'evidence_edge_index', None)
    if edge_index is None:
        edge_index = unique_undirected_edges(data.edge_index, data.num_nodes)
    edge_index = edge_index.detach().cpu()
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    src, dst = edge_index[0], edge_index[1]
    valid_edges = train_mask[src] & train_mask[dst]
    src, dst = src[valid_edges], dst[valid_edges]
    total = int(src.numel())
    same = int((labels[src] == labels[dst]).sum().item()) if total else 0
    different = total - same
    homophily = float(same) / total if total else None

    counts = Counter(int(labels[index]) for index in train_idx.tolist())
    nonzero = [count for count in counts.values() if count > 0]
    imbalance = float(max(nonzero)) / float(min(nonzero)) if nonzero else None
    entropy = 0.0
    if nonzero:
        denominator = float(sum(nonzero))
        for count in nonzero:
            probability = float(count) / denominator
            entropy -= probability * math.log(probability + 1e-12)

    profile = {
        "edge_evidence_unit": "unique_undirected_nonself_train_pair",
        "edge_homophily_train": homophily,
        "train_homophily_edges": total,
        "train_same_label_edges": same,
        "train_diff_label_edges": different,
        "class_imbalance_train": imbalance,
        "class_entropy_train": entropy if nonzero else None,
        "class_counts_train": {str(key): int(value) for key, value in sorted(counts.items())},
        "num_train_classes_observed": len(counts),
        "num_classes_for_entropy": int(num_classes),
        "uses_train_labels": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
    }
    profile.update(_calibrate(profile, selector_cfg))
    return profile


def _calibrate(profile, cfg):
    eps = 1e-12
    total = int(profile["train_homophily_edges"])
    same = int(profile["train_same_label_edges"])
    evidence_scale = max(float(cfg.get("evidence_scale", 100.0)), eps)
    reliability = total / float(total + evidence_scale)

    p_low = p_high = None
    if total > 0:
        posterior_mean = float(same + 1) / float(total + 2)
        posterior_var = posterior_mean * (1.0 - posterior_mean) / float(total + 3)
        threshold = float(cfg.get("homophily_threshold", 0.4))
        z_score = (threshold - posterior_mean) / math.sqrt(posterior_var + eps)
        p_low = min(1.0, max(0.0, _normal_cdf(z_score)))
        p_high = 1.0 - p_low

    entropy_norm = None
    num_classes = int(profile["num_classes_for_entropy"])
    entropy = profile["class_entropy_train"]
    if entropy is not None and num_classes > 1:
        entropy_norm = min(1.0, max(0.0, float(entropy) / math.log(num_classes)))

    ratio = float(profile["class_imbalance_train"] or 1.0)
    threshold = max(float(cfg.get("imbalance_threshold", 30.0)), eps)
    temperature = max(float(cfg.get("imbalance_temperature", 1.0)), eps)
    ratio_score = _sigmoid((math.log(max(ratio, eps)) - math.log(threshold)) / temperature)
    severity = ratio_score if entropy_norm is None else 0.5 * ratio_score + 0.5 * (1.0 - entropy_norm)
    confidence = [severity]
    if p_low is not None:
        confidence.extend([p_low, p_high])
    return {
        "evidence_reliability": reliability,
        "homophily_p_low": p_low,
        "homophily_p_high": p_high,
        "imbalance_entropy_norm": entropy_norm,
        "imbalance_severity": severity,
        "selector_confidence": max(confidence),
    }


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
