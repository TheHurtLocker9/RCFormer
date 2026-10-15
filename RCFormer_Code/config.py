"""Frozen paper defaults and small configuration helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PAPER_SEEDS = [2024, 2025, 2026, 2027, 2028]
BRANCH_DEPTH = {"sgc": 2, "fagcn_local": 4, "optbasis_local": 1}
PRIMARY_METRIC = {
    "questions": "ROC-AUC",
    "question": "ROC-AUC",
    "minesweeper": "ROC-AUC",
    "tolokers": "ROC-AUC",
    "amazon-ratings": "ACC",
}


def canonical_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "question": "questions",
        "amazonratings": "amazon-ratings",
        "romanempire": "roman-empire",
        "citeseer": "citeseer",
    }
    return aliases.get(normalized, normalized)


def default_config(dataset: str) -> dict:
    dataset = canonical_dataset_name(dataset)
    cfg = {
        "dataset": dataset,
        "data_root": "datasets",
        "output_root": "outputs",
        "allow_download": False,
        "device": "auto",
        "seed": PAPER_SEEDS[0],
        "split_id": 0,
        "protocol_version": "PR-canonical-undirected-v1",
        "run_kind": "benchmark",
        "method": "sc_full",
        "hidden_dim": 64,
        "num_trans": 1,
        "num_heads": 2,
        "dropout_trans": 0.3,
        "dropout": 0.5,
        "alpha": 0.1,
        "use_bn": True,
        "epochs": 300,
        "patience": 50,
        "eval_every": 1,
        "lr": 0.01,
        "weight_decay": 0.0005,
        "grad_clip": 1.0,
        "primary_metric": PRIMARY_METRIC.get(dataset, "ACC"),
        "selector": {
            "homophily_threshold": 0.4,
            "imbalance_threshold": 30.0,
            "evidence_scale": 100.0,
            "imbalance_temperature": 1.0,
            "reliability_min": 0.5,
            "posterior_confidence": 0.8,
            "imbalance_severity_threshold": 0.75,
            "imbalance_ratio_override_enabled": True,
            "imbalance_ratio_override_threshold": 30.0,
            "uncertain_fallback": "sgc",
            "sc_enabled": True,
            "sc_budget_schedule": [25, 50, 100, 150],
            "sc_decision_min_budget": 150,
            "sc_initial_folds": 2,
            "sc_max_folds": 3,
            "sc_min_valid_folds": 2,
            "sc_min_train_nodes": 20,
            "sc_min_probe_nodes": 5,
            "sc_seed_offset": 271828,
            "sc_confidence_z": 1.645,
            "sc_task_margin": 0.0,
            "sc_risk_margin": 0.0,
        },
        "safe_qk": {
            "enabled": True,
            "lr_scale": 3.0,
            "momentum": 0.95,
            "ns_steps": 5,
            "ns_variant": "polar",
            "start_epoch": 0,
            "nesterov": True,
            "eps": 1e-7,
            "weight_decay": 0.0,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
        },
        "gaussian_consistency": {
            "enabled": True,
            "sigma": 0.1,
            "weight": 0.05,
            "temperature": 1.0,
            "warmup_epochs": 20,
            "every": 2,
            "sample_ratio": 0.5,
            "kind": "kl_stopgrad_clean",
        },
    }
    if dataset in {"computers", "photo", "cs"}:
        cfg.update(hidden_dim=128, lr=0.005)
    elif dataset in {"roman-empire", "questions", "minesweeper", "amazon-ratings", "tolokers"}:
        cfg.update(hidden_dim=128, dropout=0.3, lr=0.001,
                   weight_decay=0.0001, epochs=500, patience=100)
    cfg["dropout_trans"] = cfg["dropout"]
    return cfg


def deep_update(base: dict, updates: dict) -> dict:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(dataset: str, path: str | None = None) -> dict:
    cfg = default_config(dataset)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            cfg = deep_update(cfg, json.load(handle))
    cfg["dataset"] = canonical_dataset_name(cfg["dataset"])
    cfg["primary_metric"] = PRIMARY_METRIC.get(
        cfg["dataset"], cfg.get("primary_metric", "ACC")
    )
    return cfg
