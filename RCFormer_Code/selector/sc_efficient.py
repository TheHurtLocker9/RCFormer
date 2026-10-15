"""Executable SC-Efficient train-only directional correction."""

from __future__ import annotations

import copy
import gc
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .selector_metrics import (
    class_balanced_nll,
    classification_metrics,
    paired_interval,
    task_metric_with_fallback,
)


@dataclass
class CandidateState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    trained_epochs: int = 0
    python_rng_state: object = None
    numpy_rng_state: object = None
    torch_rng_state: torch.Tensor | None = None
    cuda_rng_state: list | None = None


def challenger_policy(rc_decision):
    incumbent = rc_decision["selected_branch"]
    reason = rc_decision["reason"]
    protected = (
        "invalid_train_homophily" in reason
        or "low_evidence_reliability" in reason
        or "posterior_low_homophily" in reason
    )
    if protected:
        return incumbent, None, None, "protected_rc_v2_decision"
    if incumbent == "optbasis_local" and (
        "imbalance_ratio_override" in reason or "high_imbalance_severity" in reason
    ):
        return incumbent, "fagcn_local", "imbalance_safeguard", ""
    if incumbent == "sgc" and "posterior_high_homophily" in reason:
        return incumbent, "fagcn_local", "posterior_high_homophily", ""
    if incumbent == "sgc" and "uncertain_posterior_boundary" in reason:
        return incumbent, "fagcn_local", "posterior_boundary", ""
    return incumbent, None, None, "decision_not_in_sc_scope"


def make_inner_splits(labels, train_indices, cfg, seed):
    items = [int(index) for index in train_indices]
    folds = max(2, int(cfg.get("sc_max_folds", 3)))
    if len(items) < int(cfg.get("sc_min_train_nodes", 20)) + int(
        cfg.get("sc_min_probe_nodes", 5)
    ):
        return []
    rng = random.Random(int(seed))
    fold_probes = [[] for _ in range(folds)]
    always_train = []
    by_class = {}
    for index in items:
        by_class.setdefault(int(labels[index]), []).append(index)
    for class_offset, class_id in enumerate(sorted(by_class)):
        group = list(by_class[class_id])
        rng.shuffle(group)
        if len(group) <= 1:
            always_train.extend(group)
            continue
        for position, index in enumerate(group):
            fold_probes[(position + class_offset) % folds].append(index)
    all_items = set(items)
    result = []
    for fold_index, probe in enumerate(fold_probes):
        inner_probe = sorted(set(probe))
        inner_train = sorted(all_items.difference(inner_probe).union(always_train))
        if len(inner_train) < int(cfg.get("sc_min_train_nodes", 20)):
            continue
        if len(inner_probe) < int(cfg.get("sc_min_probe_nodes", 5)):
            continue
        result.append(
            {
                "fold_index": fold_index,
                "inner_train": inner_train,
                "inner_probe": inner_probe,
            }
        )
    return result


def run_sc_efficient(cfg, data, split, rc_decision, model_factory, device):
    sc_cfg = cfg["selector"]
    incumbent, challenger, trigger, skip_reason = challenger_policy(rc_decision)
    audit = {
        "strategy": "maturity_aligned_sequential",
        "triggered": challenger is not None,
        "incumbent": incumbent,
        "challenger": challenger,
        "trigger_type": trigger,
        "selected_branch": incumbent,
        "switched": False,
        "skip_reason": skip_reason,
        "uses_train_labels": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
        "budget_schedule": list(sc_cfg.get("sc_budget_schedule", [25, 50, 100, 150])),
        "decision_min_budget": int(sc_cfg.get("sc_decision_min_budget", 150)),
        "checkpoints": {},
        "metric_fallbacks": [],
    }
    if not bool(sc_cfg.get("sc_enabled", True)) or challenger is None:
        return audit

    labels_cpu = data.y.view(-1).detach().cpu().tolist()
    split_seed = int(cfg["seed"]) + int(sc_cfg.get("sc_seed_offset", 271828))
    folds = make_inner_splits(labels_cpu, split["train"], sc_cfg, split_seed)
    outer_train = set(int(index) for index in split["train"])
    for fold in folds:
        inner_train = set(fold["inner_train"])
        inner_probe = set(fold["inner_probe"])
        if inner_train & inner_probe or not (inner_train | inner_probe) <= outer_train:
            raise AssertionError("SC-Efficient inner split escaped the outer training set")
    audit["split_seed"] = split_seed
    audit["total_valid_folds"] = len(folds)
    audit["fold_audit"] = [
        {
            "fold_index": int(fold["fold_index"]),
            "inner_train_size": len(fold["inner_train"]),
            "inner_probe_size": len(fold["inner_probe"]),
            "outer_train_only": True,
        }
        for fold in folds
    ]
    minimum = int(sc_cfg.get("sc_min_valid_folds", 2))
    if len(folds) < minimum:
        audit["skip_reason"] = "insufficient_valid_train_only_folds"
        audit["failed"] = True
        return audit

    schedule = sorted({int(value) for value in audit["budget_schedule"]})
    decision_budget = int(audit["decision_min_budget"])
    if decision_budget not in schedule:
        schedule.append(decision_budget)
        schedule.sort()
    active = min(len(folds), int(sc_cfg.get("sc_initial_folds", 2)))
    states = {}
    evidence = {}

    def ensure_pair(fold_position):
        if (fold_position, incumbent) in states:
            return
        fold = folds[fold_position]
        pair_seed = split_seed + 1009 * int(fold["fold_index"]) + fold_position
        _seed_everything(pair_seed)
        shared_model = model_factory(incumbent).cpu()
        shared_state = copy.deepcopy(shared_model.state_dict())
        del shared_model
        for branch in (incumbent, challenger):
            _seed_everything(pair_seed)
            model = model_factory(branch)
            own_state = model.state_dict()
            compatible = {
                key: value
                for key, value in shared_state.items()
                if key in own_state and tuple(own_state[key].shape) == tuple(value.shape)
            }
            own_state.update(compatible)
            model.load_state_dict(own_state)
            model = model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            states[(fold_position, branch)] = CandidateState(
                model=model,
                optimizer=optimizer,
                python_rng_state=random.getstate(),
                numpy_rng_state=np.random.get_state(),
                torch_rng_state=torch.get_rng_state().clone(),
                cuda_rng_state=(
                    [item.clone() for item in torch.cuda.get_rng_state_all()]
                    if device.type == "cuda"
                    else []
                ),
            )

    def train_active(target_epochs):
        for fold_position in range(active):
            ensure_pair(fold_position)
            fold = folds[fold_position]
            for branch in (incumbent, challenger):
                state = states[(fold_position, branch)]
                evidence[(fold_position, branch)] = _train_to_budget(
                    state,
                    data,
                    fold["inner_train"],
                    fold["inner_probe"],
                    target_epochs,
                    cfg["primary_metric"],
                    device,
                )
                fallback = evidence[(fold_position, branch)]["metric_fallback_reason"]
                if fallback:
                    audit["metric_fallbacks"].append(
                        f"fold={fold_position},branch={branch}:{fallback}"
                    )

    final_state = "unresolved"
    final_reason = "not_evaluated"
    for budget in schedule:
        train_active(budget)
        summary = _summarize_checkpoint(
            evidence,
            active,
            incumbent,
            challenger,
            float(sc_cfg.get("sc_confidence_z", 1.645)),
        )
        audit["checkpoints"][str(budget)] = summary
        if budget < decision_budget:
            continue
        final_state, final_reason = directional_decision(
            trigger,
            summary["metric_difference"],
            summary["risk_difference"],
            float(sc_cfg.get("sc_task_margin", 0.0)),
            float(sc_cfg.get("sc_risk_margin", 0.0)),
        )
        if final_state == "unresolved" and active < len(folds):
            active += 1
            train_active(budget)
            summary = _summarize_checkpoint(
                evidence,
                active,
                incumbent,
                challenger,
                float(sc_cfg.get("sc_confidence_z", 1.645)),
            )
            audit["checkpoints"][str(budget)] = summary
            final_state, final_reason = directional_decision(
                trigger,
                summary["metric_difference"],
                summary["risk_difference"],
                float(sc_cfg.get("sc_task_margin", 0.0)),
                float(sc_cfg.get("sc_risk_margin", 0.0)),
            )
            audit["third_fold_activated"] = True
        break

    selected = challenger if final_state == "switch" else incumbent
    audit.update(
        {
            "active_folds": active,
            "provisional_decision": final_state,
            "final_decision": "switch" if final_state == "switch" else "retain",
            "decision_reason": final_reason,
            "selected_branch": selected,
            "switched": selected == challenger,
            "unresolved_mapped_to_retain": final_state == "unresolved",
            "candidate_epochs": {
                branch: sum(
                    state.trained_epochs
                    for (fold_position, name), state in states.items()
                    if name == branch and fold_position < active
                )
                for branch in (incumbent, challenger)
            },
        }
    )
    states.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return audit


def directional_decision(trigger, metric_interval, risk_interval, task_margin=0.0, risk_margin=0.0):
    if trigger == "imbalance_safeguard":
        if risk_interval["upper"] is not None and risk_interval["upper"] < -risk_margin:
            return "switch", "imbalance_risk_interval_favors_challenger"
        if risk_interval["lower"] is not None and risk_interval["lower"] > risk_margin:
            return "retain", "imbalance_risk_interval_favors_incumbent"
        return "unresolved", "imbalance_risk_interval_unresolved"
    metric_mean = metric_interval["mean"]
    risk_mean = risk_interval["mean"]
    if metric_mean is None or risk_mean is None:
        return "unresolved", "missing_paired_evidence"
    if metric_mean > task_margin and risk_mean < -risk_margin:
        return "switch", "homophily_task_and_risk_favor_challenger"
    if metric_mean < -task_margin and risk_mean > risk_margin:
        return "retain", "homophily_task_and_risk_favor_incumbent"
    return "unresolved", "homophily_task_risk_conflict"


def _train_to_budget(state, data, inner_train, inner_probe, target_epochs, primary_metric, device):
    train_idx = torch.as_tensor(inner_train, dtype=torch.long, device=device)
    probe_idx = torch.as_tensor(inner_probe, dtype=torch.long, device=device)
    graph = data.to(device)
    labels = graph.y.view(-1).long()
    random.setstate(state.python_rng_state)
    np.random.set_state(state.numpy_rng_state)
    torch.set_rng_state(state.torch_rng_state)
    if device.type == "cuda" and state.cuda_rng_state:
        torch.cuda.set_rng_state_all(state.cuda_rng_state)
    for _ in range(state.trained_epochs, int(target_epochs)):
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        log_probs = state.model(graph.x, graph.edge_index)
        loss = F.nll_loss(log_probs[train_idx], labels[train_idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite candidate loss')
        loss.backward()
        state.optimizer.step()
    state.trained_epochs = int(target_epochs)
    state.python_rng_state = random.getstate()
    state.numpy_rng_state = np.random.get_state()
    state.torch_rng_state = torch.get_rng_state().clone()
    state.cuda_rng_state = (
        [item.clone() for item in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else []
    )
    state.model.eval()
    with torch.no_grad():
        log_probs = state.model(graph.x, graph.edge_index)
        risk = float(class_balanced_nll(log_probs, labels, probe_idx).item())
        metrics = classification_metrics(log_probs, labels, probe_idx)
    metric, effective_metric, fallback = task_metric_with_fallback(
        metrics, primary_metric
    )
    if metric is None or not math.isfinite(risk):
        raise FloatingPointError('non-finite candidate evidence')
    return {
        "risk": risk,
        "metric": metric,
        "metric_name": effective_metric,
        "metric_fallback_reason": fallback,
    }


def _summarize_checkpoint(evidence, active, incumbent, challenger, z):
    incumbent_risks, challenger_risks = [], []
    incumbent_metrics, challenger_metrics = [], []
    for fold_position in range(active):
        left = evidence[(fold_position, incumbent)]
        right = evidence[(fold_position, challenger)]
        incumbent_risks.append(left["risk"])
        challenger_risks.append(right["risk"])
        incumbent_metrics.append(left["metric"])
        challenger_metrics.append(right["metric"])
    risk_differences = [
        right - left for left, right in zip(incumbent_risks, challenger_risks)
    ]
    metric_differences = [
        right - left for left, right in zip(incumbent_metrics, challenger_metrics)
    ]
    return {
        "active_folds": active,
        "incumbent_risks": incumbent_risks,
        "challenger_risks": challenger_risks,
        "incumbent_metrics": incumbent_metrics,
        "challenger_metrics": challenger_metrics,
        "risk_difference": paired_interval(risk_differences, z),
        "metric_difference": paired_interval(metric_differences, z),
    }


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
