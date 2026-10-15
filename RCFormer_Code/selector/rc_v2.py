"""Reliability-calibrated RC-v2 prior."""

from __future__ import annotations

from config import BRANCH_DEPTH
from .structural_statistics import compute_train_only_statistics


def select_rc_v2(data, split, num_classes, selector_cfg):
    profile = compute_train_only_statistics(
        data, split["train"], num_classes, selector_cfg
    )
    fallback = str(selector_cfg.get("uncertain_fallback", "sgc"))
    if fallback not in BRANCH_DEPTH:
        fallback = "sgc"

    selected = fallback
    reason = f"uncertain_posterior_boundary_fallback_{fallback}"
    uncertainty = "posterior_boundary"
    ratio_override = False
    homophily = profile["edge_homophily_train"]
    train_edges = profile["train_homophily_edges"]
    reliability = profile["evidence_reliability"]
    ratio = profile["class_imbalance_train"]

    if homophily is None or train_edges == 0:
        selected = "sgc"
        reason = "invalid_train_homophily_fallback_sgc"
        uncertainty = "invalid_homophily"
    elif reliability < float(selector_cfg.get("reliability_min", 0.5)):
        selected = "sgc"
        reason = "low_evidence_reliability_fallback_sgc"
        uncertainty = "low_evidence"
    elif (
        bool(selector_cfg.get("imbalance_ratio_override_enabled", True))
        and ratio is not None
        and ratio >= float(selector_cfg.get("imbalance_ratio_override_threshold", 30.0))
    ):
        selected = "optbasis_local"
        reason = "severe_train_class_imbalance_ratio_override_select_optbasis"
        uncertainty = ""
        ratio_override = True
    elif profile["imbalance_severity"] >= float(
        selector_cfg.get("imbalance_severity_threshold", 0.75)
    ):
        selected = "optbasis_local"
        reason = "high_imbalance_severity_select_optbasis"
        uncertainty = ""
    elif profile["homophily_p_low"] is not None and profile["homophily_p_low"] >= float(
        selector_cfg.get("posterior_confidence", 0.8)
    ):
        selected = "fagcn_local"
        reason = "posterior_low_homophily_select_fagcn"
        uncertainty = ""
    elif profile["homophily_p_high"] is not None and profile["homophily_p_high"] >= float(
        selector_cfg.get("posterior_confidence", 0.8)
    ):
        selected = "sgc"
        reason = "posterior_high_homophily_select_sgc"
        uncertainty = ""

    profile.update(
        {
            "selector_type": "reliability_calibrated_v2",
            "selector_uncertain": bool(uncertainty),
            "selector_uncertainty_type": uncertainty,
            "imbalance_ratio_override_triggered": ratio_override,
        }
    )
    return {
        "selected_branch": selected,
        "selected_layers": BRANCH_DEPTH[selected],
        "standalone_alpha": 0.3 if selected == "fagcn_local" else 0.1,
        "reason": reason,
        "profile": profile,
    }
