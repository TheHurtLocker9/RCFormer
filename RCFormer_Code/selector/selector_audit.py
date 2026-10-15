"""Selector audit assembly and leakage assertions."""

from __future__ import annotations

from datetime import datetime, timezone


def fixed_branch_reference_audit(selected_branch, selected_score, fixed_branch_scores):
    """Post-hoc diagnostic only; this function must never enter selection."""
    valid = {
        str(branch): float(score)
        for branch, score in fixed_branch_scores.items()
        if score is not None
    }
    if not valid:
        raise ValueError("fixed-branch audit requires at least one valid score")
    reference_branch = max(valid, key=valid.get)
    reference_score = valid[reference_branch]
    return {
        "selected_branch": str(selected_branch),
        "reference_best_branch": reference_branch,
        "reference_match": str(selected_branch) == reference_branch,
        "selected_score": float(selected_score),
        "reference_score": reference_score,
        "fixed_branch_regret": max(0.0, reference_score - float(selected_score)),
        "participates_in_selection": False,
    }


def assert_train_only_selection(split, sc_audit):
    train = set(int(index) for index in split["train"])
    valid = set(int(index) for index in split["valid"])
    test = set(int(index) for index in split["test"])
    if train & valid or train & test or valid & test:
        raise AssertionError("outer train/validation/test splits overlap")
    if sc_audit.get("uses_validation_labels") or sc_audit.get("uses_test_labels"):
        raise AssertionError("selector reports prohibited validation/test-label use")
    return True


def build_selector_audit(cfg, split, rc_decision, sc_audit):
    assert_train_only_selection(split, sc_audit)
    return {
        "protocol": "RC-v2_then_SC-Efficient",
        "dataset": cfg["dataset"],
        "seed": int(cfg["seed"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_split_sizes": {key: len(split[key]) for key in ("train", "valid", "test")},
        "rc_v2": rc_decision,
        "sc_efficient": sc_audit,
        "final_selected_branch": sc_audit["selected_branch"],
        "leakage_status": "NO_LEAKAGE",
        "selection_uses_validation_or_test": False,
    }
