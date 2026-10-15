"""Recompute the principal archived comparison and selector summaries."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD_NAMES = {
    "sc_full": "RCFormer-SC-Full", "baseline_gcn": "GCN", "baseline_gat": "GAT",
    "baseline_graphsage": "GraphSAGE", "baseline_appnp": "APPNP", "baseline_sgc": "SGC",
    "baseline_h2gcn_official": "H2GCN", "baseline_gprgnn_official": "GPR-GNN",
    "baseline_fagcn_official": "FAGCN", "baseline_linkx_official": "LINKX",
    "baseline_acmgcn_official": "ACM-GCN", "baseline_orderedgnn_official": "OrderedGNN",
    "baseline_gcn_renode_official": "GCN+ReNode", "baseline_graphgps": "GraphGPS",
    "baseline_nodeformer": "NodeFormer", "baseline_nagphormer": "NAGphormer",
}


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(study):
    records = rows(study / "per_run_1680.csv")
    groups = defaultdict(list)
    keys = set()
    for row in records:
        key = (row["dataset"], row["method"], int(row["split_id"]), int(row["seed"]))
        if key in keys:
            raise ValueError("Duplicate condition: " + repr(key))
        keys.add(key)
        groups[key[:2]].append(float(row["score"]))
    if len(records) != 1680:
        raise ValueError("Expected the complete 1,680-condition archive")
    cells = {}
    datasets = sorted({key[0] for key in groups})
    if len(datasets) != 11:
        raise ValueError("Expected eleven datasets")
    for (dataset, method), values in groups.items():
        expected = 10 if dataset in {"roman-empire", "questions", "minesweeper", "amazon-ratings", "tolokers"} else 5
        if len(values) != expected:
            raise ValueError("Incomplete method-dataset cell")
        cells[(dataset, method)] = {"mean": statistics.mean(values), "sd": statistics.stdev(values), "n": len(values)}
    ranks = defaultdict(list)
    full_ranks = {}
    for dataset in datasets:
        means = {method: cells[(dataset, method)]["mean"] for method in METHOD_NAMES}
        for method, mean in means.items():
            rank = 1 + sum(value > mean for value in means.values()) + (sum(value == mean for value in means.values())-1)/2
            ranks[METHOD_NAMES[method]].append(rank)
            if method == "sc_full":
                full_ranks[dataset] = rank
    rank_means = {method: statistics.mean(values) for method, values in ranks.items()}
    result = {"conditions": len(records), "main_comparison_cells": 11*len(METHOD_NAMES),
              "average_ranks": dict(sorted(rank_means.items(), key=lambda pair: pair[1])),
              "full_ranks": full_ranks, "full_top_three": sum(rank <= 3 for rank in full_ranks.values()),
              "source_fingerprints": dict(Counter(row["source_sha256"] for row in records)),
              "progressive": {"qk_minus_only_positive_datasets": sum(cells[(d, "sc_qk")]["mean"] > cells[(d, "sc_only")]["mean"] for d in datasets),
                              "full_minus_qk_positive_datasets": sum(cells[(d, "sc_full")]["mean"] > cells[(d, "sc_qk")]["mean"] for d in datasets)}}
    expected = rows(study / "baseline_cells_176.csv")
    reverse = {value: key for key, value in METHOD_NAMES.items()}
    if len(expected) != 176:
        raise ValueError("Expected 176 archived comparison cells")
    for row in expected:
        cell = cells[(row["dataset"], reverse[row["method"]])]
        if abs(cell["mean"]-float(row["new_mean"])) > 1e-12 or abs(cell["sd"]-float(row["new_sd"])) > 1e-12:
            raise ValueError("Archived summary differs from condition records")
    result["archived_cells_verified"] = len(expected)
    selections = rows(study / "selection_diagnostics_80.csv")
    if len(selections) != 80:
        raise ValueError("Expected eighty selector conditions")
    result["selector"] = {
        "conditions": len(selections),
        "prior_hits": sum(row["prior_match"] == "True" for row in selections),
        "final_hits": sum(row["final_match"] == "True" for row in selections),
        "triggered": sum(row["triggered"] == "True" for row in selections),
        "switched": sum(row["switched"] == "True" for row in selections),
        "final_branch_counts": dict(Counter(row["final"] for row in selections)),
    }
    retention = rows(study / "gaussian_retention_11.csv")
    if len(retention) != 11:
        raise ValueError("Expected eleven Gaussian summaries")
    result["gaussian_retention"] = {
        "qk_equal_dataset_mean": statistics.mean(float(row["qk_retention"]) for row in retention),
        "full_equal_dataset_mean": statistics.mean(float(row["full_retention"]) for row in retention),
        "positive_datasets": sum(float(row["delta"]) > 0 for row in retention),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "study")
    args = parser.parse_args()
    print(json.dumps(summarize(args.study), indent=2, sort_keys=True))
