"""Run the complete RCFormer operator-selection and final-training pipeline."""

from __future__ import annotations

import argparse
import copy
import json

from config import PAPER_SEEDS, load_config
from data.dataset_loader import HETEROPHILY
from training.trainer import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--config", default=None, help="optional JSON override")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument('--split-ids', type=int, nargs='+', default=None)
    parser.add_argument('--run-kind', choices=['benchmark', 'engineering_pilot'], default='benchmark')
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base = load_config(args.dataset, args.config)
    results = []
    split_ids = args.split_ids or (list(range(10)) if base['dataset'] in HETEROPHILY else [0])
    for split_id in split_ids:
      seeds = args.seeds or ([args.seed] if args.seed is not None else
                            [2024 + split_id] if base['dataset'] in HETEROPHILY else PAPER_SEEDS)
      for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg.update(
            {
                "seed": int(seed),
                'split_id': split_id,
                'run_kind': args.run_kind,
                "data_root": args.data_root,
                "output_root": args.output_root,
                "device": args.device,
                "allow_download": bool(args.allow_download),
            }
        )
        result = run_pipeline(cfg, audit_only=args.audit_only)
        results.append(
            {
                "seed": int(seed),
                'split_id': split_id,
                "run_dir": result["run_dir"],
                "selected_branch": result["selector_audit"]["final_selected_branch"],
                "test": (result.get("metrics") or {}).get("test"),
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
