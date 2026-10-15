"""Train a fixed RCFormer local branch under the final-model protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BRANCH_DEPTH, load_config
from training.trainer import train_fixed_branch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--branch", required=True, choices=sorted(BRANCH_DEPTH))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument('--split-id', type=int, required=True)
    parser.add_argument('--run-kind', choices=['benchmark', 'engineering_pilot'], default='benchmark')
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="outputs/fixed_branch")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.dataset, args.config)
    cfg.update(
        {
            "seed": args.seed,
            'split_id': args.split_id,
            'run_kind': args.run_kind,
            'method': 'fixed_' + args.branch,
            "data_root": args.data_root,
            "device": args.device,
            "allow_download": args.allow_download,
        }
    )
    output = (Path(args.output_root) / args.run_kind / cfg['method'] / cfg['dataset'] /
              f'split_{args.split_id:02d}' / f'seed_{args.seed}')
    result = train_fixed_branch(cfg, branch=args.branch, output_dir=output)
    print(json.dumps({key: result[key] for key in ("run_dir", "branch", "best_epoch", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
