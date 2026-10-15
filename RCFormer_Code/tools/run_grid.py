"""Preview or execute the paper's backbone condition grid."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import deep_update


def conditions(datasets=None, limit=None):
    values = json.loads((ROOT / "protocol/conditions_80.json").read_text(encoding="utf-8"))
    if datasets:
        unknown = set(datasets) - {row["dataset"] for row in values}
        if unknown:
            raise ValueError("Unknown datasets: " + ", ".join(sorted(unknown)))
        values = [row for row in values if row["dataset"] in datasets]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        values = values[:limit]
    return values


def resolve_config(dataset, variant):
    base = json.loads((ROOT / "presets" / (dataset + ".json")).read_text(encoding="utf-8"))
    delta = json.loads((ROOT / "presets/variants" / (variant + ".json")).read_text(encoding="utf-8"))
    return deep_update(base, delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--variants", nargs="+", default=["sc_full"], choices=["sc_only", "sc_qk", "sc_full"])
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = conditions(args.datasets, args.limit)
    print(json.dumps({"conditions_per_variant": len(rows), "fits": len(rows)*len(args.variants),
                      "execute": args.execute}))
    for row in rows:
        for variant in args.variants:
            cfg = resolve_config(row["dataset"], variant)
            config_path = Path(args.output_root).resolve() / "configs" / (row["dataset"] + "__" + variant + ".json")
            command = [sys.executable, str(ROOT / "main.py"), "--dataset", row["dataset"],
                       "--seed", str(row["seed"]), "--split-ids", str(row["split_id"]),
                       "--config", str(config_path), "--data-root", str(Path(args.data_root).resolve()),
                       "--output-root", str(Path(args.output_root).resolve()), "--device", args.device]
            print(json.dumps({"condition": row, "variant": variant, "command": command}))
            if args.execute:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
