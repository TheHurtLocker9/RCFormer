"""Evaluate a saved RCFormer checkpoint on its frozen outer split."""

from __future__ import annotations

import argparse
import json

import torch

from data.dataset_loader import load_dataset
from data.split import build_split
from training.evaluator import evaluate_all
from training.trainer import model_factory, resolve_device
from provenance import tensor_hash, json_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["config"])
    if cfg.get('protocol_version') != 'PR-canonical-undirected-v1':
        raise ValueError('legacy checkpoint: evaluate separately, not as a corrected benchmark')
    cfg.update(
        {
            "data_root": args.data_root,
            "device": args.device,
            "allow_download": args.allow_download,
        }
    )
    bundle = load_dataset(cfg["dataset"], args.data_root, args.allow_download)
    split = build_split(bundle, int(checkpoint['seed']), cfg['split_id'])
    identity = checkpoint['identity']
    if json_hash(split) != identity['split_sha256'] or tensor_hash(bundle.data.edge_index) != identity['message_edges_sha256']:
        raise ValueError('checkpoint graph or split differs from evaluation data')
    if tensor_hash(bundle.data.x) != identity['features_sha256'] or tensor_hash(bundle.data.y) != identity['labels_sha256']:
        raise ValueError('checkpoint feature or label identity differs')
    model = model_factory(cfg, bundle)(checkpoint["branch"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    data = bundle.data.to(device)
    metrics = evaluate_all(model, data, split, cfg["primary_metric"])
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
