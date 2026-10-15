"""RCFormer selection and final-training orchestration."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from config import BRANCH_DEPTH
from data.dataset_loader import load_dataset
from data.split import build_split
from data.preprocessing import train_only_view
from provenance import claim_output, run_identity
from models.rcformer import RCFormer
from optimization.safe_qk import build_safe_qk_optimizer
from selector.rc_v2 import select_rc_v2
from selector.sc_efficient import run_sc_efficient
from selector.selector_audit import build_selector_audit

from .early_stopping import EarlyStopping
from .evaluator import evaluate_all, evaluate_split
from .losses import final_training_loss


def run_pipeline(cfg, audit_only=False):
    bundle = load_dataset(
        cfg["dataset"], cfg["data_root"], bool(cfg.get("allow_download", False))
    )
    split = build_split(bundle, int(cfg["seed"]), cfg["split_id"])
    run_dir = (Path(cfg['output_root']) / cfg['run_kind'] / cfg['method'] /
               cfg['dataset'] / f"split_{cfg['split_id']:02d}" / f"seed_{cfg['seed']}")
    identity = run_identity(cfg, bundle, split)
    claim_output(run_dir, identity)
    device = resolve_device(cfg.get("device", "auto"))
    factory = model_factory(cfg, bundle)
    selector_data = train_only_view(bundle.data, split['train'])
    selector_split = {'train': split['train']}
    rc_decision = select_rc_v2(selector_data, selector_split, bundle.num_classes, cfg['selector'])
    sc_audit = run_sc_efficient(
        cfg, selector_data, selector_split, rc_decision, factory, device
    )
    audit = build_selector_audit(cfg, split, rc_decision, sc_audit)
    audit['identity'] = identity
    audit['held_out_labels_masked'] = True
    del selector_data
    write_json(run_dir / "resolved_config.json", cfg)
    write_json(run_dir / "selector_audit.json", audit)
    if audit_only:
        return {"run_dir": str(run_dir), "selector_audit": audit}

    branch = audit["final_selected_branch"]
    result = train_fixed_branch(
        cfg,
        bundle=bundle,
        split=split,
        branch=branch,
        output_dir=run_dir,
        claimed_identity=identity,
    )
    result["selector_audit"] = audit
    return result


def train_fixed_branch(cfg, bundle=None, split=None, branch=None, output_dir=None, claimed_identity=None):
    bundle = bundle or load_dataset(
        cfg["dataset"], cfg["data_root"], bool(cfg.get("allow_download", False))
    )
    split = split or build_split(bundle, int(cfg["seed"]), cfg['split_id'])
    branch = branch or cfg.get("fixed_branch")
    if branch not in BRANCH_DEPTH:
        raise ValueError(f"unsupported fixed branch: {branch}")
    output_dir = Path(output_dir or cfg["output_root"]) 
    identity = claimed_identity or run_identity(cfg, bundle, split)
    if claimed_identity is None:
        claim_output(output_dir, identity)
    write_json(output_dir / "resolved_config.json", cfg)
    device = resolve_device(cfg.get("device", "auto"))
    seed_everything(int(cfg["seed"]))
    model = model_factory(cfg, bundle)(branch).to(device)
    data = bundle.data.to(device)
    train_idx = torch.as_tensor(split["train"], dtype=torch.long, device=device)

    if bool(cfg["safe_qk"].get("enabled", True)):
        optimizer = build_safe_qk_optimizer(model, cfg)
        optimizer_name = "SafeQKWithAuxAdam"
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        optimizer_name = "Adam"
    stopper = EarlyStopping(cfg["patience"])
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(optimizer, "set_epoch"):
            optimizer.set_epoch(epoch)
        loss, loss_parts = final_training_loss(
            model,
            data,
            train_idx,
            epoch,
            cfg["gaussian_consistency"],
            device,
            int(cfg["seed"]),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite final-training loss')
        loss.backward()
        if cfg.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimizer.step()
        if (epoch + 1) % int(cfg.get("eval_every", 1)) != 0:
            continue
        valid = evaluate_split(model, data, split["valid"], cfg["primary_metric"])
        if not np.isfinite(valid['NLL']):
            raise FloatingPointError('non-finite validation loss')
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach().item()),
                "valid_nll": valid["NLL"],
                **loss_parts,
            }
        )
        if stopper.update(valid["NLL"], model, epoch + 1):
            break
    stopper.restore(model)
    metrics = evaluate_all(model, data, split, cfg["primary_metric"])
    checkpoint = {
        'identity': identity,
        'outer_split': split,
        "model_state": model.state_dict(),
        "config": cfg,
        "dataset": cfg["dataset"],
        "seed": int(cfg["seed"]),
        "branch": branch,
        "local_layers": BRANCH_DEPTH[branch],
        "num_features": bundle.num_features,
        "num_classes": bundle.num_classes,
        "best_epoch": stopper.best_epoch,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    result = {
        'identity': identity,
        'status': 'COMPLETE',
        "run_dir": str(output_dir),
        "branch": branch,
        "local_layers": BRANCH_DEPTH[branch],
        "optimizer": optimizer_name,
        "best_epoch": stopper.best_epoch,
        "best_validation_nll": stopper.best_value,
        "metrics": metrics,
        "history": history,
    }
    write_json(output_dir / "metrics.json", result)
    return result


def model_factory(cfg, bundle):
    def create(branch):
        return RCFormer(
            input_dim=bundle.num_features,
            hidden_dim=int(cfg["hidden_dim"]),
            output_dim=bundle.num_classes,
            branch=branch,
            local_layers=BRANCH_DEPTH[branch],
            num_trans=int(cfg["num_trans"]),
            num_heads=int(cfg["num_heads"]),
            dropout_trans=float(cfg["dropout_trans"]),
            dropout=float(cfg["dropout"]),
            alpha=float(cfg['alpha']),
            use_bn=bool(cfg["use_bn"]),
        )

    return create


def resolve_device(requested):
    requested = str(requested)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
