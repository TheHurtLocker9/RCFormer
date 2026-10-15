"""Fixed-maturity, same-evidence policy diagnostic; not deployed SC allocation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import BRANCH_DEPTH, default_config
from data.dataset_loader import load_dataset, HETEROPHILY
from data.preprocessing import train_only_view
from data.split import build_split
from provenance import claim_output, json_hash, run_identity, tensor_hash
from selector.rc_v2 import select_rc_v2
from selector.sc_efficient import (CandidateState, _train_to_budget, _summarize_checkpoint,
                                   challenger_policy, directional_decision, make_inner_splits)
from training.trainer import model_factory, seed_everything, resolve_device, write_json

BRANCHES = tuple(BRANCH_DEPTH)
POLICIES = ('full', 'no_evidence_reliability', 'no_homophily_confidence',
            'no_class_skew', 'empirical_homophily_with_same_cap', 'three_way')


def prior_for_policy(base, cfg, graph, split, classes, policy):
    selector = copy.deepcopy(cfg['selector'])
    if policy == 'no_evidence_reliability':
        selector['reliability_min'] = 0.0
        return select_rc_v2(graph, split, classes, selector)
    if policy == 'no_class_skew':
        selector['imbalance_ratio_override_enabled'] = False
        selector['imbalance_severity_threshold'] = 2.0
        return select_rc_v2(graph, split, classes, selector)
    result = copy.deepcopy(base)
    if policy not in ('no_homophily_confidence', 'empirical_homophily_with_same_cap'):
        return result
    h = base['profile']['edge_homophily_train']
    if h is None:
        return result
    if policy == 'no_homophily_confidence' and (
        'low_evidence_reliability' in base['reason'] or base['selected_branch'] == 'optbasis_local'
    ):
        return result
    high = h >= selector['homophily_threshold']
    branch = 'sgc' if high else 'fagcn_local'
    result.update(selected_branch=branch, selected_layers=BRANCH_DEPTH[branch],
                  reason='posterior_high_homophily_select_sgc' if high else 'posterior_low_homophily_select_fagcn')
    result['control_reason'] = 'empirical_homophily_not_posterior_confidence'
    return result


def replay_policy(prior, evidence, cfg, policy):
    if policy == 'three_way':
        means = {b: float(np.mean([evidence[(f, b)]['metric'] for f in range(2)])) for b in BRANCHES}
        chosen = max(BRANCHES, key=lambda b: (means[b], -BRANCHES.index(b)))
        return {'selected_branch': chosen, 'permitted_evidence': list(BRANCHES),
                'score_means': means, 'candidate_evidence_cells_read': 6,
                'tie_break': 'prespecified_sgc_fagcn_optbasis_order'}
    incumbent, challenger, trigger, skip = challenger_policy(prior)
    if challenger is None:
        return {'selected_branch': incumbent, 'permitted_evidence': [],
                'candidate_evidence_cells_read': 0, 'skip_reason': skip}
    permitted = {(f, b): evidence[(f, b)] for f in range(2) for b in (incumbent, challenger)}
    summary = _summarize_checkpoint(permitted, 2, incumbent, challenger,
                                    cfg['selector']['sc_confidence_z'])
    choice, reason = directional_decision(trigger, summary['metric_difference'],
        summary['risk_difference'], cfg['selector']['sc_task_margin'], cfg['selector']['sc_risk_margin'])
    return {'selected_branch': challenger if choice == 'switch' else incumbent,
            'incumbent': incumbent, 'challenger': challenger, 'trigger': trigger,
            'permitted_evidence': [incumbent, challenger], 'candidate_evidence_cells_read': 4,
            'provisional_decision': choice, 'reason': reason, 'summary': summary}


def build_bank(cfg, bundle, split, output, repeat=0, epochs=150, engineering_pilot=False):
    if epochs != 150 and not engineering_pilot:
        raise ValueError('Diagnostic maturity is prespecified at 150; short runs require engineering_pilot')
    cfg = copy.deepcopy(cfg)
    cfg['safe_qk']['enabled'] = False
    cfg['gaussian_consistency']['enabled'] = False
    cfg['method'] = 'shared_bank'
    cfg['run_kind'] = 'engineering_pilot' if engineering_pilot else 'shared_evidence_diagnostic'
    cfg['bank_repeat'] = repeat
    cfg['bank_epochs'] = epochs
    identity = run_identity(cfg, bundle, split)
    path = claim_output(output, identity)
    write_json(path / 'resolved_config.json', cfg)
    graph = train_only_view(bundle.data, split['train'])
    select_split = {'train': split['train']}
    base = select_rc_v2(graph, select_split, bundle.num_classes, cfg['selector'])
    split_seed = cfg['seed'] + cfg['selector']['sc_seed_offset']
    # Preserve the main policy's three-part partition; use only its first two folds.
    folds = make_inner_splits(graph.y.tolist(), split['train'], cfg['selector'], split_seed)[:2]
    if len(folds) != 2:
        raise ValueError('Two valid inner folds required; never invent missing folds')
    for fold in folds:
        fit, probe = set(fold['inner_train']), set(fold['inner_probe'])
        if fit & probe or not (fit | probe) <= set(split['train']):
            raise AssertionError('inner folds escaped the original training domain')
    device = resolve_device(cfg['device'])
    factory = model_factory(cfg, bundle)
    start = time.perf_counter()
    evidence = {}
    cells = []
    for position, fold in enumerate(folds):
        candidate_seed = split_seed + 1009 * fold['fold_index'] + position + 104729 * repeat
        seed_everything(candidate_seed)
        anchor = factory('sgc')
        shared = copy.deepcopy(anchor.state_dict())
        del anchor
        for branch in BRANCHES:
            seed_everything(candidate_seed)
            model = factory(branch)
            own = model.state_dict()
            shared_keys = [k for k in shared if k in own and own[k].shape == shared[k].shape]
            own.update({k: shared[k] for k in shared_keys})
            model.load_state_dict(own)
            shared_init = json_hash({k: tensor_hash(model.state_dict()[k]) for k in shared_keys})
            model = model.to(device)
            state = CandidateState(model, torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                weight_decay=cfg['weight_decay']), python_rng_state=random.getstate(),
                numpy_rng_state=np.random.get_state(), torch_rng_state=torch.get_rng_state().clone(),
                cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            score = _train_to_budget(state, graph, fold['inner_train'], fold['inner_probe'],
                                      epochs, cfg['primary_metric'], device)
            evidence[(position, branch)] = score
            cell = {'fold': position, 'branch': branch, 'seed': candidate_seed, 'epochs': epochs,
                    'local_layers': BRANCH_DEPTH[branch], 'score': score,
                    'shared_initialization_sha256': shared_init,
                    'checkpoint_sha256': json_hash({k: tensor_hash(v) for k, v in model.state_dict().items()})}
            cells.append(cell)
            print(json.dumps({'bank_cell_complete': branch, 'fold': position, 'epochs': epochs}), flush=True)
            del model, state
    policies = {}
    for policy in POLICIES:
        prior = prior_for_policy(base, cfg, graph, select_split, bundle.num_classes, policy)
        policies[policy] = replay_policy(prior, evidence, cfg, policy)
        policies[policy]['prior'] = prior
    result = {'status': 'COMPLETE', 'identity': identity, 'candidate_cells': cells,
              'folds': folds, 'policies': policies, 'bank_sha256': json_hash(cells),
              'physical_candidate_epochs': 6 * epochs, 'wall_seconds': time.perf_counter() - start,
              'held_out_labels_masked': True, 'outer_performance_evaluated': False,
              'auxiliaries_enabled_in_candidates': False,
              'scope': 'same-evidence rule diagnostic; not deployed allocation cost or test regret',
              'shared_anchor': 'sgc-compatible weights; operator depths remain 2/4/1',
              'third_fold': 'not available to any replay policy'}
    write_json(path / 'bank.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=sorted(HETEROPHILY))
    parser.add_argument('--split-ids', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--engineering-pilot', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    torch.set_num_threads(4)
    for name in args.datasets:
        bundle = load_dataset(name, args.data_root)
        for split_id in args.split_ids:
            cfg = default_config(name)
            cfg.update(seed=2024 + split_id, split_id=split_id, data_root=args.data_root, device=args.device)
            split = build_split(bundle, cfg['seed'], split_id)
            for repeat in range(args.repeats):
                kind = 'engineering_pilot' if args.engineering_pilot else 'shared_evidence_diagnostic'
                output = Path(args.output_root) / kind / cfg['dataset'] / f'split_{split_id:02d}' / f'repeat_{repeat}'
                result = build_bank(cfg, bundle, split, output, repeat, args.epochs, args.engineering_pilot)
                print(json.dumps({'bank_complete': str(output), 'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
