"""Small, self-contained checks of the reported backbone and protocol."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from config import BRANCH_DEPTH, PRIMARY_METRIC, default_config
from data.dataset_loader import DatasetBundle
from data.preprocessing import apply_graph_protocol, train_only_view
from data.split import build_split, select_mask
from models.rcformer import RCFormer
from optimization.safe_qk import build_safe_qk_optimizer, zeropower_via_newton_schulz
from regularization.gaussian_consistency import consistency_loss, consistency_weight
from selector.rc_v2 import select_rc_v2
from selector.sc_efficient import make_inner_splits, run_sc_efficient
from shared_bank import build_bank
from tools.run_grid import conditions, resolve_config
from tools.summarize import summarize
from training.trainer import model_factory, train_fixed_branch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def fixture():
    generator = torch.Generator().manual_seed(7)
    data = Data(x=torch.randn(60, 5, generator=generator), y=torch.arange(60) % 2,
                edge_index=torch.stack((torch.arange(59), torch.arange(1, 60))))
    apply_graph_protocol(data, True)
    split = {"train": list(range(40)), "valid": list(range(40, 50)), "test": list(range(50, 60))}
    masks = {key: torch.zeros(60, 10, dtype=torch.bool) for key in split}
    for index in range(10):
        for key, nodes in split.items():
            masks[key][[(node + index) % 60 for node in nodes], index] = True
    return DatasetBundle("minesweeper", data, 5, 2, masks), split


def temporary_output():
    parent = ROOT / "outputs/tests"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


class SubmissionTests(unittest.TestCase):
    def test_all_eleven_presets_match_frozen_defaults(self):
        files = list((ROOT / "presets").glob("*.json"))
        self.assertEqual(len(files), 11)
        for path in files:
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), default_config(path.stem))

    def test_grid_has_eighty_unique_conditions(self):
        values = conditions()
        keys = {(row["dataset"], row["split_id"], row["seed"]) for row in values}
        self.assertEqual(len(values), 80)
        self.assertEqual(len(keys), 80)
        for name in ("roman-empire", "questions", "minesweeper", "amazon-ratings", "tolokers"):
            self.assertEqual({row["split_id"] for row in values if row["dataset"] == name}, set(range(10)))

    def test_variant_auxiliary_switches(self):
        for name, safe, noisy in (("sc_only", False, False), ("sc_qk", True, False), ("sc_full", True, True)):
            cfg = resolve_config("cora", name)
            self.assertEqual(cfg["safe_qk"]["enabled"], safe)
            self.assertEqual(cfg["gaussian_consistency"]["enabled"], noisy)

    def test_primary_metrics(self):
        self.assertEqual(PRIMARY_METRIC["amazon-ratings"], "ACC")
        for name in ("questions", "minesweeper", "tolokers"):
            self.assertEqual(default_config(name)["primary_metric"], "ROC-AUC")

    def test_bidirected_messages_unique_nonself_evidence(self):
        data = Data(x=torch.ones(3, 2), y=torch.tensor([0, 0, 1]),
                    edge_index=torch.tensor([[0, 1, 0, 1, 2], [1, 0, 1, 2, 2]]))
        apply_graph_protocol(data, True)
        self.assertEqual(data.edge_index.shape[1], 4)
        self.assertEqual(data.evidence_edge_index.shape[1], 2)
        before = data.edge_index.clone()
        apply_graph_protocol(data, True)
        self.assertTrue(torch.equal(before, data.edge_index))

    def test_split_index_is_independent_of_initialization(self):
        bundle, _ = fixture()
        self.assertEqual(build_split(bundle, 2024, 3), build_split(bundle, 9999, 3))
        self.assertNotEqual(build_split(bundle, 2024, 0), build_split(bundle, 2024, 1))

    def test_invalid_split_does_not_fall_back(self):
        bundle, _ = fixture()
        with self.assertRaises(ValueError):
            build_split(bundle, 2024, 10)
        with self.assertRaises(ValueError):
            select_mask(torch.ones(5, dtype=torch.bool), 1)

    def test_selector_graph_masks_held_out_labels(self):
        bundle, split = fixture()
        original = bundle.data.y.clone()
        view = train_only_view(bundle.data, split["train"])
        self.assertTrue((view.y[split["test"]] == -999).all())
        self.assertTrue(torch.equal(bundle.data.y, original))
        self.assertTrue(all(not key.endswith("_mask") for key in view.keys()))

    def test_prior_and_inner_folds_ignore_held_out_label_changes(self):
        bundle, split = fixture()
        cfg = default_config("minesweeper")["selector"]
        prior = select_rc_v2(bundle.data, {"train": split["train"]}, 2, cfg)
        folds = make_inner_splits(bundle.data.y.tolist(), split["train"], cfg, 42)
        bundle.data.y[40:] = 1234
        self.assertEqual(prior, select_rc_v2(bundle.data, {"train": split["train"]}, 2, cfg))
        self.assertEqual(folds, make_inner_splits(bundle.data.y.tolist(), split["train"], cfg, 42))

    def test_all_branches_forward_and_backward(self):
        bundle, _ = fixture()
        cfg = default_config("minesweeper")
        cfg["hidden_dim"] = 8
        for branch in BRANCH_DEPTH:
            with self.subTest(branch=branch):
                model = model_factory(cfg, bundle)(branch)
                result = model(bundle.data.x, bundle.data.edge_index)
                self.assertEqual(tuple(result.shape), (60, 2))
                self.assertTrue(torch.isfinite(result).all())
                torch.nn.functional.nll_loss(result[:40], bundle.data.y[:40]).backward()

    def test_only_query_key_matrices_receive_polar(self):
        model = RCFormer(5, 8, 2, "sgc", 2)
        optimizer = build_safe_qk_optimizer(model, default_config("cora"))
        self.assertTrue(optimizer.safe_qk_parameter_names)
        self.assertTrue(all("Wq" in name or "Wk" in name for name in optimizer.safe_qk_parameter_names))
        self.assertTrue(any("Wv" in name for name in optimizer.auxiliary_parameter_names))

    def test_polar_rectangular_and_zero_gradients_are_finite(self):
        for shape in ((3, 7), (7, 3), (4, 4)):
            for gradient in (torch.randn(shape), torch.zeros(shape)):
                value = zeropower_via_newton_schulz(gradient)
                self.assertEqual(value.shape, gradient.shape)
                self.assertTrue(torch.isfinite(value).all())

    def test_zero_polar_steps_retains_normalization(self):
        gradient = torch.randn(3, 7)
        self.assertTrue(torch.equal(zeropower_via_newton_schulz(gradient, steps=0), gradient / gradient.norm()))

    def test_consistency_stops_clean_target_gradient(self):
        clean = torch.randn(5, 2, requires_grad=True)
        noisy = torch.randn(5, 2, requires_grad=True)
        loss = consistency_loss(clean.log_softmax(1), noisy.log_softmax(1), [0, 1, 2])
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNotNone(noisy.grad)

    def test_consistency_ramp_matches_protocol(self):
        cfg = default_config("cora")["gaussian_consistency"]
        self.assertEqual(consistency_weight(0, cfg), 0)
        self.assertAlmostEqual(consistency_weight(10, cfg), .025)
        self.assertAlmostEqual(consistency_weight(20, cfg), .05)

    def test_sc_tie_activates_third_fold(self):
        bundle, split = fixture()
        cfg = default_config("minesweeper")
        cfg["hidden_dim"] = 8
        cfg["selector"].update(sc_min_train_nodes=5, sc_min_probe_nodes=2)
        prior = {"selected_branch": "sgc", "reason": "posterior_high_homophily_select_sgc"}
        equal = {"risk": 1., "metric": .5, "metric_fallback_reason": "", "metric_name": "ROC-AUC"}
        with patch("selector.sc_efficient._train_to_budget", return_value=equal):
            result = run_sc_efficient(cfg, train_only_view(bundle.data, split["train"]),
                                     {"train": split["train"]}, prior, model_factory(cfg, bundle), torch.device("cpu"))
        self.assertTrue(result["third_fold_activated"])
        self.assertEqual(result["active_folds"], 3)

    def test_shared_bank_real_small_optimizer_run(self):
        bundle, split = fixture()
        cfg = default_config("minesweeper")
        cfg.update(hidden_dim=8, device="cpu")
        cfg["selector"].update(sc_min_train_nodes=5, sc_min_probe_nodes=2)
        with temporary_output() as root:
            result = build_bank(cfg, bundle, split, root, epochs=2, engineering_pilot=True)
        self.assertEqual(len(result["candidate_cells"]), 6)
        self.assertEqual(len(result["policies"]), 6)
        self.assertFalse(result["outer_performance_evaluated"])
        self.assertFalse(result["auxiliaries_enabled_in_candidates"])

    def test_final_training_checkpoint_reload(self):
        bundle, split = fixture()
        cfg = default_config("minesweeper")
        cfg.update(hidden_dim=8, epochs=3, patience=3, device="cpu", run_kind="engineering_pilot")
        with temporary_output() as root:
            result = train_fixed_branch(cfg, bundle, split, "fagcn_local", root)
            checkpoint = torch.load(Path(root) / "model.pt", map_location="cpu", weights_only=False)
            model = model_factory(cfg, bundle)(checkpoint["branch"])
            model.load_state_dict(checkpoint["model_state"], strict=True)
            from training.evaluator import evaluate_all
            replay = evaluate_all(model, bundle.data, split, cfg["primary_metric"])
        self.assertEqual(replay, result["metrics"])
        self.assertEqual(min(row["valid_nll"] for row in result["history"]), result["best_validation_nll"])

    def test_archived_results_match_main_claims(self):
        result = summarize(ROOT / "study")
        self.assertEqual(result["archived_cells_verified"], 176)
        self.assertAlmostEqual(result["average_ranks"]["RCFormer-SC-Full"], 3.909090909090909)
        self.assertEqual(result["full_top_three"], 7)
        self.assertEqual(result["selector"]["prior_hits"], 35)
        self.assertEqual(result["selector"]["final_hits"], 73)
        self.assertEqual(result["gaussian_retention"]["positive_datasets"], 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
