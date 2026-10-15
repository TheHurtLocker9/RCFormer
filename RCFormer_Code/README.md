# RCFormer: Code for Pattern Recognition Review

This compact submission contains the reported RCFormer backbone, train-only
operator selection, final training, fixed-branch training, checkpoint evaluation,
and equal-maturity shared-candidate diagnostic. Eleven dataset presets, the
80-condition protocol, three progressive variants, and tabular study records
are included. Public datasets and large checkpoint tensors are not bundled.

## Environment

Python 3.9 or newer is required. The tested research environment used Python
3.9.23, PyTorch 2.8.0+cu129, PyG 2.6.1, NumPy 1.26.3, SciPy 1.13.1, and
scikit-learn 1.6.1. The numerical package versions are recorded in
`ENVIRONMENT.json`. Install a PyTorch build appropriate for the machine, then
install the packages in `requirements.txt`. CUDA is optional; CPU is supported.
Hardware and random-number implementations can affect numerical results.

```text
python -m pip install -r requirements.txt
python tools/verify_manifest.py
python -m unittest discover -s tests -v
```

## Training and Evaluation

Run commands from this directory. A full Cora condition is:

```text
python main.py --dataset cora --config presets/cora.json --data-root datasets --output-root outputs --seed 2024 --split-ids 0 --device auto --allow-download
```

Existing datasets are preferred; downloads require explicit permission through
`--allow-download`. The loader accepts the original heterophily NPZ files and
the PyG dataset caches. For Coauthor CS it also accepts `ms_academic_cs.npz`.
The six Planetoid/Amazon/Coauthor datasets use five initializations; the five
heterophily datasets use ten official split indices, each with its specified
initialization. Amazon-Ratings uses accuracy; Questions, Minesweeper, and
Tolokers use ROC-AUC. Macro-F1 is exported from the same checkpoint.

Preview the 80 Full conditions, or run one selected condition:

```text
python tools/run_grid.py
python tools/run_grid.py --datasets cora --variants sc_full --limit 1 --data-root datasets --execute
python tools/run_grid.py --variants sc_only sc_qk sc_full --data-root datasets --execute
```

The final command schedules 240 full-budget fits and can be expensive. Nothing
runs in the background. The preview does not train or create output files.
Grid execution requires cached datasets; use the single-condition command to
download datasets where needed. Existing run identities are never overwritten.
Use a fresh output directory for a new run.

```text
python train.py --dataset cora --branch sgc --seed 2024 --split-id 0 --data-root datasets --output-root outputs/fixed --device auto
python evaluate.py --checkpoint outputs/benchmark/sc_full/cora/split_00/seed_2024/model.pt --data-root datasets --device auto
python shared_bank.py --datasets minesweeper --split-ids 0 --data-root datasets --output-root outputs/shared --device auto
```

Only load trusted checkpoint files. The shared-bank diagnostic is not the
adaptive deployed selector: it uses fixed two-fold, 150-epoch candidate
evidence. Short engineering checks are explicitly separated from benchmark
results. Candidate comparisons disable both final-training auxiliaries.

## Files and Scope

- `models/`: TransConv and the SGC, FAGCN-inspired, and OptBasisLocal candidates.
- `selector/`: RC-v2 statistics, ordered prior, and directional SC-Efficient.
- `optimization/`, `regularization/`, `training/`: five-step polar Q/K updates,
  Gaussian consistency, and minimum-validation-NLL checkpoint selection.
- `presets/`: eleven frozen defaults and Only/QK/Full variant switches.
- `protocol/`: condition grid, prespecified additions, and 80 archived Full
  execution configurations with portable paths. Original file hashes remain
  recorded; runtime paths/device fields are normalized, not claimed byte-identical.
- `study/`: recorded results, baseline execution metadata, and recipe mappings.
- `tools/`: manifest verification, grid execution, and compact result checks.

```text
python tools/summarize.py
```

This checks all 176 main-comparison means and sample SDs from the 1,680
condition records, recomputes average ranks, and checks the selector and
Gaussian-retention summaries. Historical V1 results and aligned V2 runs retain
their original source fingerprints; the release is not represented as having
generated all those historical results. CSV exports omit local run-directory
paths without changing numerical values.

External baseline implementations, original checkpoints, publication artwork,
and the checkpoint-dependent extension adapter are not included. The archived
scope, matched-polar, profiling, repartition, and perturbation tables support
inspection, not standalone rerunning of those extensions. Archived inferential
tables are not all recomputed by `tools/summarize.py`. In particular, the CS
matched-polar raw Wilcoxon value is precision-sensitive; the recorded Holm
result and scientific conclusion are unchanged. This is a backbone-and-records
release, not a one-command reproduction of every table in the article.

## Version and Rights

`RELEASE_PROVENANCE.json` records the frozen input hashes and three narrowly
pruned files. The unused alternative optimizer polynomial and unused
compatibility helpers were removed; the executed polar algorithm and reported
model paths are preserved and checked against the frozen implementation.
The package has a new distribution fingerprint, not the historical full-project
fingerprint. `FILE_MANIFEST.json` covers all distributed files except itself.

The TransConv encoder is adapted from DUALFormer, ICLR 2025:
https://github.com/JiamingZhuo/DUALFormer
The upstream MIT notice is retained in `THIRD_PARTY_LICENSES/DUALFormer-MIT.txt`.
This local review/reproduction packet does not assert a new license for
third-party datasets or a public repository/DOI. Authors should approve the
licensing of their original additions before public redistribution.
