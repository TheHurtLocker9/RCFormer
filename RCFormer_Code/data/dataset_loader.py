"""Load only the eleven datasets used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch_geometric.data import Data

from config import canonical_dataset_name
from .preprocessing import apply_graph_protocol


PLANETOID = {"cora", "citeseer", "pubmed"}
AMAZON = {"computers", "photo"}
COAUTHOR = {"cs"}
HETEROPHILY = {
    "roman-empire",
    "questions",
    "minesweeper",
    "amazon-ratings",
    "tolokers",
}


@dataclass
class DatasetBundle:
    name: str
    data: Data
    num_features: int
    num_classes: int
    official_split: dict | None = None


def load_dataset(name, root="datasets", allow_download=False):
    name = canonical_dataset_name(name)
    if name not in PLANETOID | AMAZON | COAUTHOR | HETEROPHILY:
        raise ValueError(f"dataset is outside the paper benchmark: {name}")
    root_path = Path(root)
    local = _load_local_npz(name, root_path)
    if local is not None:
        return local
    if not allow_download and not _has_pyg_cache(name, root_path):
        raise FileNotFoundError(
            f"no cached data found for {name} under {root_path}; "
            "supply the paper data or pass --allow-download"
        )
    return _load_pyg(name, root_path)


def _load_pyg(name, root):
    import torch_geometric.transforms as transforms
    from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, Planetoid

    transform = transforms.NormalizeFeatures()
    if name in PLANETOID:
        dataset = Planetoid(str(root), name, split="public", transform=transform)
    elif name in AMAZON:
        dataset = Amazon(str(root), name.capitalize(), transform=transform)
    elif name in COAUTHOR:
        dataset = Coauthor(str(root), "CS", transform=transform)
    else:
        dataset = HeterophilousGraphDataset(
            root=str(root), name=name.replace("-", "_")
        )
    data = apply_graph_protocol(dataset[0], bidirected=name in HETEROPHILY)
    official = None
    if all(hasattr(data, key) for key in ("train_mask", "val_mask", "test_mask")):
        official = {
            "train": data.train_mask,
            "valid": data.val_mask,
            "test": data.test_mask,
        }
    return DatasetBundle(
        name=name,
        data=data,
        num_features=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
        official_split=official,
    )


def _load_local_npz(name, root):
    filenames = {
        "cs": "ms_academic_cs.npz",
        "roman-empire": "roman_empire.npz",
        "questions": "questions.npz",
        "minesweeper": "minesweeper.npz",
        "amazon-ratings": "amazon_ratings.npz",
        "tolokers": "tolokers.npz",
    }
    filename = filenames.get(name)
    if not filename or not (root / filename).exists():
        return None
    payload = np.load(str(root / filename), allow_pickle=True)
    if name == "cs":
        adjacency = sparse.csr_matrix(
            (payload["adj_data"], payload["adj_indices"], payload["adj_indptr"]),
            shape=tuple(payload["adj_shape"]),
        ).tocoo()
        attributes = sparse.csr_matrix(
            (payload["attr_data"], payload["attr_indices"], payload["attr_indptr"]),
            shape=tuple(payload["attr_shape"]),
        )
        data = Data(
            x=torch.from_numpy(attributes.toarray()).float(),
            edge_index=torch.from_numpy(np.vstack([adjacency.row, adjacency.col])).long(),
            y=torch.from_numpy(payload["labels"]).long(),
        )
        return _bundle(name, data, None)

    data = Data(
        x=torch.from_numpy(payload["node_features"]).float(),
        edge_index=torch.from_numpy(payload["edges"]).long().t().contiguous(),
        y=torch.from_numpy(payload["node_labels"]).long(),
    )
    # NPZ is [split, node]; use [node, split] consistently with PyG.
    official = {
        "train": torch.from_numpy(payload["train_masks"].T.copy()).bool(),
        "valid": torch.from_numpy(payload["val_masks"].T.copy()).bool(),
        "test": torch.from_numpy(payload["test_masks"].T.copy()).bool(),
    }
    return _bundle(name, data, official)


def _bundle(name, data, official):
    data = apply_graph_protocol(data, bidirected=name in HETEROPHILY)
    return DatasetBundle(
        name=name,
        data=data,
        num_features=int(data.x.shape[1]),
        num_classes=int(data.y.max().item()) + 1,
        official_split=official,
    )


def _has_pyg_cache(name, root):
    candidates = [root / name, root / name.capitalize(), root / name.replace("-", "_")]
    if name in PLANETOID:
        candidates.append(root)
    for candidate in candidates:
        if (candidate / "processed" / "data.pt").exists():
            return True
        raw = candidate / "raw"
        if raw.exists() and any(raw.iterdir()):
            return True
    return False
