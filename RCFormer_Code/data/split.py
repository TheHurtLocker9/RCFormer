"""Frozen outer-split handling."""

from __future__ import annotations

import random

import torch

from config import canonical_dataset_name


RATIO_60_20_20 = {"computers", "photo", "cs"}


def build_split(bundle, seed, split_id=0):
    split_id = int(split_id)
    if split_id < 0:
        raise ValueError('split_id must be nonnegative')
    if bundle.official_split is not None:
        split = {
            key: _to_indices(select_mask(value, split_id))
            for key, value in bundle.official_split.items()
        }
    elif canonical_dataset_name(bundle.name) in RATIO_60_20_20:
        if split_id != 0:
            raise ValueError('ratio-split datasets use seed, not an official split_id')
        split = ratio_split(bundle.data.num_nodes, 0.60, 0.20, seed)
    else:
        raise ValueError(f"no split policy available for {bundle.name}")
    assert_valid_split(split, bundle.data.num_nodes)
    return split


def select_mask(value, split_id):
    mask = torch.as_tensor(value)
    if mask.ndim == 1:
        if split_id != 0:
            raise ValueError('dataset has only one official split')
        return mask
    if mask.ndim != 2 or split_id >= mask.shape[1]:
        raise ValueError('official split_id outside available mask columns')
    return mask[:, split_id]


def ratio_split(num_nodes, train_ratio, validation_ratio, seed):
    indices = list(range(int(num_nodes)))
    random.Random(int(seed)).shuffle(indices)
    train_size = int(num_nodes * train_ratio)
    validation_size = int(num_nodes * validation_ratio)
    return {
        "train": indices[:train_size],
        "valid": indices[train_size : train_size + validation_size],
        "test": indices[train_size + validation_size :],
    }


def assert_valid_split(split, num_nodes):
    seen = set()
    for key in ("train", "valid", "test"):
        values = [int(value) for value in split[key]]
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate node in {key} split")
        if seen.intersection(values):
            raise AssertionError(f"split overlap detected at {key}")
        if any(value < 0 or value >= num_nodes for value in values):
            raise AssertionError(f"out-of-range node in {key} split")
        seen.update(values)
    return True


def _to_indices(value):
    tensor = torch.as_tensor(value)
    if tensor.dtype == torch.bool:
        return tensor.nonzero(as_tuple=False).view(-1).tolist()
    return tensor.view(-1).long().tolist()
