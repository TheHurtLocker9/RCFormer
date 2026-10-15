"""Small graph preprocessing helpers."""

from __future__ import annotations

import torch


def prepare_graph(data):
    data.x = data.x.float().contiguous()
    data.y = data.y.view(-1).long().contiguous()
    data.edge_index = data.edge_index.long().contiguous()
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if data.x.shape[0] != data.y.numel():
        raise ValueError("feature and label node counts do not match")
    return data


def unique_undirected_edges(edge_index, num_nodes):
    """One non-self-loop evidence observation per unordered node pair."""
    edges = edge_index.detach().cpu().long()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('edge_index must be [2, E]')
    if edges.numel() and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError('edge endpoint outside node range')
    left, right = torch.minimum(edges[0], edges[1]), torch.maximum(edges[0], edges[1])
    keep = left != right
    keys = torch.unique(left[keep] * int(num_nodes) + right[keep], sorted=True)
    return torch.stack((keys // int(num_nodes), keys % int(num_nodes)))


def apply_graph_protocol(data, bidirected=False):
    data = prepare_graph(data)
    evidence = unique_undirected_edges(data.edge_index, data.num_nodes)
    data.evidence_edge_index = evidence
    if bidirected:
        # Operators add their own propagation loops; these never enter RC-v2 evidence.
        data.edge_index = torch.cat((evidence, evidence.flip(0)), dim=1).contiguous()
    data.graph_protocol = 'bidirected_simple_no_loader_loops' if bidirected else 'preserved_message_edges'
    data.evidence_protocol = 'unique_undirected_nonself_train_pairs'
    return data


def train_only_view(data, train_indices):
    view = data.clone().cpu()
    view.y.fill_(-999)
    indices = torch.as_tensor(train_indices, dtype=torch.long)
    view.y[indices] = data.y.detach().cpu()[indices]
    for key in list(view.keys()):
        if key.endswith('_mask'):
            del view[key]
    return view


