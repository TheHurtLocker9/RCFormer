"""Learned multi-order local propagation candidate."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .sgc import GraphConv


class OptBasisLocal(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        self.convs = nn.ModuleList(GraphConv() for _ in range(self.num_layers))
        self.coeff = nn.Parameter(torch.zeros(self.num_layers + 1))
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = float(dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.zeros_(self.coeff)
        self.project.reset_parameters()

    def forward(self, x, edge_index):
        basis = [x]
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            basis.append(h)
        weights = torch.softmax(self.coeff[: len(basis)], dim=0)
        mixed = torch.zeros_like(x)
        for weight, value in zip(weights, basis):
            mixed = mixed + weight * value
        return F.dropout(self.project(mixed), p=self.dropout, training=self.training)
