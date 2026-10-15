"""FAGCN-inspired local propagation candidate."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .sgc import GraphConv


class FAGCNLocal(nn.Module):
    def __init__(self, hidden_dim, num_layers=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList(GraphConv() for _ in range(int(num_layers)))
        self.linears = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(int(num_layers))
        )
        self.eps = nn.Parameter(torch.zeros(int(num_layers)))
        self.dropout = float(dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for linear in self.linears:
            linear.reset_parameters()
        nn.init.zeros_(self.eps)

    def forward(self, x, edge_index):
        h0 = x
        h = x
        for index, (conv, linear) in enumerate(zip(self.convs, self.linears)):
            propagated = conv(h, edge_index)
            transformed = torch.tanh(linear(propagated))
            eps = torch.sigmoid(self.eps[index])
            h = (1.0 - eps) * transformed + eps * h0
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
