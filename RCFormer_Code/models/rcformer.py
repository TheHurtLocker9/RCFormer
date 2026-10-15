"""Single-path RCFormer model after local-operator selection."""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from optimization.safe_qk import is_qk_parameter

from .fagcn_local import FAGCNLocal
from .optbasis_local import OptBasisLocal
from .sgc import GraphConv
from .transconv import TransConv


class RCFormer(nn.Module):
    VALID_BRANCHES = {"sgc", "fagcn_local", "optbasis_local"}

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        branch,
        local_layers,
        num_trans=1,
        num_heads=2,
        dropout_trans=0.3,
        dropout=0.5,
        alpha=0.1,
        use_bn=True,
    ):
        super().__init__()
        if branch not in self.VALID_BRANCHES:
            raise ValueError(f"unsupported local branch: {branch}")
        activation = nn.ReLU()
        self.layers_trans = TransConv(
            input_dim,
            hidden_dim,
            activation,
            num_layers=num_trans,
            num_heads=num_heads,
            alpha=alpha,
            dropout=dropout_trans,
            use_bn=use_bn,
        )
        if branch == "sgc":
            self.convs = nn.ModuleList(
                GraphConv() for _ in range(int(local_layers))
            )
        elif branch == "fagcn_local":
            self.convs = FAGCNLocal(
                hidden_dim, num_layers=local_layers, dropout=dropout
            )
        else:
            self.convs = OptBasisLocal(
                hidden_dim, num_layers=local_layers, dropout=dropout
            )
        self.branch = branch
        self.dropout = float(dropout)
        self.linear_project = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.layers_trans.reset_parameters()
        if hasattr(self.convs, "reset_parameters"):
            self.convs.reset_parameters()
        else:
            for layer in self.convs:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        self.linear_project.reset_parameters()

    def forward(self, x, edge_index):
        z = self.layers_trans(x)
        if self.branch == "sgc":
            for conv in self.convs:
                z = conv(z, edge_index)
        else:
            z = self.convs(z, edge_index)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return F.log_softmax(self.linear_project(z), dim=1)

    def named_qk_parameters(self):
        for name, parameter in self.named_parameters():
            if is_qk_parameter(name, parameter):
                yield name, parameter
