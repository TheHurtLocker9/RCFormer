"""Feature-level global encoder adapted from DUALFormer.

Upstream source: https://github.com/JiamingZhuo/DUALFormer
Upstream paper: "DUALFormer: Dual Graph Transformer," ICLR 2025.
License notice: ../THIRD_PARTY_LICENSES/DUALFormer-MIT.txt

RCFormer uses this encoder as a supporting backbone before its selected local
operator; local-operator configuration is implemented separately.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def full_attention_conv(query, key, value, output_attention=False):
    scale = math.sqrt(float(query.shape[0]))
    interaction = torch.einsum("lmh,ldh->mdh", key / scale, value)
    attention = torch.softmax(interaction, dim=0)
    output = torch.einsum("lmh,mdh->ldh", query, attention)
    return (output, attention) if output_attention else output


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = (
            nn.Linear(in_channels, out_channels * num_heads)
            if use_weight
            else None
        )
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.Wv is not None:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attention=False):
        query = self.Wq(query_input).reshape(
            -1, self.out_channels, self.num_heads
        )
        key = self.Wk(source_input).reshape(-1, self.out_channels, self.num_heads)
        value = (
            self.Wv(source_input).reshape(-1, self.out_channels, self.num_heads)
            if self.Wv is not None
            else source_input.reshape(-1, self.out_channels, 1)
        )
        result = full_attention_conv(
            query, key, value, output_attention=output_attention
        )
        if output_attention:
            output, attention = result
            return output.mean(dim=-1), attention
        return result.mean(dim=-1)


class TransConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        activation,
        num_layers=1,
        num_heads=2,
        alpha=0.1,
        dropout=0.3,
        use_bn=True,
    ):
        super().__init__()
        # Registration and reset order intentionally match the experiment code.
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels)])
        for _ in range(int(num_layers)):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads)
            )
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.activation = activation
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for norm in self.bns:
            norm.reset_parameters()
        for projection in self.fcs:
            projection.reset_parameters()

    def forward(self, x):
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for index, conv in enumerate(self.convs):
            residual = x
            x = conv(x, x)
            x = self.alpha * x + (1.0 - self.alpha) * residual
            if self.use_bn:
                x = self.bns[index + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
