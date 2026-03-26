from __future__ import annotations

import torch
import torch.nn as nn


class QueryRelationLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(query_features, query_features, query_features, need_weights=False)
        query_features = self.norm1(query_features + self.dropout(attn_output))
        ffn_output = self.ffn(query_features)
        query_features = self.norm2(query_features + self.dropout(ffn_output))
        return query_features


class QueryRelationBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        ff_dim = int(ff_dim or hidden_dim * 2)
        self.layers = nn.ModuleList(
            [
                QueryRelationLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query_features = layer(query_features)
        return query_features
