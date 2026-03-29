from __future__ import annotations

import torch
import torch.nn as nn


class FixedLaneAnchorBank(nn.Module):
    def __init__(self, num_queries: int, mode: str = "bottom_dx_rowspan") -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.mode = str(mode).lower()
        if self.mode != "bottom_dx_rowspan":
            raise ValueError(f"Unsupported dense_geo_anchor_mode={mode!r}")
        self.register_buffer("anchor_bank", self._build_anchor_bank(), persistent=True)

    def _build_anchor_bank(self) -> torch.Tensor:
        if self.num_queries == 20:
            bottom_xs = [0.10, 0.30, 0.50, 0.70, 0.90]
            delta_xs = [-0.12, 0.12]
            row_starts = [0.05, 0.25]
            row_end = 1.0
            anchors = []
            for bottom_x in bottom_xs:
                for delta_x in delta_xs:
                    for row_start in row_starts:
                        anchors.append([bottom_x, delta_x, row_start, row_end])
            return torch.tensor(anchors, dtype=torch.float32)

        bottom_x = torch.linspace(0.05, 0.95, steps=max(self.num_queries, 1), dtype=torch.float32)
        delta_x = torch.zeros_like(bottom_x)
        row_start = torch.full_like(bottom_x, 0.10)
        row_end = torch.ones_like(bottom_x)
        return torch.stack((bottom_x, delta_x, row_start, row_end), dim=1)

    def forward(self) -> torch.Tensor:
        return self.anchor_bank


class LaneAnchorEncoder(nn.Module):
    def __init__(self, anchor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(anchor_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, anchor_bank: torch.Tensor) -> torch.Tensor:
        if anchor_bank.dim() != 2:
            raise ValueError(f"anchor_bank must be 2D [Q, A], got {tuple(anchor_bank.shape)}")
        return self.net(anchor_bank)
