"""
BÖLÜM 2 (REG) — DynamicRegHead.

Çıktılar:
  - heatmap: (B, L, H, W)
  - reg    : (B, L, H)          # normalize residual in [-1, 1]
  - offset : (B, L, H, W)       # legacy adapter (reg * W broadcast)
  - v_range: (B, L, 2)
  - scores : (B, L, 2)
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRegHead(nn.Module):
    """
    T:       (L, B, C) or (B, L, C)
    M_prime: (B, C, H, W)
    """

    def __init__(
        self,
        num_queries: int = 7,
        feat_dim: int = 32,
        feat_h: int = 10,
        feat_w: int = 26,
        use_coords: bool = False,
        prior_prob: float = None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feat_dim = feat_dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.use_coords = use_coords

        self.kernel_dim = feat_dim + (2 if use_coords else 0)

        self.mlp_heatmap = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.kernel_dim),
        )
        self.mlp_reg = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_h),
            nn.Tanh(),
        )
        self.mlp_vrange = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )
        self.mlp_score = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )

        if prior_prob is not None:
            if not (0.0 < float(prior_prob) < 1.0):
                raise ValueError(f"prior_prob must be in (0,1), got {prior_prob}")
            bias_value = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
            with torch.no_grad():
                last = self.mlp_score[-1]
                if isinstance(last, nn.Linear) and last.out_features == 2:
                    # 0: bg, 1: fg
                    last.bias[0] = 0.0
                    last.bias[1] = float(bias_value)

    @staticmethod
    def _inject_coords(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError(f"feat must be 4D, got shape={tuple(feat.shape)}")
        b, _c, h, w = feat.shape
        ys = torch.linspace(0.0, 1.0, steps=h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(0.0, 1.0, steps=w, device=feat.device, dtype=feat.dtype)
        y_plane = ys.view(1, 1, h, 1).expand(b, 1, h, w)
        x_plane = xs.view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([feat, y_plane, x_plane], dim=1)

    def forward(self, T: torch.Tensor, M_prime: torch.Tensor) -> Dict[str, torch.Tensor]:
        if T.dim() != 3:
            raise ValueError(f"T must be 3D, got shape={tuple(T.shape)}")
        if M_prime.dim() != 4:
            raise ValueError(f"M_prime must be 4D, got shape={tuple(M_prime.shape)}")

        bsz, _c, h, w = M_prime.shape
        if T.shape[1] == bsz:  # (L,B,C)
            t_blc = T.permute(1, 0, 2)
            lanes = T.shape[0]
        elif T.shape[0] == bsz:  # (B,L,C)
            t_blc = T
            lanes = T.shape[1]
        else:
            raise ValueError(f"Cannot infer batch dim from T={tuple(T.shape)} and M_prime={tuple(M_prime.shape)}")

        feat = self._inject_coords(M_prime) if self.use_coords else M_prime
        m_flat = feat.flatten(2)  # (B, C(+2), HW)

        k_hm = self.mlp_heatmap(t_blc)       # (B, L, C(+2))
        b_raw = torch.bmm(k_hm, m_flat)      # (B, L, HW)
        heatmap = F.softmax(b_raw.view(bsz, lanes, h, w), dim=-1)

        reg = self.mlp_reg(t_blc)            # (B, L, H)
        offset = reg.unsqueeze(-1).expand(-1, -1, -1, w) * float(w)

        v_range = self.mlp_vrange(t_blc)
        scores = self.mlp_score(t_blc)

        return {
            "heatmap": heatmap,
            "reg": reg,
            "offset": offset,  # legacy adapter
            "v_range": v_range,
            "vrange": v_range,
            "scores": scores,
        }

