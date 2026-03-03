"""
BÖLÜM 2 — DynamicMaskHead (izole test ile).
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMaskHead(nn.Module):
    """
    T:       (L, B, C) or (B, L, C)
    M_prime: (B, C, H, W)
    """

    def __init__(
        self,
        num_queries: int = 7,
        feat_dim: int = 32,
        feat_h: int = 12,
        feat_w: int = 20,
        use_coords: bool = False,
        prior_prob: float = None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feat_dim = feat_dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.use_coords = use_coords

        # CondLSTR benzeri coord injection: kernel kanalı C yerine C+2 olur.
        self.kernel_dim = feat_dim + (2 if use_coords else 0)

        self.mlp_heatmap = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.kernel_dim),
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.kernel_dim),
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

        # CondLSTR benzeri prior bias (opsiyonel): obj/fg başlangıç olasılığını düşürür.
        if prior_prob is not None:
            if not (0.0 < float(prior_prob) < 1.0):
                raise ValueError(f"prior_prob must be in (0,1), got {prior_prob}")
            bias_value = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
            with torch.no_grad():
                last = self.mlp_score[-1]
                if isinstance(last, nn.Linear) and last.out_features == 2:
                    # index 0: background, index 1: foreground
                    last.bias[0] = 0.0
                    last.bias[1] = float(bias_value)

    @staticmethod
    def _inject_coords(feat: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,C+2,H,W), coord order: [y, x] in [0,1]."""
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

        B, C, H, W = M_prime.shape
        if T.shape[1] == B:  # (L, B, C)
            T_blc = T.permute(1, 0, 2)
            L = T.shape[0]
        elif T.shape[0] == B:  # (B, L, C)
            T_blc = T
            L = T.shape[1]
        else:
            raise ValueError(
                f"Cannot infer batch dim from T={tuple(T.shape)} and M_prime={tuple(M_prime.shape)}"
            )

        Kb = self.mlp_heatmap(T_blc)  # (B, L, C or C+2)
        Kz = self.mlp_offset(T_blc)   # (B, L, C or C+2)

        M_feat = self._inject_coords(M_prime) if self.use_coords else M_prime
        M_flat = M_feat.flatten(2)  # (B, C(+2), HW)

        B_raw = torch.bmm(Kb, M_flat)  # (B, L, HW)
        Z_raw = torch.bmm(Kz, M_flat)  # (B, L, HW)

        B_spatial = B_raw.view(B, L, H, W)
        Z_spatial = Z_raw.view(B, L, H, W)

        heatmap = F.softmax(B_spatial, dim=-1)
        v_range = self.mlp_vrange(T_blc)
        scores = self.mlp_score(T_blc)

        return {
            "heatmap": heatmap,
            "offset": Z_spatial,
            "v_range": v_range,
            "vrange": v_range,
            "scores": scores,
        }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script.")

    print("=" * 80)
    print("BÖLÜM 2 — DynamicMaskHead İZOLE TEST")
    print("=" * 80)

    NUM_QUERIES = 7
    FEAT_DIM = 32
    # Bölüm 0 / 1b gerçek model shape: input_proj -> (B, 32, 10, 26)
    FEAT_H = 10
    FEAT_W = 26
    BATCH = 2

    head = DynamicMaskHead(NUM_QUERIES, FEAT_DIM, FEAT_H, FEAT_W).cuda()
    print(f"Parametre sayısı: {sum(p.numel() for p in head.parameters()):,}")

    T = torch.randn(NUM_QUERIES, BATCH, FEAT_DIM, device="cuda")
    M_prime = torch.randn(BATCH, FEAT_DIM, FEAT_H, FEAT_W, device="cuda")

    print(f"\nGirdi:\n  T       : {tuple(T.shape)}\n  M_prime : {tuple(M_prime.shape)}")

    out = head(T, M_prime)
    print("\nÇıktı:")
    print(f"  heatmap : {tuple(out['heatmap'].shape)}")
    print(f"  offset  : {tuple(out['offset'].shape)}")
    print(f"  v_range : {tuple(out['v_range'].shape)}")
    print(f"  scores  : {tuple(out['scores'].shape)}")

    row_sums = out["heatmap"].sum(dim=-1)
    print("\nHEATMAP satır toplamları:")
    print(f"  mean={row_sums.mean().item():.6f}, std={row_sums.std().item():.6f}")
    print("  {} Softmax doğru".format("✅" if abs(row_sums.mean().item() - 1.0) < 1e-3 else "🚨"))

    for k, v in out.items():
        has_nan = torch.isnan(v).any().item()
        has_inf = torch.isinf(v).any().item()
        print(f"  {k}: NaN={has_nan}, Inf={has_inf} {'✅' if not (has_nan or has_inf) else '🚨'}")

    print("\nGRADIENT TESTİ:")
    T2 = torch.randn(NUM_QUERIES, BATCH, FEAT_DIM, device="cuda", requires_grad=True)
    M2 = torch.randn(BATCH, FEAT_DIM, FEAT_H, FEAT_W, device="cuda", requires_grad=True)
    out2 = head(T2, M2)
    # Tüm dalları loss'a bağla ki sahte dead-gradient alarmı oluşmasın.
    loss = (
        out2["heatmap"].sum()
        + out2["offset"].sum()
        + out2["v_range"].sum()
        + out2["scores"].sum()
    )
    loss.backward()

    grad_ok = True
    for name, param in head.named_parameters():
        if param.grad is None or param.grad.abs().mean().item() < 1e-12:
            grad_ok = False
            print(f"  🚨 {name}: grad yok/dead")
        else:
            print(f"  ✅ {name}: grad OK ({param.grad.abs().mean().item():.2e})")

    print(f"\n  T2.grad: {'✅ var' if T2.grad is not None else '🚨 yok'}")
    print(f"  M2.grad: {'✅ var' if M2.grad is not None else '🚨 yok'}")
    print("\n✅ DynamicMaskHead izole test GEÇTİ" if grad_ok else "\n🚨 Gradient sorunu var")


if __name__ == "__main__":
    main()

