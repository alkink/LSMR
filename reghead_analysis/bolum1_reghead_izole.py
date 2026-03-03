"""
BÖLÜM 1 — RegHead izole test.

Yeni başlık:
  T:   (B, L, C)
  reg: (B, L, H)
"""

import argparse
from typing import Dict

import torch
import torch.nn as nn


class RegHead(nn.Module):
    """
    Query tabanlı satır-residual başlığı.

    Girdi:
      t_blc: (B, L, C)
    Çıktı:
      reg:   (B, L, H), normalize residual in [-1, 1]
    """

    def __init__(self, feat_dim: int = 32, feat_h: int = 10, num_queries: int = 7):
        super().__init__()
        self.feat_dim = feat_dim
        self.feat_h = feat_h
        self.num_queries = num_queries

        self.mlp_reg = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_h),
            nn.Tanh(),
        )

    def forward(self, t_blc: torch.Tensor) -> torch.Tensor:
        if t_blc.dim() != 3:
            raise ValueError(f"t_blc must be 3D, got {tuple(t_blc.shape)}")
        bsz, lanes, channels = t_blc.shape
        if lanes != self.num_queries:
            raise ValueError(f"lane dim={lanes}, expected num_queries={self.num_queries}")
        if channels != self.feat_dim:
            raise ValueError(f"channel dim={channels}, expected feat_dim={self.feat_dim}")
        return self.mlp_reg(t_blc)


def run(batch: int = 2, num_queries: int = 7, feat_dim: int = 32, feat_h: int = 10, seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = RegHead(feat_dim=feat_dim, feat_h=feat_h, num_queries=num_queries).to(device)

    t = torch.randn(batch, num_queries, feat_dim, requires_grad=True, device=device)
    out = head(t)

    checks: Dict[str, bool] = {}
    checks["shape"] = tuple(out.shape) == (batch, num_queries, feat_h)
    checks["range"] = float(out.min().item()) >= -1.000001 and float(out.max().item()) <= 1.000001
    checks["finite"] = bool(torch.isfinite(out).all().item())

    loss = out.sum()
    loss.backward()
    grad_params_ok = True
    for p in head.parameters():
        if p.grad is None or float(p.grad.abs().mean().item()) < 1e-12:
            grad_params_ok = False
            break
    checks["grad_params"] = grad_params_ok
    checks["grad_input"] = t.grad is not None and float(t.grad.abs().mean().item()) > 1e-12

    t2 = torch.randn_like(t)
    out2 = head(t2)
    checks["sensitive_to_input"] = float((out - out2).abs().mean().item()) > 1e-6

    print("=" * 90)
    print("BÖLÜM 1 — REGHEAD İZOLE TEST")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Input T: {tuple(t.shape)}")
    print(f"Output reg: {tuple(out.shape)}")
    print(f"Range: [{float(out.min().item()):.4f}, {float(out.max().item()):.4f}]")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<22}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=7)
    parser.add_argument("--feat-dim", type=int, default=32)
    parser.add_argument("--feat-h", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        num_queries=args.num_queries,
        feat_dim=args.feat_dim,
        feat_h=args.feat_h,
        seed=args.seed,
    )
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 1 — RegHead izole: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

