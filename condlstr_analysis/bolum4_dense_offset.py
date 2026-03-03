"""
BÖLÜM 4 — Dense offset supervision analizi (izole).

Amaç:
1) Mevcut offset loss (n_idx tek örnekleme) ile dense offset loss'u karşılaştırmak
2) Farkın sayısal büyüklüğünü raporlamak
3) Dense loss altında gradient akışını doğrulamak
"""

import argparse
from typing import Dict, Optional

import numpy as np
import torch

from mask_migration.bolum1_gt_mask import lanes_to_mask_gt
from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.bolum3_loss import compute_mask_loss


def dense_offset_loss_weak(
    pred_heatmap: torch.Tensor,  # (B,L,H,W)
    pred_offset: torch.Tensor,   # (B,L,H,W)
    gt_heatmap: torch.Tensor,    # (B,L,H,W)
    gt_offset: torch.Tensor,     # (B,L,H,W)
    gt_valid_mask: torch.Tensor, # (B,L,H)
) -> torch.Tensor:
    """
    Row-wise dense supervision:
      valid satırlarda W boyunca |pred_offset - gt_offset|, gt_heatmap ile ağırlıklı ortalama.
    """
    diff = torch.abs(pred_offset - gt_offset)
    w = gt_heatmap / gt_heatmap.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    row_loss = (diff * w).sum(dim=-1)  # (B,L,H)
    m = gt_valid_mask.float()
    return (row_loss * m).sum() / m.sum().clamp(min=1.0)


def dense_offset_loss_strong(
    pred_offset: torch.Tensor,   # (B,L,H,W)
    gt_offset: torch.Tensor,     # (B,L,H,W)
    gt_valid_mask: torch.Tensor, # (B,L,H)
    pred_heatmap: Optional[torch.Tensor] = None,
    gt_heatmap: Optional[torch.Tensor] = None,
    weak_mix: float = 0.25,
) -> torch.Tensor:
    """
    Daha güçlü dense sinyal:
      1) valid satırlarda tüm W boyunca L1 (full-row supervision)
      2) opsiyonel zayıf merkez-odaklı terim ile karışım

    Not:
      - full-row terimi, merkezde offset≈0 olmasından kaynaklı sinyal zayıflamasını azaltır.
    """
    diff = torch.abs(pred_offset - gt_offset)  # (B,L,H,W)
    m = gt_valid_mask.float().unsqueeze(-1)    # (B,L,H,1)

    full_row = (diff * m).sum() / m.sum().clamp(min=1.0) / pred_offset.shape[-1]

    if (
        pred_heatmap is not None
        and gt_heatmap is not None
        and weak_mix > 0.0
    ):
        weak = dense_offset_loss_weak(
            pred_heatmap=pred_heatmap,
            pred_offset=pred_offset,
            gt_heatmap=gt_heatmap,
            gt_offset=gt_offset,
            gt_valid_mask=gt_valid_mask,
        )
        return full_row + weak_mix * weak

    return full_row


# Geri uyumluluk: önceki isim güçlü sürüme yönlensin.
def dense_offset_loss(
    pred_heatmap: torch.Tensor,
    pred_offset: torch.Tensor,
    gt_heatmap: torch.Tensor,
    gt_offset: torch.Tensor,
    gt_valid_mask: torch.Tensor,
) -> torch.Tensor:
    return dense_offset_loss_strong(
        pred_offset=pred_offset,
        gt_offset=gt_offset,
        gt_valid_mask=gt_valid_mask,
        pred_heatmap=pred_heatmap,
        gt_heatmap=gt_heatmap,
    )


def _build_toy_gt(batch: int, lanes: int, h: int, w: int, device: torch.device):
    gt_list = []
    for b in range(batch):
        dx = 12 * b
        fake_lanes = [
            [(140 + dx, 300), (152 + dx, 250), (164 + dx, 200), (176 + dx, 150)],
            [(420 - dx, 300), (408 - dx, 250), (396 - dx, 200), (384 - dx, 150)],
        ]
        gt_list.append(lanes_to_mask_gt(fake_lanes, img_h=360, img_w=640, feat_h=h, feat_w=w, num_lanes=lanes))

    gt_hm = torch.stack([g["heatmap"] for g in gt_list]).to(device)
    gt_off = torch.stack([g["offset"] for g in gt_list]).to(device)
    gt_vr = torch.stack([g["v_range"] for g in gt_list]).to(device)
    gt_lbl = torch.stack([g["labels"] for g in gt_list]).to(device)
    gt_vm = torch.stack([g["valid_mask"] for g in gt_list]).to(device)
    return gt_hm, gt_off, gt_vr, gt_lbl, gt_vm


def run(seed: int = 0, dense_weight: float = 1.0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, c, h, w = 2, 7, 32, 10, 26
    head = DynamicMaskHead(num_queries=l, feat_dim=c, feat_h=h, feat_w=w).to(device)

    t = torch.randn(l, b, c, device=device, requires_grad=True)
    m = torch.randn(b, c, h, w, device=device, requires_grad=True)
    gt_hm, gt_off, gt_vr, gt_lbl, gt_vm = _build_toy_gt(b, l, h, w, device)

    pred = head(t, m)
    base_losses = compute_mask_loss(
        pred["heatmap"], pred["offset"], pred["v_range"], pred["scores"], gt_hm, gt_off, gt_vr, gt_lbl, gt_vm
    )

    dense_weak = dense_offset_loss_weak(pred["heatmap"], pred["offset"], gt_hm, gt_off, gt_vm)
    dense_strong = dense_offset_loss_strong(
        pred_offset=pred["offset"],
        gt_offset=gt_off,
        gt_valid_mask=gt_vm,
        pred_heatmap=pred["heatmap"],
        gt_heatmap=gt_hm,
        weak_mix=0.25,
    )
    total_dense = base_losses["total"] + dense_weight * dense_strong

    head.zero_grad(set_to_none=True)
    if t.grad is not None:
        t.grad.zero_()
    if m.grad is not None:
        m.grad.zero_()

    total_dense.backward()

    grad_alive = True
    for p in head.parameters():
        if p.grad is None or float(p.grad.abs().mean().item()) < 1e-12:
            grad_alive = False
            break

    checks: Dict[str, bool] = {}
    checks["dense_weak_positive"] = float(dense_weak.item()) >= 0.0
    checks["dense_strong_positive"] = float(dense_strong.item()) >= 0.0
    checks["dense_strong_not_tiny_vs_weak"] = float(dense_strong.item()) >= 0.5 * float(dense_weak.item())
    checks["dense_strong_vs_sparse_reasonable"] = float(dense_strong.item()) >= 0.75 * float(base_losses["offset_loss"].item())
    checks["dense_changes_total"] = abs(float(total_dense.item()) - float(base_losses["total"].item())) > 1e-9
    checks["finite"] = np.isfinite(float(total_dense.item()))
    checks["gradient_alive"] = grad_alive and (t.grad is not None) and (m.grad is not None)

    print("=" * 90)
    print("BÖLÜM 4 — DENSE OFFSET ANALİZİ")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"dense_weight: {dense_weight}")
    print(f"base_total : {float(base_losses['total'].item()):.6f}")
    print(f"base_offset: {float(base_losses['offset_loss'].item()):.6f}")
    print(f"dense_weak : {float(dense_weak.item()):.6f}")
    print(f"dense_strng: {float(dense_strong.item()):.6f}")
    print(f"total_dense: {float(total_dense.item()):.6f}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<24}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    args = parser.parse_args()

    checks = run(seed=args.seed, dense_weight=args.dense_weight)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 4 — Dense Offset: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

