"""
BÖLÜM 3 — Yeni reg-loss (B,L,H) izole testi.

Offset (B,L,H,W) yerine reg (B,L,H) kullanılır.
"""

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mask_migration.bolum1_gt_mask import lanes_to_mask_gt
from reghead_analysis.bolum2_gt_reg import compute_gt_reg


def compute_reg_loss(pred_reg: torch.Tensor, gt_reg: torch.Tensor, gt_valid_mask: torch.Tensor) -> torch.Tensor:
    if pred_reg.shape != gt_reg.shape:
        raise ValueError(f"pred_reg shape={tuple(pred_reg.shape)} != gt_reg shape={tuple(gt_reg.shape)}")
    mask = gt_valid_mask.float()
    diff = torch.abs(pred_reg - gt_reg)
    return (diff * mask).sum() / mask.sum().clamp(min=1.0)


def compute_new_mask_loss(
    pred_heatmap: torch.Tensor,   # (B,L,H,W)
    pred_reg: torch.Tensor,       # (B,L,H)
    pred_vrange: torch.Tensor,    # (B,L,2)
    pred_scores: torch.Tensor,    # (B,L,2)
    gt_heatmap: torch.Tensor,     # (B,L,H,W)
    gt_reg: torch.Tensor,         # (B,L,H)
    gt_vrange: torch.Tensor,      # (B,L,2)
    gt_labels: torch.Tensor,      # (B,L)
    gt_valid_mask: torch.Tensor,  # (B,L,H)
    w_heat: float = 1.0,
    w_reg: float = 1.0,
    w_vr: float = 1.0,
    w_cls: float = 1.0,
) -> Dict[str, torch.Tensor]:
    device = pred_heatmap.device
    bsz, lq, h, w = pred_heatmap.shape
    cols = torch.arange(w, device=device, dtype=pred_heatmap.dtype)

    pred_x_coarse = (pred_heatmap * cols.view(1, 1, 1, w)).sum(dim=-1)  # (B,L,H)
    gt_x_coarse = (gt_heatmap * cols.view(1, 1, 1, w)).sum(dim=-1)      # (B,L,H)

    pred_x_fine = pred_x_coarse + pred_reg * w
    gt_x_fine = gt_x_coarse + gt_reg * w

    heat_total = pred_heatmap.new_zeros(())
    reg_total = pred_heatmap.new_zeros(())
    vr_total = pred_heatmap.new_zeros(())
    cls_total = pred_heatmap.new_zeros(())
    class_error_total = pred_heatmap.new_zeros(())
    n_matches = 0

    for b in range(bsz):
        gt_idx = torch.nonzero(gt_labels[b] > 0, as_tuple=False).squeeze(-1)
        cls_target = torch.zeros(lq, dtype=torch.long, device=device)

        if gt_idx.numel() > 0:
            fg_prob = F.softmax(pred_scores[b], dim=-1)[:, 1]
            cost = torch.zeros((lq, gt_idx.numel()), device=device)
            for j, g in enumerate(gt_idx):
                valid = gt_valid_mask[b, g]
                if bool(valid.any().item()):
                    heat_cost = torch.abs(pred_x_coarse[b, :, valid] - gt_x_coarse[b, g, valid].unsqueeze(0)).mean(dim=1)
                    reg_cost = torch.abs(pred_x_fine[b, :, valid] - gt_x_fine[b, g, valid].unsqueeze(0)).mean(dim=1)
                else:
                    heat_cost = torch.full((lq,), 2.0, device=device)
                    reg_cost = torch.full((lq,), 2.0, device=device)

                range_cost = torch.abs(pred_vrange[b] - gt_vrange[b, g].float().unsqueeze(0)).mean(dim=1)
                cls_cost = 1.0 - fg_prob
                cost[:, j] = cls_cost + heat_cost + reg_cost + 0.25 * range_cost

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            pred_match = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            gt_match = gt_idx[torch.as_tensor(col_ind, dtype=torch.long, device=device)]
            cls_target[pred_match] = 1

            for p_i, g_i in zip(pred_match.tolist(), gt_match.tolist()):
                valid = gt_valid_mask[b, g_i]
                if bool(valid.any().item()):
                    heat_total = heat_total + torch.abs(pred_x_coarse[b, p_i, valid] - gt_x_coarse[b, g_i, valid]).mean()
                    reg_total = reg_total + torch.abs(pred_x_fine[b, p_i, valid] - gt_x_fine[b, g_i, valid]).mean()

                vr_total = vr_total + F.l1_loss(
                    pred_vrange[b, p_i],
                    gt_vrange[b, g_i].float(),
                    reduction="mean",
                )
                n_matches += 1

        cls_total = cls_total + F.cross_entropy(pred_scores[b], cls_target, reduction="mean")
        class_error_total = class_error_total + (pred_scores[b].argmax(dim=-1) != cls_target).float().mean() * 100.0

    cls_loss = cls_total / max(bsz, 1)
    class_error = class_error_total / max(bsz, 1)

    if n_matches > 0:
        heat_loss = heat_total / n_matches
        reg_loss = reg_total / n_matches
        vrange_loss = vr_total / n_matches
    else:
        heat_loss = pred_heatmap.new_zeros(())
        reg_loss = pred_heatmap.new_zeros(())
        vrange_loss = pred_heatmap.new_zeros(())

    total = w_heat * heat_loss + w_reg * reg_loss + w_vr * vrange_loss + w_cls * cls_loss
    return {
        "total": total,
        "heat_loss": heat_loss,
        "reg_loss": reg_loss,
        "vrange_loss": vrange_loss,
        "cls_loss": cls_loss,
        "class_error": class_error,
    }


def run(seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz, lq, h, w = 2, 7, 10, 26

    pred_hm = F.softmax(torch.randn(bsz, lq, h, w, device=device), dim=-1)
    pred_reg = torch.tanh(torch.randn(bsz, lq, h, device=device))
    pred_vr = torch.randn(bsz, lq, 2, device=device)
    pred_sc = torch.randn(bsz, lq, 2, device=device)

    fake_lanes = [[(150, 250), (160, 200), (170, 150)], [(400, 250), (410, 200), (420, 150)]]
    gt = lanes_to_mask_gt(fake_lanes, img_h=295, img_w=820, feat_h=h, feat_w=w, num_lanes=lq)
    gt_hm = gt["heatmap"].unsqueeze(0).expand(bsz, -1, -1, -1).to(device)
    gt_vr = gt["v_range"].unsqueeze(0).expand(bsz, -1, -1).to(device)
    gt_lbl = gt["labels"].unsqueeze(0).expand(bsz, -1).to(device)
    gt_vm = gt["valid_mask"].unsqueeze(0).expand(bsz, -1, -1).to(device)
    gt_reg = compute_gt_reg(gt["heatmap"], gt["valid_mask"], feat_w=w, gt_offset=gt["offset"]).unsqueeze(0).expand(bsz, -1, -1).to(device)

    losses = compute_new_mask_loss(
        pred_hm,
        pred_reg,
        pred_vr,
        pred_sc,
        gt_hm,
        gt_reg,
        gt_vr,
        gt_lbl,
        gt_vm,
    )

    # gradient sanity
    hm_logits = torch.randn(bsz, lq, h, w, device=device, requires_grad=True)
    reg_raw = torch.randn(bsz, lq, h, device=device, requires_grad=True)
    vr_raw = torch.randn(bsz, lq, 2, device=device, requires_grad=True)
    sc_raw = torch.randn(bsz, lq, 2, device=device, requires_grad=True)

    losses_g = compute_new_mask_loss(
        F.softmax(hm_logits, dim=-1),
        torch.tanh(reg_raw),
        vr_raw,
        sc_raw,
        gt_hm,
        gt_reg,
        gt_vr,
        gt_lbl,
        gt_vm,
    )
    losses_g["total"].backward()

    checks: Dict[str, bool] = {}
    checks["finite_losses"] = all(bool(torch.isfinite(v).item()) for v in losses.values())
    checks["reg_loss_positive"] = float(losses["reg_loss"].item()) > 0.0
    checks["grad_hm"] = hm_logits.grad is not None and float(hm_logits.grad.abs().mean().item()) > 1e-12
    checks["grad_reg"] = reg_raw.grad is not None and float(reg_raw.grad.abs().mean().item()) > 1e-12
    checks["grad_vr"] = vr_raw.grad is not None and float(vr_raw.grad.abs().mean().item()) > 1e-12
    checks["grad_sc"] = sc_raw.grad is not None and float(sc_raw.grad.abs().mean().item()) > 1e-12

    print("=" * 90)
    print("BÖLÜM 3 — YENİ REG LOSS İZOLE TEST")
    print("=" * 90)
    for k, v in losses.items():
        print(f"  {k:<14}: {float(v.item()):.6f}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<18}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 3 — Yeni loss: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

