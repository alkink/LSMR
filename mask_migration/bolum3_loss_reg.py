"""
BÖLÜM 3 (REG) — Reg tabanlı loss.

Offset (B,L,H,W) yerine reg (B,L,H) kullanır.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_mask_loss_reg(
    pred_heatmap: torch.Tensor,   # (B,L,H,W)
    pred_reg: torch.Tensor,       # (B,L,H)
    pred_vrange: torch.Tensor,    # (B,L,2)
    pred_scores: torch.Tensor,    # (B,L,2)
    gt_heatmap: torch.Tensor,     # (B,L,H,W)
    gt_reg: torch.Tensor,         # (B,L,H)
    gt_vrange: torch.Tensor,      # (B,L,2)
    gt_labels: torch.Tensor,      # (B,L)
    gt_valid_mask: torch.Tensor,  # (B,L,H)
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
    vrange_total = pred_heatmap.new_zeros(())
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

                vrange_total = vrange_total + F.l1_loss(
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
        vrange_loss = vrange_total / n_matches
    else:
        heat_loss = pred_heatmap.new_zeros(())
        reg_loss = pred_heatmap.new_zeros(())
        vrange_loss = pred_heatmap.new_zeros(())

    total = cls_loss + heat_loss + reg_loss + vrange_loss
    return {
        "total": total,
        "heat_loss": heat_loss,
        "reg_loss": reg_loss,
        "vrange_loss": vrange_loss,
        "cls_loss": cls_loss,
        "class_error": class_error,
    }

