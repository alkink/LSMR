"""
BÖLÜM 3 — Mask tabanlı loss.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_mask_loss(
    pred_heatmap: torch.Tensor,
    pred_offset: torch.Tensor,
    pred_vrange: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_heatmap: torch.Tensor,
    gt_offset: torch.Tensor,
    gt_vrange: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
        total, heat_loss, offset_loss, vrange_loss, cls_loss, class_error
    """
    del gt_offset  # signature parity; current loss uses gt_x derived from heatmap

    device = pred_heatmap.device
    B, L, H, W = pred_heatmap.shape
    cols = torch.arange(W, device=device, dtype=pred_heatmap.dtype)

    pred_x = (pred_heatmap * cols.view(1, 1, 1, W)).sum(dim=-1)  # (B, L, H)
    gt_x = (gt_heatmap * cols.view(1, 1, 1, W)).sum(dim=-1)      # (B, L, H)

    heat_total = pred_heatmap.new_zeros(())
    offset_total = pred_heatmap.new_zeros(())
    vrange_total = pred_heatmap.new_zeros(())
    cls_total = pred_heatmap.new_zeros(())
    class_error_total = pred_heatmap.new_zeros(())
    n_matches = 0

    for b in range(B):
        gt_fg = (gt_labels[b] > 0)
        gt_idx = torch.nonzero(gt_fg, as_tuple=False).squeeze(-1)

        cls_target = torch.zeros(L, dtype=torch.long, device=device)

        if gt_idx.numel() > 0:
            fg_prob = F.softmax(pred_scores[b], dim=-1)[:, 1]  # (L,)
            cost = torch.zeros((L, gt_idx.numel()), device=device)

            for j, g in enumerate(gt_idx):
                valid = gt_valid_mask[b, g]  # (H,)
                if valid.any():
                    heat_cost = torch.abs(pred_x[b, :, valid] - gt_x[b, g, valid].unsqueeze(0)).mean(dim=1)
                else:
                    heat_cost = torch.full((L,), 2.0, device=device)

                range_cost = torch.abs(pred_vrange[b] - gt_vrange[b, g].float().unsqueeze(0)).mean(dim=1)
                cls_cost = 1.0 - fg_prob
                cost[:, j] = cls_cost + heat_cost + 0.25 * range_cost

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            pred_match = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            gt_match = gt_idx[torch.as_tensor(col_ind, dtype=torch.long, device=device)]

            cls_target[pred_match] = 1

            for p_i, g_i in zip(pred_match.tolist(), gt_match.tolist()):
                valid_rows = gt_valid_mask[b, g_i]

                if valid_rows.any():
                    heat_total = heat_total + torch.abs(pred_x[b, p_i, valid_rows] - gt_x[b, g_i, valid_rows]).mean()

                    row_ids = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
                    fine_losses = []
                    for r in row_ids.tolist():
                        n_col = int(torch.round(pred_x[b, p_i, r]).detach().item())
                        n_col = max(0, min(W - 1, n_col))
                        pred_fine_x = pred_x[b, p_i, r] + pred_offset[b, p_i, r, n_col]
                        fine_losses.append(torch.abs(pred_fine_x - gt_x[b, g_i, r]))

                    if fine_losses:
                        offset_total = offset_total + torch.stack(fine_losses).mean()

                vrange_total = vrange_total + F.l1_loss(
                    pred_vrange[b, p_i],
                    gt_vrange[b, g_i].float(),
                    reduction="mean",
                )
                n_matches += 1

        cls_total = cls_total + F.cross_entropy(pred_scores[b], cls_target, reduction="mean")
        class_error_total = class_error_total + (pred_scores[b].argmax(dim=-1) != cls_target).float().mean() * 100.0

    cls_loss = cls_total / max(B, 1)
    class_error = class_error_total / max(B, 1)

    if n_matches > 0:
        heat_loss = heat_total / n_matches
        offset_loss = offset_total / n_matches
        vrange_loss = vrange_total / n_matches
    else:
        heat_loss = pred_heatmap.new_zeros(())
        offset_loss = pred_heatmap.new_zeros(())
        vrange_loss = pred_heatmap.new_zeros(())

    total = cls_loss + heat_loss + offset_loss + vrange_loss
    return {
        "total": total,
        "heat_loss": heat_loss,
        "offset_loss": offset_loss,
        "vrange_loss": vrange_loss,
        "cls_loss": cls_loss,
        "class_error": class_error,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script.")

    from mask_migration.bolum1_gt_mask import lanes_to_mask_gt
    from mask_migration.bolum2_mask_head import DynamicMaskHead

    print("=" * 80)
    print("BÖLÜM 3 — LOSS İZOLE TEST")
    print("=" * 80)

    B, L, H, W = 2, 7, 12, 20
    head = DynamicMaskHead(num_queries=L, feat_dim=32, feat_h=H, feat_w=W).cuda()
    T = torch.randn(L, B, 32, device="cuda", requires_grad=True)
    M = torch.randn(B, 32, H, W, device="cuda", requires_grad=True)
    pred = head(T, M)

    fake_lanes_b0 = [[(150, 300), (160, 250), (170, 200)], [(400, 300), (410, 250), (420, 200)]]
    fake_lanes_b1 = [[(120, 280), (130, 230), (140, 180)]]
    gt_list = [
        lanes_to_mask_gt(fake_lanes_b0, feat_h=H, feat_w=W, num_lanes=L),
        lanes_to_mask_gt(fake_lanes_b1, feat_h=H, feat_w=W, num_lanes=L),
    ]

    gt_hm = torch.stack([g["heatmap"] for g in gt_list]).cuda()
    gt_off = torch.stack([g["offset"] for g in gt_list]).cuda()
    gt_vr = torch.stack([g["v_range"] for g in gt_list]).cuda()
    gt_lbl = torch.stack([g["labels"] for g in gt_list]).cuda()
    gt_vm = torch.stack([g["valid_mask"] for g in gt_list]).cuda()

    losses = compute_mask_loss(
        pred["heatmap"],
        pred["offset"],
        pred["v_range"],
        pred["scores"],
        gt_hm,
        gt_off,
        gt_vr,
        gt_lbl,
        gt_vm,
    )

    print("\nLOSS DEĞERLERİ:")
    for k, v in losses.items():
        val = float(v.item())
        ok = "✅" if not (np.isnan(val) or np.isinf(val)) else "🚨"
        print(f"  {k:<15}: {val:.6f} {ok}")

    losses["total"].backward()
    print("\nBackward sonrası:")
    print(f"  T.grad: {'✅' if T.grad is not None else '🚨'}")
    print(f"  M.grad: {'✅' if M.grad is not None else '🚨'}")

    all_finite = all(np.isfinite(float(v.item())) for v in losses.values())
    print("\n✅ Loss izole test GEÇTİ" if all_finite else "\n🚨 Loss'ta NaN/Inf var")


if __name__ == "__main__":
    main()

