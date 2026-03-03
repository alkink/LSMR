"""
BÖLÜM 5 (REG) — Reg postprocess + legacy adapter.

Not:
 - `test/culane.py` şu an `mask_to_lane_coords` (offset) çağırıyor.
 - Reg modeli `pred_offset` alanını adapter olarak verdiği için mevcut test pipeline bozulmaz.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _decode_vrange_rows(v_range: torch.Tensor, feat_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if v_range.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=v_range.device)
        return z, z

    if float(v_range.min().item()) >= -0.5 and float(v_range.max().item()) <= 1.5:
        vr = torch.sigmoid(v_range)
        s = (vr[:, 0] * (feat_h - 1)).round().long().clamp(0, feat_h - 1)
        e = (vr[:, 1] * (feat_h - 1)).round().long().clamp(0, feat_h - 1)
    else:
        s = v_range[:, 0].round().long().clamp(0, feat_h - 1)
        e = v_range[:, 1].round().long().clamp(0, feat_h - 1)
    return s, e


def reg_to_lane_coords(
    heatmap: torch.Tensor,  # (L,H,W)
    reg: torch.Tensor,      # (L,H)
    v_range: torch.Tensor,  # (L,2)
    scores: torch.Tensor,   # (L,2)
    score_thresh: float = 0.5,
    img_h: int = 360,
    img_w: int = 640,
) -> List[List[Tuple[int, int]]]:
    if heatmap.dim() != 3:
        raise ValueError(f"heatmap must be (L,H,W), got {tuple(heatmap.shape)}")
    if reg.dim() != 2:
        raise ValueError(f"reg must be (L,H), got {tuple(reg.shape)}")

    lanes, feat_h, feat_w = heatmap.shape
    if reg.shape != (lanes, feat_h):
        raise ValueError(f"reg shape {tuple(reg.shape)} != {(lanes, feat_h)}")

    col_idx = torch.arange(feat_w, device=heatmap.device, dtype=torch.float32)
    fg_prob = F.softmax(scores, dim=-1)[:, 1]
    row_start, row_end = _decode_vrange_rows(v_range, feat_h)

    scale_x = img_w / float(max(feat_w - 1, 1))
    scale_y = img_h / float(feat_h)

    out: List[List[Tuple[int, int]]] = []
    for l in range(lanes):
        if float(fg_prob[l].item()) < score_thresh:
            continue

        s = int(row_start[l].item())
        e = int(row_end[l].item())
        if s > e:
            s, e = e, s

        pts: List[Tuple[int, int]] = []
        for r in range(s, e + 1):
            row_hm = heatmap[l, r]
            if float(row_hm.sum().item()) < 1e-6:
                continue

            x_coarse = float((row_hm * col_idx).sum().item())
            x_fine = x_coarse + float(reg[l, r].item()) * feat_w
            x_fine = max(0.0, min(float(feat_w - 1), x_fine))

            x_img = int(round(x_fine * scale_x))
            y_img = int(round((r + 0.5) * scale_y))
            pts.append((x_img, y_img))

        if len(pts) >= 2:
            out.append(pts)

    return out

