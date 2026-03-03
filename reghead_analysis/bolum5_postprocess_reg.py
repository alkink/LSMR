"""
BÖLÜM 5 — Yeni reg postprocess testi.

ESKİ:
  x_final = x_coarse + offset[row, n_col]

YENİ:
  x_final = x_coarse + reg[row] * feat_w
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mask_migration.bolum1_gt_mask import lanes_to_mask_gt
from reghead_analysis.bolum2_gt_reg import compute_gt_reg


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


def new_postprocess(
    heatmap: torch.Tensor,   # (L,H,W)
    reg: torch.Tensor,       # (L,H)
    vrange: torch.Tensor,    # (L,2)
    scores: torch.Tensor,    # (L,2)
    img_h: int,
    img_w: int,
    feat_h: int,
    feat_w: int,
    score_thresh: float = 0.5,
) -> List[List[Tuple[int, int]]]:
    if heatmap.dim() != 3:
        raise ValueError(f"heatmap must be (L,H,W), got {tuple(heatmap.shape)}")
    if reg.dim() != 2:
        raise ValueError(f"reg must be (L,H), got {tuple(reg.shape)}")

    lanes, h, w = heatmap.shape
    if h != feat_h or w != feat_w:
        raise ValueError(f"heatmap size {(h, w)} != given feat size {(feat_h, feat_w)}")

    col_idx = torch.arange(feat_w, device=heatmap.device, dtype=torch.float32)
    fg_prob = F.softmax(scores, dim=-1)[:, 1]
    row_start, row_end = _decode_vrange_rows(vrange, feat_h)

    scale_x = img_w / float(max(feat_w - 1, 1))
    scale_y = img_h / float(max(feat_h - 1, 1))

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
            y_img = int(round(float(r) * scale_y))
            pts.append((x_img, y_img))

        if len(pts) >= 2:
            out.append(pts)

    return out


def run() -> bool:
    feat_h, feat_w = 10, 26
    img_h, img_w = 295, 820
    num_lanes = 7

    fake_lane = [(160, 250), (170, 200), (180, 150), (190, 100)]
    gt = lanes_to_mask_gt([fake_lane], img_h=img_h, img_w=img_w, feat_h=feat_h, feat_w=feat_w, num_lanes=num_lanes)

    hm = gt["heatmap"]
    vm = gt["valid_mask"]
    vr = gt["v_range"].float()
    reg = compute_gt_reg(hm, vm, feat_w=feat_w, gt_offset=gt["offset"])

    scores = torch.full((num_lanes, 2), -8.0)
    scores[0, 1] = 8.0

    lanes = new_postprocess(
        hm,
        reg,
        vr,
        scores,
        img_h=img_h,
        img_w=img_w,
        feat_h=feat_h,
        feat_w=feat_w,
        score_thresh=0.5,
    )

    print("=" * 90)
    print("BÖLÜM 5 — REG POSTPROCESS TEST")
    print("=" * 90)
    print(f"Orijinal lane: {fake_lane}")
    print(f"Çıkarılan lane: {lanes[0] if lanes else 'BOŞ'}")

    if not lanes:
        print("🚨 Lane çıkarılamadı")
        return False

    orig = np.array(fake_lane, dtype=np.float32)
    errs = []
    for x_pred, y_pred in lanes[0]:
        idx = int(np.argmin(np.abs(orig[:, 1] - float(y_pred))))
        if abs(float(orig[idx, 1]) - float(y_pred)) <= 40.0:
            errs.append(abs(float(orig[idx, 0]) - float(x_pred)))

    if not errs:
        print("⚠️ Uygun y-eşleşmesi bulunamadı")
        return False

    mean_err = float(np.mean(errs))
    print(f"Ortalama x hatası: {mean_err:.2f} px")
    ok = mean_err < 30.0
    print(f"BÖLÜM 5: {'✅ GEÇTİ' if ok else '🚨 BAŞARISIZ'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    ok = run()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

