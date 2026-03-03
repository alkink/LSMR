"""
BÖLÜM 5 — Mask çıktısından koordinata dönüşüm.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _decode_vrange_rows(v_range: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits/prob-like ise sigmoid + ölçekle, satır index ise doğrudan kullan
    if v_range.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=v_range.device)
        return z, z

    if v_range.min() >= -0.5 and v_range.max() <= 1.5:
        vr = torch.sigmoid(v_range)
        s = (vr[:, 0] * (H - 1)).round().long().clamp(0, H - 1)
        e = (vr[:, 1] * (H - 1)).round().long().clamp(0, H - 1)
    else:
        s = v_range[:, 0].round().long().clamp(0, H - 1)
        e = v_range[:, 1].round().long().clamp(0, H - 1)

    return s, e


def mask_to_lane_coords(
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    v_range: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float = 0.5,
    img_h: int = 360,
    img_w: int = 640,
) -> List[List[Tuple[int, int]]]:
    """
    heatmap: (L, H, W)
    offset:  (L, H, W)
    v_range: (L, 2)
    scores:  (L, 2)
    """
    if heatmap.dim() != 3:
        raise ValueError(f"heatmap must be (L,H,W), got {tuple(heatmap.shape)}")

    L, H, W = heatmap.shape
    col_idx = torch.arange(W, dtype=torch.float32, device=heatmap.device)

    fg_prob = F.softmax(scores, dim=-1)[:, 1]
    start_rows, end_rows = _decode_vrange_rows(v_range, H)

    # GT oluştururken x -> feat dönüşümü (feat_w - 1) ile yapılıyor.
    # Ters dönüşüm de simetrik olmalı.
    scale_x = img_w / float(max(W - 1, 1))
    scale_y = img_h / float(H)

    lanes = []
    for lane_id in range(L):
        if float(fg_prob[lane_id].item()) < score_thresh:
            continue

        s = int(start_rows[lane_id].item())
        e = int(end_rows[lane_id].item())
        if s > e:
            s, e = e, s

        pts = []
        for r in range(s, e + 1):
            row_hm = heatmap[lane_id, r]
            # Geçersiz satırlar (GT yok) x=0 üretmesin.
            if float(row_hm.sum().item()) < 1e-6:
                continue

            n_float = float((row_hm * col_idx).sum().item())
            n_idx = max(0, min(W - 1, int(n_float)))
            z = float(offset[lane_id, r, n_idx].item())
            x_feat = max(0.0, min(float(W - 1), n_float + z))

            x_img = int(round(x_feat * scale_x))
            y_img = int(round((r + 0.5) * scale_y))
            pts.append((x_img, y_img))

        if len(pts) >= 2:
            lanes.append(pts)

    return lanes


def main():
    from mask_migration.bolum1_gt_mask import lanes_to_mask_gt

    print("=" * 80)
    print("BÖLÜM 5 — POSTPROCESS TESTİ")
    print("=" * 80)

    FEAT_H, FEAT_W = 10, 26

    fake_lane = [(160, 250), (170, 200), (180, 150), (190, 100)]
    gt = lanes_to_mask_gt([fake_lane], img_h=360, img_w=640, feat_h=FEAT_H, feat_w=FEAT_W, num_lanes=7)

    hm = gt["heatmap"]
    off = gt["offset"]
    vr = gt["v_range"].float()

    scores_fake = torch.zeros(7, 2)
    scores_fake[0, 1] = 5.0
    scores_fake[0, 0] = -5.0

    lanes = mask_to_lane_coords(hm, off, vr, scores_fake, score_thresh=0.5, img_h=360, img_w=640)

    print(f"Orijinal lane: {fake_lane}")
    print(f"Çıkarılan lane: {lanes[0] if lanes else 'BOŞ'}")

    if lanes:
        orig_x = [p[0] for p in fake_lane]
        pred_x = [p[0] for p in lanes[0]]
        n = min(len(orig_x), len(pred_x))
        mean_err = float(np.mean([abs(orig_x[i] - pred_x[i]) for i in range(n)])) if n > 0 else 1e9
        print(f"Ortalama x hatası: {mean_err:.2f} px")
        print("✅ Postprocess çalışıyor" if mean_err < 30 else "⚠️  Hata büyük")
    else:
        print("🚨 Lane çıkarılamadı")


if __name__ == "__main__":
    main()

