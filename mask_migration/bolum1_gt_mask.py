"""
BÖLÜM 1 — GT mask üretici.

Bu modül hem script olarak çalışır hem de eğitim/loss tarafında utility olarak kullanılır.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def _to_np_label(label_tensor) -> np.ndarray:
    if isinstance(label_tensor, torch.Tensor):
        arr = label_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(label_tensor)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)


def label_tensor_to_lane_points(label_tensor) -> List[List[Tuple[float, float]]]:
    """
    LSTR label format -> normalized lane points list.

    label row format:
      [cls, lower, upper, x_0..x_{P-1}, y_0..y_{P-1}]
    where x/y are normalized [0,1], invalids are large negative values.
    """
    arr = _to_np_label(label_tensor)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label tensor, got shape={arr.shape}")

    dim = arr.shape[1]
    max_points = (dim - 3) // 2

    lanes = []
    for lane in arr:
        if lane[0] <= 0:
            continue
        xs = lane[3 : 3 + max_points]
        ys = lane[3 + max_points : 3 + 2 * max_points]
        valid = (xs >= 0.0) & (ys >= 0.0)
        if not np.any(valid):
            continue
        pts = [(float(x), float(y)) for x, y in zip(xs[valid], ys[valid])]
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def lanes_to_mask_gt(
    lanes_xy_list: Sequence[Sequence[Tuple[float, float]]],
    img_h: int = 360,
    img_w: int = 640,
    feat_h: int = 12,
    feat_w: int = 20,
    num_lanes: int = 7,
    normalized: bool = False,
    sigma: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """
    lanes_xy_list:
      - normalized=False: points are pixel coordinates (x,y)
      - normalized=True:  points are normalized coordinates in [0,1]
    """
    heatmap = torch.zeros(num_lanes, feat_h, feat_w, dtype=torch.float32)
    offset = torch.zeros(num_lanes, feat_h, feat_w, dtype=torch.float32)
    v_range = torch.zeros(num_lanes, 2, dtype=torch.long)
    labels = torch.zeros(num_lanes, dtype=torch.long)
    valid_mask = torch.zeros(num_lanes, feat_h, dtype=torch.bool)

    cols = torch.arange(feat_w, dtype=torch.float32)

    for lane_idx, lane_pts in enumerate(lanes_xy_list):
        if lane_idx >= num_lanes:
            break
        if lane_pts is None or len(lane_pts) < 2:
            continue

        labels[lane_idx] = 1
        row_to_x = {}

        for x_in, y_in in lane_pts:
            if normalized:
                x_norm = float(x_in)
                y_norm = float(y_in)
            else:
                x_norm = float(x_in) / float(img_w)
                y_norm = float(y_in) / float(img_h)

            if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                continue

            fr = int(round(y_norm * (feat_h - 1)))
            fx = x_norm * (feat_w - 1)
            if 0 <= fr < feat_h:
                row_to_x.setdefault(fr, []).append(fx)

        if not row_to_x:
            labels[lane_idx] = 0
            continue

        rows = sorted(row_to_x.keys())
        v_range[lane_idx, 0] = rows[0]
        v_range[lane_idx, 1] = rows[-1]

        for fr in rows:
            fx = float(np.mean(row_to_x[fr]))
            valid_mask[lane_idx, fr] = True

            dist = cols - fx
            row_hm = torch.exp(-0.5 * (dist / sigma) ** 2)
            row_sum = row_hm.sum()
            if row_sum > 0:
                row_hm = row_hm / row_sum

            heatmap[lane_idx, fr, :] = row_hm
            offset[lane_idx, fr, :] = fx - cols

    return {
        "heatmap": heatmap,
        "offset": offset,
        "v_range": v_range,
        "labels": labels,
        "valid_mask": valid_mask,
    }


def labels_to_mask_batch_gt(
    label_tensors: Sequence,
    feat_h: int,
    feat_w: int,
    num_lanes: int = 7,
) -> Dict[str, torch.Tensor]:
    hm, off, vr, lbl, vm = [], [], [], [], []
    for lt in label_tensors:
        lanes_norm = label_tensor_to_lane_points(lt)
        gt = lanes_to_mask_gt(
            lanes_norm,
            feat_h=feat_h,
            feat_w=feat_w,
            num_lanes=num_lanes,
            normalized=True,
        )
        hm.append(gt["heatmap"])
        off.append(gt["offset"])
        vr.append(gt["v_range"])
        lbl.append(gt["labels"])
        vm.append(gt["valid_mask"])

    return {
        "heatmap": torch.stack(hm, dim=0),
        "offset": torch.stack(off, dim=0),
        "v_range": torch.stack(vr, dim=0),
        "labels": torch.stack(lbl, dim=0),
        "valid_mask": torch.stack(vm, dim=0),
    }


def main():
    # Bölüm 0 / 1b gerçek model shape: input_proj -> (B, 32, 10, 26)
    FEAT_H, FEAT_W = 10, 26
    IMG_H, IMG_W = 360, 640
    NUM_LANES = 7

    print("=" * 80)
    print("BÖLÜM 1.1 — GT FORMAT TESTİ")
    print("=" * 80)

    fake_lane_1 = [(150, 300), (160, 250), (170, 200), (180, 150)]
    fake_lane_2 = [(400, 300), (410, 250), (420, 200), (430, 150)]

    gt = lanes_to_mask_gt(
        [fake_lane_1, fake_lane_2],
        img_h=IMG_H,
        img_w=IMG_W,
        feat_h=FEAT_H,
        feat_w=FEAT_W,
        num_lanes=NUM_LANES,
    )

    print(f"heatmap shape:  {tuple(gt['heatmap'].shape)}   beklenen: ({NUM_LANES}, {FEAT_H}, {FEAT_W})")
    print(f"offset shape:   {tuple(gt['offset'].shape)}   beklenen: ({NUM_LANES}, {FEAT_H}, {FEAT_W})")
    print(f"v_range shape:  {tuple(gt['v_range'].shape)}   beklenen: ({NUM_LANES}, 2)")
    print(f"labels shape:   {tuple(gt['labels'].shape)}   beklenen: ({NUM_LANES},)")

    print(f"\nlabels: {gt['labels'].tolist()}")
    print(f"v_range[0]: {gt['v_range'][0].tolist()}")
    print(f"v_range[1]: {gt['v_range'][1].tolist()}")

    print("\nHEATMAP SATIR TOPLAMLARI (valid satırlarda 1.0 olmalı):")
    for lane_idx in range(2):
        valid_rows = gt["valid_mask"][lane_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
        for fr in valid_rows:
            row_sum = gt["heatmap"][lane_idx, fr].sum().item()
            ok = "✅" if abs(row_sum - 1.0) < 1e-2 else "🚨"
            print(f"  Lane {lane_idx}, Row {fr:2d}: sum={row_sum:.4f} {ok}")

    print("\nWEIGHTED ARGMAX TESTİ:")
    cols = torch.arange(FEAT_W, dtype=torch.float32)
    for lane_idx in range(2):
        valid_rows = gt["valid_mask"][lane_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
        for fr in valid_rows:
            row_hm = gt["heatmap"][lane_idx, fr]
            n = float((row_hm * cols).sum().item())
            n_int = max(0, min(FEAT_W - 1, int(round(n))))
            z = float(gt["offset"][lane_idx, fr, n_int].item())
            x_feat = n + z
            x_img = x_feat * (IMG_W / max(FEAT_W - 1, 1))
            print(f"  Lane {lane_idx}, Row {fr:2d}: pred_feat_x={x_feat:.2f}, pred_img_x={x_img:.1f}")

    print("\n✅ GT üretici ÇALIŞIYOR")


if __name__ == "__main__":
    main()

