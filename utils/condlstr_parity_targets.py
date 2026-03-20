from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from utils.condlstr_dense_targets import convert_culane_points_to_rowwise_targets


def legacy_label_tensor_to_lane_points(
    label_tensor: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
    if label_tensor.dim() == 3:
        if label_tensor.size(0) == 0:
            return [], []
        label_tensor = label_tensor[0]
    if label_tensor.dim() != 2:
        raise ValueError(f'label_tensor must be 2D [L, D], got shape {tuple(label_tensor.shape)}')

    image_h, image_w = image_size
    num_lane_points = max((label_tensor.size(1) - 3) // 2, 0)

    lanes: List[List[Tuple[float, float]]] = []
    lane_attrs: List[int] = []
    for lane in label_tensor:
        category = int(round(float(lane[0].item())))
        if category <= 0:
            continue

        xs = lane[3 : 3 + num_lane_points]
        ys = lane[3 + num_lane_points : 3 + 2 * num_lane_points]
        valid = torch.isfinite(xs) & torch.isfinite(ys) & (xs >= 0.0) & (ys >= 0.0)
        xs = xs[valid]
        ys = ys[valid]
        if xs.numel() < 2:
            continue

        lane_points = [
            (float(x.item()) * float(image_w), float(y.item()) * float(image_h))
            for x, y in zip(xs, ys)
        ]
        lanes.append(lane_points)
        lane_attrs.append(max(category - 1, 0))

    return lanes, lane_attrs


def build_parity_targets_from_legacy_targets(
    targets: Sequence[torch.Tensor],
    target_size: Tuple[int, int],
    device: torch.device,
    line_width: float = 16.0,
    min_valid_rows: int = 2,
) -> List[Dict[str, torch.Tensor]]:
    if len(targets) == 0:
        raise ValueError('targets must contain at least the image batch tensor')

    images = targets[0]
    if images.dim() != 4:
        raise ValueError(f'targets[0] must be image batch [B, C, H, W], got {tuple(images.shape)}')

    batch_size = int(images.size(0))
    image_size = (int(images.shape[-2]), int(images.shape[-1]))
    gt_label_tensors = [tgt[0] for tgt in targets[1:]]

    if len(gt_label_tensors) < batch_size and gt_label_tensors:
        gt_label_tensors = gt_label_tensors + [gt_label_tensors[-1]] * (batch_size - len(gt_label_tensors))
    elif len(gt_label_tensors) > batch_size:
        gt_label_tensors = gt_label_tensors[:batch_size]

    dense_targets: List[Dict[str, torch.Tensor]] = []
    for batch_index in range(batch_size):
        if batch_index < len(gt_label_tensors):
            lanes, lane_attrs = legacy_label_tensor_to_lane_points(
                gt_label_tensors[batch_index],
                image_size=image_size,
            )
        else:
            lanes, lane_attrs = [], []

        dense_targets.append(
            convert_culane_points_to_rowwise_targets(
                lanes=lanes,
                image_size=image_size,
                target_size=target_size,
                lane_attrs=lane_attrs,
                line_width=line_width,
                min_valid_rows=min_valid_rows,
                device=device,
            )
        )

    return dense_targets
