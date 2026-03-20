from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _require_single_channel_dense_output(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 5 and tensor.size(2) == 1:
        return tensor.squeeze(2)
    raise ValueError(f'{name} must have shape [B, Q, H, W] or [B, Q, 1, H, W], got {tuple(tensor.shape)}')


def _decode_query_lane_points(
    row_locations: torch.Tensor,
    pred_range: torch.Tensor,
    feature_width: int,
    image_h: int,
    image_w: int,
    min_points: int,
) -> List[Tuple[float, float]]:
    feat_h = int(row_locations.size(0))
    feat_w = int(feature_width)
    row_start = int(torch.round(pred_range[0] * float(feat_h - 1)).item())
    row_end = int(torch.round(pred_range[1] * float(feat_h - 1)).item())
    row_start = max(0, min(feat_h - 1, row_start))
    row_end = max(0, min(feat_h - 1, row_end))
    if row_start > row_end:
        row_start, row_end = row_end, row_start

    y_scale = float(max(image_h - 1, 0)) / float(max(feat_h - 1, 1))
    x_scale = float(max(image_w - 1, 0)) / float(max(feat_w - 1, 1))

    points: List[Tuple[float, float]] = []
    for row in range(row_start, row_end + 1):
        x_value = float(row_locations[row].item())
        if x_value < 0.0 or x_value > float(max(feat_w - 1, 0)):
            continue
        points.append((x_value * x_scale, float(row) * y_scale))

    if len(points) < int(min_points):
        return []
    return points


def parity_outputs_to_lane_coords(
    outputs: Dict[str, torch.Tensor],
    target_sizes: torch.Tensor,
    score_thresh: float = 0.7,
    min_points: int = 2,
    max_lanes: int | None = None,
) -> List[List[List[Tuple[float, float]]]]:
    required = {'pred_object_logits', 'pred_ranges', 'pred_dense_mask', 'pred_dense_reg'}
    missing = sorted(required.difference(outputs.keys()))
    if missing:
        raise KeyError(f'parity outputs missing required keys: {missing}')

    pred_object_logits = outputs['pred_object_logits']
    pred_ranges = outputs['pred_ranges']
    pred_dense_mask = _require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
    pred_dense_reg = _require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')

    object_probs = F.softmax(pred_object_logits, dim=-1)[..., 0]
    row_probabilities = pred_dense_mask.softmax(dim=-1)
    column_positions = torch.arange(pred_dense_mask.size(-1), dtype=pred_dense_mask.dtype, device=pred_dense_mask.device)
    row_centers = (row_probabilities * column_positions.view(1, 1, 1, -1)).sum(dim=-1)
    rounded_centers = row_centers.round().long().clamp(min=0, max=pred_dense_mask.size(-1) - 1)
    row_offsets = pred_dense_reg.gather(dim=-1, index=rounded_centers.unsqueeze(-1)).squeeze(-1)
    row_locations = rounded_centers.float() + row_offsets

    lanes_batch: List[List[List[Tuple[float, float]]]] = []
    for batch_index in range(pred_object_logits.size(0)):
        image_h = int(target_sizes[batch_index, 0].item())
        image_w = int(target_sizes[batch_index, 1].item())

        keep = torch.nonzero(object_probs[batch_index] >= float(score_thresh), as_tuple=False).flatten()
        if keep.numel() == 0:
            lanes_batch.append([])
            continue

        keep_scores = object_probs[batch_index, keep]
        order = torch.argsort(keep_scores, descending=True)
        keep = keep[order]
        if max_lanes is not None:
            keep = keep[: max(0, int(max_lanes))]

        image_lanes: List[List[Tuple[float, float]]] = []
        for query_index in keep.tolist():
            lane_points = _decode_query_lane_points(
                row_locations=row_locations[batch_index, query_index],
                pred_range=pred_ranges[batch_index, query_index],
                feature_width=int(pred_dense_mask.size(-1)),
                image_h=image_h,
                image_w=image_w,
                min_points=min_points,
            )
            if lane_points:
                image_lanes.append(lane_points)
        lanes_batch.append(image_lanes)

    return lanes_batch
