from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _require_single_channel_dense_output(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 5 and tensor.size(2) == 1:
        return tensor.squeeze(2)
    raise ValueError(f'{name} must have shape [B, Q, H, W] or [B, Q, 1, H, W], got {tuple(tensor.shape)}')


def _decode_row_range(pred_range: torch.Tensor, feat_h: int, range_format: str = 'normalized') -> Tuple[int, int]:
    if pred_range.numel() != 2:
        raise ValueError(f'pred_range must contain 2 values, got shape {tuple(pred_range.shape)}')

    start_v = float(pred_range[0].item())
    end_v = float(pred_range[1].item())

    if range_format not in {'normalized', 'absolute', 'auto'}:
        raise ValueError(f"range_format must be one of ['normalized', 'absolute', 'auto'], got {range_format!r}")

    if range_format == 'normalized':
        start_v = start_v * float(feat_h)
        end_v = end_v * float(feat_h)
    elif range_format == 'auto' and max(abs(start_v), abs(end_v)) <= 1.5:
        start_v = start_v * float(feat_h)
        end_v = end_v * float(feat_h)

    row_start = int(torch.floor(torch.tensor(start_v)).item())
    row_end = int(torch.ceil(torch.tensor(end_v)).item())
    row_start = max(0, min(feat_h - 1, row_start))
    row_end = max(0, min(feat_h - 1, row_end))
    if row_start > row_end:
        row_start, row_end = row_end, row_start
    return row_start, row_end


def _decode_query_lane_points(
    mask_logits: torch.Tensor,
    reg_map: torch.Tensor,
    pred_range: torch.Tensor,
    range_format: str,
    image_h: int,
    image_w: int,
    min_points: int,
) -> List[Tuple[float, float]]:
    feat_h, feat_w = int(mask_logits.shape[0]), int(mask_logits.shape[1])
    row_start, row_end = _decode_row_range(pred_range=pred_range, feat_h=feat_h, range_format=range_format)

    points: List[Tuple[float, float]] = []
    x_scale = float(max(image_w - 1, 0)) / float(max(feat_w - 1, 1))
    y_scale = float(max(image_h - 1, 0)) / float(max(feat_h - 1, 1))

    for row in range(row_start, row_end + 1):
        row_logits = mask_logits[row]
        col = int(torch.argmax(row_logits).item())
        x_dense = float(col) + float(reg_map[row, col].item())
        if x_dense < 0.0 or x_dense > float(feat_w - 1):
            continue

        x_img = x_dense * x_scale
        y_img = float(row) * y_scale
        if x_img < 0.0 or x_img > float(max(image_w - 1, 0)):
            continue
        points.append((x_img, y_img))

    if len(points) < int(min_points):
        return []
    return points


def dense_outputs_to_lane_coords(
    outputs: Dict[str, torch.Tensor],
    target_sizes: torch.Tensor,
    score_thresh: float = 0.5,
    min_points: int = 2,
    max_lanes: int | None = None,
    range_format: str = 'normalized',
) -> List[List[List[Tuple[float, float]]]]:
    """
    Decode CondLSTR-style dense outputs to per-image lane point lists.

    Expected outputs:
        pred_object_logits: [B, Q, 2]
        pred_ranges:        [B, Q, 2]
        pred_dense_mask:    [B, Q, H, W] or [B, Q, 1, H, W]
        pred_dense_reg:     [B, Q, H, W] or [B, Q, 1, H, W]
    """
    required = {'pred_object_logits', 'pred_ranges', 'pred_dense_mask', 'pred_dense_reg'}
    missing = sorted(required.difference(outputs.keys()))
    if missing:
        raise KeyError(f'dense outputs missing required keys: {missing}')

    pred_object_logits = outputs['pred_object_logits']
    pred_ranges = outputs['pred_ranges']
    pred_dense_mask = _require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
    pred_dense_reg = _require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')

    if pred_object_logits.dim() != 3 or pred_object_logits.size(-1) != 2:
        raise ValueError(f'pred_object_logits must have shape [B, Q, 2], got {tuple(pred_object_logits.shape)}')
    if pred_ranges.dim() != 3 or pred_ranges.size(-1) != 2:
        raise ValueError(f'pred_ranges must have shape [B, Q, 2], got {tuple(pred_ranges.shape)}')
    if pred_dense_mask.dim() != 4:
        raise ValueError(f'pred_dense_mask must have shape [B, Q, H, W], got {tuple(pred_dense_mask.shape)}')
    if pred_dense_reg.dim() != 4:
        raise ValueError(f'pred_dense_reg must have shape [B, Q, H, W], got {tuple(pred_dense_reg.shape)}')
    if pred_dense_mask.shape != pred_dense_reg.shape:
        raise ValueError('pred_dense_mask and pred_dense_reg must have identical shapes after channel normalization')
    if pred_ranges.shape[:2] != pred_object_logits.shape[:2] or pred_dense_mask.shape[:2] != pred_object_logits.shape[:2]:
        raise ValueError('all dense prediction tensors must share the same [B, Q] dimensions')
    if target_sizes.dim() != 2 or target_sizes.size(1) != 2:
        raise ValueError(f'target_sizes must have shape [B, 2], got {tuple(target_sizes.shape)}')
    if int(target_sizes.size(0)) != int(pred_object_logits.size(0)):
        raise ValueError('target_sizes batch size must match dense output batch size')

    object_probs = F.softmax(pred_object_logits, dim=-1)[..., 0]

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
                mask_logits=pred_dense_mask[batch_index, query_index],
                reg_map=pred_dense_reg[batch_index, query_index],
                pred_range=pred_ranges[batch_index, query_index],
                range_format=range_format,
                image_h=image_h,
                image_w=image_w,
                min_points=min_points,
            )
            if lane_points:
                image_lanes.append(lane_points)

        lanes_batch.append(image_lanes)

    return lanes_batch

