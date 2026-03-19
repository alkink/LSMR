from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


LanePoints = Sequence[Tuple[float, float]]


def _scale_lane_points(
    lane: LanePoints,
    image_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    image_h, image_w = image_size
    target_h, target_w = target_size

    lane_array = np.asarray(lane, dtype=np.float32)
    if lane_array.ndim != 2 or lane_array.shape[0] == 0 or lane_array.shape[1] != 2:
        return None

    finite_mask = np.isfinite(lane_array).all(axis=1)
    lane_array = lane_array[finite_mask]
    if lane_array.shape[0] == 0:
        return None

    valid_y = (lane_array[:, 1] >= 0.0) & (lane_array[:, 1] <= float(max(image_h - 1, 0)))
    lane_array = lane_array[valid_y]
    if lane_array.shape[0] == 0:
        return None

    order = np.argsort(lane_array[:, 1], kind='mergesort')
    lane_array = lane_array[order]

    y_scale = 0.0 if image_h <= 1 else float(target_h - 1) / float(image_h - 1)
    x_scale = 0.0 if image_w <= 1 else float(target_w - 1) / float(image_w - 1)

    ys = lane_array[:, 1] * y_scale
    xs = lane_array[:, 0] * x_scale

    unique_ys, inverse = np.unique(ys, return_inverse=True)
    unique_xs = np.zeros_like(unique_ys, dtype=np.float32)
    counts = np.zeros_like(unique_ys, dtype=np.float32)
    np.add.at(unique_xs, inverse, xs)
    np.add.at(counts, inverse, 1.0)
    unique_xs = unique_xs / np.clip(counts, a_min=1.0, a_max=None)

    return unique_ys.astype(np.float32), unique_xs.astype(np.float32)


def _sample_lane_rows(
    scaled_lane: Tuple[np.ndarray, np.ndarray],
    target_height: int,
    target_width: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ys, xs = scaled_lane
    if ys.size == 0:
        return None

    if ys.size == 1:
        row_index = int(round(float(ys[0])))
        x_value = float(xs[0])
        if 0 <= row_index < target_height and 0.0 <= x_value <= float(target_width - 1):
            return (
                np.asarray([row_index], dtype=np.int64),
                np.asarray([x_value], dtype=np.float32),
            )
        return None

    row_coords = np.arange(target_height, dtype=np.float32)
    in_span = (row_coords >= ys[0]) & (row_coords <= ys[-1])
    if not np.any(in_span):
        return None

    sampled_rows = row_coords[in_span]
    sampled_xs = np.interp(sampled_rows, ys, xs).astype(np.float32)
    inside_image = (sampled_xs >= 0.0) & (sampled_xs <= float(target_width - 1))
    if not np.any(inside_image):
        return None

    return sampled_rows[inside_image].astype(np.int64), sampled_xs[inside_image]


def convert_culane_points_to_rowwise_targets(
    lanes: Sequence[LanePoints],
    image_size: Tuple[int, int],
    target_size: Tuple[int, int],
    lane_attrs: Optional[Sequence[int]] = None,
    line_width: float = 1.0,
    min_valid_rows: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert CULane-style lane point annotations into CondLSTR-style row-wise targets.

    Assumptions:
    - `lanes` are point lists already expressed in the coordinate frame given by `image_size`.
    - Piecewise-linear interpolation is performed as x=f(y) after sorting points by y.
    - `target_size` is the dense supervision grid for the future CondLSTR-style loss.
    - `gt_row_rng` is normalized by `target_height`, matching the reference loss convention.
    - Rows whose interpolated x falls outside the target width are discarded.
    """
    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError('target_size must contain positive height and width')
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError('image_size must contain positive height and width')
    if min_valid_rows <= 0:
        raise ValueError('min_valid_rows must be positive')
    if lane_attrs is not None and len(lane_attrs) != len(lanes):
        raise ValueError('lane_attrs length must match lanes length')

    device = device or torch.device('cpu')
    regression_radius = float(line_width) * 4.0
    column_coords = torch.arange(target_w, dtype=torch.float32, device=device)

    row_ranges = []
    row_locations = []
    row_regressions = []
    row_location_masks = []
    row_regression_masks = []
    label_objects = []
    label_classes = []

    for lane_index, lane in enumerate(lanes):
        scaled_lane = _scale_lane_points(lane, image_size=image_size, target_size=target_size)
        if scaled_lane is None:
            continue

        sampled_lane = _sample_lane_rows(scaled_lane, target_height=target_h, target_width=target_w)
        if sampled_lane is None:
            continue

        lane_rows_np, lane_xs_np = sampled_lane
        if lane_rows_np.size < min_valid_rows:
            continue

        lane_rows = torch.as_tensor(lane_rows_np, dtype=torch.long, device=device)
        lane_xs = torch.as_tensor(lane_xs_np, dtype=torch.float32, device=device)

        row_loc = torch.zeros(target_h, dtype=torch.float32, device=device)
        row_loc_mask = torch.zeros(target_h, dtype=torch.float32, device=device)
        row_reg = torch.zeros((target_h, target_w), dtype=torch.float32, device=device)
        row_reg_mask = torch.zeros((target_h, target_w), dtype=torch.float32, device=device)

        row_loc[lane_rows] = lane_xs
        row_loc_mask[lane_rows] = 1.0

        lane_reg = lane_xs[:, None] - column_coords[None, :]
        lane_reg_mask = (lane_reg >= -regression_radius) & (lane_reg <= regression_radius)
        row_reg[lane_rows] = lane_reg
        row_reg_mask[lane_rows] = lane_reg_mask.float()

        row_ranges.append(
            torch.tensor(
                [float(lane_rows[0].item()) / float(target_h), float(lane_rows[-1].item()) / float(target_h)],
                dtype=torch.float32,
                device=device,
            )
        )
        row_locations.append(row_loc)
        row_regressions.append(row_reg)
        row_location_masks.append(row_loc_mask)
        row_regression_masks.append(row_reg_mask)
        label_objects.append(torch.tensor(0, dtype=torch.int64, device=device))
        label_classes.append(
            torch.tensor(
                0 if lane_attrs is None else int(lane_attrs[lane_index]),
                dtype=torch.int64,
                device=device,
            )
        )

    if len(row_ranges) == 0:
        return {
            'gt_row_rng': torch.zeros((0, 2), dtype=torch.float32, device=device),
            'gt_row_loc': torch.zeros((0, target_h), dtype=torch.float32, device=device),
            'gt_row_reg': torch.zeros((0, target_h, target_w), dtype=torch.float32, device=device),
            'gt_row_loc_mask': torch.zeros((0, target_h), dtype=torch.float32, device=device),
            'gt_row_reg_mask': torch.zeros((0, target_h, target_w), dtype=torch.float32, device=device),
            'gt_label_obj': torch.zeros((0,), dtype=torch.int64, device=device),
            'gt_label_cls': torch.zeros((0,), dtype=torch.int64, device=device),
        }

    return {
        'gt_row_rng': torch.stack(row_ranges, dim=0),
        'gt_row_loc': torch.stack(row_locations, dim=0),
        'gt_row_reg': torch.stack(row_regressions, dim=0),
        'gt_row_loc_mask': torch.stack(row_location_masks, dim=0),
        'gt_row_reg_mask': torch.stack(row_regression_masks, dim=0),
        'gt_label_obj': torch.stack(label_objects, dim=0),
        'gt_label_cls': torch.stack(label_classes, dim=0),
    }
