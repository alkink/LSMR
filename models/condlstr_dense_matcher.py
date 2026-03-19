from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class DenseMatchCostBreakdown:
    cost_object: torch.Tensor
    cost_class: torch.Tensor
    cost_row_location: torch.Tensor
    cost_row_iou: torch.Tensor
    cost_row_reg: torch.Tensor
    cost_row_range: torch.Tensor
    total_cost: torch.Tensor


class CondLSTRDenseHungarianMatcher(nn.Module):
    """
    Local standalone Hungarian matcher for dense CondLSTR-style row-wise targets.

    Supported prediction contract:
        pred_object_logits: [B, Q, 2]
        pred_class_logits:  [B, Q, C]
        pred_ranges:        [B, Q, 2]
        pred_dense_mask:    [B, Q, H, W] or [B, Q, 1, H, W]
        pred_dense_reg:     [B, Q, H, W] or [B, Q, 1, H, W]

    Supported target contract per image:
        gt_row_rng:      [M, 2]
        gt_row_loc:      [M, H]
        gt_row_reg:      [M, H, W]
        gt_row_loc_mask: [M, H]
        gt_row_reg_mask: [M, H, W]
        gt_label_obj:    [M]
        gt_label_cls:    [M]

    Intentional scope limits:
    - Only the single-channel dense mask/reg contract is supported for now.
    - The reference row-classification cost based on full mask CE is deferred.
    - Costs are computed independently per image and solved with Hungarian matching.
    - The dense regression cost currently uses a full [Q, M, H, W] broadcast, which is
      acceptable for small local tests but should be revisited before large dense grids.
    """

    def __init__(
        self,
        line_width: float,
        object_weight: float = 1.0,
        class_weight: float = 1.0,
        location_weight: float = 1.0,
        location_iou_weight: float = 2.0,
        regression_weight: float = 1.0,
        range_weight: float = 1.0,
        ignore_class_index: int = 255,
    ):
        super().__init__()
        self.object_weight = float(object_weight)
        self.class_weight = float(class_weight)
        self.location_weight = float(location_weight)
        self.location_iou_weight = float(location_iou_weight)
        self.regression_weight = float(regression_weight)
        self.range_weight = float(range_weight)
        self.line_width = float(line_width)
        self.ignore_class_index = int(ignore_class_index)

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
        return_costs: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]] | Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[DenseMatchCostBreakdown]]:
        pred_object_logits = outputs['pred_object_logits']
        pred_class_logits = outputs['pred_class_logits']
        pred_ranges = outputs['pred_ranges']
        pred_dense_mask = self._require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
        pred_dense_reg = self._require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')

        if pred_object_logits.dim() != 3 or pred_object_logits.size(-1) != 2:
            raise ValueError(
                f"pred_object_logits must have shape [B, Q, 2], got {tuple(pred_object_logits.shape)}"
            )
        if pred_class_logits.dim() != 3:
            raise ValueError(f"pred_class_logits must have shape [B, Q, C], got {tuple(pred_class_logits.shape)}")
        if pred_ranges.dim() != 3 or pred_ranges.size(-1) != 2:
            raise ValueError(f"pred_ranges must have shape [B, Q, 2], got {tuple(pred_ranges.shape)}")
        if pred_dense_mask.dim() != 4:
            raise ValueError(f"pred_dense_mask must have shape [B, Q, H, W], got {tuple(pred_dense_mask.shape)}")
        if pred_dense_reg.dim() != 4:
            raise ValueError(f"pred_dense_reg must have shape [B, Q, H, W], got {tuple(pred_dense_reg.shape)}")

        batch_size, num_queries = pred_object_logits.shape[:2]
        expected_dense_shape = (batch_size, num_queries)
        if pred_class_logits.shape[:2] != expected_dense_shape:
            raise ValueError('pred_class_logits batch/query dimensions must match pred_object_logits')
        if pred_ranges.shape[:2] != expected_dense_shape:
            raise ValueError('pred_ranges batch/query dimensions must match pred_object_logits')
        if pred_dense_mask.shape[:2] != expected_dense_shape:
            raise ValueError('pred_dense_mask batch/query dimensions must match pred_object_logits')
        if pred_dense_reg.shape[:2] != expected_dense_shape:
            raise ValueError('pred_dense_reg batch/query dimensions must match pred_object_logits')
        if len(targets) != batch_size:
            raise ValueError(f'targets length must equal batch size {batch_size}, got {len(targets)}')
        if pred_dense_mask.shape[-2:] != pred_dense_reg.shape[-2:]:
            raise ValueError('pred_dense_mask and pred_dense_reg must share the same spatial shape')

        row_locations = self._dense_mask_logits_to_row_locations(pred_dense_mask)

        match_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        cost_breakdowns: List[DenseMatchCostBreakdown] = []

        for batch_index, target in enumerate(targets):
            self._validate_target(target=target, spatial_shape=pred_dense_mask.shape[-2:], target_index=batch_index)
            num_targets = int(target['gt_row_rng'].size(0))
            if num_targets == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=pred_object_logits.device)
                match_indices.append((empty, empty))
                if return_costs:
                    empty_cost = pred_object_logits.new_zeros((num_queries, 0))
                    cost_breakdowns.append(
                        DenseMatchCostBreakdown(
                            cost_object=empty_cost,
                            cost_class=empty_cost,
                            cost_row_location=empty_cost,
                            cost_row_iou=empty_cost,
                            cost_row_reg=empty_cost,
                            cost_row_range=empty_cost,
                            total_cost=empty_cost,
                        )
                    )
                continue

            costs = self.compute_cost_matrix_for_image(
                pred_object_logits=pred_object_logits[batch_index],
                pred_class_logits=pred_class_logits[batch_index],
                pred_ranges=pred_ranges[batch_index],
                pred_row_locations=row_locations[batch_index],
                pred_dense_reg=pred_dense_reg[batch_index],
                target=target,
            )
            row_ind, col_ind = linear_sum_assignment(costs.total_cost.detach().cpu().numpy())
            match_indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=pred_object_logits.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=pred_object_logits.device),
                )
            )
            if return_costs:
                cost_breakdowns.append(costs)

        if return_costs:
            return match_indices, cost_breakdowns
        return match_indices

    def compute_cost_matrix_for_image(
        self,
        pred_object_logits: torch.Tensor,
        pred_class_logits: torch.Tensor,
        pred_ranges: torch.Tensor,
        pred_row_locations: torch.Tensor,
        pred_dense_reg: torch.Tensor,
        target: Dict[str, torch.Tensor],
        line_width: float | None = None,
    ) -> DenseMatchCostBreakdown:
        resolved_line_width = self.line_width if line_width is None else float(line_width)
        object_probabilities = pred_object_logits.softmax(dim=-1)
        target_object_labels = target['gt_label_obj'].long()
        cost_object = -object_probabilities[:, target_object_labels]

        class_probabilities = pred_class_logits.softmax(dim=-1)
        target_class_labels = target['gt_label_cls'].long().clone()
        ignored_class_mask = target_class_labels == self.ignore_class_index
        target_class_labels[ignored_class_mask] = 0
        cost_class = -class_probabilities[:, target_class_labels]
        if ignored_class_mask.any():
            cost_class = cost_class * (~ignored_class_mask).float().unsqueeze(0)

        target_row_locations = target['gt_row_loc']
        target_row_location_mask = target['gt_row_loc_mask']
        location_errors = torch.abs(
            pred_row_locations.unsqueeze(1) - target_row_locations.unsqueeze(0)
        )
        weighted_location_errors = location_errors * target_row_location_mask.unsqueeze(0)
        valid_location_counts = target_row_location_mask.sum(dim=1).clamp(min=1.0).unsqueeze(0)
        cost_row_location = weighted_location_errors.sum(dim=2) / valid_location_counts

        pred_row_left = pred_row_locations.unsqueeze(1) - resolved_line_width
        pred_row_right = pred_row_locations.unsqueeze(1) + resolved_line_width
        target_row_left = target_row_locations.unsqueeze(0) - resolved_line_width
        target_row_right = target_row_locations.unsqueeze(0) + resolved_line_width
        overlap = (
            torch.minimum(pred_row_right, target_row_right) - torch.maximum(pred_row_left, target_row_left)
        ).clamp(min=0.0)
        overlap = overlap * target_row_location_mask.unsqueeze(0)
        union = (
            torch.maximum(pred_row_right, target_row_right) - torch.minimum(pred_row_left, target_row_left)
        )
        union = union * target_row_location_mask.unsqueeze(0)
        row_iou = overlap.sum(dim=2) / (union.sum(dim=2) + 1e-9)
        cost_row_iou = 1.0 - row_iou

        target_row_reg = target['gt_row_reg']
        target_row_reg_mask = target['gt_row_reg_mask']
        regression_errors = torch.abs(pred_dense_reg.unsqueeze(1) - target_row_reg.unsqueeze(0))
        weighted_regression_errors = regression_errors * target_row_reg_mask.unsqueeze(0)
        valid_regression_counts = target_row_reg_mask.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(0)
        cost_row_reg = weighted_regression_errors.sum(dim=(2, 3)) / valid_regression_counts

        target_row_ranges = target['gt_row_rng']
        cost_row_range = torch.abs(pred_ranges.unsqueeze(1) - target_row_ranges.unsqueeze(0)).sum(dim=2)

        total_cost = (
            self.object_weight * cost_object
            + self.class_weight * cost_class
            + self.location_weight * cost_row_location
            + (self.location_weight * self.location_iou_weight) * cost_row_iou
            + self.regression_weight * cost_row_reg
            + self.range_weight * cost_row_range
        )

        return DenseMatchCostBreakdown(
            cost_object=cost_object,
            cost_class=cost_class,
            cost_row_location=cost_row_location,
            cost_row_iou=cost_row_iou,
            cost_row_reg=cost_row_reg,
            cost_row_range=cost_row_range,
            total_cost=total_cost,
        )

    @staticmethod
    def _require_single_channel_dense_output(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 5 and tensor.size(2) == 1:
            return tensor.squeeze(2)
        raise ValueError(
            f'{name} must have shape [B, Q, H, W] or [B, Q, 1, H, W], got {tuple(tensor.shape)}'
        )

    @staticmethod
    def _dense_mask_logits_to_row_locations(mask_logits: torch.Tensor) -> torch.Tensor:
        width = mask_logits.size(-1)
        row_probabilities = mask_logits.softmax(dim=-1)
        column_positions = torch.arange(width, dtype=mask_logits.dtype, device=mask_logits.device)
        return (row_probabilities * column_positions.view(1, 1, 1, width)).sum(dim=-1)

    @staticmethod
    def _validate_target(
        target: Dict[str, torch.Tensor],
        spatial_shape: Tuple[int, int],
        target_index: int,
    ) -> None:
        required_keys = {
            'gt_row_rng',
            'gt_row_loc',
            'gt_row_reg',
            'gt_row_loc_mask',
            'gt_row_reg_mask',
            'gt_label_obj',
            'gt_label_cls',
        }
        missing_keys = sorted(required_keys.difference(target.keys()))
        if missing_keys:
            raise ValueError(f'target {target_index} is missing keys: {missing_keys}')

        height, width = spatial_shape
        num_targets = target['gt_row_rng'].size(0)
        expected_shapes = {
            'gt_row_rng': (num_targets, 2),
            'gt_row_loc': (num_targets, height),
            'gt_row_reg': (num_targets, height, width),
            'gt_row_loc_mask': (num_targets, height),
            'gt_row_reg_mask': (num_targets, height, width),
            'gt_label_obj': (num_targets,),
            'gt_label_cls': (num_targets,),
        }
        for key, expected_shape in expected_shapes.items():
            if tuple(target[key].shape) != expected_shape:
                raise ValueError(
                    f'target {target_index} key {key} must have shape {expected_shape}, got {tuple(target[key].shape)}'
                )

