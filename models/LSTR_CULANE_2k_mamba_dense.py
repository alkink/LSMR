"""
LSTR_CULANE_2k_mamba_dense
==========================

Dense CondLSTR-style lane head integrated into the existing 2k Mamba branch.

Design goals:
1) Keep the base LSTR+Mamba encoder branch untouched.
2) Reuse the existing sample target format produced by `sample/culane.py`.
3) Emit the dense prediction contract already supported by the local
   matcher/criterion/postprocess stack.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import system_configs
from models.LSTR_CULANE_MAMBA import model as _BaseMambaModel
from models.condlstr_dense_criterion import CondLSTRDenseSetCriterion
from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher
from models.dynamic_lane_head import DynamicLaneHead
from models.py_utils.misc import reduce_dict
from utils.condlstr_dense_targets import convert_culane_points_to_rowwise_targets


def _legacy_label_tensor_to_lane_points(
    label_tensor: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
    """
    Convert the legacy augmented label tensor back to lane point lists.

    Legacy lane format per row:
        [class, lower, upper, x_0..x_n, y_0..y_n]
    where x/y are normalized into the augmented image frame.
    """

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


def build_dense_targets_from_legacy_targets(
    targets: Sequence[torch.Tensor],
    target_size: Tuple[int, int],
    device: torch.device,
    line_width: float = 1.5,
    min_valid_rows: int = 2,
) -> List[Dict[str, torch.Tensor]]:
    """
    Convert the current trainer `ys` package to the dense target contract.

    Current training targets arrive as:
        [images, gt_lane_tensor_0, ..., gt_lane_tensor_{B-1}]

    Each lane tensor keeps a fake leading batch dimension because the sampler
    historically duplicated the per-sample labels to match the old loss path.
    """

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
            lanes, lane_attrs = _legacy_label_tensor_to_lane_points(
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


class model(_BaseMambaModel):
    def __init__(self, flag: bool = False):
        super().__init__(flag=flag)

        dense_num_classes = max(int(system_configs.full.get('dense_num_classes', 1)), 1)
        branch_hidden_dim = int(system_configs.full.get('dense_branch_hidden_dim', 64))
        use_coords = bool(system_configs.full.get('dense_use_coords', True))

        self.dense_head = DynamicLaneHead(
            feature_channels=int(system_configs.res_dims[1]),
            query_channels=int(system_configs.attn_dim),
            num_classes=dense_num_classes,
            branch_hidden_dim=branch_hidden_dim,
            mask_out_channels=1,
            reg_out_channels=1,
            use_coords=use_coords,
        )

        print(
            "[LSTR_CULANE_2k_mamba_dense] "
            f"head -> DynamicLaneHead(num_classes={dense_num_classes}, hidden={branch_hidden_dim}, use_coords={use_coords})"
        )

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        dense_feature = self.layer2(p)
        p = self.layer3(dense_feature)
        p = self.layer4(p)

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)
        hs, _, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)

        out = self.dense_head(feature_map=dense_feature, query_features=hs[-1])
        out['aux_outputs'] = []
        return out, weights

    def _test(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class loss(nn.Module):
    def __init__(self):
        super().__init__()
        snapshot_name = system_configs.snapshot_name
        base_result_dir = system_configs.full.get('result_dir', './results')
        self.debug_path = os.path.join(base_result_dir, snapshot_name) if snapshot_name else base_result_dir
        self.line_width = float(system_configs.full.get('dense_line_width', 1.5))
        self.min_valid_rows = int(system_configs.full.get('dense_min_valid_rows', 2))

        self.weight_dict = {
            'loss_object': float(system_configs.full.get('dense_loss_object_weight', 1.0)),
            'loss_class': float(system_configs.full.get('dense_loss_class_weight', 1.0)),
            'loss_row_location': float(system_configs.full.get('dense_loss_row_location_weight', 1.0)),
            'loss_row_iou': float(system_configs.full.get('dense_loss_row_iou_weight', 1.0)),
            'loss_dense_mask': float(system_configs.full.get('dense_loss_dense_mask_weight', 1.0)),
            'loss_row_reg': float(system_configs.full.get('dense_loss_row_reg_weight', 1.0)),
            'loss_row_range': float(system_configs.full.get('dense_loss_row_range_weight', 1.0)),
        }

        matcher = CondLSTRDenseHungarianMatcher(
            line_width=self.line_width,
            object_weight=float(system_configs.full.get('dense_match_object_weight', 1.0)),
            class_weight=float(system_configs.full.get('dense_match_class_weight', 1.0)),
            location_weight=float(system_configs.full.get('dense_match_location_weight', 1.0)),
            location_iou_weight=float(system_configs.full.get('dense_match_location_iou_weight', 2.0)),
            regression_weight=float(system_configs.full.get('dense_match_regression_weight', 1.0)),
            range_weight=float(system_configs.full.get('dense_match_range_weight', 1.0)),
        )
        self.criterion = CondLSTRDenseSetCriterion(
            matcher=matcher,
            line_width=self.line_width,
            object_eos_coef=float(system_configs.full.get('dense_object_eos_coef', 0.1)),
            enable_row_iou_loss=bool(system_configs.full.get('dense_enable_row_iou_loss', False)),
        )

        print(f"[LSTR_CULANE_2k_mamba_dense] weight_dict: {self.weight_dict}")

    def forward(self, iteration, save, viz_split, outputs, targets):
        del iteration, save, viz_split

        if 'pred_dense_mask' not in outputs or 'pred_object_logits' not in outputs:
            raise KeyError('Dense loss requires pred_object_logits and pred_dense_mask outputs')

        spatial_size = (int(outputs['pred_dense_mask'].shape[-2]), int(outputs['pred_dense_mask'].shape[-1]))
        dense_targets = build_dense_targets_from_legacy_targets(
            targets=targets,
            target_size=spatial_size,
            device=outputs['pred_object_logits'].device,
            line_width=self.line_width,
            min_valid_rows=self.min_valid_rows,
        )

        loss_dict, _ = self.criterion(outputs, dense_targets)
        loss_dict['class_error'] = outputs['pred_object_logits'].new_tensor(0.0)
        total = sum(
            loss_dict[name] * self.weight_dict[name]
            for name in self.weight_dict.keys()
            if name in loss_dict
        )

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * self.weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in self.weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            raise RuntimeError('Non-finite dense loss encountered')

        return (
            total,
            loss_dict_reduced,
            loss_dict_reduced_unscaled,
            loss_dict_reduced_scaled,
            loss_value,
        )


