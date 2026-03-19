from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher


class CondLSTRDenseSetCriterion(nn.Module):
    """
    Local standalone dense criterion for CondLSTR-style row-wise supervision.

    Current loss terms:
    - object classification over all queries
    - lane class classification over matched queries only
    - row location L1 over valid rows
    - row location IoU using a configurable line width
    - row-wise dense mask cross entropy over valid rows
    - dense regression L1 over masked support
    - normalized row range L1

    Scope notes:
    - This intentionally uses the existing local matcher and dense target contract only.
    - It is not wired into the broader training path yet.
    - The matcher's dense regression cost still uses broadcasted [Q, M, H, W] tensors;
      that is kept as-is and should be revisited for larger dense grids.
    """

    def __init__(
        self,
        matcher: CondLSTRDenseHungarianMatcher,
        line_width: float,
        object_eos_coef: float = 0.1,
        ignore_class_index: int = 255,
        enable_row_iou_loss: bool = False,
    ):
        super().__init__()
        self.matcher = matcher
        self.line_width = float(line_width)
        self.ignore_class_index = int(ignore_class_index)
        self.enable_row_iou_loss = bool(enable_row_iou_loss)

        object_empty_weight = torch.ones(2, dtype=torch.float32)
        object_empty_weight[1] = float(object_eos_coef)
        self.register_buffer('object_empty_weight', object_empty_weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        outputs_without_aux = {key: value for key, value in outputs.items() if key != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        pred_object_logits = outputs_without_aux['pred_object_logits']
        pred_class_logits = outputs_without_aux['pred_class_logits']
        pred_ranges = outputs_without_aux['pred_ranges']
        pred_dense_mask = self.matcher._require_single_channel_dense_output(
            outputs_without_aux['pred_dense_mask'],
            'pred_dense_mask',
        )
        pred_dense_reg = self.matcher._require_single_channel_dense_output(
            outputs_without_aux['pred_dense_reg'],
            'pred_dense_reg',
        )
        pred_row_locations = self.matcher._dense_mask_logits_to_row_locations(pred_dense_mask)

        num_targets = float(sum(int(target['gt_row_rng'].size(0)) for target in targets))
        normalizer = max(num_targets, 1.0)

        losses = {
            'loss_object': self.loss_object(pred_object_logits, targets, indices),
            'loss_class': pred_object_logits.new_tensor(0.0),
            'loss_row_location': pred_object_logits.new_tensor(0.0),
            'loss_row_iou': pred_object_logits.new_tensor(0.0),
            'loss_dense_mask': pred_object_logits.new_tensor(0.0),
            'loss_row_reg': pred_object_logits.new_tensor(0.0),
            'loss_row_range': pred_object_logits.new_tensor(0.0),
        }

        matched_class_logits: List[torch.Tensor] = []
        matched_class_targets: List[torch.Tensor] = []

        total_row_location = pred_object_logits.new_tensor(0.0)
        total_row_iou = pred_object_logits.new_tensor(0.0)
        total_dense_mask = pred_object_logits.new_tensor(0.0)
        total_row_reg = pred_object_logits.new_tensor(0.0)
        total_row_range = pred_object_logits.new_tensor(0.0)

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            target = targets[batch_index]
            batch_pred_class_logits = pred_class_logits[batch_index, src_idx]
            batch_pred_ranges = pred_ranges[batch_index, src_idx]
            batch_pred_mask_logits = pred_dense_mask[batch_index, src_idx]
            batch_pred_row_locations = pred_row_locations[batch_index, src_idx]
            batch_pred_dense_reg = pred_dense_reg[batch_index, src_idx]

            target_classes = target['gt_label_cls'][tgt_idx].long()
            valid_class_mask = target_classes != self.ignore_class_index
            if valid_class_mask.any():
                matched_class_logits.append(batch_pred_class_logits[valid_class_mask])
                matched_class_targets.append(target_classes[valid_class_mask])

            target_row_locations = target['gt_row_loc'][tgt_idx]
            target_row_location_mask = target['gt_row_loc_mask'][tgt_idx]
            target_row_ranges = target['gt_row_rng'][tgt_idx]
            target_row_reg = target['gt_row_reg'][tgt_idx]
            target_row_reg_mask = target['gt_row_reg_mask'][tgt_idx]

            valid_location_counts = target_row_location_mask.sum(dim=1).clamp(min=1.0)
            row_location_error = torch.abs(batch_pred_row_locations - target_row_locations)
            total_row_location = total_row_location + (
                (row_location_error * target_row_location_mask).sum(dim=1) / valid_location_counts
            ).sum()

            if self.enable_row_iou_loss:
                pred_row_left = batch_pred_row_locations - self.line_width
                pred_row_right = batch_pred_row_locations + self.line_width
                target_row_left = target_row_locations - self.line_width
                target_row_right = target_row_locations + self.line_width
                overlap = (
                    torch.minimum(pred_row_right, target_row_right) - torch.maximum(pred_row_left, target_row_left)
                ).clamp(min=0.0)
                overlap = overlap * target_row_location_mask
                union = (
                    torch.maximum(pred_row_right, target_row_right) - torch.minimum(pred_row_left, target_row_left)
                )
                union = union * target_row_location_mask
                row_iou = overlap.sum(dim=1) / (union.sum(dim=1) + 1e-9)
                total_row_iou = total_row_iou + (1.0 - row_iou).sum()

            valid_regression_counts = target_row_reg_mask.sum(dim=(1, 2)).clamp(min=1.0)
            row_reg_error = torch.abs(batch_pred_dense_reg - target_row_reg)
            total_row_reg = total_row_reg + (
                (row_reg_error * target_row_reg_mask).sum(dim=(1, 2)) / valid_regression_counts
            ).sum()

            total_row_range = total_row_range + F.l1_loss(
                batch_pred_ranges,
                target_row_ranges,
                reduction='none',
            ).sum(dim=1).sum()

            valid_rows = torch.nonzero(target_row_location_mask > 0.5, as_tuple=False)
            if valid_rows.numel() > 0:
                valid_row_mask_logits = batch_pred_mask_logits[valid_rows[:, 0], valid_rows[:, 1]]
                valid_row_mask_targets = target_row_locations[valid_rows[:, 0], valid_rows[:, 1]].round().long().clamp(
                        min=0,
                        max=batch_pred_mask_logits.size(-1) - 1,
                    )
                valid_row_mask_loss = F.cross_entropy(
                    valid_row_mask_logits,
                    valid_row_mask_targets,
                    reduction='none',
                )
                # Keep dense-mask scaling aligned with the other lane-level terms:
                # first average over valid rows within each matched lane, then sum over
                # matched lanes, and finally divide once by the global lane normalizer.
                # This avoids overweighting lanes that simply contribute more valid rows.
                lane_ids = valid_rows[:, 0]
                num_lanes = batch_pred_mask_logits.size(0)
                lane_loss_sum = torch.zeros(num_lanes, dtype=valid_row_mask_loss.dtype, device=valid_row_mask_loss.device)
                lane_loss_count = torch.zeros(num_lanes, dtype=valid_row_mask_loss.dtype, device=valid_row_mask_loss.device)
                lane_loss_sum.scatter_add_(0, lane_ids, valid_row_mask_loss)
                lane_loss_count.scatter_add_(0, lane_ids, torch.ones_like(valid_row_mask_loss))
                total_dense_mask = total_dense_mask + (lane_loss_sum / lane_loss_count.clamp(min=1.0)).sum()

        if matched_class_logits:
            losses['loss_class'] = F.cross_entropy(
                torch.cat(matched_class_logits, dim=0),
                torch.cat(matched_class_targets, dim=0),
            )

        losses['loss_row_location'] = total_row_location / normalizer
        losses['loss_row_iou'] = total_row_iou / normalizer
        losses['loss_dense_mask'] = total_dense_mask / normalizer
        losses['loss_row_reg'] = total_row_reg / normalizer
        losses['loss_row_range'] = total_row_range / normalizer

        return losses, indices

    def loss_object(
        self,
        pred_object_logits: torch.Tensor,
        targets: Sequence[Dict[str, torch.Tensor]],
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        target_objects = torch.full(
            pred_object_logits.shape[:2],
            1,
            dtype=torch.int64,
            device=pred_object_logits.device,
        )
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_objects[batch_index, src_idx] = targets[batch_index]['gt_label_obj'][tgt_idx].long()

        return F.cross_entropy(
            pred_object_logits.transpose(1, 2),
            target_objects,
            weight=self.object_empty_weight,
        )


class CondLSTRDenseLoss(nn.Module):
    """
    Thin wrapper around the standalone dense criterion.

    Returns a small local package that is easy to test before any broader trainer wiring.
    """

    def __init__(
        self,
        criterion: CondLSTRDenseSetCriterion,
        weight_dict: Dict[str, float] | None = None,
    ):
        super().__init__()
        self.criterion = criterion
        self.weight_dict = {
            'loss_object': 1.0,
            'loss_class': 1.0,
            'loss_row_location': 1.0,
            'loss_row_iou': 1.0,
            'loss_dense_mask': 1.0,
            'loss_row_reg': 1.0,
            'loss_row_range': 1.0,
        }
        if weight_dict is not None:
            self.weight_dict.update({key: float(value) for key, value in weight_dict.items()})

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, object]:
        loss_dict, indices = self.criterion(outputs, targets)
        total_loss = sum(
            loss_dict[name] * self.weight_dict[name]
            for name in self.weight_dict.keys()
            if name in loss_dict
        )
        return {
            'loss_total': total_loss,
            'loss_dict': loss_dict,
            'weight_dict': dict(self.weight_dict),
            'indices': indices,
        }
