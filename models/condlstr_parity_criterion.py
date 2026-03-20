from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher


class CondLSTRParitySetCriterion(nn.Module):
    def __init__(
        self,
        matcher: CondLSTRDenseHungarianMatcher,
        line_width: float,
        object_eos_coef: float = 0.4,
        ignore_class_index: int = 255,
    ):
        super().__init__()
        self.matcher = matcher
        self.line_width = float(line_width)
        self.ignore_class_index = int(ignore_class_index)

        object_empty_weight = torch.ones(2, dtype=torch.float32)
        object_empty_weight[1] = float(object_eos_coef)
        self.register_buffer('object_empty_weight', object_empty_weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        main_outputs = {key: value for key, value in outputs.items() if key not in {'aux_outputs', 'postprocess_mode'}}
        loss_dict, indices = self._compute_losses(main_outputs, targets)

        aux_outputs = outputs.get('aux_outputs', [])
        for layer_index, aux_output in enumerate(aux_outputs):
            aux_loss_dict, _ = self._compute_losses(aux_output, targets)
            for name, value in aux_loss_dict.items():
                loss_dict[f'{name}_{layer_index}'] = value

        return loss_dict, indices

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        indices = self.matcher(outputs, targets)
        pred_object_logits = outputs['pred_object_logits']
        pred_class_logits = outputs['pred_class_logits']
        pred_ranges = outputs['pred_ranges']
        pred_dense_mask = self.matcher._require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
        pred_dense_reg = self.matcher._require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')
        pred_row_locations = self.matcher._dense_mask_logits_to_row_locations(pred_dense_mask)

        num_targets = float(sum(int(target['gt_row_rng'].size(0)) for target in targets))
        normalizer = max(num_targets, 1.0)

        losses = {
            'loss_object': self.loss_object(pred_object_logits, targets, indices),
            'loss_class': pred_object_logits.new_tensor(0.0),
            'loss_loc': pred_object_logits.new_tensor(0.0),
            'loss_reg': pred_object_logits.new_tensor(0.0),
            'loss_range': pred_object_logits.new_tensor(0.0),
        }

        matched_class_logits: List[torch.Tensor] = []
        matched_class_targets: List[torch.Tensor] = []
        total_row_l1 = pred_object_logits.new_tensor(0.0)
        total_row_iou = pred_object_logits.new_tensor(0.0)
        total_row_reg = pred_object_logits.new_tensor(0.0)
        total_row_range = pred_object_logits.new_tensor(0.0)

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            target = targets[batch_index]
            batch_pred_class_logits = pred_class_logits[batch_index, src_idx]
            batch_pred_ranges = pred_ranges[batch_index, src_idx]
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
            total_row_l1 = total_row_l1 + (
                (row_location_error * target_row_location_mask).sum(dim=1) / valid_location_counts
            ).sum()

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

        if matched_class_logits:
            losses['loss_class'] = F.cross_entropy(
                torch.cat(matched_class_logits, dim=0),
                torch.cat(matched_class_targets, dim=0),
            )

        losses['loss_loc'] = (total_row_l1 + (2.0 * total_row_iou)) / normalizer
        losses['loss_reg'] = total_row_reg / normalizer
        losses['loss_range'] = total_row_range / normalizer
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
