from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher, DenseMatchCostBreakdown


class CondLSTRParitySetCriterion(nn.Module):
    PREDICTION_KEYS = (
        'pred_object_logits',
        'pred_class_logits',
        'pred_ranges',
        'pred_dense_mask',
        'pred_dense_reg',
    )

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
        image_keys: Sequence[str] | None = None,
        collect_diagnostics: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, object]]]:
        dn_meta = outputs.get('dn_meta')
        total_queries = int(outputs['pred_object_logits'].shape[1])
        num_main_queries = int(dn_meta['num_main_queries']) if dn_meta is not None else total_queries
        main_outputs = self._slice_outputs_for_queries(outputs, 0, num_main_queries)

        loss_dict, indices, diagnostics = self._compute_losses(
            main_outputs,
            targets,
            image_keys=image_keys,
            collect_diagnostics=collect_diagnostics,
        )

        aux_outputs = outputs.get('aux_outputs', [])
        for layer_index, aux_output in enumerate(aux_outputs):
            aux_main_outputs = self._slice_outputs_for_queries(aux_output, 0, num_main_queries)
            aux_loss_dict, _, _ = self._compute_losses(aux_main_outputs, targets)
            for name, value in aux_loss_dict.items():
                loss_dict[f'{name}_{layer_index}'] = value

        if dn_meta is not None and total_queries > num_main_queries:
            dn_outputs = self._slice_outputs_for_queries(outputs, num_main_queries, total_queries)
            dn_loss_dict = self._compute_dn_losses(
                dn_outputs,
                dn_targets=dn_meta['targets'],
                dn_valid_counts=dn_meta['valid_counts'],
            )
            loss_dict.update(dn_loss_dict)

        return loss_dict, indices, diagnostics

    def _slice_outputs_for_queries(
        self,
        outputs: Dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> Dict[str, torch.Tensor]:
        sliced: Dict[str, torch.Tensor] = {}
        for key in self.PREDICTION_KEYS:
            if key in outputs:
                sliced[key] = outputs[key][:, start:end]
        return sliced

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
        image_keys: Sequence[str] | None = None,
        collect_diagnostics: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, object]]]:
        cost_breakdowns: List[DenseMatchCostBreakdown] | None = None
        if collect_diagnostics:
            indices, cost_breakdowns = self.matcher(outputs, targets, return_costs=True)
        else:
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
            if valid_class_mask.any() and batch_pred_class_logits.size(-1) > 1:
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

        diagnostics: List[Dict[str, object]] = []
        if collect_diagnostics and cost_breakdowns is not None:
            diagnostics = self._build_match_diagnostics(
                indices=indices,
                cost_breakdowns=cost_breakdowns,
                targets=targets,
                image_keys=image_keys,
            )

        return losses, indices, diagnostics

    def _compute_dn_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        dn_targets: Sequence[Dict[str, torch.Tensor]],
        dn_valid_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred_object_logits = outputs['pred_object_logits']
        pred_class_logits = outputs['pred_class_logits']
        pred_ranges = outputs['pred_ranges']
        pred_dense_mask = self.matcher._require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
        pred_dense_reg = self.matcher._require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')
        pred_row_locations = self.matcher._dense_mask_logits_to_row_locations(pred_dense_mask)

        device = pred_object_logits.device
        batch_size, num_dn_queries = pred_object_logits.shape[:2]
        target_objects = torch.full((batch_size, num_dn_queries), 1, dtype=torch.int64, device=device)

        total_row_l1 = pred_object_logits.new_tensor(0.0)
        total_row_iou = pred_object_logits.new_tensor(0.0)
        total_row_reg = pred_object_logits.new_tensor(0.0)
        total_row_range = pred_object_logits.new_tensor(0.0)
        matched_class_logits: List[torch.Tensor] = []
        matched_class_targets: List[torch.Tensor] = []
        valid_total = 0.0

        for batch_index in range(batch_size):
            valid_count = int(min(int(dn_valid_counts[batch_index].item()), num_dn_queries))
            if valid_count <= 0:
                continue

            target_objects[batch_index, :valid_count] = 0
            target = dn_targets[batch_index]
            valid_total += float(valid_count)

            batch_pred_class_logits = pred_class_logits[batch_index, :valid_count]
            batch_pred_ranges = pred_ranges[batch_index, :valid_count]
            batch_pred_row_locations = pred_row_locations[batch_index, :valid_count]
            batch_pred_dense_reg = pred_dense_reg[batch_index, :valid_count]

            target_classes = target['gt_label_cls'][:valid_count].long()
            valid_class_mask = target_classes != self.ignore_class_index
            if valid_class_mask.any() and batch_pred_class_logits.size(-1) > 1:
                matched_class_logits.append(batch_pred_class_logits[valid_class_mask])
                matched_class_targets.append(target_classes[valid_class_mask])

            target_row_locations = target['gt_row_loc'][:valid_count]
            target_row_location_mask = target['gt_row_loc_mask'][:valid_count]
            target_row_ranges = target['gt_row_rng'][:valid_count]
            target_row_reg = target['gt_row_reg'][:valid_count]
            target_row_reg_mask = target['gt_row_reg_mask'][:valid_count]

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

        normalizer = max(valid_total, 1.0)
        losses = {
            'loss_dn_object': F.cross_entropy(
                pred_object_logits.transpose(1, 2),
                target_objects,
                weight=self.object_empty_weight,
            ),
            'loss_dn_class': pred_object_logits.new_tensor(0.0),
            'loss_dn_loc': (total_row_l1 + (2.0 * total_row_iou)) / normalizer,
            'loss_dn_reg': total_row_reg / normalizer,
            'loss_dn_range': total_row_range / normalizer,
        }

        if matched_class_logits:
            losses['loss_dn_class'] = F.cross_entropy(
                torch.cat(matched_class_logits, dim=0),
                torch.cat(matched_class_targets, dim=0),
            )

        return losses

    def _build_match_diagnostics(
        self,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        cost_breakdowns: Sequence[DenseMatchCostBreakdown],
        targets: Sequence[Dict[str, torch.Tensor]],
        image_keys: Sequence[str] | None = None,
    ) -> List[Dict[str, object]]:
        diagnostics: List[Dict[str, object]] = []
        for batch_index, ((src_idx, tgt_idx), breakdown, target) in enumerate(zip(indices, cost_breakdowns, targets)):
            image_key = None
            if image_keys is not None and batch_index < len(image_keys):
                image_key = image_keys[batch_index]

            assignments = []
            margins = []
            for query_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
                total_row = breakdown.total_cost[query_index]
                match_cost = float(total_row[target_index].item())
                if total_row.numel() > 1:
                    competing = torch.cat((total_row[:target_index], total_row[target_index + 1:]))
                    second_best = float(competing.min().item())
                    margin = second_best - match_cost
                    margins.append(margin)
                else:
                    second_best = None
                    margin = None

                assignments.append(
                    {
                        'query': int(query_index),
                        'target': int(target_index),
                        'cost_total': match_cost,
                        'margin_to_second': margin,
                        'cost_object': float(breakdown.cost_object[query_index, target_index].item()),
                        'cost_class': float(breakdown.cost_class[query_index, target_index].item()),
                        'cost_row_location': float(breakdown.cost_row_location[query_index, target_index].item()),
                        'cost_row_iou': float(breakdown.cost_row_iou[query_index, target_index].item()),
                        'cost_row_reg': float(breakdown.cost_row_reg[query_index, target_index].item()),
                        'cost_row_range': float(breakdown.cost_row_range[query_index, target_index].item()),
                    }
                )

            mean_margin = float(sum(margins) / len(margins)) if margins else None
            min_margin = float(min(margins)) if margins else None
            low_margin_ratio = (
                float(sum(1 for margin in margins if margin < 1.0) / len(margins))
                if margins else None
            )

            diagnostics.append(
                {
                    'image_key': image_key,
                    'num_targets': int(target['gt_row_rng'].size(0)),
                    'num_matches': int(src_idx.numel()),
                    'mean_margin': mean_margin,
                    'min_margin': min_margin,
                    'low_margin_ratio': low_margin_ratio,
                    'assignments': assignments,
                }
            )

        return diagnostics

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
