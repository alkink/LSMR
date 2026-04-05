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
        'pred_quality_logits',
        'pred_ranges',
        'pred_row_visibility_logits',
        'pred_dense_mask',
        'pred_dense_reg',
    )

    def __init__(
        self,
        matcher: CondLSTRDenseHungarianMatcher,
        line_width: float,
        object_eos_coef: float = 0.4,
        ignore_class_index: int = 255,
        object_target_mode: str = 'binary',
        object_quality_power: float = 1.0,
        quality_target_power: float = 1.0,
        quality_soft_positive_enabled: bool = False,
        quality_soft_positive_iou_thresh: float = 0.5,
        quality_soft_positive_decay: float = 0.5,
    ):
        super().__init__()
        self.matcher = matcher
        self.line_width = float(line_width)
        self.ignore_class_index = int(ignore_class_index)
        self.object_target_mode = str(object_target_mode).lower()
        self.object_quality_power = float(object_quality_power)
        self.quality_target_power = float(quality_target_power)
        self.quality_soft_positive_enabled = bool(quality_soft_positive_enabled)
        self.quality_soft_positive_iou_thresh = float(quality_soft_positive_iou_thresh)
        self.quality_soft_positive_decay = float(quality_soft_positive_decay)
        if self.object_target_mode not in {'binary', 'row_iou'}:
            raise ValueError(f'unsupported object_target_mode={object_target_mode!r}')
        if self.object_quality_power <= 0.0:
            raise ValueError('object_quality_power must be positive')
        if self.quality_target_power <= 0.0:
            raise ValueError('quality_target_power must be positive')
        if not (0.0 <= self.quality_soft_positive_iou_thresh <= 1.0):
            raise ValueError('quality_soft_positive_iou_thresh must be in [0, 1]')
        if not (0.0 <= self.quality_soft_positive_decay <= 1.0):
            raise ValueError('quality_soft_positive_decay must be in [0, 1]')

        object_empty_weight = torch.ones(2, dtype=torch.float32)
        object_empty_weight[1] = float(object_eos_coef)
        self.register_buffer('object_empty_weight', object_empty_weight)
        self.register_buffer(
            'quality_pos_weight',
            torch.tensor([1.0 / max(float(object_eos_coef), 1e-6)], dtype=torch.float32),
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
        image_keys: Sequence[str] | None = None,
        collect_diagnostics: bool = False,
        object_quality_mix: float = 1.0,
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
            object_quality_mix=object_quality_mix,
        )

        aux_outputs = outputs.get('aux_outputs', [])
        for layer_index, aux_output in enumerate(aux_outputs):
            aux_main_outputs = self._slice_outputs_for_queries(aux_output, 0, num_main_queries)
            aux_loss_dict, _, _ = self._compute_losses(
                aux_main_outputs,
                targets,
                object_quality_mix=object_quality_mix,
            )
            for name, value in aux_loss_dict.items():
                loss_dict[f'{name}_{layer_index}'] = value

        if dn_meta is not None and total_queries > num_main_queries:
            dn_outputs = self._slice_outputs_for_queries(outputs, num_main_queries, total_queries)
            dn_loss_dict = self._compute_dn_losses(
                dn_outputs,
                dn_targets=dn_meta['targets'],
                dn_valid_counts=dn_meta['valid_counts'],
                object_quality_mix=object_quality_mix,
            )
            loss_dict.update(dn_loss_dict)
        else:
            dn_outputs = None

        if diagnostics:
            self._attach_output_diagnostics(
                diagnostics=diagnostics,
                main_outputs=main_outputs,
                dn_outputs=dn_outputs,
                dn_valid_counts=dn_meta['valid_counts'] if dn_meta is not None else None,
            )
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

    def _compute_pairwise_row_iou(
        self,
        pred_row_locations: torch.Tensor,
        target_row_locations: torch.Tensor,
        target_row_location_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_row_left = pred_row_locations.unsqueeze(1) - self.line_width
        pred_row_right = pred_row_locations.unsqueeze(1) + self.line_width
        target_row_left = target_row_locations.unsqueeze(0) - self.line_width
        target_row_right = target_row_locations.unsqueeze(0) + self.line_width

        overlap = (
            torch.minimum(pred_row_right, target_row_right) - torch.maximum(pred_row_left, target_row_left)
        ).clamp(min=0.0)
        overlap = overlap * target_row_location_mask.unsqueeze(0)

        union = (
            torch.maximum(pred_row_right, target_row_right) - torch.minimum(pred_row_left, target_row_left)
        )
        union = union * target_row_location_mask.unsqueeze(0)
        return overlap.sum(dim=2) / (union.sum(dim=2) + 1e-9)

    def _assign_soft_quality_targets(
        self,
        quality_targets: torch.Tensor,
        pairwise_row_iou: torch.Tensor,
        matched_src_idx: torch.Tensor,
    ) -> None:
        if not self.quality_soft_positive_enabled:
            return
        if pairwise_row_iou.numel() == 0:
            return

        num_queries, num_targets = pairwise_row_iou.shape
        if num_queries == 0 or num_targets == 0:
            return

        candidate_mask = pairwise_row_iou >= self.quality_soft_positive_iou_thresh
        if matched_src_idx.numel() > 0:
            unmatched_query_mask = torch.ones((num_queries,), dtype=torch.bool, device=pairwise_row_iou.device)
            unmatched_query_mask[matched_src_idx] = False
            candidate_mask = candidate_mask & unmatched_query_mask.unsqueeze(1)

        candidate_pairs = torch.nonzero(candidate_mask, as_tuple=False)
        if candidate_pairs.numel() == 0:
            return

        candidate_scores = pairwise_row_iou[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        order = torch.argsort(candidate_scores, descending=True)
        used_queries: set[int] = set()
        used_targets: set[int] = set()

        for order_index in order.tolist():
            query_index = int(candidate_pairs[order_index, 0].item())
            target_index = int(candidate_pairs[order_index, 1].item())
            if query_index in used_queries or target_index in used_targets:
                continue

            iou_value = float(candidate_scores[order_index].item())
            target_value = self.quality_soft_positive_decay * (iou_value ** self.quality_target_power)
            quality_targets[query_index] = max(float(quality_targets[query_index].item()), target_value)
            used_queries.add(query_index)
            used_targets.add(target_index)

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
        image_keys: Sequence[str] | None = None,
        collect_diagnostics: bool = False,
        object_quality_mix: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, object]]]:
        cost_breakdowns: List[DenseMatchCostBreakdown] | None = None
        if collect_diagnostics:
            indices, cost_breakdowns = self.matcher(outputs, targets, return_costs=True)
        else:
            indices = self.matcher(outputs, targets)

        pred_object_logits = outputs['pred_object_logits']
        pred_class_logits = outputs['pred_class_logits']
        pred_quality_logits = outputs.get('pred_quality_logits')
        pred_ranges = outputs.get('pred_ranges')
        pred_row_visibility_logits = outputs.get('pred_row_visibility_logits')
        pred_dense_mask = self.matcher._require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
        pred_dense_reg = self.matcher._require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')
        pred_row_locations = self.matcher._dense_mask_logits_to_row_locations(pred_dense_mask)

        num_targets = float(sum(int(target['gt_row_rng'].size(0)) for target in targets))
        normalizer = max(num_targets, 1.0)
        object_quality_targets = pred_object_logits.new_zeros(pred_object_logits.shape[:2])
        quality_targets = pred_object_logits.new_zeros(pred_object_logits.shape[:2])

        losses = {
            'loss_object': pred_object_logits.new_tensor(0.0),
            'loss_class': pred_object_logits.new_tensor(0.0),
            'loss_loc': pred_object_logits.new_tensor(0.0),
            'loss_reg': pred_object_logits.new_tensor(0.0),
            'loss_range': pred_object_logits.new_tensor(0.0),
        }
        if pred_quality_logits is not None:
            losses['loss_quality'] = pred_object_logits.new_tensor(0.0)

        matched_class_logits: List[torch.Tensor] = []
        matched_class_targets: List[torch.Tensor] = []
        total_row_l1 = pred_object_logits.new_tensor(0.0)
        total_row_iou = pred_object_logits.new_tensor(0.0)
        total_row_reg = pred_object_logits.new_tensor(0.0)
        total_row_range = pred_object_logits.new_tensor(0.0)

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            target = targets[batch_index]
            batch_all_pred_row_locations = pred_row_locations[batch_index]
            batch_all_target_row_locations = target['gt_row_loc']
            batch_all_target_row_location_mask = target['gt_row_loc_mask']

            if self.quality_soft_positive_enabled:
                pairwise_row_iou = self._compute_pairwise_row_iou(
                    pred_row_locations=batch_all_pred_row_locations,
                    target_row_locations=batch_all_target_row_locations,
                    target_row_location_mask=batch_all_target_row_location_mask,
                ).detach().clamp(0.0, 1.0)
                self._assign_soft_quality_targets(
                    quality_targets=quality_targets[batch_index],
                    pairwise_row_iou=pairwise_row_iou,
                    matched_src_idx=src_idx,
                )

            if src_idx.numel() == 0:
                continue

            batch_pred_class_logits = pred_class_logits[batch_index, src_idx]
            batch_pred_ranges = pred_ranges[batch_index, src_idx] if pred_ranges is not None else None
            batch_pred_visibility_logits = (
                pred_row_visibility_logits[batch_index, src_idx] if pred_row_visibility_logits is not None else None
            )
            batch_pred_row_locations = batch_all_pred_row_locations[src_idx]
            batch_pred_dense_reg = pred_dense_reg[batch_index, src_idx]

            target_classes = target['gt_label_cls'][tgt_idx].long()
            valid_class_mask = target_classes != self.ignore_class_index
            if valid_class_mask.any() and batch_pred_class_logits.size(-1) > 1:
                matched_class_logits.append(batch_pred_class_logits[valid_class_mask])
                matched_class_targets.append(target_classes[valid_class_mask])

            target_row_locations = batch_all_target_row_locations[tgt_idx]
            target_row_location_mask = batch_all_target_row_location_mask[tgt_idx]
            target_row_ranges = target['gt_row_rng'][tgt_idx]
            target_row_reg = target['gt_row_reg'][tgt_idx]
            target_row_reg_mask = target['gt_row_reg_mask'][tgt_idx]

            valid_location_counts = target_row_location_mask.sum(dim=1).clamp(min=1.0)
            row_location_error = torch.abs(batch_pred_row_locations - target_row_locations)
            total_row_l1 = total_row_l1 + (
                (row_location_error * target_row_location_mask).sum(dim=1) / valid_location_counts
            ).sum()

            row_iou = self._compute_pairwise_row_iou(
                pred_row_locations=batch_pred_row_locations,
                target_row_locations=target_row_locations,
                target_row_location_mask=target_row_location_mask,
            ).diag()
            object_quality_targets[batch_index, src_idx] = row_iou.detach().clamp(0.0, 1.0).pow(
                self.object_quality_power
            )
            quality_targets[batch_index, src_idx] = row_iou.detach().clamp(0.0, 1.0).pow(
                self.quality_target_power
            )
            total_row_iou = total_row_iou + (1.0 - row_iou).sum()

            valid_regression_counts = target_row_reg_mask.sum(dim=(1, 2)).clamp(min=1.0)
            row_reg_error = torch.abs(batch_pred_dense_reg - target_row_reg)
            total_row_reg = total_row_reg + (
                (row_reg_error * target_row_reg_mask).sum(dim=(1, 2)) / valid_regression_counts
            ).sum()

            if batch_pred_visibility_logits is not None:
                total_row_range = total_row_range + F.binary_cross_entropy_with_logits(
                    batch_pred_visibility_logits,
                    target_row_location_mask,
                    reduction='none',
                ).mean(dim=1).sum()
            else:
                if batch_pred_ranges is None:
                    raise ValueError('pred_ranges must be provided when pred_row_visibility_logits is absent')
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

        losses['loss_object'] = self.loss_object(
            pred_object_logits,
            object_quality_targets,
            object_quality_mix=object_quality_mix,
        )
        if pred_quality_logits is not None:
            losses['loss_quality'] = F.binary_cross_entropy_with_logits(
                pred_quality_logits[..., 0],
                quality_targets,
                pos_weight=self.quality_pos_weight,
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
        object_quality_mix: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        pred_object_logits = outputs['pred_object_logits']
        pred_class_logits = outputs['pred_class_logits']
        pred_ranges = outputs.get('pred_ranges')
        pred_row_visibility_logits = outputs.get('pred_row_visibility_logits')
        pred_dense_mask = self.matcher._require_single_channel_dense_output(outputs['pred_dense_mask'], 'pred_dense_mask')
        pred_dense_reg = self.matcher._require_single_channel_dense_output(outputs['pred_dense_reg'], 'pred_dense_reg')
        pred_row_locations = self.matcher._dense_mask_logits_to_row_locations(pred_dense_mask)

        device = pred_object_logits.device
        batch_size, num_dn_queries = pred_object_logits.shape[:2]
        target_objects = torch.full((batch_size, num_dn_queries), 1, dtype=torch.int64, device=device)
        object_quality_targets = pred_object_logits.new_zeros((batch_size, num_dn_queries))

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
            batch_pred_ranges = pred_ranges[batch_index, :valid_count] if pred_ranges is not None else None
            batch_pred_visibility_logits = (
                pred_row_visibility_logits[batch_index, :valid_count]
                if pred_row_visibility_logits is not None else None
            )
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
            object_quality_targets[batch_index, :valid_count] = row_iou.detach().clamp(0.0, 1.0).pow(
                self.object_quality_power
            )
            total_row_iou = total_row_iou + (1.0 - row_iou).sum()

            valid_regression_counts = target_row_reg_mask.sum(dim=(1, 2)).clamp(min=1.0)
            row_reg_error = torch.abs(batch_pred_dense_reg - target_row_reg)
            total_row_reg = total_row_reg + (
                (row_reg_error * target_row_reg_mask).sum(dim=(1, 2)) / valid_regression_counts
            ).sum()

            if batch_pred_visibility_logits is not None:
                total_row_range = total_row_range + F.binary_cross_entropy_with_logits(
                    batch_pred_visibility_logits,
                    target_row_location_mask,
                    reduction='none',
                ).mean(dim=1).sum()
            else:
                if batch_pred_ranges is None:
                    raise ValueError('pred_ranges must be provided when pred_row_visibility_logits is absent')
                total_row_range = total_row_range + F.l1_loss(
                    batch_pred_ranges,
                    target_row_ranges,
                    reduction='none',
                ).sum(dim=1).sum()

        normalizer = max(valid_total, 1.0)
        losses = {
            'loss_dn_object': self.loss_object(
                pred_object_logits,
                object_quality_targets,
                object_quality_mix=object_quality_mix,
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
            query_margins = []
            target_competition = []
            target_margins = []
            for query_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
                total_row = breakdown.total_cost[query_index]
                match_cost = float(total_row[target_index].item())
                if total_row.numel() > 1:
                    competing = torch.cat((total_row[:target_index], total_row[target_index + 1:]))
                    second_best_target_cost = float(competing.min().item())
                    query_margin = second_best_target_cost - match_cost
                    query_margins.append(query_margin)
                else:
                    second_best_target_cost = None
                    query_margin = None

                total_col = breakdown.total_cost[:, target_index]
                if total_col.numel() > 1:
                    competing_queries = torch.cat((total_col[:query_index], total_col[query_index + 1:]))
                    second_best_query_cost = float(competing_queries.min().item())
                    target_margin = second_best_query_cost - match_cost
                    target_margins.append(target_margin)
                else:
                    second_best_query_cost = None
                    target_margin = None

                assignments.append(
                    {
                        'query': int(query_index),
                        'target': int(target_index),
                        'cost_total': match_cost,
                        'query_margin_to_second_target': query_margin,
                        'target_margin_to_second_query': target_margin,
                        'cost_object': float(breakdown.cost_object[query_index, target_index].item()),
                        'cost_class': float(breakdown.cost_class[query_index, target_index].item()),
                        'cost_row_location': float(breakdown.cost_row_location[query_index, target_index].item()),
                        'cost_row_iou': float(breakdown.cost_row_iou[query_index, target_index].item()),
                        'cost_row_reg': float(breakdown.cost_row_reg[query_index, target_index].item()),
                        'cost_row_range': float(breakdown.cost_row_range[query_index, target_index].item()),
                        'cost_order': float(breakdown.cost_order[query_index, target_index].item()),
                    }
                )
                target_competition.append(
                    {
                        'target': int(target_index),
                        'query': int(query_index),
                        'best_query_cost': match_cost,
                        'second_best_query_cost': second_best_query_cost,
                        'margin_to_second_query': target_margin,
                    }
                )

            mean_query_margin = float(sum(query_margins) / len(query_margins)) if query_margins else None
            min_query_margin = float(min(query_margins)) if query_margins else None
            low_query_margin_ratio = (
                float(sum(1 for margin in query_margins if margin < 1.0) / len(query_margins))
                if query_margins else None
            )
            mean_target_margin = float(sum(target_margins) / len(target_margins)) if target_margins else None
            min_target_margin = float(min(target_margins)) if target_margins else None
            low_target_margin_ratio = (
                float(sum(1 for margin in target_margins if margin < 1.0) / len(target_margins))
                if target_margins else None
            )

            diagnostics.append(
                {
                    'image_key': image_key,
                    'num_targets': int(target['gt_row_rng'].size(0)),
                    'num_matches': int(src_idx.numel()),
                    'mean_query_margin': mean_query_margin,
                    'min_query_margin': min_query_margin,
                    'low_query_margin_ratio': low_query_margin_ratio,
                    'mean_target_margin': mean_target_margin,
                    'min_target_margin': min_target_margin,
                    'low_target_margin_ratio': low_target_margin_ratio,
                    'assignments': assignments,
                    'target_competition': target_competition,
                }
            )

        return diagnostics

    def _attach_output_diagnostics(
        self,
        diagnostics: List[Dict[str, object]],
        main_outputs: Dict[str, torch.Tensor],
        dn_outputs: Dict[str, torch.Tensor] | None,
        dn_valid_counts: torch.Tensor | None,
    ) -> None:
        main_fg_probs = F.softmax(main_outputs['pred_object_logits'], dim=-1)[..., 0]
        main_quality_probs = None
        if 'pred_quality_logits' in main_outputs:
            main_quality_probs = torch.sigmoid(main_outputs['pred_quality_logits'][..., 0])

        def _stats(values: torch.Tensor) -> Dict[str, float | int]:
            if values.numel() == 0:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'p_gt_03': 0,
                    'p_gt_05': 0,
                    'top3_mean': 0.0,
                }
            flat = values.reshape(-1).float()
            topk = min(3, int(flat.numel()))
            return {
                'count': int(flat.numel()),
                'mean': float(flat.mean().item()),
                'std': float(flat.std(unbiased=False).item()),
                'max': float(flat.max().item()),
                'p_gt_03': int((flat > 0.3).sum().item()),
                'p_gt_05': int((flat > 0.5).sum().item()),
                'top3_mean': float(flat.topk(topk).values.mean().item()),
            }

        for batch_index, payload in enumerate(diagnostics):
            payload['main_object_fg_stats'] = _stats(main_fg_probs[batch_index])
            if main_quality_probs is not None:
                payload['main_quality_stats'] = _stats(main_quality_probs[batch_index])
            if dn_outputs is None or dn_valid_counts is None:
                continue

            dn_fg_probs = F.softmax(dn_outputs['pred_object_logits'][batch_index], dim=-1)[..., 0]
            dn_quality_probs = None
            if 'pred_quality_logits' in dn_outputs:
                dn_quality_probs = torch.sigmoid(dn_outputs['pred_quality_logits'][batch_index, ..., 0])
            valid_count = int(min(int(dn_valid_counts[batch_index].item()), int(dn_fg_probs.numel())))
            payload['dn_valid_count'] = valid_count
            payload['dn_object_fg_stats_all'] = _stats(dn_fg_probs)
            payload['dn_object_fg_stats_valid'] = _stats(dn_fg_probs[:valid_count])
            payload['dn_object_fg_stats_unused'] = _stats(dn_fg_probs[valid_count:])
            if dn_quality_probs is not None:
                payload['dn_quality_stats_all'] = _stats(dn_quality_probs)
                payload['dn_quality_stats_valid'] = _stats(dn_quality_probs[:valid_count])
                payload['dn_quality_stats_unused'] = _stats(dn_quality_probs[valid_count:])

    def loss_object(
        self,
        pred_object_logits: torch.Tensor,
        object_quality_targets: torch.Tensor,
        object_quality_mix: float = 1.0,
    ) -> torch.Tensor:
        object_quality_mix = float(min(max(object_quality_mix, 0.0), 1.0))
        if self.object_target_mode == 'binary':
            target_objects = (object_quality_targets <= 0.0).to(torch.int64)
            return F.cross_entropy(
                pred_object_logits.transpose(1, 2),
                target_objects,
                weight=self.object_empty_weight,
            )

        fg_quality = object_quality_targets.clamp(0.0, 1.0)
        binary_fg = (fg_quality > 0.0).float()
        fg_target = ((1.0 - object_quality_mix) * binary_fg) + (object_quality_mix * fg_quality)
        target_probs = torch.stack((fg_target, 1.0 - fg_target), dim=-1)
        log_probs = F.log_softmax(pred_object_logits, dim=-1)
        weighted_targets = target_probs * self.object_empty_weight.view(1, 1, -1)
        denom = weighted_targets.sum(dim=-1).clamp_min(1e-6)
        loss = -(weighted_targets * log_probs).sum(dim=-1) / denom
        return loss.mean()
