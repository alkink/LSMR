import copy
import sys

import torch

sys.path.insert(0, '.')

from models.condlstr_parity_criterion import CondLSTRParitySetCriterion
from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher
from utils.condlstr_parity_targets import build_parity_targets_from_legacy_targets


def _build_mask_logits_from_target_locations(
    target_locations: torch.Tensor,
    target_mask: torch.Tensor,
    width: int,
) -> torch.Tensor:
    height = target_locations.size(0)
    logits = torch.full((height, width), -8.0, dtype=torch.float32)
    active_rows = torch.nonzero(target_mask > 0.5, as_tuple=False).flatten()
    for row_index in active_rows.tolist():
        column_index = int(round(float(target_locations[row_index].item())))
        column_index = max(0, min(width - 1, column_index))
        logits[row_index, column_index] = 8.0
    return logits


def _build_legacy_label(max_lanes: int = 4, max_points: int = 6) -> torch.Tensor:
    label = torch.full((max_lanes, 1 + 2 + 2 * max_points), -1e5, dtype=torch.float32)
    label[0, 0] = 1.0
    label[0, 1] = 0.2
    label[0, 2] = 0.6
    label[0, 3:6] = torch.tensor([0.20, 0.35, 0.50], dtype=torch.float32)
    label[0, 3 + max_points : 3 + max_points + 3] = torch.tensor([0.20, 0.40, 0.65], dtype=torch.float32)
    return label


def main() -> None:
    image_h = 64
    image_w = 128

    images = torch.zeros((1, 3, image_h, image_w), dtype=torch.float32)
    legacy_label = _build_legacy_label()
    targets = [images, legacy_label.unsqueeze(0)]

    dense_targets = build_parity_targets_from_legacy_targets(
        targets=targets,
        target_size=(image_h, image_w),
        device=torch.device('cpu'),
        line_width=16.0,
        min_valid_rows=2,
    )
    target = dense_targets[0]

    outputs = {
        'pred_object_logits': torch.tensor([[[6.0, -6.0], [-6.0, 6.0]]], dtype=torch.float32),
        'pred_class_logits': torch.zeros((1, 2, 1), dtype=torch.float32),
        'pred_ranges': torch.zeros((1, 2, 2), dtype=torch.float32),
        'pred_dense_mask': torch.zeros((1, 2, image_h, image_w), dtype=torch.float32),
        'pred_dense_reg': torch.zeros((1, 2, image_h, image_w), dtype=torch.float32),
        'aux_outputs': [],
    }
    outputs['pred_ranges'][0, 0] = target['gt_row_rng'][0]
    outputs['pred_ranges'][0, 1] = torch.tensor([0.7, 0.9], dtype=torch.float32)
    outputs['pred_dense_mask'][0, 0] = _build_mask_logits_from_target_locations(
        target['gt_row_loc'][0],
        target['gt_row_loc_mask'][0],
        image_w,
    )
    outputs['pred_dense_mask'][0, 1].fill_(-1.0)
    outputs['pred_dense_reg'][0, 0] = target['gt_row_reg'][0]
    outputs['pred_dense_reg'][0, 1].fill_(8.0)

    matcher = CondLSTRDenseHungarianMatcher(
        line_width=16.0,
        object_weight=10.0,
        class_weight=10.0,
        location_weight=1.0,
        location_iou_weight=2.0,
        regression_weight=1.0,
        range_weight=20.0,
    )
    criterion = CondLSTRParitySetCriterion(
        matcher=matcher,
        line_width=16.0,
        object_eos_coef=0.4,
    )
    loss_dict, _ = criterion(outputs, dense_targets)

    assert set(loss_dict.keys()) == {
        'loss_object',
        'loss_class',
        'loss_loc',
        'loss_reg',
        'loss_range',
    }
    for value in loss_dict.values():
        assert torch.isfinite(value)

    aux_outputs = copy.deepcopy(outputs)
    aux_outputs['aux_outputs'] = [copy.deepcopy({k: v for k, v in outputs.items() if k != 'aux_outputs'})]
    aux_loss_dict, _ = criterion(aux_outputs, dense_targets)
    assert 'loss_object_0' in aux_loss_dict
    assert 'loss_loc_0' in aux_loss_dict

    print('CondLSTR parity integration contract OK')


if __name__ == '__main__':
    main()
