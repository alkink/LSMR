import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from models.condlstr_dense_criterion import CondLSTRDenseLoss, CondLSTRDenseSetCriterion
from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher
from utils.condlstr_dense_targets import convert_culane_points_to_rowwise_targets


def _build_mask_logits_from_target_locations(target_locations: torch.Tensor, target_mask: torch.Tensor, width: int) -> torch.Tensor:
    height = target_locations.size(0)
    logits = torch.full((height, width), -8.0, dtype=torch.float32)
    active_rows = torch.nonzero(target_mask > 0.5, as_tuple=False).flatten()
    for row_index in active_rows.tolist():
        column_index = int(round(float(target_locations[row_index].item())))
        column_index = max(0, min(width - 1, column_index))
        logits[row_index, column_index] = 8.0
    return logits


def _build_synthetic_batch():
    target_size = (10, 20)
    height, width = target_size
    targets = [
        convert_culane_points_to_rowwise_targets(
            lanes=[[(5.0, 2.0), (7.0, 4.0), (9.0, 6.0)]],
            image_size=(10, 20),
            target_size=target_size,
            lane_attrs=[2],
            line_width=1.5,
        )
    ]

    pred_object_logits = torch.tensor([[[6.0, -6.0], [-6.0, 6.0]]], dtype=torch.float32)
    pred_class_logits = torch.full((1, 2, 5), -4.0, dtype=torch.float32)
    pred_class_logits[0, 0, 2] = 7.0
    pred_ranges = torch.zeros((1, 2, 2), dtype=torch.float32)
    pred_dense_mask = torch.zeros((1, 2, height, width), dtype=torch.float32)
    pred_dense_reg = torch.zeros((1, 2, height, width), dtype=torch.float32)

    pred_ranges[0, 0] = targets[0]['gt_row_rng'][0]
    pred_ranges[0, 1] = torch.tensor([0.7, 0.9], dtype=torch.float32)
    pred_dense_mask[0, 0] = _build_mask_logits_from_target_locations(
        targets[0]['gt_row_loc'][0],
        targets[0]['gt_row_loc_mask'][0],
        width,
    )
    pred_dense_mask[0, 1].fill_(-1.0)
    pred_dense_reg[0, 0] = targets[0]['gt_row_reg'][0]
    pred_dense_reg[0, 1].fill_(8.0)

    outputs = {
        'pred_object_logits': pred_object_logits,
        'pred_class_logits': pred_class_logits,
        'pred_ranges': pred_ranges,
        'pred_dense_mask': pred_dense_mask,
        'pred_dense_reg': pred_dense_reg,
    }
    return outputs, targets


def _build_two_lane_batch_for_dense_mask_normalization():
    width = 8
    height = 4
    targets = [{
        'gt_label_obj': torch.tensor([0, 0], dtype=torch.int64),
        'gt_label_cls': torch.tensor([1, 2], dtype=torch.int64),
        'gt_row_loc': torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        'gt_row_loc_mask': torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        'gt_row_reg': torch.zeros((2, height, width), dtype=torch.float32),
        'gt_row_reg_mask': torch.zeros((2, height, width), dtype=torch.float32),
        'gt_row_rng': torch.tensor([[0.0, 0.5], [0.0, 0.25]], dtype=torch.float32),
    }]

    pred_object_logits = torch.tensor([[[8.0, -8.0], [8.0, -8.0]]], dtype=torch.float32)
    pred_class_logits = torch.full((1, 2, 4), -6.0, dtype=torch.float32)
    pred_class_logits[0, 0, 1] = 6.0
    pred_class_logits[0, 1, 2] = 6.0
    pred_ranges = targets[0]['gt_row_rng'].unsqueeze(0).clone()
    pred_dense_mask = torch.full((1, 2, height, width), -6.0, dtype=torch.float32)
    pred_dense_reg = torch.zeros((1, 2, height, width), dtype=torch.float32)

    pred_dense_mask[0, 0, 0, 1] = 6.0
    pred_dense_mask[0, 0, 1, 0] = 6.0
    pred_dense_mask[0, 1, 0, 5] = 6.0

    outputs = {
        'pred_object_logits': pred_object_logits,
        'pred_class_logits': pred_class_logits,
        'pred_ranges': pred_ranges,
        'pred_dense_mask': pred_dense_mask,
        'pred_dense_reg': pred_dense_reg,
    }
    return outputs, targets


def main() -> None:
    outputs, targets = _build_synthetic_batch()
    matcher = CondLSTRDenseHungarianMatcher(line_width=1.5)
    criterion = CondLSTRDenseSetCriterion(matcher=matcher, line_width=1.5)
    loss_module = CondLSTRDenseLoss(criterion=criterion)

    loss_dict, indices = criterion(outputs, targets)
    expected_keys = {
        'loss_object',
        'loss_class',
        'loss_row_location',
        'loss_row_iou',
        'loss_dense_mask',
        'loss_row_reg',
        'loss_row_range',
    }
    assert set(loss_dict.keys()) == expected_keys
    assert len(indices) == 1
    assert indices[0][0].tolist() == [0]
    assert indices[0][1].tolist() == [0]
    for value in loss_dict.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

    assert loss_dict['loss_object'].item() < 1e-3
    assert loss_dict['loss_class'].item() < 1e-3
    assert loss_dict['loss_row_location'].item() < 1e-3
    assert loss_dict['loss_row_iou'].item() == 0.0
    assert loss_dict['loss_dense_mask'].item() < 1e-3
    assert loss_dict['loss_row_reg'].item() < 1e-6
    assert loss_dict['loss_row_range'].item() < 1e-6

    criterion_with_row_iou = CondLSTRDenseSetCriterion(
        matcher=matcher,
        line_width=1.5,
        enable_row_iou_loss=True,
    )
    loss_dict_with_row_iou, _ = criterion_with_row_iou(outputs, targets)
    assert loss_dict_with_row_iou['loss_row_iou'].item() < 5e-3

    packaged = loss_module(outputs, targets)
    assert set(packaged.keys()) == {'loss_total', 'loss_dict', 'weight_dict', 'indices'}
    assert packaged['indices'][0][0].tolist() == [0]
    assert torch.isclose(
        packaged['loss_total'],
        sum(packaged['loss_dict'][name] * packaged['weight_dict'][name] for name in packaged['loss_dict'].keys()),
    )

    perturbed_outputs, perturbed_targets = _build_synthetic_batch()
    perturbed_outputs['pred_ranges'][0, 0] = torch.tensor([0.0, 0.1], dtype=torch.float32)
    perturbed_outputs['pred_dense_reg'][0, 0].fill_(4.0)
    perturbed_outputs['pred_dense_mask'][0, 0].fill_(-4.0)
    perturbed_outputs['pred_dense_mask'][0, 0, :, 0] = 4.0
    perturbed_loss_dict, _ = criterion(perturbed_outputs, perturbed_targets)

    assert perturbed_loss_dict['loss_dense_mask'] > loss_dict['loss_dense_mask']
    assert perturbed_loss_dict['loss_row_reg'] > loss_dict['loss_row_reg']
    assert perturbed_loss_dict['loss_row_range'] > loss_dict['loss_row_range']

    dense_mask_outputs, dense_mask_targets = _build_two_lane_batch_for_dense_mask_normalization()
    dense_mask_loss_dict, dense_mask_indices = criterion(dense_mask_outputs, dense_mask_targets)
    assert dense_mask_indices[0][0].tolist() == [0, 1]
    assert dense_mask_indices[0][1].tolist() == [0, 1]

    lane0_row0 = F.cross_entropy(
        dense_mask_outputs['pred_dense_mask'][0, 0, 0].unsqueeze(0),
        torch.tensor([1], dtype=torch.int64),
    )
    lane0_row1 = F.cross_entropy(
        dense_mask_outputs['pred_dense_mask'][0, 0, 1].unsqueeze(0),
        torch.tensor([2], dtype=torch.int64),
    )
    lane1_row0 = F.cross_entropy(
        dense_mask_outputs['pred_dense_mask'][0, 1, 0].unsqueeze(0),
        torch.tensor([5], dtype=torch.int64),
    )
    expected_dense_mask_loss = (((lane0_row0 + lane0_row1) / 2.0) + lane1_row0) / 2.0
    flat_row_mean = (lane0_row0 + lane0_row1 + lane1_row0) / 3.0

    assert torch.isclose(dense_mask_loss_dict['loss_dense_mask'], expected_dense_mask_loss, atol=1e-6)
    assert not torch.isclose(dense_mask_loss_dict['loss_dense_mask'], flat_row_mean, atol=1e-6)

    print('CondLSTR dense criterion contract OK')


if __name__ == '__main__':
    main()
