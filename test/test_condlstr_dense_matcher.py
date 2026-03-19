import sys

import torch

sys.path.insert(0, '.')

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


def main() -> None:
    target_size = (10, 20)
    width = target_size[1]
    lanes_image0 = [
        [(5.0, 2.0), (7.0, 4.0), (9.0, 6.0)],
        [(12.0, 1.0), (12.0, 8.0)],
    ]
    lanes_image1 = [
        [(3.0, 1.0), (5.0, 4.0), (7.0, 7.0)],
    ]

    targets = [
        convert_culane_points_to_rowwise_targets(
            lanes=lanes_image0,
            image_size=(10, 20),
            target_size=target_size,
            lane_attrs=[2, 4],
        ),
        convert_culane_points_to_rowwise_targets(
            lanes=lanes_image1,
            image_size=(10, 20),
            target_size=target_size,
            lane_attrs=[1],
        ),
    ]

    batch_size = 2
    num_queries = 4
    num_classes = 6
    height = target_size[0]

    pred_object_logits = torch.full((batch_size, num_queries, 2), -5.0, dtype=torch.float32)
    pred_class_logits = torch.full((batch_size, num_queries, num_classes), -4.0, dtype=torch.float32)
    pred_ranges = torch.zeros((batch_size, num_queries, 2), dtype=torch.float32)
    pred_dense_mask = torch.zeros((batch_size, num_queries, height, width), dtype=torch.float32)
    pred_dense_reg = torch.zeros((batch_size, num_queries, height, width), dtype=torch.float32)

    pred_object_logits[0, 0] = torch.tensor([6.0, -6.0])
    pred_object_logits[0, 1] = torch.tensor([6.0, -6.0])
    pred_object_logits[0, 2] = torch.tensor([-6.0, 6.0])
    pred_object_logits[0, 3] = torch.tensor([-6.0, 6.0])
    pred_object_logits[1, 0] = torch.tensor([6.0, -6.0])
    pred_object_logits[1, 1] = torch.tensor([-6.0, 6.0])
    pred_object_logits[1, 2] = torch.tensor([-6.0, 6.0])
    pred_object_logits[1, 3] = torch.tensor([-6.0, 6.0])

    pred_class_logits[0, 0, 2] = 7.0
    pred_class_logits[0, 1, 4] = 7.0
    pred_class_logits[1, 0, 1] = 7.0

    pred_ranges[0, 0] = targets[0]['gt_row_rng'][0]
    pred_ranges[0, 1] = targets[0]['gt_row_rng'][1]
    pred_ranges[1, 0] = targets[1]['gt_row_rng'][0]
    pred_ranges[0, 2] = torch.tensor([0.0, 0.1])
    pred_ranges[0, 3] = torch.tensor([0.8, 0.9])
    pred_ranges[1, 1] = torch.tensor([0.8, 0.9])
    pred_ranges[1, 2] = torch.tensor([0.0, 0.2])
    pred_ranges[1, 3] = torch.tensor([0.2, 0.3])

    pred_dense_mask[0, 0] = _build_mask_logits_from_target_locations(
        targets[0]['gt_row_loc'][0],
        targets[0]['gt_row_loc_mask'][0],
        width,
    )
    pred_dense_mask[0, 1] = _build_mask_logits_from_target_locations(
        targets[0]['gt_row_loc'][1],
        targets[0]['gt_row_loc_mask'][1],
        width,
    )
    pred_dense_mask[1, 0] = _build_mask_logits_from_target_locations(
        targets[1]['gt_row_loc'][0],
        targets[1]['gt_row_loc_mask'][0],
        width,
    )

    pred_dense_reg[0, 0] = targets[0]['gt_row_reg'][0]
    pred_dense_reg[0, 1] = targets[0]['gt_row_reg'][1]
    pred_dense_reg[1, 0] = targets[1]['gt_row_reg'][0]
    pred_dense_reg[0, 2].fill_(15.0)
    pred_dense_reg[0, 3].fill_(-15.0)
    pred_dense_reg[1, 1].fill_(9.0)
    pred_dense_reg[1, 2].fill_(-9.0)
    pred_dense_reg[1, 3].fill_(4.0)

    outputs = {
        'pred_object_logits': pred_object_logits,
        'pred_class_logits': pred_class_logits,
        'pred_ranges': pred_ranges,
        'pred_dense_mask': pred_dense_mask,
        'pred_dense_reg': pred_dense_reg,
    }

    matcher = CondLSTRDenseHungarianMatcher(line_width=1.0)
    matches, costs = matcher(outputs, targets, return_costs=True)

    assert len(matches) == batch_size
    assert len(costs) == batch_size

    matched_pred0, matched_tgt0 = matches[0]
    matched_pred1, matched_tgt1 = matches[1]

    assert matched_pred0.tolist() == [0, 1]
    assert matched_tgt0.tolist() == [0, 1]
    assert matched_pred1.tolist() == [0]
    assert matched_tgt1.tolist() == [0]

    assert costs[0].total_cost.shape == (num_queries, 2)
    assert costs[1].total_cost.shape == (num_queries, 1)
    assert costs[0].cost_row_location[0, 0] < 1e-3
    assert costs[0].cost_row_location[1, 1] < 1e-3
    assert costs[0].cost_row_reg[0, 0] < 1e-6
    assert costs[0].cost_row_reg[1, 1] < 1e-6
    assert costs[0].cost_row_range[0, 0] < 1e-6
    assert costs[0].cost_row_range[1, 1] < 1e-6
    assert costs[0].cost_row_location[0, 1] > costs[0].cost_row_location[0, 0]
    assert costs[0].cost_row_location[1, 0] > costs[0].cost_row_location[1, 1]
    assert costs[0].total_cost[2, 0] > costs[0].total_cost[0, 0]
    assert costs[0].total_cost[3, 1] > costs[0].total_cost[1, 1]

    empty_target = convert_culane_points_to_rowwise_targets(
        lanes=[],
        image_size=(10, 20),
        target_size=target_size,
    )
    empty_matches, empty_costs = matcher(outputs={
        'pred_object_logits': pred_object_logits[:1],
        'pred_class_logits': pred_class_logits[:1],
        'pred_ranges': pred_ranges[:1],
        'pred_dense_mask': pred_dense_mask[:1],
        'pred_dense_reg': pred_dense_reg[:1],
    }, targets=[empty_target], return_costs=True)
    assert empty_matches[0][0].numel() == 0
    assert empty_matches[0][1].numel() == 0
    assert empty_costs[0].total_cost.shape == (num_queries, 0)

    invalid_outputs = dict(outputs)
    invalid_outputs['pred_dense_mask'] = pred_dense_mask.unsqueeze(2).repeat(1, 1, 2, 1, 1)
    try:
        matcher(invalid_outputs, targets)
        raise AssertionError('Expected multi-channel dense mask validation to fail')
    except ValueError as exc:
        assert 'pred_dense_mask must have shape [B, Q, H, W] or [B, Q, 1, H, W]' in str(exc)

    print('CondLSTR dense Hungarian matcher contract OK')


if __name__ == '__main__':
    main()
