import sys

import torch

sys.path.insert(0, '.')

from utils.condlstr_dense_targets import convert_culane_points_to_rowwise_targets


def main() -> None:
    lanes = [
        [(5.0, 2.0), (7.0, 4.0), (9.0, 6.0)],
        [(12.0, 1.0), (12.0, 8.0)],
        [(-4.0, 1.0), (-2.0, 5.0)],
    ]
    targets = convert_culane_points_to_rowwise_targets(
        lanes=lanes,
        image_size=(10, 20),
        target_size=(10, 20),
        lane_attrs=[3, 5, 9],
        line_width=0.5,
        min_valid_rows=2,
    )

    assert targets['gt_row_rng'].shape == (2, 2)
    assert targets['gt_row_loc'].shape == (2, 10)
    assert targets['gt_row_reg'].shape == (2, 10, 20)
    assert targets['gt_row_loc_mask'].shape == (2, 10)
    assert targets['gt_row_reg_mask'].shape == (2, 10, 20)
    assert targets['gt_label_obj'].shape == (2,)
    assert targets['gt_label_cls'].shape == (2,)

    torch.testing.assert_close(targets['gt_row_rng'][0], torch.tensor([0.2, 0.6]))
    torch.testing.assert_close(targets['gt_row_rng'][1], torch.tensor([0.1, 0.8]))

    expected_lane0_mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
    expected_lane1_mask = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.float32)
    torch.testing.assert_close(targets['gt_row_loc_mask'][0], expected_lane0_mask)
    torch.testing.assert_close(targets['gt_row_loc_mask'][1], expected_lane1_mask)

    expected_lane0_xs = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
    torch.testing.assert_close(targets['gt_row_loc'][0, 2:7], expected_lane0_xs)
    torch.testing.assert_close(targets['gt_row_loc'][1, 1:9], torch.full((8,), 12.0))

    assert torch.all(targets['gt_label_obj'] == 0)
    assert targets['gt_label_cls'].tolist() == [3, 5]

    assert targets['gt_row_reg_mask'][0, 4, 7].item() == 1.0
    assert targets['gt_row_reg_mask'][0, 4, 0].item() == 0.0
    assert abs(targets['gt_row_reg'][0, 4, 7].item()) < 1e-6
    assert targets['gt_row_reg'][0, 4, 6].item() > 0.0
    assert targets['gt_row_reg'][0, 4, 8].item() < 0.0

    empty_targets = convert_culane_points_to_rowwise_targets(
        lanes=[],
        image_size=(10, 20),
        target_size=(6, 8),
    )
    assert empty_targets['gt_row_rng'].shape == (0, 2)
    assert empty_targets['gt_row_loc'].shape == (0, 6)
    assert empty_targets['gt_row_reg'].shape == (0, 6, 8)

    print('CondLSTR-style dense target conversion contract OK')


if __name__ == '__main__':
    main()
