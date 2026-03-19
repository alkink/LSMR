import sys

import torch

sys.path.insert(0, '.')

from models.LSTR_CULANE_2k_mamba_dense import build_dense_targets_from_legacy_targets, loss as DenseLoss


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
    image_h = 295
    image_w = 820
    dense_h = 45
    dense_w = 80

    images = torch.zeros((1, 3, image_h, image_w), dtype=torch.float32)
    legacy_label = _build_legacy_label()
    targets = [images, legacy_label.unsqueeze(0)]

    dense_targets = build_dense_targets_from_legacy_targets(
        targets=targets,
        target_size=(dense_h, dense_w),
        device=torch.device('cpu'),
        line_width=1.5,
        min_valid_rows=2,
    )
    assert len(dense_targets) == 1
    target = dense_targets[0]

    assert target['gt_row_rng'].shape == (1, 2)
    assert target['gt_row_loc'].shape == (1, dense_h)
    assert target['gt_row_reg'].shape == (1, dense_h, dense_w)
    assert target['gt_label_obj'].tolist() == [0]
    assert target['gt_label_cls'].tolist() == [0]
    assert int(target['gt_row_loc_mask'].sum().item()) >= 2

    outputs = {
        'pred_object_logits': torch.tensor([[[6.0, -6.0], [-6.0, 6.0]]], dtype=torch.float32),
        'pred_class_logits': torch.zeros((1, 2, 1), dtype=torch.float32),
        'pred_ranges': torch.zeros((1, 2, 2), dtype=torch.float32),
        'pred_dense_mask': torch.zeros((1, 2, dense_h, dense_w), dtype=torch.float32),
        'pred_dense_reg': torch.zeros((1, 2, dense_h, dense_w), dtype=torch.float32),
        'aux_outputs': [],
    }
    outputs['pred_ranges'][0, 0] = target['gt_row_rng'][0]
    outputs['pred_ranges'][0, 1] = torch.tensor([0.7, 0.9], dtype=torch.float32)
    outputs['pred_dense_mask'][0, 0] = _build_mask_logits_from_target_locations(
        target['gt_row_loc'][0],
        target['gt_row_loc_mask'][0],
        dense_w,
    )
    outputs['pred_dense_mask'][0, 1].fill_(-1.0)
    outputs['pred_dense_reg'][0, 0] = target['gt_row_reg'][0]
    outputs['pred_dense_reg'][0, 1].fill_(8.0)

    loss_module = DenseLoss()
    total, reduced, unscaled, scaled, loss_value = loss_module(
        iteration=0,
        save=False,
        viz_split='train',
        outputs=outputs,
        targets=targets,
    )

    assert torch.isfinite(total)
    assert loss_value < 0.3
    assert reduced['loss_object'].item() < 1e-3
    assert reduced['loss_dense_mask'].item() < 1e-3
    assert reduced['loss_row_location'].item() < 0.3
    assert set(reduced.keys()) == {
        'loss_object',
        'loss_class',
        'loss_row_location',
        'loss_row_iou',
        'loss_dense_mask',
        'loss_row_reg',
        'loss_row_range',
        'class_error',
    }
    assert set(unscaled.keys()) == {f'{name}_unscaled' for name in reduced.keys()}
    assert set(scaled.keys()) == (set(reduced.keys()) - {'class_error'})

    print('LSTR_CULANE_2k_mamba_dense integration contract OK')


if __name__ == '__main__':
    main()



