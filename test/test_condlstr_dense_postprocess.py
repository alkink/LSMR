import sys

import torch

sys.path.insert(0, '.')

from models.condlstr_dense_postprocess import dense_outputs_to_lane_coords
from test.culane import PostProcess
from db.utils.evaluator import Evaluator


def _build_dense_outputs(batch_size: int = 1):
    height, width = 6, 10
    num_queries = 3

    pred_object_logits = torch.tensor(
        [[
            [8.0, -8.0],   # strong fg
            [-8.0, 8.0],   # background
            [8.0, -8.0],   # fg but too short (min_points filter)
        ]],
        dtype=torch.float32,
    )
    pred_ranges = torch.tensor(
        [[
            [1.0 / height, 5.0 / height],
            [0.0, 1.0],
            [2.0 / height, 2.0 / height],
        ]],
        dtype=torch.float32,
    )

    pred_dense_mask = torch.full((batch_size, num_queries, height, width), -5.0, dtype=torch.float32)
    pred_dense_reg = torch.zeros((batch_size, num_queries, height, width), dtype=torch.float32)

    # Query-0: clean diagonal lane in decoded row range.
    for row, col in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
        pred_dense_mask[0, 0, row, col] = 5.0
        pred_dense_reg[0, 0, row, col] = 0.25

    # Query-2: single-row lane, should be filtered by min_points=2.
    pred_dense_mask[0, 2, 2, 7] = 5.0

    return {
        'pred_object_logits': pred_object_logits,
        'pred_ranges': pred_ranges,
        'pred_dense_mask': pred_dense_mask,
        'pred_dense_reg': pred_dense_reg,
    }


def test_dense_postprocess_core_behavior() -> None:
    outputs = _build_dense_outputs()
    target_sizes = torch.tensor([[120, 240]], dtype=torch.float32)

    lanes_batch = dense_outputs_to_lane_coords(
        outputs=outputs,
        target_sizes=target_sizes,
        score_thresh=0.5,
        min_points=2,
    )

    assert isinstance(lanes_batch, list)
    assert len(lanes_batch) == 1
    assert len(lanes_batch[0]) == 1  # only query-0 survives (q1 score, q2 min_points)

    lane = lanes_batch[0][0]
    assert len(lane) >= 4
    xs = [point[0] for point in lane]
    ys = [point[1] for point in lane]
    assert all(0.0 <= x <= 239.0 for x in xs)
    assert all(0.0 <= y <= 119.0 for y in ys)
    assert ys == sorted(ys)


def test_dense_postprocess_accepts_5d_single_channel_inputs() -> None:
    outputs = _build_dense_outputs()
    outputs['pred_dense_mask'] = outputs['pred_dense_mask'].unsqueeze(2)
    outputs['pred_dense_reg'] = outputs['pred_dense_reg'].unsqueeze(2)

    target_sizes = torch.tensor([[120, 240]], dtype=torch.float32)
    lanes_batch = dense_outputs_to_lane_coords(outputs=outputs, target_sizes=target_sizes)
    assert len(lanes_batch) == 1
    assert len(lanes_batch[0]) == 1


def test_dense_postprocess_range_format_normalized_vs_absolute() -> None:
    outputs = _build_dense_outputs()
    target_sizes = torch.tensor([[120, 240]], dtype=torch.float32)

    # Baseline uses normalized ranges from dense targets.
    lanes_normalized = dense_outputs_to_lane_coords(
        outputs=outputs,
        target_sizes=target_sizes,
        range_format='normalized',
    )
    assert len(lanes_normalized[0]) == 1

    # Convert ranges to absolute feature-row indices and decode with absolute mode.
    abs_outputs = dict(outputs)
    abs_outputs['pred_ranges'] = outputs['pred_ranges'] * outputs['pred_dense_mask'].shape[-2]
    lanes_absolute = dense_outputs_to_lane_coords(
        outputs=abs_outputs,
        target_sizes=target_sizes,
        range_format='absolute',
    )
    assert len(lanes_absolute[0]) == 1
    assert len(lanes_absolute[0][0]) == len(lanes_normalized[0][0])


def test_dense_postprocess_range_format_auto_backcompat() -> None:
    outputs = _build_dense_outputs()
    target_sizes = torch.tensor([[120, 240]], dtype=torch.float32)

    lanes_auto = dense_outputs_to_lane_coords(
        outputs=outputs,
        target_sizes=target_sizes,
        range_format='auto',
    )
    lanes_norm = dense_outputs_to_lane_coords(
        outputs=outputs,
        target_sizes=target_sizes,
        range_format='normalized',
    )

    assert len(lanes_auto[0]) == len(lanes_norm[0])
    assert len(lanes_auto[0][0]) == len(lanes_norm[0][0])


def test_culane_postprocess_dense_integration_path() -> None:
    outputs = _build_dense_outputs()
    target_sizes = torch.tensor([[120, 240]], dtype=torch.float32)

    post = PostProcess()
    results = post(outputs, target_sizes)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], list)
    assert len(results[0]) == 1
    assert len(results[0][0]) >= 4


def test_dense_predictions_are_compatible_with_evaluator_add_prediction() -> None:
    class _DummyDataset:
        def __init__(self) -> None:
            self._annotations = [object()]

        def __len__(self) -> int:
            return len(self._annotations)

        def eval(self, exp_dir, predictions, runtimes, **kwargs):
            return 0

    outputs = _build_dense_outputs()
    target_sizes = torch.tensor([[120, 240]], dtype=torch.float32)
    post = PostProcess()
    results = post(outputs, target_sizes)

    evaluator = Evaluator(dataset=_DummyDataset(), exp_dir='.')
    evaluator.add_prediction(0, results[0], runtime=0.01)

    assert evaluator.predictions_mode == 'lanes'
    assert isinstance(evaluator.predictions[0], list)
    assert len(evaluator.predictions[0]) == 1
    assert len(evaluator.predictions[0][0]) >= 2


def main() -> None:
    test_dense_postprocess_core_behavior()
    test_dense_postprocess_accepts_5d_single_channel_inputs()
    test_dense_postprocess_range_format_normalized_vs_absolute()
    test_dense_postprocess_range_format_auto_backcompat()
    test_culane_postprocess_dense_integration_path()
    test_dense_predictions_are_compatible_with_evaluator_add_prediction()
    print('CondLSTR dense postprocess contract OK')


if __name__ == '__main__':
    main()

