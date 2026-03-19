import sys

import torch

sys.path.insert(0, '.')

from models.dynamic_lane_head import DynamicLaneHead, LOGIT_BIAS_INIT


def main() -> None:
    batch_size = 2
    num_queries = 7
    feature_channels = 16
    query_channels = 32
    num_classes = 4
    mask_out_channels = 3
    reg_out_channels = 2
    height = 9
    width = 13

    head = DynamicLaneHead(
        feature_channels=feature_channels,
        query_channels=query_channels,
        num_classes=num_classes,
        branch_hidden_dim=24,
        mask_out_channels=mask_out_channels,
        reg_out_channels=reg_out_channels,
        use_coords=True,
    )

    feature_map = torch.randn(batch_size, feature_channels, height, width)
    query_features = torch.randn(batch_size, num_queries, query_channels)

    outputs = head(feature_map, query_features)

    assert outputs['pred_object_logits'].shape == (batch_size, num_queries, 2)
    assert outputs['pred_class_logits'].shape == (batch_size, num_queries, num_classes)
    assert outputs['pred_dense_reg'].shape == (batch_size, num_queries, reg_out_channels, height, width)
    assert outputs['pred_dense_mask'].shape == (batch_size, num_queries, mask_out_channels, height, width)
    assert outputs['pred_ranges'].shape == (batch_size, num_queries, 2)
    assert outputs['pred_mask_params'].shape == (batch_size, num_queries, head.mask_branch.param_dim)
    assert outputs['pred_reg_params'].shape == (batch_size, num_queries, head.reg_branch.param_dim)

    expected_mask_bias = torch.full((mask_out_channels,), LOGIT_BIAS_INIT)
    assert torch.allclose(head.query_branch_head.mask_params[-1].bias[-mask_out_channels:], expected_mask_bias)

    for name, tensor in outputs.items():
        assert torch.isfinite(tensor).all(), f'{name} contains non-finite values'

    aux_feature_maps = [
        torch.randn(batch_size, feature_channels, height, width),
        torch.randn(batch_size, feature_channels, height // 3 + 1, width // 2),
    ]
    aux_query_features = [
        torch.randn(batch_size, num_queries, query_channels),
        torch.randn(batch_size, num_queries, query_channels),
    ]

    aux_outputs = head(aux_feature_maps, aux_query_features)

    for name, value in aux_outputs.items():
        assert isinstance(value, list), f'{name} should return a per-layer list for sequence inputs'
        assert len(value) == len(aux_feature_maps), f'{name} should have one entry per layer'
        for tensor in value:
            assert torch.isfinite(tensor).all(), f'{name} contains non-finite values'

    assert aux_outputs['pred_object_logits'][0].shape == (batch_size, num_queries, 2)
    assert aux_outputs['pred_class_logits'][1].shape == (batch_size, num_queries, num_classes)
    assert aux_outputs['pred_dense_reg'][0].shape == (batch_size, num_queries, reg_out_channels, height, width)
    assert aux_outputs['pred_dense_mask'][1].shape == (
        batch_size,
        num_queries,
        mask_out_channels,
        height // 3 + 1,
        width // 2,
    )
    assert aux_outputs['pred_ranges'][0].shape == (batch_size, num_queries, 2)
    assert aux_outputs['pred_mask_params'][0].shape == (batch_size, num_queries, head.mask_branch.param_dim)
    assert aux_outputs['pred_reg_params'][1].shape == (batch_size, num_queries, head.reg_branch.param_dim)

    print('DynamicLaneHead shape contract OK')


if __name__ == '__main__':
    main()
