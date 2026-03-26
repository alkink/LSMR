from __future__ import annotations

import math
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn


PRIOR_PROB = 0.01
LOGIT_BIAS_INIT = -math.log((1.0 - PRIOR_PROB) / PRIOR_PROB)


def _build_normalized_locations(height: int, width: int, device: torch.device) -> torch.Tensor:
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    if height > 1:
        grid_y = grid_y / float(height - 1)
    else:
        grid_y = torch.zeros_like(grid_y)

    if width > 1:
        grid_x = grid_x / float(width - 1)
    else:
        grid_x = torch.zeros_like(grid_x)

    return torch.stack((grid_x, grid_y), dim=0)


class QueryBranchHead(nn.Module):
    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        num_classes: int,
        mask_param_dim: int,
        reg_param_dim: int,
        mask_bias_channels: int,
        predict_ranges: bool = True,
        visibility_dim: int = 0,
    ):
        super().__init__()
        self.object_logits = self._make_branch(query_dim, hidden_dim, 2, init_logits_bias=True)
        self.class_logits = self._make_branch(query_dim, hidden_dim, num_classes)
        self.predict_ranges = bool(predict_ranges)
        self.visibility_dim = int(visibility_dim)
        self.ranges = self._make_branch(query_dim, hidden_dim, 2) if self.predict_ranges else None
        self.visibility_logits = (
            self._make_branch(query_dim, hidden_dim, self.visibility_dim)
            if self.visibility_dim > 0 else None
        )
        self.mask_params = self._make_branch(
            query_dim,
            hidden_dim,
            mask_param_dim,
            output_bias_init=self._make_mask_param_bias_init(mask_param_dim, mask_bias_channels),
        )
        self.reg_params = self._make_branch(query_dim, hidden_dim, reg_param_dim)

    @staticmethod
    def _make_branch(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        init_logits_bias: bool = False,
        output_bias_init: torch.Tensor | None = None,
    ) -> nn.Sequential:
        branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(branch[-1].bias)
        if init_logits_bias:
            nn.init.constant_(branch[-1].bias, LOGIT_BIAS_INIT)
        if output_bias_init is not None:
            branch[-1].bias.data.copy_(output_bias_init)
        return branch

    @staticmethod
    def _make_mask_param_bias_init(mask_param_dim: int, mask_bias_channels: int) -> torch.Tensor:
        bias_init = torch.zeros(mask_param_dim, dtype=torch.float32)
        if mask_bias_channels > 0:
            bias_init[-mask_bias_channels:] = LOGIT_BIAS_INIT
        return bias_init

    def forward(self, query_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {
            'pred_object_logits': self.object_logits(query_features),
            'pred_class_logits': self.class_logits(query_features),
            'pred_mask_params': self.mask_params(query_features),
            'pred_reg_params': self.reg_params(query_features),
        }
        if self.ranges is not None:
            outputs['pred_ranges'] = self.ranges(query_features).sigmoid()
        if self.visibility_logits is not None:
            outputs['pred_row_visibility_logits'] = self.visibility_logits(query_features)
        return outputs


class DynamicSpatialBranch(nn.Module):
    def __init__(self, feature_channels: int, out_channels: int = 1, use_coords: bool = False):
        super().__init__()
        self.feature_channels = feature_channels
        self.out_channels = out_channels
        self.use_coords = use_coords
        self.input_channels = feature_channels + (2 if use_coords else 0)
        self.param_dim = (self.input_channels * out_channels) + out_channels

    def forward(self, feature_map: torch.Tensor, dynamic_params: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(f'feature_map must be 4D [B, C, H, W], got shape {tuple(feature_map.shape)}')
        if dynamic_params.dim() != 3:
            raise ValueError(f'dynamic_params must be 3D [B, Q, P], got shape {tuple(dynamic_params.shape)}')
        if feature_map.size(0) != dynamic_params.size(0):
            raise ValueError('feature_map and dynamic_params batch size must match')
        if feature_map.size(1) != self.feature_channels:
            raise ValueError(
                f'feature_map channel mismatch: expected {self.feature_channels}, got {feature_map.size(1)}'
            )
        if dynamic_params.size(-1) != self.param_dim:
            raise ValueError(
                f'dynamic_params last dimension mismatch: expected {self.param_dim}, got {dynamic_params.size(-1)}'
            )

        batch_size, _, height, width = feature_map.shape
        augmented_feature_map = self._augment_with_coords(feature_map)
        weights, biases = self._split_dynamic_params(dynamic_params)
        flattened_features = augmented_feature_map.flatten(2)
        outputs = weights.bmm(flattened_features) + biases
        outputs = outputs.view(batch_size, dynamic_params.size(1), self.out_channels, height, width)
        return outputs

    def _augment_with_coords(self, feature_map: torch.Tensor) -> torch.Tensor:
        if not self.use_coords:
            return feature_map

        _, _, height, width = feature_map.shape
        locations = _build_normalized_locations(height, width, feature_map.device)
        locations = locations.unsqueeze(0).expand(feature_map.size(0), -1, -1, -1)
        return torch.cat((feature_map, locations), dim=1)

    def _split_dynamic_params(self, dynamic_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_param_count = self.input_channels * self.out_channels
        weight_params = dynamic_params[..., :weight_param_count]
        bias_params = dynamic_params[..., weight_param_count:]

        batch_size, num_queries = dynamic_params.shape[:2]
        weights = weight_params.reshape(batch_size, num_queries * self.out_channels, self.input_channels).contiguous()
        biases = bias_params.reshape(batch_size, num_queries * self.out_channels, 1).contiguous()
        return weights, biases


class CondLSTRParityHead(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        query_channels: int,
        num_classes: int,
        branch_hidden_dim: int = 256,
        mask_out_channels: int = 1,
        reg_out_channels: int = 1,
        use_coords: bool = False,
        predict_ranges: bool = True,
        visibility_dim: int = 0,
    ):
        super().__init__()
        self.mask_out_channels = mask_out_channels
        self.reg_out_channels = reg_out_channels
        self.mask_branch = DynamicSpatialBranch(
            feature_channels=feature_channels,
            out_channels=mask_out_channels,
            use_coords=use_coords,
        )
        self.reg_branch = DynamicSpatialBranch(
            feature_channels=feature_channels,
            out_channels=reg_out_channels,
            use_coords=use_coords,
        )
        self.query_branch_head = QueryBranchHead(
            query_dim=query_channels,
            hidden_dim=branch_hidden_dim,
            num_classes=num_classes,
            mask_param_dim=self.mask_branch.param_dim,
            reg_param_dim=self.reg_branch.param_dim,
            mask_bias_channels=mask_out_channels,
            predict_ranges=predict_ranges,
            visibility_dim=visibility_dim,
        )

    def forward(
        self,
        feature_map: Union[torch.Tensor, Sequence[torch.Tensor]],
        query_features: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        if isinstance(feature_map, (list, tuple)) or isinstance(query_features, (list, tuple)):
            return self._forward_sequence(feature_map, query_features)
        return self._forward_single(feature_map, query_features)

    def _forward_single(self, feature_map: torch.Tensor, query_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.query_branch_head(query_features)
        outputs['pred_dense_reg'] = self._format_dense_output(
            self.reg_branch(feature_map, outputs['pred_reg_params']),
            self.reg_out_channels,
        )
        outputs['pred_dense_mask'] = self._format_dense_output(
            self.mask_branch(feature_map, outputs['pred_mask_params']),
            self.mask_out_channels,
        )
        return outputs

    def _forward_sequence(
        self,
        feature_maps: Sequence[torch.Tensor],
        query_features: Sequence[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        if len(feature_maps) != len(query_features):
            raise ValueError('feature_map and query_features sequence lengths must match')
        if len(feature_maps) == 0:
            raise ValueError('feature_map and query_features sequences must be non-empty')

        per_layer_outputs = [
            self._forward_single(single_feature_map, single_query_features)
            for single_feature_map, single_query_features in zip(feature_maps, query_features)
        ]

        return {
            key: [layer_output[key] for layer_output in per_layer_outputs]
            for key in per_layer_outputs[0]
        }

    @staticmethod
    def _format_dense_output(dense_output: torch.Tensor, out_channels: int) -> torch.Tensor:
        if out_channels == 1:
            return dense_output.squeeze(2)
        return dense_output
