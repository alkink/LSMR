from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import system_configs
from models.LSTR_CULANE import model as _BaseTransformerModel
from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher
from models.condlstr_parity_criterion import CondLSTRParitySetCriterion
from models.condlstr_parity_head import CondLSTRParityHead
from models.parity_stdc_res34_backbone import STDCResNet34Backbone
from models.py_utils.misc import reduce_dict
from utils.condlstr_parity_targets import build_parity_targets_from_legacy_targets


def _resize_dense_map(tensor: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    batch_size, num_queries, height, width = tensor.shape
    tensor = tensor.view(batch_size * num_queries, 1, height, width)
    tensor = F.interpolate(tensor, size=output_size, mode='bilinear', align_corners=True)
    return tensor.view(batch_size, num_queries, output_size[0], output_size[1])


class model(_BaseTransformerModel):
    def __init__(self, flag: bool = False):
        super().__init__(flag=flag)
        dense_num_classes = max(int(system_configs.full.get('dense_num_classes', 1)), 1)
        branch_hidden_dim = int(system_configs.full.get('dense_branch_hidden_dim', 256))
        use_coords = bool(system_configs.full.get('dense_use_coords', False))
        self.mask_downscale = int(system_configs.full.get('condlstr_mask_downscale', 1))
        self.parity_backbone_mode = str(system_configs.full.get('parity_backbone', 'lstr')).lower()
        self.parity_backbone = None
        if self.parity_backbone_mode == 'stdc_res34':
            self.parity_backbone = STDCResNet34Backbone(
                pretrained=bool(system_configs.full.get('parity_backbone_pretrained', False)),
                norm_layer=nn.BatchNorm2d,
            )
        elif self.parity_backbone_mode != 'lstr':
            raise ValueError(f"Unsupported parity_backbone={self.parity_backbone_mode!r}")

        self.parity_head = CondLSTRParityHead(
            feature_channels=int(system_configs.attn_dim),
            query_channels=int(system_configs.attn_dim),
            num_classes=dense_num_classes,
            branch_hidden_dim=branch_hidden_dim,
            mask_out_channels=1,
            reg_out_channels=1,
            use_coords=use_coords,
        )

        print(
            "[LSTR_CULANE_condlstr_parity_base] "
            f"backbone={self.parity_backbone_mode} "
            f"head -> CondLSTRParityHead(num_classes={dense_num_classes}, hidden={branch_hidden_dim}, use_coords={use_coords})"
        )

    def _extract_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.parity_backbone is not None:
            return self.parity_backbone(images)

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)
        return p

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]

        p = self._extract_backbone_features(images)

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)
        hs, memory, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)

        query_features_per_layer = [layer_output for layer_output in hs]
        feature_maps = [memory] * len(query_features_per_layer)
        per_layer_outputs = self.parity_head(feature_map=feature_maps, query_features=query_features_per_layer)
        output_size = (
            int(images.shape[-2] // max(self.mask_downscale, 1)),
            int(images.shape[-1] // max(self.mask_downscale, 1)),
        )

        formatted_outputs: List[Dict[str, torch.Tensor]] = []
        num_layers = len(per_layer_outputs['pred_object_logits'])
        for layer_index in range(num_layers):
            layer_output = {key: value[layer_index] for key, value in per_layer_outputs.items()}
            layer_output['pred_dense_mask'] = _resize_dense_map(layer_output['pred_dense_mask'], output_size)
            layer_output['pred_dense_reg'] = _resize_dense_map(layer_output['pred_dense_reg'], output_size)
            formatted_outputs.append(layer_output)

        final_output = dict(formatted_outputs[-1])
        final_output['aux_outputs'] = formatted_outputs[:-1]
        final_output['postprocess_mode'] = 'condlstr_parity'
        return final_output, weights

    def _test(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class loss(nn.Module):
    def __init__(self):
        super().__init__()
        snapshot_name = system_configs.snapshot_name
        base_result_dir = system_configs.full.get('result_dir', './results')
        self.debug_path = os.path.join(base_result_dir, snapshot_name) if snapshot_name else base_result_dir
        self.line_width = float(system_configs.full.get('dense_line_width', 16.0))
        self.min_valid_rows = int(system_configs.full.get('dense_min_valid_rows', 2))

        self.weight_dict = {
            'loss_object': float(system_configs.full.get('dense_loss_object_weight', 10.0)),
            'loss_class': float(system_configs.full.get('dense_loss_class_weight', 10.0)),
            'loss_loc': float(system_configs.full.get('dense_loss_loc_weight', 1.0)),
            'loss_reg': float(system_configs.full.get('dense_loss_reg_weight', 1.0)),
            'loss_range': float(system_configs.full.get('dense_loss_range_weight', 20.0)),
        }
        base_weight_dict = dict(self.weight_dict)
        aux_weight_dict = {}
        for layer_index in range(max(int(system_configs.dec_layers) - 1, 0)):
            for name, value in base_weight_dict.items():
                aux_weight_dict[f'{name}_{layer_index}'] = value
        self.weight_dict.update(aux_weight_dict)

        matcher = CondLSTRDenseHungarianMatcher(
            line_width=self.line_width,
            object_weight=float(system_configs.full.get('dense_match_object_weight', 10.0)),
            class_weight=float(system_configs.full.get('dense_match_class_weight', 10.0)),
            location_weight=float(system_configs.full.get('dense_match_location_weight', 1.0)),
            location_iou_weight=float(system_configs.full.get('dense_match_location_iou_weight', 2.0)),
            regression_weight=float(system_configs.full.get('dense_match_regression_weight', 1.0)),
            range_weight=float(system_configs.full.get('dense_match_range_weight', 20.0)),
        )
        self.criterion = CondLSTRParitySetCriterion(
            matcher=matcher,
            line_width=self.line_width,
            object_eos_coef=float(system_configs.full.get('dense_object_eos_coef', 0.4)),
        )

        print(f"[LSTR_CULANE_condlstr_parity_base] weight_dict: {self.weight_dict}")

    def forward(self, iteration, save, viz_split, outputs, targets):
        del iteration, save, viz_split

        spatial_size = (int(outputs['pred_dense_mask'].shape[-2]), int(outputs['pred_dense_mask'].shape[-1]))
        dense_targets = build_parity_targets_from_legacy_targets(
            targets=targets,
            target_size=spatial_size,
            device=outputs['pred_object_logits'].device,
            line_width=self.line_width,
            min_valid_rows=self.min_valid_rows,
        )

        loss_dict, _ = self.criterion(outputs, dense_targets)
        loss_dict['class_error'] = outputs['pred_object_logits'].new_tensor(0.0)
        total = sum(
            loss_dict[name] * self.weight_dict[name]
            for name in self.weight_dict.keys()
            if name in loss_dict
        )

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * self.weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in self.weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            raise RuntimeError('Non-finite parity loss encountered')

        return (
            total,
            loss_dict_reduced,
            loss_dict_reduced_unscaled,
            loss_dict_reduced_scaled,
            loss_value,
        )
