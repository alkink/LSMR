from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import system_configs
from models.LSTR_CULANE import model as _BaseTransformerModel
from models.condlstr_dense_matcher import CondLSTRDenseHungarianMatcher
from models.condlstr_dn_lane import build_dn_lane_queries
from models.condlstr_parity_criterion import CondLSTRParitySetCriterion
from models.condlstr_parity_head import CondLSTRParityHead
from models.condlstr_query_relation import QueryRelationBlock
from models.parity_stdc_res34_backbone import STDCResNet34Backbone
from models.py_utils.misc import reduce_dict
from utils.condlstr_parity_targets import build_parity_targets_from_legacy_targets


def _resize_dense_map(tensor: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    batch_size, num_queries, height, width = tensor.shape
    tensor = tensor.view(batch_size * num_queries, 1, height, width)
    tensor = F.interpolate(tensor, size=output_size, mode='bilinear', align_corners=True)
    return tensor.view(batch_size, num_queries, output_size[0], output_size[1])


def _build_dn_decoder_attention_mask(
    num_main_queries: int,
    num_dn_queries: int,
    device: torch.device,
) -> torch.Tensor | None:
    if num_dn_queries <= 0:
        return None

    total_queries = int(num_main_queries) + int(num_dn_queries)
    attn_mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)
    main_slice = slice(0, int(num_main_queries))
    dn_slice = slice(int(num_main_queries), total_queries)

    # Keep the matching queries isolated from denoising queries.
    attn_mask[main_slice, dn_slice] = True
    attn_mask[dn_slice, main_slice] = True
    return attn_mask


class model(_BaseTransformerModel):
    def __init__(self, flag: bool = False):
        super().__init__(flag=flag)
        dense_num_classes = max(int(system_configs.full.get('dense_num_classes', 1)), 1)
        branch_hidden_dim = int(system_configs.full.get('dense_branch_hidden_dim', 256))
        use_coords = bool(system_configs.full.get('dense_use_coords', False))
        self.dense_decoder_init_mode = str(
            system_configs.full.get('dense_decoder_init_mode', 'legacy_query_embed')
        ).lower()
        self.dense_relation_mode = str(system_configs.full.get('dense_relation_mode', 'none')).lower()
        self.dense_relation_layers = int(system_configs.full.get('dense_relation_layers', 0))
        self.dense_relation_heads = int(system_configs.full.get('dense_relation_heads', 4))
        self.dense_relation_ff_dim = int(system_configs.full.get('dense_relation_ff_dim', int(system_configs.attn_dim) * 2))
        self.dense_relation_dropout = float(system_configs.full.get('dense_relation_dropout', 0.1))
        self.dense_range_mode = str(system_configs.full.get('dense_range_mode', 'range')).lower()
        self.dense_visibility_dim = int(system_configs.full.get('dense_visibility_dim', 0))
        if self.dense_decoder_init_mode not in {'legacy_query_embed', 'learned_target_embed'}:
            raise ValueError(f"Unsupported dense_decoder_init_mode={self.dense_decoder_init_mode!r}")
        if self.dense_relation_mode not in {'none', 'self_attn'}:
            raise ValueError(f"Unsupported dense_relation_mode={self.dense_relation_mode!r}")
        if self.dense_range_mode not in {'range', 'visibility'}:
            raise ValueError(f"Unsupported dense_range_mode={self.dense_range_mode!r}")
        if self.dense_range_mode == 'visibility' and self.dense_visibility_dim <= 0:
            raise ValueError('dense_visibility_dim must be positive when dense_range_mode=visibility')
        self.mask_downscale = int(system_configs.full.get('condlstr_mask_downscale', 1))
        self.dn_lane_num_queries = int(system_configs.full.get('dn_lane_num_queries', 0))
        self.dn_lane_x_noise_scale = float(system_configs.full.get('dn_lane_x_noise_scale', 0.05))
        self.dn_lane_range_noise_scale = float(system_configs.full.get('dn_lane_range_noise_scale', 0.03))
        self.dn_lane_enabled = self.dn_lane_num_queries > 0
        self.parity_backbone_mode = str(system_configs.full.get('parity_backbone', 'lstr')).lower()
        self.parity_backbone = None
        if self.parity_backbone_mode == 'stdc_res34':
            self.parity_backbone = STDCResNet34Backbone(
                pretrained=bool(system_configs.full.get('parity_backbone_pretrained', False)),
                norm_layer=nn.BatchNorm2d,
            )
        elif self.parity_backbone_mode != 'lstr':
            raise ValueError(f"Unsupported parity_backbone={self.parity_backbone_mode!r}")

        if self.dense_decoder_init_mode == 'learned_target_embed':
            self.decoder_target_embed = nn.Embedding(int(system_configs.num_queries), int(system_configs.attn_dim))
        else:
            self.decoder_target_embed = None

        if self.dense_relation_mode == 'self_attn' and self.dense_relation_layers > 0:
            self.query_relation = QueryRelationBlock(
                hidden_dim=int(system_configs.attn_dim),
                num_heads=self.dense_relation_heads,
                num_layers=self.dense_relation_layers,
                ff_dim=self.dense_relation_ff_dim,
                dropout=self.dense_relation_dropout,
            )
        else:
            self.query_relation = None

        self.parity_head = CondLSTRParityHead(
            feature_channels=int(system_configs.attn_dim),
            query_channels=int(system_configs.attn_dim),
            num_classes=dense_num_classes,
            branch_hidden_dim=branch_hidden_dim,
            mask_out_channels=1,
            reg_out_channels=1,
            use_coords=use_coords,
            predict_ranges=self.dense_range_mode != 'visibility',
            visibility_dim=self.dense_visibility_dim if self.dense_range_mode == 'visibility' else 0,
        )
        if self.dn_lane_enabled:
            self.dn_query_encoder = nn.Sequential(
                nn.Linear(8, int(system_configs.attn_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(system_configs.attn_dim), int(system_configs.attn_dim)),
            )
            self.dn_query_type_embed = nn.Parameter(torch.zeros(int(system_configs.attn_dim)))
        else:
            self.dn_query_encoder = None
            self.dn_query_type_embed = None

        print(
            "[LSTR_CULANE_condlstr_parity_base] "
            f"backbone={self.parity_backbone_mode} "
            f"head -> CondLSTRParityHead(num_classes={dense_num_classes}, hidden={branch_hidden_dim}, "
            f"use_coords={use_coords}, decoder_init_mode={self.dense_decoder_init_mode}, "
            f"range_mode={self.dense_range_mode}, visibility_dim={self.dense_visibility_dim}, "
            f"relation_mode={self.dense_relation_mode}, relation_layers={self.dense_relation_layers})"
        )
        if self.dn_lane_enabled:
            print(
                "[LSTR_CULANE_condlstr_parity_base] "
                f"dn_lane enabled: num_queries={self.dn_lane_num_queries}, "
                f"x_noise={self.dn_lane_x_noise_scale}, range_noise={self.dn_lane_range_noise_scale}"
            )

    def _apply_query_relation(self, query_features: torch.Tensor) -> torch.Tensor:
        if self.query_relation is None:
            return query_features
        return self.query_relation(query_features)

    def _build_decoder_query_inputs(
        self,
        batch_size: int,
        learned_queries: torch.Tensor,
        dn_query_embed: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dense_decoder_init_mode == 'learned_target_embed':
            main_query_pos = self.decoder_target_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            main_decoder_tgt = main_query_pos
        else:
            main_query_pos = learned_queries
            main_decoder_tgt = learned_queries * 0.1

        if dn_query_embed is None:
            return main_query_pos, main_decoder_tgt

        query_pos = torch.cat((main_query_pos, dn_query_embed), dim=1)
        decoder_tgt = torch.cat((main_decoder_tgt, dn_query_embed), dim=1)
        return query_pos, decoder_tgt

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
        targets = kwargs.get('targets')
        output_size = (
            int(images.shape[-2] // max(self.mask_downscale, 1)),
            int(images.shape[-1] // max(self.mask_downscale, 1)),
        )

        p = self._extract_backbone_features(images)

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)
        learned_queries = self.query_embed.weight.unsqueeze(0).expand(images.size(0), -1, -1)
        query_embed, decoder_tgt = self._build_decoder_query_inputs(
            batch_size=images.size(0),
            learned_queries=learned_queries,
        )
        dn_meta = None
        dn_tgt_mask = None
        force_dn = bool(kwargs.get('force_dn', False))
        if (self.training or force_dn) and self.dn_lane_enabled and targets is not None:
            dense_targets = build_parity_targets_from_legacy_targets(
                targets=targets,
                target_size=output_size,
                device=images.device,
                line_width=float(system_configs.full.get('dense_line_width', 16.0)),
                min_valid_rows=int(system_configs.full.get('dense_min_valid_rows', 2)),
            )
            dn_queries, dn_targets, dn_valid_counts = build_dn_lane_queries(
                targets=dense_targets,
                num_dn_queries=self.dn_lane_num_queries,
                feature_width=output_size[1],
                x_noise_scale=self.dn_lane_x_noise_scale,
                range_noise_scale=self.dn_lane_range_noise_scale,
            )
            dn_query_embed = self.dn_query_encoder(dn_queries) + self.dn_query_type_embed.view(1, 1, -1)
            query_embed, decoder_tgt = self._build_decoder_query_inputs(
                batch_size=images.size(0),
                learned_queries=learned_queries,
                dn_query_embed=dn_query_embed,
            )
            dn_tgt_mask = _build_dn_decoder_attention_mask(
                num_main_queries=int(self.query_embed.weight.shape[0]),
                num_dn_queries=int(dn_queries.shape[1]),
                device=images.device,
            )
            dn_meta = {
                'num_main_queries': int(self.query_embed.weight.shape[0]),
                'num_dn_queries': int(dn_queries.shape[1]),
                'valid_counts': dn_valid_counts,
                'targets': dn_targets,
            }

        hs, memory, weights = self.transformer(
            self.input_proj(p),
            pmasks,
            query_embed,
            pos,
            tgt_mask=dn_tgt_mask,
            decoder_tgt=decoder_tgt,
            decoder_query_pos=query_embed,
        )

        query_features_per_layer = [self._apply_query_relation(layer_output) for layer_output in hs]
        feature_maps = [memory] * len(query_features_per_layer)
        per_layer_outputs = self.parity_head(feature_map=feature_maps, query_features=query_features_per_layer)

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
        if self.dense_range_mode == 'visibility':
            final_output['visibility_thresh'] = float(system_configs.full.get('condlstr_visibility_thresh', 0.5))
        if dn_meta is not None:
            final_output['dn_meta'] = dn_meta
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
        self.dn_lane_num_queries = int(system_configs.full.get('dn_lane_num_queries', 0))
        self.dn_lane_loss_weight = float(system_configs.full.get('dn_lane_loss_weight', 1.0))
        self.match_diag_enabled = bool(system_configs.full.get('match_diag_enabled', False))
        self.match_diag_interval = max(int(system_configs.full.get('match_diag_interval', 500)), 1)
        self.match_diag_path = os.path.join(self.debug_path, 'match_diag_train.jsonl')
        os.makedirs(self.debug_path, exist_ok=True)

        self.weight_dict = {
            'loss_object': float(system_configs.full.get('dense_loss_object_weight', 10.0)),
            'loss_class': float(system_configs.full.get('dense_loss_class_weight', 10.0)),
            'loss_loc': float(system_configs.full.get('dense_loss_loc_weight', 1.0)),
            'loss_reg': float(system_configs.full.get('dense_loss_reg_weight', 1.0)),
            'loss_range': float(system_configs.full.get('dense_loss_range_weight', 20.0)),
        }
        if self.dn_lane_num_queries > 0 and self.dn_lane_loss_weight > 0.0:
            self.weight_dict.update(
                {
                    'loss_dn_object': self.weight_dict['loss_object'] * self.dn_lane_loss_weight,
                    'loss_dn_class': self.weight_dict['loss_class'] * self.dn_lane_loss_weight,
                    'loss_dn_loc': self.weight_dict['loss_loc'] * self.dn_lane_loss_weight,
                    'loss_dn_reg': self.weight_dict['loss_reg'] * self.dn_lane_loss_weight,
                    'loss_dn_range': self.weight_dict['loss_range'] * self.dn_lane_loss_weight,
                }
            )
        base_weight_dict = dict(self.weight_dict)
        aux_weight_dict = {}
        for layer_index in range(max(int(system_configs.dec_layers) - 1, 0)):
            for name, value in base_weight_dict.items():
                if name.startswith('loss_dn_'):
                    continue
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

    def _append_match_diagnostics(self, iteration: int, diagnostics: List[Dict[str, object]]) -> None:
        if not diagnostics:
            return
        os.makedirs(os.path.dirname(self.match_diag_path), exist_ok=True)
        with open(self.match_diag_path, 'a', encoding='utf-8') as handle:
            for payload in diagnostics:
                record = {'iteration': int(iteration)}
                record.update(payload)
                handle.write(json.dumps(record, ensure_ascii=True) + '\n')

    def forward(self, iteration, save, viz_split, outputs, targets, **kwargs):
        del save, viz_split

        spatial_size = (int(outputs['pred_dense_mask'].shape[-2]), int(outputs['pred_dense_mask'].shape[-1]))
        dense_targets = build_parity_targets_from_legacy_targets(
            targets=targets,
            target_size=spatial_size,
            device=outputs['pred_object_logits'].device,
            line_width=self.line_width,
            min_valid_rows=self.min_valid_rows,
        )
        collect_diagnostics = self.match_diag_enabled and (int(iteration) % self.match_diag_interval == 0)

        loss_dict, _, diagnostics = self.criterion(
            outputs,
            dense_targets,
            image_keys=kwargs.get('image_keys'),
            collect_diagnostics=collect_diagnostics,
        )
        if diagnostics:
            self._append_match_diagnostics(int(iteration), diagnostics)
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
