"""
LSTR_CULANE_2k_mamba_mask_c
===========================

Option-2 production variant:
  - Coord injection: ON
  - Prior bias: ON (p=0.01)
  - Dense offset extra loss: OFF (same loss pipeline as current mask model)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import system_configs
from models.py_utils import kp
from models.py_utils.mamba_encoder import BidirectionalMambaEncoder
from models.py_utils.misc import reduce_dict

from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt
from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.bolum3_loss import compute_mask_loss
from mask_migration.bolum5_postprocess import mask_to_lane_coords


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class model(kp):
    def __init__(self, flag=False):
        layers = system_configs.res_layers
        res_dims = system_configs.res_dims
        res_strides = system_configs.res_strides
        attn_dim = system_configs.attn_dim
        dim_feedforward = system_configs.dim_feedforward
        num_queries = system_configs.num_queries
        drop_out = system_configs.drop_out
        num_heads = system_configs.num_heads
        enc_layers = system_configs.enc_layers
        dec_layers = system_configs.dec_layers
        lsp_dim = system_configs.lsp_dim
        mlp_layers = system_configs.mlp_layers
        lane_cls = system_configs.lane_categories
        aux_loss = system_configs.aux_loss
        pos_type = system_configs.pos_type
        pre_norm = system_configs.pre_norm
        return_intermediate = system_configs.return_intermediate

        if system_configs.block == "BasicBlock":
            block = BasicBlock
        elif system_configs.block == "BottleNeck":
            block = Bottleneck
        else:
            raise ValueError(f"invalid system_configs.block: {system_configs.block}")

        super().__init__(
            flag=flag,
            block=block,
            layers=layers,
            res_dims=res_dims,
            res_strides=res_strides,
            attn_dim=attn_dim,
            num_queries=num_queries,
            aux_loss=aux_loss,
            pos_type=pos_type,
            drop_out=drop_out,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate=return_intermediate,
            num_cls=lane_cls,
            lsp_dim=lsp_dim,
            mlp_layers=mlp_layers,
        )

        self.transformer.encoder = BidirectionalMambaEncoder()
        self.mask_head = DynamicMaskHead(
            num_queries=num_queries,
            feat_dim=attn_dim,
            feat_h=10,
            feat_w=26,
            use_coords=True,
            prior_prob=0.01,
        )

        print("[LSTR_CULANE_2k_mamba_mask_c] encoder -> BidirectionalMambaEncoder")
        print("[LSTR_CULANE_2k_mamba_mask_c] head -> DynamicMaskHead(use_coords=True, prior_prob=0.01)")

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)
        hs, memory, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)

        t_blc = hs[-1]  # (B, num_queries, C)
        mask_out = self.mask_head(t_blc, memory)

        out = {
            "pred_heatmap": mask_out["heatmap"],
            "pred_offset": mask_out["offset"],
            "pred_vrange": mask_out["v_range"],
            "pred_scores": mask_out["scores"],
            "aux_outputs": [],
        }
        return out, weights

    def _test(self, *xs, decode_lanes=False, score_thresh=0.5, **kwargs):
        out, weights = self._train(*xs, **kwargs)
        if not decode_lanes:
            return out, weights

        images = xs[0]
        img_h, img_w = int(images.shape[-2]), int(images.shape[-1])
        lanes_batch = []
        for b in range(out["pred_heatmap"].shape[0]):
            lanes = mask_to_lane_coords(
                out["pred_heatmap"][b],
                out["pred_offset"][b],
                out["pred_vrange"][b],
                out["pred_scores"][b],
                score_thresh=score_thresh,
                img_h=img_h,
                img_w=img_w,
            )
            lanes_batch.append(lanes)
        return lanes_batch, weights

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.debug_path = system_configs.result_dir

        self.weight_dict = {
            "loss_ce": 1.0,
            "loss_heatmap": 1.0,
            "loss_offset": 1.0,
            "loss_vrange": 1.0,
        }
        print(f"[LSTR_CULANE_2k_mamba_mask_c] weight_dict: {self.weight_dict}")

    def forward(self, iteration, save, viz_split, outputs, targets):
        del iteration, save, viz_split

        required = ["pred_heatmap", "pred_offset", "pred_vrange", "pred_scores"]
        missing = [k for k in required if k not in outputs]
        if missing:
            raise KeyError(f"Mask loss missing outputs: {missing}")

        pred_hm = outputs["pred_heatmap"]
        pred_off = outputs["pred_offset"]
        pred_vr = outputs["pred_vrange"]
        pred_sc = outputs["pred_scores"]

        bsz, lq, h, w = pred_hm.shape
        device = pred_hm.device

        gt_label_tensors = [tgt[0] for tgt in targets[1:]]
        if len(gt_label_tensors) < bsz:
            gt_label_tensors = gt_label_tensors + [gt_label_tensors[-1]] * (bsz - len(gt_label_tensors))
        elif len(gt_label_tensors) > bsz:
            gt_label_tensors = gt_label_tensors[:bsz]

        gt = labels_to_mask_batch_gt(gt_label_tensors, feat_h=h, feat_w=w, num_lanes=lq)
        gt_hm = gt["heatmap"].to(device)
        gt_off = gt["offset"].to(device)
        gt_vr = gt["v_range"].to(device)
        gt_lbl = gt["labels"].to(device)
        gt_vm = gt["valid_mask"].to(device)

        mask_losses = compute_mask_loss(
            pred_hm,
            pred_off,
            pred_vr,
            pred_sc,
            gt_hm,
            gt_off,
            gt_vr,
            gt_lbl,
            gt_vm,
        )

        total = (
            self.weight_dict["loss_ce"] * mask_losses["cls_loss"]
            + self.weight_dict["loss_heatmap"] * mask_losses["heat_loss"]
            + self.weight_dict["loss_offset"] * mask_losses["offset_loss"]
            + self.weight_dict["loss_vrange"] * mask_losses["vrange_loss"]
        )

        loss_dict = {
            "loss_ce": mask_losses["cls_loss"],
            "loss_heatmap": mask_losses["heat_loss"],
            "loss_offset": mask_losses["offset_loss"],
            "loss_vrange": mask_losses["vrange_loss"],
            "class_error": mask_losses["class_error"],
        }

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
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
            raise RuntimeError("Non-finite loss encountered")

        return (
            total,
            loss_dict_reduced,
            loss_dict_reduced_unscaled,
            loss_dict_reduced_scaled,
            loss_value,
        )

