"""
LSTR_CULANE_2k_mamba_dynamic
============================

Contract-compatible dynamic enhancement for LSTR+Mamba.

Design goals:
1) Keep full training/eval compatibility with existing pipeline:
   - Output keys remain: pred_logits, pred_curves, aux_outputs
   - Loss remains AELoss / Hungarian matcher / SetCriterion path
2) Improve representational power with dynamic, query-conditioned feature pooling
   WITHOUT breaking the polynomial lane parameterization used by CULANE eval.

Compared to the previous failed draft:
- No API break in loss signature
- No key mismatch (pred_xcoords removed)
- No channel mismatch in head
- No premature global collapse before lane-parameter prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .py_utils import kp, AELoss
from .py_utils.mamba_encoder import BidirectionalMambaEncoder
from config import system_configs


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
        super(BasicBlock, self).__init__()
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
        self.stride = stride

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
        super(Bottleneck, self).__init__()
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
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class DynamicLaneRefinementHead(nn.Module):
    """
    Query-conditioned dynamic refinement over high-resolution features.

    Input:
      - queries: (B, N, D) decoder embeddings
      - feat_map: (B, C, H, W) early spatial features (layer2)
      - base_curves: (B, N, K) baseline polynomial lane params from LSTR head

    Output:
      - refined_curves: (B, N, K) as residual refinement of base_curves

    Key idea:
      1) Generate lane-specific dynamic kernels from queries.
      2) Produce lane activation map over spatial tokens (no early spatial collapse).
      3) Softmax pool projected feature tokens into per-lane descriptor.
      4) Predict residual delta for polynomial parameters.
    """

    def __init__(self, query_dim=32, feat_channels=32, hidden_dim=64, curve_dim=8):
        super().__init__()
        self.query_dim = query_dim
        self.feat_channels = feat_channels
        self.hidden_dim = hidden_dim
        self.curve_dim = curve_dim

        # Query -> dynamic kernel (same channel dimensionality as feat map)
        self.kernel_proj = nn.Linear(query_dim, feat_channels)

        # Feature projection before weighted pooling
        self.feat_proj = nn.Conv2d(feat_channels, hidden_dim, kernel_size=1)

        # Query context branch
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Residual predictor for curve params
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, curve_dim),
        )

        # Keep refinement conservative initially
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, queries, feat_map, base_curves):
        # queries: (B, N, D), feat_map: (B, C, H, W), base_curves: (B, N, K)
        B, N, D = queries.shape
        Bf, C, H, W = feat_map.shape
        if B != Bf:
            raise ValueError(f"Batch mismatch in dynamic head: queries B={B}, feat B={Bf}")

        # (B, C, HW)
        feat_flat = feat_map.view(B, C, H * W)

        # (B, N, C)
        kernels = self.kernel_proj(queries)

        # Lane activation logits over spatial tokens: (B, N, HW)
        # bmm: (B,N,C) @ (B,C,HW)
        attn_logits = torch.bmm(kernels, feat_flat)
        attn = F.softmax(attn_logits, dim=-1)

        # Project features: (B, hidden, H, W) -> (B, HW, hidden)
        feat_embed = self.feat_proj(feat_map).flatten(2).transpose(1, 2)

        # Weighted lane descriptor from full spatial support: (B, N, hidden)
        lane_desc = torch.bmm(attn, feat_embed)

        # Query branch: (B, N, hidden)
        q_desc = self.query_proj(queries)

        # Fuse and predict residual delta: (B, N, K)
        fused = torch.cat([lane_desc, q_desc], dim=-1)
        delta = self.delta_head(fused)

        # Residual refinement
        refined = base_curves + delta
        return refined


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

        super(model, self).__init__(
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

        # Swap encoder to Mamba as in mamba branch
        self.transformer.encoder = BidirectionalMambaEncoder()

        # Dynamic refinement head over high-res layer2 features
        # curve_dim = lsp_dim (e.g., 8 = [lower, upper, 6 poly params])
        self.dynamic_refine = DynamicLaneRefinementHead(
            query_dim=attn_dim,
            feat_channels=res_dims[1],  # layer2 channels
            hidden_dim=max(attn_dim * 2, 64),
            curve_dim=lsp_dim,
        )

        print("[LSTR_CULANE_2k_mamba_dynamic] encoder -> BidirectionalMambaEncoder")
        print("[LSTR_CULANE_2k_mamba_dynamic] DynamicLaneRefinementHead enabled")

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks = xs[1]

        # Backbone with layer2 tap for dynamic refinement
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p2 = self.layer2(p)   # high-res map for dynamic head
        p = self.layer3(p2)
        p = self.layer4(p)

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)

        hs, _, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        # hs: (L, B, N, D)

        # Base LSTR heads
        output_class = self.class_embed(hs)          # (L, B, N, C+1)
        output_specific = self.specific_embed(hs)    # (L, B, N, lsp_dim-4)
        output_shared = self.shared_embed(hs)        # (L, B, N, 4)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)  # (L,B,1,4)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat(
            [output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]],
            dim=-1,
        )

        # Dynamic residual refinement (preserve output contract: pred_curves)
        refined_specific = []
        for lid in range(output_specific.shape[0]):
            q = hs[lid]  # (B, N, D)
            base = output_specific[lid]  # (B, N, lsp_dim)
            refined = self.dynamic_refine(q, p2, base)
            refined_specific.append(refined)
        output_specific = torch.stack(refined_specific, dim=0)

        out = {
            "pred_logits": output_class[-1],
            "pred_curves": output_specific[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(output_class, output_specific)
        return out, weights

    def _test(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class loss(AELoss):
    def __init__(self):
        super(loss, self).__init__(
            debug_path=system_configs.result_dir,
            aux_loss=system_configs.aux_loss,
            num_classes=system_configs.lane_categories,
            dec_layers=system_configs.dec_layers,
        )

        # Keep proven spike fix from mamba branch
        for k in list(self.criterion.weight_dict.keys()):
            if "loss_curves" in k:
                self.criterion.weight_dict[k] = 2.5
        print(f"[LSTR_CULANE_2k_mamba_dynamic] weight_dict: {self.criterion.weight_dict}")

