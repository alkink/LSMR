"""
LSTR_CULANE_MAMBA_FPN — Mamba + FPN Entegrasyonu
==================================================

Değişiklikler:
1. SimpleFPNNeck eklendi (layer2, layer3, layer4 → P2, P3, P4)
2. input_proj (128→32) KALDIRıLDI
3. fpn_proj (256→32) eklendi - kalıcı
4. P2 (45×80) yüksek çözünürlük üretiliyor (future: dynamic kernel için)
5. P4 (12×20) Mamba encoder'a besleniyor

Amaç: FPN çok-seviyeli özellik piramidi + Mamba encoder = MambaLane
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .py_utils import kp, AELoss
from .py_utils.mamba_encoder import BidirectionalMambaEncoder
from config import system_configs


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = norm_layer(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1  = conv1x1(inplanes, width)
        self.bn1    = norm_layer(width)
        self.conv2  = conv3x3(width, width, stride, groups, dilation)
        self.bn2    = norm_layer(width)
        self.conv3  = conv1x1(width, planes * self.expansion)
        self.bn3    = norm_layer(planes * self.expansion)
        self.relu   = nn.ReLU(inplace=True)
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


class SimpleFPNNeck(nn.Module):
    """
    Lean Multi-scale Feature Pyramid Network for LSTR.
    
    Input: [layer2, layer3, layer4]
           (B, 32, 45, 80), (B, 64, 23, 40), (B, 128, 12, 20)
    
    Output: {
        'P2': (B, 64, 45, 80),   # High-res for dynamic kernel (future)
        'P3': (B, 64, 23, 40),
        'P4': (B, 64, 12, 20),   # For Mamba encoder
    }
    
    Design choices:
    - out_channels=64 (instead of 256) to match attn_dim=32 scale
    - 1x1 convs instead of 3x3 to minimize parameters
    - No fused output (redundant, use P4 directly)
    - Total params: ~30K (vs 1.8M with 256ch+3x3)
    """
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for idx in range(len(in_channels_list)):
            in_c = in_channels_list[idx]
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            # 1x1 conv instead of 3x3 for efficiency
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 1))
            
    def forward(self, inputs):
        # inputs: [layer2, layer3, layer4]
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        used_backbone_levels = len(laterals)
        
        # Top-down fusion (coarse to fine)
        outs = [laterals[-1]]
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]
            out_up = F.interpolate(outs[0], size=prev_shape, mode="nearest")
            outs.insert(0, laterals[i - 1] + out_up)
        
        # FPN conv per level
        fpn_outs = [
            self.fpn_convs[i](outs[i])
            for i in range(used_backbone_levels)
        ]
        
        return {
            'P2': fpn_outs[0],      # (B, 64, 45, 80)
            'P3': fpn_outs[1],      # (B, 64, 23, 40)
            'P4': fpn_outs[2],      # (B, 64, 12, 20)
        }


class model(kp):
    def __init__(self, flag=False):
        layers          = system_configs.res_layers
        res_dims        = system_configs.res_dims
        res_strides     = system_configs.res_strides
        attn_dim        = system_configs.attn_dim
        dim_feedforward = system_configs.dim_feedforward
        num_queries     = system_configs.num_queries
        drop_out        = system_configs.drop_out
        num_heads       = system_configs.num_heads
        enc_layers      = system_configs.enc_layers
        dec_layers      = system_configs.dec_layers
        lsp_dim         = system_configs.lsp_dim
        mlp_layers      = system_configs.mlp_layers
        lane_cls        = system_configs.lane_categories
        aux_loss        = system_configs.aux_loss
        pos_type        = system_configs.pos_type
        pre_norm        = system_configs.pre_norm
        return_intermediate = system_configs.return_intermediate

        if system_configs.block == 'BasicBlock':
            block = BasicBlock
        elif system_configs.block == 'BottleNeck':
            block = Bottleneck
        else:
            raise ValueError('invalid system_configs.block: {}'.format(system_configs.block))

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
            mlp_layers=mlp_layers
        )

        # ── FPN + Mamba encoder entegrasyonu ──────────────────────────────────
        # Lean FPN: 64 channels, 1x1 convs (~30K params)
        self.fpn = SimpleFPNNeck(
            in_channels_list=[32, 64, 128],  # layer2, layer3, layer4
            out_channels=64  # 64 (not 256) to keep params low
        )
        
        # FPN projection: 64 → attn_dim (32)
        self.fpn_proj = nn.Conv2d(64, attn_dim, 1)
        
        # Mamba encoder
        self.transformer.encoder = BidirectionalMambaEncoder()
        
        fpn_params = sum(p.numel() for p in self.fpn.parameters())
        print("[LSTR_CULANE_MAMBA_FPN] Entegrasyon tamamlandı:")
        print(f"  - FPN: 64ch, 1x1 convs → {fpn_params//1000}K params")
        print(f"  - P2(45×80), P3(23×40), P4(12×20)")
        print(f"  - fpn_proj: 64 → {attn_dim}")
        print(f"  - Mamba encoder: d_state=16, bidirectional")

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks  = xs[1]

        # Backbone
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p1 = self.layer1(p)
        p2 = self.layer2(p1)  # (B, 32, 45, 80)
        p3 = self.layer3(p2)  # (B, 64, 23, 40)
        p4 = self.layer4(p3)  # (B, 128, 12, 20)
        
        # FPN: Multi-scale features (64 channels)
        fpn_outs = self.fpn([p2, p3, p4])
        # fpn_outs = {'P2': (B,64,45,80), 'P3': (B,64,23,40), 'P4': (B,64,12,20)}
        
        # Use P4 for Mamba encoder (already fused via top-down)
        fpn_features = fpn_outs['P4']  # (B, 64, 12, 20)
        
        # Project to transformer dimension (64 → 32)
        transformer_input = self.fpn_proj(fpn_features)  # (B, 32, 12, 20)
        
        # Prepare masks and position encoding
        pmasks = F.interpolate(masks[:, 0, :, :][None], 
                               size=transformer_input.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(transformer_input, pmasks)
        
        # Mamba Encoder + Decoder
        hs, _, weights = self.transformer(
            transformer_input,
            pmasks,
            self.query_embed.weight,
            pos
        )
        
        # Prediction heads
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat(
            [output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]],
            dim=-1
        )
        
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
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
            dec_layers=system_configs.dec_layers
        )
        # Loss weight fix
        for k in list(self.criterion.weight_dict.keys()):
            if 'loss_curves' in k:
                self.criterion.weight_dict[k] = 2.5
        print(f"[LSTR_CULANE_MAMBA_FPN] weight_dict: {self.criterion.weight_dict}")
