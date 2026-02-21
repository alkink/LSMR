"""
LSTR_CULANE_MAMBA — LSTR with BidirectionalMambaEncoder
========================================================

Drop-in replacement for LSTR_CULANE.py.
Değişiklikler:
  1. transformer.encoder → BidirectionalMambaEncoder (gerçek mamba-ssm kernel)
  2. loss_curves weight 5 → 2.5 (spike fix, analiz testi ile doğrulandı)

Pre-integration analiz sonuçları:
  - Mamba d_state=16, B=1: %85.7 drop  (vs Baseline %81.1)
  - Mamba d_state=16, B=16: %92.8 drop ✅
  - 4 testin tamamında spike=0
  - enc_attn_weights shape: dinamik (B * num_heads, HW, HW)
"""
import torch
import torch.nn as nn

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

        # ── Mamba encoder'ı yükle ──────────────────────────────────────────
        # Analiz bulguları: d_state=16 optimal, B'ye göre dinamik wts shape
        self.transformer.encoder = BidirectionalMambaEncoder()
        print("[LSTR_CULANE_MAMBA] transformer.encoder → BidirectionalMambaEncoder (d_state=16)")


class loss(AELoss):
    def __init__(self):
        super(loss, self).__init__(
            debug_path=system_configs.result_dir,
            aux_loss=system_configs.aux_loss,
            num_classes=system_configs.lane_categories,
            dec_layers=system_configs.dec_layers
        )
        # ── Spike fix: loss_curves 5 → 2.5 ───────────────────────────────
        # Analiz sonucu: loss_curves=2.5 ile 4 testin tamamında spike=0
        for k in list(self.criterion.weight_dict.keys()):
            if 'loss_curves' in k:
                self.criterion.weight_dict[k] = 2.5
        print(f"[LSTR_CULANE_MAMBA] weight_dict: {self.criterion.weight_dict}")
