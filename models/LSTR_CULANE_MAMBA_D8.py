"""
LSTR_CULANE_MAMBA_D8 — Mamba d_state=8 ile parametre eşitliği ablation

Ablation amacı: Transformer encoder ile parametre sayısını eşitlemek.
d_state=8 → Mamba encoder ~25,400 param ≈ Transformer encoder ~25,400 param

Karşılaştırma:
  Transformer encoder: ~25,400 (d_model=32, 2-layer self_attn+FFN)
  Mamba d_state=16   : ~39,800 (+57% encoder, +1.9% total) — ana deney
  Mamba d_state=8    : ~25,400 (eşit encoder param)        — bu deney
"""
import torch
import torch.nn as nn

from .py_utils import kp, AELoss
from config import system_configs

try:
    from mamba_ssm import Mamba as MambaSSM
except ImportError:
    raise ImportError("pip install mamba-ssm causal-conv1d --no-build-isolation")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
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
        return self.relu(out + identity)


class BidirMambaD8(nn.Module):
    """d_state=8 — Transformer encoder ile eşit parametre sayısı."""
    def __init__(self):
        super().__init__()
        d_model    = system_configs.attn_dim   # 32
        d_state    = 8                          # ← KEY: 8 instead of 16
        enc_layers = system_configs.enc_layers  # 2
        self.num_heads = system_configs.num_heads  # 2

        self.fwd = nn.ModuleList([
            MambaSSM(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(enc_layers)])
        self.bwd = nn.ModuleList([
            MambaSSM(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(enc_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(enc_layers)])

        enc_params = sum(p.numel() for p in self.parameters())
        print(f"[MAMBA_D8] encoder params: {enc_params:,} (d_state=8)")

    def forward(self, src, src_key_padding_mask=None, pos=None):
        if pos is not None:
            src = src + pos
        x = src.permute(1, 0, 2)  # (HW,B,D) → (B,HW,D)
        for f, b, n in zip(self.fwd, self.bwd, self.norms):
            x = n(x + f(x) + torch.flip(b(torch.flip(x, [1])), [1]))
        out = x.permute(1, 0, 2)
        B = out.shape[1]
        wts = torch.zeros(B * self.num_heads, out.shape[0], out.shape[0], device=out.device)
        return out, wts


class model(kp):
    def __init__(self, flag=False):
        layers  = system_configs.res_layers
        super(model, self).__init__(
            flag=flag, block=BasicBlock, layers=layers,
            res_dims=system_configs.res_dims,
            res_strides=system_configs.res_strides,
            attn_dim=system_configs.attn_dim,
            num_queries=system_configs.num_queries,
            aux_loss=system_configs.aux_loss,
            pos_type=system_configs.pos_type,
            drop_out=system_configs.drop_out,
            num_heads=system_configs.num_heads,
            dim_feedforward=system_configs.dim_feedforward,
            enc_layers=system_configs.enc_layers,
            dec_layers=system_configs.dec_layers,
            pre_norm=system_configs.pre_norm,
            return_intermediate=system_configs.return_intermediate,
            num_cls=system_configs.lane_categories,
            lsp_dim=system_configs.lsp_dim,
            mlp_layers=system_configs.mlp_layers
        )
        self.transformer.encoder = BidirMambaD8()
        print("[MAMBA_D8] model hazır — param-matched ablation")


class loss(AELoss):
    def __init__(self):
        super(loss, self).__init__(
            debug_path=system_configs.result_dir,
            aux_loss=system_configs.aux_loss,
            num_classes=system_configs.lane_categories,
            dec_layers=system_configs.dec_layers
        )
        for k in list(self.criterion.weight_dict.keys()):
            if 'loss_curves' in k:
                self.criterion.weight_dict[k] = 2.5
        print(f"[MAMBA_D8] weight_dict: {self.criterion.weight_dict}")
