from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _batch_norm_1x1_safe(x: torch.Tensor, bn_module: nn.BatchNorm2d) -> torch.Tensor:
    """Use running stats for degenerate 1x1 features without toggling module mode."""
    return F.batch_norm(
        x,
        bn_module.running_mean,
        bn_module.running_var,
        bn_module.weight,
        bn_module.bias,
        training=False,
        eps=bn_module.eps,
    )


def _build_resnet34(pretrained: bool, norm_layer: type[nn.Module]) -> nn.Module:
    if pretrained:
        try:
            weights = torchvision.models.ResNet34_Weights.DEFAULT
            return torchvision.models.resnet34(weights=weights, norm_layer=norm_layer)
        except AttributeError:
            return torchvision.models.resnet34(pretrained=True, norm_layer=norm_layer)
    try:
        return torchvision.models.resnet34(weights=None, norm_layer=norm_layer)
    except TypeError:
        return torchvision.models.resnet34(pretrained=False, norm_layer=norm_layer)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if x.dim() == 4 and x.size(0) == 1 and x.size(2) == 1 and x.size(3) == 1 and self.bn.training:
            x = _batch_norm_1x1_safe(x, self.bn)  # pragma: no cover - rare runtime path
        else:
            x = self.bn(x)
        return self.relu(x)


class AttentionRefinementModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = norm_layer(out_channels)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv_atten.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        if atten.dim() == 4 and atten.size(0) == 1 and atten.size(2) == 1 and atten.size(3) == 1 and self.bn_atten.training:
            atten = _batch_norm_1x1_safe(atten, self.bn_atten)  # pragma: no cover - rare runtime path
        else:
            atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        return feat * atten


class FeatureFusionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(in_channels, out_channels, ks=1, stride=1, padding=0, norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv1.weight, a=1)
        nn.init.kaiming_normal_(self.conv2.weight, a=1)

    def forward(self, spatial_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        feat = self.convblk(torch.cat([spatial_feat, context_feat], dim=1))
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.relu(self.conv1(atten))
        atten = self.sigmoid(self.conv2(atten))
        return feat + (feat * atten)


class ContextPathResNet34(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.backbone = _build_resnet34(pretrained=pretrained, norm_layer=norm_layer)
        del self.backbone.fc

        self.arm16 = AttentionRefinementModule(256, 128, norm_layer=norm_layer)
        self.arm32 = AttentionRefinementModule(512, 128, norm_layer=norm_layer)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        _feat4 = x
        x = self.backbone.layer2(x)
        feat8 = x
        x = self.backbone.layer3(x)
        feat16 = x
        x = self.backbone.layer4(x)
        feat32 = x

        h8, w8 = feat8.shape[-2:]
        h16, w16 = feat16.shape[-2:]
        h32, w32 = feat32.shape[-2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_up = self.conv_head32(F.interpolate(feat32_arm + avg_up, (h16, w16), mode="nearest"))

        feat16_arm = self.arm16(feat16)
        feat16_up = self.conv_head16(F.interpolate(feat16_arm + feat32_up, (h8, w8), mode="nearest"))
        return feat8, feat16_up


class STDCResNet34Backbone(nn.Module):
    """CondLSTR-style STDC/BiSeNet wrapper around a ResNet34 backbone.

    The output is a 256-channel fused feature map, matching the parity head input
    contract while leaving the LSTR transformer untouched.
    """

    def __init__(
        self,
        pretrained: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.context_path = ContextPathResNet34(pretrained=pretrained, norm_layer=norm_layer)
        self.fusion = FeatureFusionModule(128 + 128, 256, norm_layer=norm_layer)
        self.output = ConvBNReLU(256, 256, ks=3, stride=1, padding=1, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat8, context8 = self.context_path(x)
        fused = self.fusion(feat8, context8)
        return self.output(fused)
