"""
Düzeltilmiş SimpleFPNNeck'in gradient akışını doğrular.
Beklenti: TÜM lateral_convs ve fpn_convs gradient almalı.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch: {torch.__version__}")

# ── ESKI FPN (bozuk) ───────────────────────────────────────────────────────────
class BrokenFPNNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        laterals = [lc(inputs[i]) for i, lc in enumerate(self.lateral_convs)]
        n = len(laterals)
        outs = [laterals[-1]]
        for i in range(n - 1, 0, -1):
            up = F.interpolate(outs[0], size=laterals[i-1].shape[-2:], mode="nearest")
            outs.insert(0, laterals[i-1] + up)
        fpn_outs = [self.fpn_convs[i](outs[i]) for i in range(n)]
        return fpn_outs[-1]   # ← HATA: sadece layer4 kolu


# ── YENİ FPN (düzeltilmiş) ────────────────────────────────────────────────────
class FixedFPNNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        laterals = [lc(inputs[i]) for i, lc in enumerate(self.lateral_convs)]
        n = len(laterals)

        # Top-down birleştirme
        outs = [laterals[-1]]
        for i in range(n - 1, 0, -1):
            up = F.interpolate(outs[0], size=laterals[i-1].shape[-2:], mode="nearest")
            outs.insert(0, laterals[i-1] + up)

        fpn_outs = [self.fpn_convs[i](outs[i]) for i in range(n)]

        # Tüm seviyeleri layer4 boyutuna indir ve topla  ← DÜZELTME
        target_size = inputs[-1].shape[-2:]
        fused = fpn_outs[-1]
        for i in range(n - 1):
            fused = fused + F.interpolate(fpn_outs[i], size=target_size, mode="nearest")
        return fused


# ── TEST YARDIMCISI ───────────────────────────────────────────────────────────
def test_gradients(model, label):
    inputs = [
        torch.randn(1, 32, 45, 80).cuda(),
        torch.randn(1, 64, 23, 40).cuda(),
        torch.randn(1, 128, 12, 20).cuda(),
    ]

    out = model(inputs)
    proj = nn.Conv2d(128, 32, 1).cuda()
    lin  = nn.Linear(32, 7).cuda()
    loss = lin(proj(out).flatten(2).permute(0,2,1)).sum()

    opt = torch.optim.SGD(
        list(model.parameters()) + list(proj.parameters()) + list(lin.parameters()),
        lr=1e-3
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    all_ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  {name:40s}: gradient YOK")
            all_ok = False
        elif param.grad.abs().mean() < 1e-9:
            print(f"  ⚠️  {name:40s}: dead gradient (mean={param.grad.abs().mean():.2e})")
            all_ok = False
        else:
            print(f"  ✅ {name:40s}: mean={param.grad.abs().mean():.6f}")
    print(f"\n  {'✅ TÜM GRADIENTLER AKIYOR' if all_ok else '🚨 BAZI GRADIENTLER YOK'}")
    return all_ok


broken = BrokenFPNNeck([32, 64, 128], 128).cuda()
fixed  = FixedFPNNeck([32, 64, 128], 128).cuda()

test_gradients(broken, "ESKİ FPN (bozuk) — fpn_outs[-1]")
test_gradients(fixed,  "YENİ FPN (düzeltilmiş) — fused tüm seviyeler")
