"""
BÖLÜM 1b — Coord-aware head analizi (izole).

Amaç:
1) Coord injection destekli başlık (head) varyantını bağımsız test etmek
2) Sıfır coord-kernel ile baseline ile birebir uyum göstermek
3) Coord-kernel etkinleştirildiğinde çıktının değiştiğini doğrulamak
4) Gradient akışını kontrol etmek
"""

import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.common import build_model_from_cfg, forward_to_decoder, load_system_config


def inject_coord_channels(memory: torch.Tensor) -> torch.Tensor:
    if memory.dim() != 4:
        raise ValueError(f"memory must be 4D (B,C,H,W), got {tuple(memory.shape)}")
    bsz, _c, h, w = memory.shape
    ys = torch.linspace(0.0, 1.0, steps=h, device=memory.device, dtype=memory.dtype)
    xs = torch.linspace(0.0, 1.0, steps=w, device=memory.device, dtype=memory.dtype)
    y_plane = ys.view(1, 1, h, 1).expand(bsz, 1, h, w)
    x_plane = xs.view(1, 1, 1, w).expand(bsz, 1, h, w)
    return torch.cat([memory, y_plane, x_plane], dim=1)


class DynamicMaskHeadWithCoords(nn.Module):
    """
    T:       (L, B, C) or (B, L, C)
    M_prime: (B, C, H, W)
    """

    def __init__(self, num_queries: int = 7, feat_dim: int = 32, use_coords: bool = True):
        super().__init__()
        self.num_queries = num_queries
        self.feat_dim = feat_dim
        self.use_coords = use_coords
        self.kernel_dim = feat_dim + (2 if use_coords else 0)

        self.mlp_heatmap = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.kernel_dim),
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.kernel_dim),
        )
        self.mlp_vrange = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )
        self.mlp_score = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )

    @staticmethod
    def _to_blc(t: torch.Tensor, batch: int) -> torch.Tensor:
        if t.dim() != 3:
            raise ValueError(f"T must be 3D, got {tuple(t.shape)}")
        if t.shape[1] == batch:  # (L,B,C)
            return t.permute(1, 0, 2)
        if t.shape[0] == batch:  # (B,L,C)
            return t
        raise ValueError(f"Cannot infer batch dim from T={tuple(t.shape)} and batch={batch}")

    def forward(self, t: torch.Tensor, m_prime: torch.Tensor) -> Dict[str, torch.Tensor]:
        if m_prime.dim() != 4:
            raise ValueError(f"M_prime must be 4D, got {tuple(m_prime.shape)}")

        bsz, channels, h, w = m_prime.shape
        if channels != self.feat_dim:
            raise ValueError(f"M_prime channels={channels}, expected feat_dim={self.feat_dim}")

        t_blc = self._to_blc(t, bsz)
        feats = inject_coord_channels(m_prime) if self.use_coords else m_prime

        m_flat = feats.flatten(2)  # (B, C(+2), HW)
        kb = self.mlp_heatmap(t_blc)  # (B, L, C(+2))
        kz = self.mlp_offset(t_blc)   # (B, L, C(+2))

        b_raw = torch.bmm(kb, m_flat)
        z_raw = torch.bmm(kz, m_flat)

        b_spatial = b_raw.view(bsz, t_blc.shape[1], h, w)
        z_spatial = z_raw.view(bsz, t_blc.shape[1], h, w)

        heatmap = F.softmax(b_spatial, dim=-1)
        v_range = self.mlp_vrange(t_blc)
        scores = self.mlp_score(t_blc)

        return {
            "heatmap": heatmap,
            "offset": z_spatial,
            "v_range": v_range,
            "scores": scores,
        }


@torch.no_grad()
def copy_from_baseline(baseline: DynamicMaskHead, coord_head: DynamicMaskHeadWithCoords) -> None:
    """Baseline ağırlıklarını coord-head'e taşır, extra coord kernel satırlarını 0 başlatır."""

    def _copy_kernel_mlp(src: nn.Sequential, dst: nn.Sequential, feat_dim: int) -> None:
        dst[0].weight.copy_(src[0].weight)
        dst[0].bias.copy_(src[0].bias)
        dst[2].weight.zero_()
        dst[2].bias.zero_()
        dst[2].weight[:feat_dim].copy_(src[2].weight)
        dst[2].bias[:feat_dim].copy_(src[2].bias)

    _copy_kernel_mlp(baseline.mlp_heatmap, coord_head.mlp_heatmap, coord_head.feat_dim)
    _copy_kernel_mlp(baseline.mlp_offset, coord_head.mlp_offset, coord_head.feat_dim)

    coord_head.mlp_vrange.load_state_dict(baseline.mlp_vrange.state_dict())
    coord_head.mlp_score.load_state_dict(baseline.mlp_score.state_dict())


@torch.no_grad()
def enable_coord_effect(coord_head: DynamicMaskHeadWithCoords, scale: float = 5e-2) -> None:
    """Extra coord kernel satırlarını sıfırdan çıkararak coord etkisini görünür yapar."""
    fd = coord_head.feat_dim

    # heatmap kernel (y ve x coord satırları)
    coord_head.mlp_heatmap[2].weight[fd + 0].fill_(+scale)
    coord_head.mlp_heatmap[2].weight[fd + 1].fill_(-scale)

    # offset kernel
    coord_head.mlp_offset[2].weight[fd + 0].fill_(-scale)
    coord_head.mlp_offset[2].weight[fd + 1].fill_(+scale)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _mean_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def run(cfg_name: str = "LSTR_CULANE_2k_mamba_mask", batch: int = 2, seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_system_config(cfg_name)
    in_h, in_w = cfg["db"]["input_size"]

    net = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)

    images = torch.randn(batch, 3, in_h, in_w, device=device)
    masks = torch.zeros(batch, 1, in_h, in_w, device=device)

    with torch.no_grad():
        hs, memory, _ = forward_to_decoder(net, images, masks)
        t_blc = hs[-1]  # (B, L, C)

    bsz, c, h, w = memory.shape
    lq = t_blc.shape[1]

    baseline_head = DynamicMaskHead(num_queries=lq, feat_dim=c, feat_h=h, feat_w=w).to(device)
    baseline_head.load_state_dict(net.mask_head.state_dict(), strict=True)
    baseline_head.eval()

    coord_head = DynamicMaskHeadWithCoords(num_queries=lq, feat_dim=c, use_coords=True).to(device)
    coord_head.eval()
    copy_from_baseline(baseline_head, coord_head)

    with torch.no_grad():
        out_base = baseline_head(t_blc, memory)
        out_coord_zero = coord_head(t_blc, memory)

    zero_diffs = {
        "heatmap": _max_diff(out_base["heatmap"], out_coord_zero["heatmap"]),
        "offset": _max_diff(out_base["offset"], out_coord_zero["offset"]),
        "v_range": _max_diff(out_base["v_range"], out_coord_zero["v_range"]),
        "scores": _max_diff(out_base["scores"], out_coord_zero["scores"]),
    }

    enable_coord_effect(coord_head, scale=5e-2)
    with torch.no_grad():
        out_coord_on = coord_head(t_blc, memory)

    effect_diffs = {
        "heatmap": _mean_diff(out_coord_zero["heatmap"], out_coord_on["heatmap"]),
        "offset": _mean_diff(out_coord_zero["offset"], out_coord_on["offset"]),
    }

    # Gradient testi
    coord_head.train()
    t2 = t_blc.detach().clone().requires_grad_(True)
    m2 = memory.detach().clone().requires_grad_(True)
    out_grad = coord_head(t2, m2)
    grad_loss = (
        out_grad["heatmap"].sum()
        + out_grad["offset"].sum()
        + out_grad["v_range"].sum()
        + out_grad["scores"].sum()
    )
    grad_loss.backward()

    grad_ok = True
    for p in coord_head.parameters():
        if p.grad is None or float(p.grad.abs().mean().item()) < 1e-12:
            grad_ok = False
            break

    checks: Dict[str, bool] = {}
    checks["shape_ok"] = tuple(out_coord_on["heatmap"].shape) == (bsz, lq, h, w)
    checks["finite_outputs"] = all(
        bool(torch.isfinite(v).all().item())
        for v in [
            out_coord_on["heatmap"],
            out_coord_on["offset"],
            out_coord_on["v_range"],
            out_coord_on["scores"],
        ]
    )
    checks["zero_coord_matches_baseline"] = max(zero_diffs.values()) < 1e-6
    checks["coords_change_output"] = (effect_diffs["heatmap"] > 1e-8) or (effect_diffs["offset"] > 1e-8)
    checks["gradient_flow"] = grad_ok and (t2.grad is not None) and (m2.grad is not None)

    print("=" * 90)
    print("BÖLÜM 1b — HEAD WITH COORDS")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"T shape: {tuple(t_blc.shape)}")
    print(f"M_prime shape: {tuple(memory.shape)}")
    print(f"Output shape: {tuple(out_coord_on['heatmap'].shape)}")

    print("\nZero-coord compatibility max |diff|:")
    for k, v in zero_diffs.items():
        print(f"  {k:<8}: {v:.3e}")

    print("\nCoord effect mean |diff| (after enabling coord kernels):")
    for k, v in effect_diffs.items():
        print(f"  {k:<8}: {v:.3e}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<30}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, batch=args.batch, seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 1b — Head+Coords: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

