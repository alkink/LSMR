"""
BÖLÜM 1 — Coord injection analizi (izole).

Amaç:
1) Decoder memory (M_prime) üzerine koordinat kanalı eklemek
2) Kanal aralığı ve monotonluk doğrulaması yapmak
3) Bu dönüşümün mevcut kanalları bozmadığını doğrulamak
"""

import argparse
from typing import Dict

import torch

from mask_migration.common import build_model_from_cfg, forward_to_decoder, load_system_config


def inject_coord_channels(memory: torch.Tensor, order: str = "yx") -> torch.Tensor:
    """
    Args:
        memory: (B, C, H, W)
        order : "yx" => [y_norm, x_norm], "xy" => [x_norm, y_norm]

    Returns:
        (B, C+2, H, W)
    """
    if memory.dim() != 4:
        raise ValueError(f"memory must be 4D (B,C,H,W), got {tuple(memory.shape)}")

    bsz, _c, h, w = memory.shape
    ys = torch.linspace(0.0, 1.0, steps=h, device=memory.device, dtype=memory.dtype)
    xs = torch.linspace(0.0, 1.0, steps=w, device=memory.device, dtype=memory.dtype)

    y_plane = ys.view(1, 1, h, 1).expand(bsz, 1, h, w)
    x_plane = xs.view(1, 1, 1, w).expand(bsz, 1, h, w)

    if order == "yx":
        coords = torch.cat([y_plane, x_plane], dim=1)
    elif order == "xy":
        coords = torch.cat([x_plane, y_plane], dim=1)
    else:
        raise ValueError(f"Unsupported coord order: {order}")

    return torch.cat([memory, coords], dim=1)


@torch.no_grad()
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

    hs, memory, _weights = forward_to_decoder(net, images, masks)
    m_coord = inject_coord_channels(memory, order="yx")

    y_plane = m_coord[:, -2, :, :]  # y_norm
    x_plane = m_coord[:, -1, :, :]  # x_norm

    checks: Dict[str, bool] = {}
    checks["shape_plus2"] = m_coord.shape[1] == memory.shape[1] + 2
    checks["orig_channels_preserved"] = float((m_coord[:, : memory.shape[1]] - memory).abs().max().item()) < 1e-7

    eps = 1e-6
    checks["coord_range_0_1"] = (
        float(y_plane.min().item()) >= -eps
        and float(y_plane.max().item()) <= 1.0 + eps
        and float(x_plane.min().item()) >= -eps
        and float(x_plane.max().item()) <= 1.0 + eps
    )

    x_diff = x_plane[:, :, 1:] - x_plane[:, :, :-1]
    y_diff = y_plane[:, 1:, :] - y_plane[:, :-1, :]
    checks["x_monotonic"] = float(x_diff.min().item()) >= -1e-7
    checks["y_monotonic"] = float(y_diff.min().item()) >= -1e-7

    checks["x_constant_across_rows"] = float((x_plane[:, 1:, :] - x_plane[:, :-1, :]).abs().max().item()) < 1e-7
    checks["y_constant_across_cols"] = float((y_plane[:, :, 1:] - y_plane[:, :, :-1]).abs().max().item()) < 1e-7
    checks["finite"] = bool(torch.isfinite(m_coord).all().item())

    print("=" * 90)
    print("BÖLÜM 1 — COORD INJECTION")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Input: {(in_h, in_w)}")
    print(f"hs[-1] shape: {tuple(hs[-1].shape)}")
    print(f"memory shape: {tuple(memory.shape)}")
    print(f"memory+coords shape: {tuple(m_coord.shape)}")
    print(
        f"coord mins/maxs: y=[{float(y_plane.min().item()):.4f}, {float(y_plane.max().item()):.4f}], "
        f"x=[{float(x_plane.min().item()):.4f}, {float(x_plane.max().item()):.4f}]"
    )

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<28}: {'✅' if v else '🚨'}")

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
    print(f"BÖLÜM 1 — Coord Injection: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

