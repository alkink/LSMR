"""
BÖLÜM 3 — Prior bias analizi (izole).

Amaç:
1) CondLSTR benzeri prior-bias başlatmasını skor başlığına uygulamak
2) Başlangıç fg olasılık dağılımını sayısallaştırmak
3) Bias öncesi/sonrası loss-gradient davranışını karşılaştırmak
"""

import argparse
import math
from typing import Dict

import numpy as np
import torch

from mask_migration.bolum1_gt_mask import lanes_to_mask_gt
from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.bolum3_loss import compute_mask_loss


def init_score_prior_bias(head: DynamicMaskHead, prior_prob: float = 0.01) -> None:
    if not (0.0 < prior_prob < 1.0):
        raise ValueError(f"prior_prob must be in (0,1), got {prior_prob}")
    bias_value = -math.log((1.0 - prior_prob) / prior_prob)
    with torch.no_grad():
        last = head.mlp_score[-1]
        if not isinstance(last, torch.nn.Linear) or last.out_features != 2:
            raise ValueError("mlp_score son katmanı 2 çıkışlı Linear olmalı")
        last.bias[0] = 0.0
        last.bias[1] = float(bias_value)


def _build_toy_gt(batch: int, lanes: int, h: int, w: int, device: torch.device):
    gt_list = []
    for b in range(batch):
        dx = 10 * b
        fake_lanes = [
            [(150 + dx, 300), (160 + dx, 250), (170 + dx, 200), (180 + dx, 150)],
            [(390 - dx, 300), (400 - dx, 250), (410 - dx, 200), (420 - dx, 150)],
        ]
        gt_list.append(lanes_to_mask_gt(fake_lanes, img_h=360, img_w=640, feat_h=h, feat_w=w, num_lanes=lanes))

    gt_hm = torch.stack([g["heatmap"] for g in gt_list]).to(device)
    gt_off = torch.stack([g["offset"] for g in gt_list]).to(device)
    gt_vr = torch.stack([g["v_range"] for g in gt_list]).to(device)
    gt_lbl = torch.stack([g["labels"] for g in gt_list]).to(device)
    gt_vm = torch.stack([g["valid_mask"] for g in gt_list]).to(device)
    return gt_hm, gt_off, gt_vr, gt_lbl, gt_vm


def _collect_stats(
    head: DynamicMaskHead,
    t: torch.Tensor,
    m: torch.Tensor,
    gt_hm: torch.Tensor,
    gt_off: torch.Tensor,
    gt_vr: torch.Tensor,
    gt_lbl: torch.Tensor,
    gt_vm: torch.Tensor,
) -> Dict[str, float]:
    out = head(t, m)
    probs = torch.softmax(out["scores"], dim=-1)[..., 1]

    losses = compute_mask_loss(
        out["heatmap"],
        out["offset"],
        out["v_range"],
        out["scores"],
        gt_hm,
        gt_off,
        gt_vr,
        gt_lbl,
        gt_vm,
    )

    return {
        "fg_prob_mean": float(probs.mean().item()),
        "fg_prob_std": float(probs.std().item()),
        "fg_prob_min": float(probs.min().item()),
        "fg_prob_max": float(probs.max().item()),
        "loss_total": float(losses["total"].item()),
        "loss_cls": float(losses["cls_loss"].item()),
        "loss_heat": float(losses["heat_loss"].item()),
        "loss_off": float(losses["offset_loss"].item()),
        "loss_vr": float(losses["vrange_loss"].item()),
    }


def run(seed: int = 0, prior_prob: float = 0.01) -> Dict[str, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, c, h, w = 2, 7, 32, 10, 26
    t = torch.randn(l, b, c, device=device)
    m = torch.randn(b, c, h, w, device=device)
    gt_hm, gt_off, gt_vr, gt_lbl, gt_vm = _build_toy_gt(b, l, h, w, device)

    # Aynı başlangıçtan iki head
    base = DynamicMaskHead(num_queries=l, feat_dim=c, feat_h=h, feat_w=w).to(device)
    with_prior = DynamicMaskHead(num_queries=l, feat_dim=c, feat_h=h, feat_w=w).to(device)
    with_prior.load_state_dict(base.state_dict(), strict=True)

    init_score_prior_bias(with_prior, prior_prob=prior_prob)

    stats_base = _collect_stats(base, t, m, gt_hm, gt_off, gt_vr, gt_lbl, gt_vm)
    stats_prior = _collect_stats(with_prior, t, m, gt_hm, gt_off, gt_vr, gt_lbl, gt_vm)

    # gradient kıyası (classification head odaklı)
    for mod in (base, with_prior):
        mod.zero_grad(set_to_none=True)

    out_b = base(t, m)
    loss_b = compute_mask_loss(out_b["heatmap"], out_b["offset"], out_b["v_range"], out_b["scores"], gt_hm, gt_off, gt_vr, gt_lbl, gt_vm)["total"]
    loss_b.backward()
    grad_b = float(base.mlp_score[-1].weight.grad.abs().mean().item())

    out_p = with_prior(t, m)
    loss_p = compute_mask_loss(out_p["heatmap"], out_p["offset"], out_p["v_range"], out_p["scores"], gt_hm, gt_off, gt_vr, gt_lbl, gt_vm)["total"]
    loss_p.backward()
    grad_p = float(with_prior.mlp_score[-1].weight.grad.abs().mean().item())

    checks: Dict[str, bool] = {}
    checks["prior_reduces_initial_fg_prob"] = stats_prior["fg_prob_mean"] < stats_base["fg_prob_mean"]
    checks["prior_fg_near_target_scale"] = abs(stats_prior["fg_prob_mean"] - prior_prob) < 0.05
    checks["loss_finite"] = np.isfinite(stats_base["loss_total"]) and np.isfinite(stats_prior["loss_total"])
    checks["gradient_alive"] = grad_b > 0.0 and grad_p > 0.0

    print("=" * 90)
    print("BÖLÜM 3 — PRIOR BIAS ANALİZİ")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Prior prob: {prior_prob}")

    print("\nBase stats:")
    for k, v in stats_base.items():
        print(f"  {k:<14}: {v:.6f}")

    print("\nPrior stats:")
    for k, v in stats_prior.items():
        print(f"  {k:<14}: {v:.6f}")

    print("\nGrad mean |mlp_score[-1].weight|:")
    print(f"  base : {grad_b:.6e}")
    print(f"  prior: {grad_p:.6e}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<30}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior", type=float, default=0.01)
    args = parser.parse_args()

    checks = run(seed=args.seed, prior_prob=args.prior)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 3 — Prior Bias: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

