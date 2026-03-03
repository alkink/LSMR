"""
BÖLÜM 5 — A/B/C/D ablation overfit testi (izole).

Amaç:
1) Her değişikliği ayrı ölçmek: A→B→C→D
   A: baseline head
   B: baseline + coord injection
   C: baseline + coord injection + prior bias
   D: baseline + coord injection + prior bias + dense offset
2) Hangi adımda overfit gücü düştüğünü netleştirmek
3) Entegrasyon öncesi riskli bileşeni izole etmek
"""

import argparse
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mask_migration.bolum1_gt_mask import lanes_to_mask_gt
from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.bolum3_loss import compute_mask_loss
from condlstr_analysis.bolum4_dense_offset import dense_offset_loss_strong


def inject_coord_channels(memory: torch.Tensor) -> torch.Tensor:
    b, _c, h, w = memory.shape
    ys = torch.linspace(0.0, 1.0, steps=h, device=memory.device, dtype=memory.dtype)
    xs = torch.linspace(0.0, 1.0, steps=w, device=memory.device, dtype=memory.dtype)
    y = ys.view(1, 1, h, 1).expand(b, 1, h, w)
    x = xs.view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([memory, y, x], dim=1)


class CoordHead(nn.Module):
    def __init__(self, num_queries: int = 7, feat_dim: int = 32):
        super().__init__()
        self.feat_dim = feat_dim
        kd = feat_dim + 2
        self.mlp_heat = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, kd))
        self.mlp_off = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, kd))
        self.mlp_vr = nn.Sequential(nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(inplace=True), nn.Linear(feat_dim // 2, 2))
        self.mlp_sc = nn.Sequential(nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(inplace=True), nn.Linear(feat_dim // 2, 2))

    def forward(self, t: torch.Tensor, m: torch.Tensor):
        # t: (L,B,C) or (B,L,C), m: (B,C,H,W)
        b, _c, h, w = m.shape
        if t.shape[1] == b:
            t = t.permute(1, 0, 2)
        f = inject_coord_channels(m)
        flat = f.flatten(2)
        kb = self.mlp_heat(t)
        kz = self.mlp_off(t)
        b_raw = torch.bmm(kb, flat).view(b, t.shape[1], h, w)
        z_raw = torch.bmm(kz, flat).view(b, t.shape[1], h, w)
        return {
            "heatmap": F.softmax(b_raw, dim=-1),
            "offset": z_raw,
            "v_range": self.mlp_vr(t),
            "scores": self.mlp_sc(t),
        }


def _t_to_blc(t: torch.Tensor, batch: int) -> torch.Tensor:
    if t.dim() != 3:
        raise ValueError(f"T must be 3D, got {tuple(t.shape)}")
    if t.shape[1] == batch:  # (L,B,C)
        return t.permute(1, 0, 2)
    if t.shape[0] == batch:  # (B,L,C)
        return t
    raise ValueError(f"Cannot infer batch dim from T={tuple(t.shape)} and batch={batch}")


@torch.no_grad()
def copy_baseline_to_coord(baseline: DynamicMaskHead, coord: CoordHead):
    fd = coord.feat_dim

    # heat
    coord.mlp_heat[0].weight.copy_(baseline.mlp_heatmap[0].weight)
    coord.mlp_heat[0].bias.copy_(baseline.mlp_heatmap[0].bias)
    coord.mlp_heat[2].weight.zero_()
    coord.mlp_heat[2].bias.zero_()
    coord.mlp_heat[2].weight[:fd].copy_(baseline.mlp_heatmap[2].weight)
    coord.mlp_heat[2].bias[:fd].copy_(baseline.mlp_heatmap[2].bias)

    # offset
    coord.mlp_off[0].weight.copy_(baseline.mlp_offset[0].weight)
    coord.mlp_off[0].bias.copy_(baseline.mlp_offset[0].bias)
    coord.mlp_off[2].weight.zero_()
    coord.mlp_off[2].bias.zero_()
    coord.mlp_off[2].weight[:fd].copy_(baseline.mlp_offset[2].weight)
    coord.mlp_off[2].bias[:fd].copy_(baseline.mlp_offset[2].bias)

    coord.mlp_vr.load_state_dict(baseline.mlp_vrange.state_dict())
    coord.mlp_sc.load_state_dict(baseline.mlp_score.state_dict())


class BaselineHeadAdapter(nn.Module):
    """[mask_migration.bolum2_mask_head.DynamicMaskHead](mask_migration/bolum2_mask_head.py:11) için ortak arayüz."""

    def __init__(self, head: DynamicMaskHead):
        super().__init__()
        self.head = head

    def forward(self, t: torch.Tensor, m: torch.Tensor):
        # [mask_migration.bolum2_mask_head.DynamicMaskHead.forward()](mask_migration/bolum2_mask_head.py:45)
        # hem (L,B,C) hem (B,L,C) kabul eder; burada (B,L,C) veriyoruz.
        # Adaptörde bilinçli olarak (L,B,C)'ye çevirip baseline yoluyla birebir çalıştırıyoruz.
        if t.dim() != 3:
            raise ValueError(f"t must be 3D, got {tuple(t.shape)}")
        if t.shape[0] == m.shape[0]:
            t = t.permute(1, 0, 2)
        out = self.head(t, m)
        return {
            "heatmap": out["heatmap"],
            "offset": out["offset"],
            "v_range": out["v_range"],
            "scores": out["scores"],
        }


def init_prior(head: CoordHead, prior_prob: float = 0.01):
    bias_value = -math.log((1.0 - prior_prob) / prior_prob)
    with torch.no_grad():
        head.mlp_sc[-1].bias[0] = 0.0
        head.mlp_sc[-1].bias[1] = float(bias_value)


def _train_stage(
    stage_name: str,
    model: nn.Module,
    t: torch.Tensor,
    m: torch.Tensor,
    gt_hm: torch.Tensor,
    gt_off: torch.Tensor,
    gt_vr: torch.Tensor,
    gt_lbl: torch.Tensor,
    gt_vm: torch.Tensor,
    epochs: int,
    lr: float,
    dense_weight: float = 0.0,
):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []

    for ep in range(epochs):
        optim.zero_grad()
        out = model(t, m)
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

        dense = out["offset"].new_zeros(())
        total = losses["total"]
        if dense_weight > 0.0:
            dense = dense_offset_loss_strong(
                pred_offset=out["offset"],
                gt_offset=gt_off,
                gt_valid_mask=gt_vm,
                pred_heatmap=out["heatmap"],
                gt_heatmap=gt_hm,
                weak_mix=0.25,
            )
            total = total + dense_weight * dense

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        hist.append(float(total.item()))
        if ep % 20 == 0 or ep == epochs - 1:
            print(
                f"[{stage_name}] Epoch {ep:3d}: total={float(total.item()):.4f} "
                f"base={float(losses['total'].item()):.4f} dense={float(dense.item()):.4f}"
            )

    init_loss = hist[0]
    final_loss = hist[-1]
    reduction = (init_loss - final_loss) / max(init_loss, 1e-6) * 100.0
    return {
        "stage": stage_name,
        "init": init_loss,
        "final": final_loss,
        "reduction": reduction,
    }


def run(epochs: int = 120, lr: float = 1e-3, seed: int = 0, dense_weight: float = 1.0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, l, c, h, w = 1, 7, 32, 10, 26

    t = torch.randn(l, b, c, device=device)
    m = torch.randn(b, c, h, w, device=device)
    t_blc = _t_to_blc(t, batch=b)

    gt = lanes_to_mask_gt(
        [
            [(150, 300), (160, 250), (170, 200), (180, 150)],
            [(410, 300), (400, 250), (390, 200), (380, 150)],
        ],
        img_h=360,
        img_w=640,
        feat_h=h,
        feat_w=w,
        num_lanes=l,
    )

    gt_hm = gt["heatmap"].unsqueeze(0).to(device)
    gt_off = gt["offset"].unsqueeze(0).to(device)
    gt_vr = gt["v_range"].unsqueeze(0).to(device)
    gt_lbl = gt["labels"].unsqueeze(0).to(device)
    gt_vm = gt["valid_mask"].unsqueeze(0).to(device)

    # ortak başlangıç ağırlığı
    baseline_template = DynamicMaskHead(num_queries=l, feat_dim=c, feat_h=h, feat_w=w).to(device)

    # A) baseline
    head_a = DynamicMaskHead(num_queries=l, feat_dim=c, feat_h=h, feat_w=w).to(device)
    head_a.load_state_dict(baseline_template.state_dict(), strict=True)
    stage_a = _train_stage(
        stage_name="A-baseline",
        model=BaselineHeadAdapter(head_a),
        t=t_blc,
        m=m,
        gt_hm=gt_hm,
        gt_off=gt_off,
        gt_vr=gt_vr,
        gt_lbl=gt_lbl,
        gt_vm=gt_vm,
        epochs=epochs,
        lr=lr,
        dense_weight=0.0,
    )

    # B) +coord
    head_b = CoordHead(num_queries=l, feat_dim=c).to(device)
    copy_baseline_to_coord(baseline_template, head_b)
    stage_b = _train_stage(
        stage_name="B+coord",
        model=head_b,
        t=t_blc,
        m=m,
        gt_hm=gt_hm,
        gt_off=gt_off,
        gt_vr=gt_vr,
        gt_lbl=gt_lbl,
        gt_vm=gt_vm,
        epochs=epochs,
        lr=lr,
        dense_weight=0.0,
    )

    # C) +coord +prior
    head_c = CoordHead(num_queries=l, feat_dim=c).to(device)
    copy_baseline_to_coord(baseline_template, head_c)
    init_prior(head_c, prior_prob=0.01)
    stage_c = _train_stage(
        stage_name="C+coord+prior",
        model=head_c,
        t=t_blc,
        m=m,
        gt_hm=gt_hm,
        gt_off=gt_off,
        gt_vr=gt_vr,
        gt_lbl=gt_lbl,
        gt_vm=gt_vm,
        epochs=epochs,
        lr=lr,
        dense_weight=0.0,
    )

    # D) +coord +prior +dense
    head_d = CoordHead(num_queries=l, feat_dim=c).to(device)
    copy_baseline_to_coord(baseline_template, head_d)
    init_prior(head_d, prior_prob=0.01)
    stage_d = _train_stage(
        stage_name="D+coord+prior+dense",
        model=head_d,
        t=t_blc,
        m=m,
        gt_hm=gt_hm,
        gt_off=gt_off,
        gt_vr=gt_vr,
        gt_lbl=gt_lbl,
        gt_vm=gt_vm,
        epochs=epochs,
        lr=lr,
        dense_weight=dense_weight,
    )

    stages = [stage_a, stage_b, stage_c, stage_d]

    checks: Dict[str, bool] = {}
    checks["all_loss_finite"] = all(np.isfinite(s["init"]) and np.isfinite(s["final"]) for s in stages)
    checks["all_loss_reduced"] = all(s["final"] < s["init"] for s in stages)
    checks["all_reduction_over_30pct"] = all(s["reduction"] > 30.0 for s in stages)

    print("=" * 90)
    print("BÖLÜM 5 — A/B/C/D ABLATION OVERFIT")
    print("=" * 90)
    print(f"Device: {device}")
    print("\nStage sonuçları:")
    for s in stages:
        print(
            f"  {s['stage']:<22} init={s['init']:.6f} "
            f"final={s['final']:.6f} reduction={s['reduction']:.2f}%"
        )

    print("\nAşama-delta (reduction farkları):")
    print(f"  B - A: {stage_b['reduction'] - stage_a['reduction']:+.2f} puan")
    print(f"  C - B: {stage_c['reduction'] - stage_b['reduction']:+.2f} puan")
    print(f"  D - C: {stage_d['reduction'] - stage_c['reduction']:+.2f} puan")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<22}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    args = parser.parse_args()

    checks = run(epochs=args.epochs, lr=args.lr, seed=args.seed, dense_weight=args.dense_weight)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 5 — A/B/C/D Overfit: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

