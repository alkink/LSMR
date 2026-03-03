"""
BÖLÜM 2 — GT residual-reg üretimi.

Hedef:
  gt_reg[lane, row] = (gt_x_true - gt_x_coarse) / feat_w

Burada:
  - gt_x_coarse: gt heatmap weighted-argmax (sürekli)
  - gt_x_true  : tercihen gt_offset map'ten türetilen x (varsa)
"""

import argparse
from typing import Dict, Optional

import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt, lanes_to_mask_gt
from mask_migration.common import load_system_config


def compute_gt_reg(
    gt_heatmap: torch.Tensor,     # (L,H,W)
    gt_valid_mask: torch.Tensor,  # (L,H)
    feat_w: int,
    gt_offset: Optional[torch.Tensor] = None,  # (L,H,W), optional
) -> torch.Tensor:
    if gt_heatmap.dim() != 3:
        raise ValueError(f"gt_heatmap must be (L,H,W), got {tuple(gt_heatmap.shape)}")
    if gt_valid_mask.dim() != 2:
        raise ValueError(f"gt_valid_mask must be (L,H), got {tuple(gt_valid_mask.shape)}")

    lanes, h, w = gt_heatmap.shape
    if w != feat_w:
        raise ValueError(f"feat_w mismatch: heatmap W={w}, feat_w={feat_w}")

    cols = torch.arange(w, dtype=gt_heatmap.dtype, device=gt_heatmap.device)
    gt_x_coarse = (gt_heatmap * cols.view(1, 1, w)).sum(dim=-1)  # (L,H)

    if gt_offset is not None:
        if gt_offset.dim() != 3 or gt_offset.shape != gt_heatmap.shape:
            raise ValueError(f"gt_offset shape must be {tuple(gt_heatmap.shape)}, got {tuple(gt_offset.shape)}")
        # gt_offset[l,r,c] = gt_x_true - c -> c=0'dan gt_x_true alınabilir.
        gt_x_true = gt_offset[:, :, 0]
    else:
        gt_x_true = gt_x_coarse

    reg = (gt_x_true - gt_x_coarse) / float(max(feat_w, 1))
    reg = reg * gt_valid_mask.float()
    return reg


def _round_trip_error(
    gt_heatmap: torch.Tensor,
    gt_reg: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    feat_w: int,
    gt_offset: Optional[torch.Tensor] = None,
) -> float:
    cols = torch.arange(feat_w, dtype=gt_heatmap.dtype, device=gt_heatmap.device)
    x_coarse = (gt_heatmap * cols.view(1, 1, feat_w)).sum(dim=-1)
    x_recovered = x_coarse + gt_reg * feat_w

    if gt_offset is not None:
        x_true = gt_offset[:, :, 0]
    else:
        x_true = x_coarse

    valid = gt_valid_mask.bool()
    if not bool(valid.any().item()):
        return 0.0
    return float(torch.abs(x_recovered[valid] - x_true[valid]).mean().item())


def run_synthetic(feat_h: int = 10, feat_w: int = 26, num_lanes: int = 7) -> Dict[str, bool]:
    fake_lanes = [[(150, 250), (160, 200), (170, 150)], [(400, 250), (410, 200), (420, 150)]]
    gt = lanes_to_mask_gt(
        fake_lanes,
        img_h=295,
        img_w=820,
        feat_h=feat_h,
        feat_w=feat_w,
        num_lanes=num_lanes,
    )

    gt_reg = compute_gt_reg(
        gt_heatmap=gt["heatmap"],
        gt_valid_mask=gt["valid_mask"],
        feat_w=feat_w,
        gt_offset=gt["offset"],
    )
    err = _round_trip_error(
        gt_heatmap=gt["heatmap"],
        gt_reg=gt_reg,
        gt_valid_mask=gt["valid_mask"],
        feat_w=feat_w,
        gt_offset=gt["offset"],
    )

    valid_vals = gt_reg[gt["valid_mask"]]
    checks: Dict[str, bool] = {}
    checks["shape"] = tuple(gt_reg.shape) == (num_lanes, feat_h)
    checks["range"] = (not bool(gt["valid_mask"].any().item())) or (float(valid_vals.abs().max().item()) <= 1.0 + 1e-3)
    checks["invalid_zero"] = (not bool((~gt["valid_mask"]).any().item())) or (float(gt_reg[~gt["valid_mask"]].abs().max().item()) < 1e-6)
    checks["roundtrip"] = err < 0.5

    print("=" * 90)
    print("BÖLÜM 2.1 — GT REG SENTETİK TEST")
    print("=" * 90)
    print(f"gt_reg shape: {tuple(gt_reg.shape)}")
    if bool(gt["valid_mask"].any().item()):
        print(f"reg valid range: [{float(valid_vals.min().item()):.4f}, {float(valid_vals.max().item()):.4f}]")
    print(f"round-trip hata (feat px): {err:.6f}")

    for k, v in checks.items():
        print(f"  {k:<18}: {'✅' if v else '🚨'}")

    return checks


def run_real(cfg_name: str = "LSTR_CULANE_2k_mamba_mask_c", samples: int = 8) -> Dict[str, bool]:
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)

    feat_h = 10
    feat_w = 26
    num_lanes = int(system_configs.num_queries)

    success = 0
    fail = 0
    for i in range(min(samples, len(db.db_inds))):
        db_ind = int(db.db_inds[i])
        item = db.detections(db_ind)
        label_t = torch.from_numpy(item["label"]).float()

        gt_batch = labels_to_mask_batch_gt([label_t], feat_h=feat_h, feat_w=feat_w, num_lanes=num_lanes)
        hm = gt_batch["heatmap"][0]
        vm = gt_batch["valid_mask"][0]
        off = gt_batch["offset"][0]

        reg = compute_gt_reg(hm, vm, feat_w=feat_w, gt_offset=off)

        has_valid = bool(vm.any().item())
        in_range = (not has_valid) or (float(reg[vm].abs().max().item()) <= 1.0 + 1e-3)
        zero_inv = (not bool((~vm).any().item())) or (float(reg[~vm].abs().max().item()) < 1e-6)
        if has_valid and in_range and zero_inv:
            success += 1
            rng_txt = f"[{float(reg[vm].min().item()):.3f},{float(reg[vm].max().item()):.3f}]"
            print(f"  Sample {i}: ✅ valid_rows={int(vm.sum().item())} reg_range={rng_txt}")
        else:
            fail += 1
            print(f"  Sample {i}: 🚨 has_valid={has_valid} in_range={in_range} zero_inv={zero_inv}")

    checks: Dict[str, bool] = {}
    checks["all_samples_ok"] = fail == 0
    checks["at_least_one_success"] = success > 0

    print("\n" + "=" * 90)
    print("BÖLÜM 2.2 — GT REG GERÇEK CULANE TEST")
    print("=" * 90)
    print(f"Başarılı: {success}, Hatalı: {fail}")
    for k, v in checks.items():
        print(f"  {k:<20}: {'✅' if v else '🚨'}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask_c")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--feat-h", type=int, default=10)
    parser.add_argument("--feat-w", type=int, default=26)
    parser.add_argument("--num-lanes", type=int, default=7)
    args = parser.parse_args()

    c1 = run_synthetic(feat_h=args.feat_h, feat_w=args.feat_w, num_lanes=args.num_lanes)
    c2 = run_real(cfg_name=args.cfg, samples=args.samples)
    ok = all(c1.values()) and all(c2.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 2 — GT reg formatı: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

