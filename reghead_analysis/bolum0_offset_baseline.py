"""
BÖLÜM 0 — Offset baseline referans ölçümü.

Amaç:
1) Mevcut offset-temelli (B,L,H,W) temsilin decode hatasını ölçmek
2) Eski formül için referans metrik üretmek (reg geçişinden önce)
3) Model output kontratını hızlıca doğrulamak
"""

import argparse
from typing import Dict, Tuple

import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt, lanes_to_mask_gt
from mask_migration.common import build_model_from_cfg, load_system_config


def decode_old_error(
    heatmap: torch.Tensor,     # (L,H,W)
    offset: torch.Tensor,      # (L,H,W)
    valid_mask: torch.Tensor,  # (L,H)
) -> Tuple[float, float]:
    """
    Returns:
      mae_coarse, mae_old  (feature-pixel space)
    """
    if heatmap.dim() != 3 or offset.dim() != 3:
        raise ValueError(f"heatmap/offset must be (L,H,W), got {tuple(heatmap.shape)} / {tuple(offset.shape)}")
    if valid_mask.dim() != 2:
        raise ValueError(f"valid_mask must be (L,H), got {tuple(valid_mask.shape)}")

    lanes, h, w = heatmap.shape
    if offset.shape != heatmap.shape:
        raise ValueError(f"offset shape {tuple(offset.shape)} != heatmap shape {tuple(heatmap.shape)}")
    if valid_mask.shape != (lanes, h):
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} != ({lanes}, {h})")

    cols = torch.arange(w, dtype=heatmap.dtype, device=heatmap.device)
    x_coarse = (heatmap * cols.view(1, 1, w)).sum(dim=-1)  # (L,H)

    n_col = torch.round(x_coarse).long().clamp(0, w - 1)
    off_at_n = torch.gather(offset, dim=-1, index=n_col.unsqueeze(-1)).squeeze(-1)
    x_old = x_coarse + off_at_n

    # GT offset tanımı: offset[..., c] = x_true - c -> c=0 => x_true
    x_true = offset[:, :, 0]

    valid = valid_mask.bool()
    if not bool(valid.any().item()):
        return 0.0, 0.0

    mae_coarse = float(torch.abs(x_coarse[valid] - x_true[valid]).mean().item())
    mae_old = float(torch.abs(x_old[valid] - x_true[valid]).mean().item())
    return mae_coarse, mae_old


@torch.no_grad()
def run_model_contract(cfg_name: str = "LSTR_CULANE_2k_mamba_mask_c", batch: int = 2, seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_system_config(cfg_name)
    in_h, in_w = cfg["db"]["input_size"]

    net = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)
    images = torch.randn(batch, 3, in_h, in_w, device=device)
    masks = torch.zeros(batch, 1, in_h, in_w, device=device)

    out, _ = net._train(images, masks)

    checks: Dict[str, bool] = {}
    required = ["pred_heatmap", "pred_offset", "pred_vrange", "pred_scores"]
    checks["required_keys"] = all(k in out for k in required)

    if checks["required_keys"]:
        phm = out["pred_heatmap"]
        pof = out["pred_offset"]
        pvr = out["pred_vrange"]
        psc = out["pred_scores"]

        checks["heat_offset_shape_match"] = tuple(phm.shape) == tuple(pof.shape)
        checks["vrange_score_lastdim2"] = pvr.shape[-1] == 2 and psc.shape[-1] == 2
        checks["finite_outputs"] = bool(
            torch.isfinite(phm).all().item()
            and torch.isfinite(pof).all().item()
            and torch.isfinite(pvr).all().item()
            and torch.isfinite(psc).all().item()
        )
    else:
        checks["heat_offset_shape_match"] = False
        checks["vrange_score_lastdim2"] = False
        checks["finite_outputs"] = False

    print("=" * 90)
    print("BÖLÜM 0.0 — MODEL KONTRATI (OFFSET MODE)")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    if checks["required_keys"]:
        print(f"pred_heatmap: {tuple(out['pred_heatmap'].shape)}")
        print(f"pred_offset : {tuple(out['pred_offset'].shape)}")
        print(f"pred_vrange : {tuple(out['pred_vrange'].shape)}")
        print(f"pred_scores : {tuple(out['pred_scores'].shape)}")
    for k, v in checks.items():
        print(f"  {k:<24}: {'✅' if v else '🚨'}")

    return checks


def run_synthetic(feat_h: int = 10, feat_w: int = 26, num_lanes: int = 7) -> Dict[str, bool]:
    fake_lanes = [
        [(150, 300), (160, 250), (170, 200), (180, 150)],
        [(410, 300), (400, 250), (390, 200), (380, 150)],
    ]
    gt = lanes_to_mask_gt(
        fake_lanes,
        img_h=360,
        img_w=640,
        feat_h=feat_h,
        feat_w=feat_w,
        num_lanes=num_lanes,
    )

    mae_coarse, mae_old = decode_old_error(
        heatmap=gt["heatmap"],
        offset=gt["offset"],
        valid_mask=gt["valid_mask"],
    )

    checks: Dict[str, bool] = {}
    checks["old_not_worse_than_coarse"] = mae_old <= mae_coarse + 0.25
    checks["old_mae_lt_1px_feat"] = mae_old < 1.0

    print("\n" + "=" * 90)
    print("BÖLÜM 0.1 — SENTETİK OFFSET REFERANS")
    print("=" * 90)
    print(f"MAE coarse: {mae_coarse:.4f} feat-px")
    print(f"MAE old   : {mae_old:.4f} feat-px")
    for k, v in checks.items():
        print(f"  {k:<24}: {'✅' if v else '🚨'}")

    return checks


def run_real(cfg_name: str = "LSTR_CULANE_2k_mamba_mask_c", samples: int = 8) -> Dict[str, bool]:
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)

    feat_h = 10
    feat_w = 26
    num_lanes = int(system_configs.num_queries)

    mae_old_all = []
    mae_coarse_all = []

    print("\n" + "=" * 90)
    print("BÖLÜM 0.2 — GERÇEK CULANE OFFSET REFERANS")
    print("=" * 90)
    for i in range(min(samples, len(db.db_inds))):
        db_ind = int(db.db_inds[i])
        item = db.detections(db_ind)
        label_t = torch.from_numpy(item["label"]).float()

        gt_batch = labels_to_mask_batch_gt([label_t], feat_h=feat_h, feat_w=feat_w, num_lanes=num_lanes)
        hm = gt_batch["heatmap"][0]
        off = gt_batch["offset"][0]
        vm = gt_batch["valid_mask"][0]

        if not bool(vm.any().item()):
            print(f"  Sample {i}: ⚠️ valid row yok, atlandı")
            continue

        mae_coarse, mae_old = decode_old_error(hm, off, vm)
        mae_coarse_all.append(mae_coarse)
        mae_old_all.append(mae_old)
        print(f"  Sample {i}: coarse={mae_coarse:.4f}, old={mae_old:.4f} feat-px")

    checks: Dict[str, bool] = {}
    checks["used_samples_gt0"] = len(mae_old_all) > 0

    if checks["used_samples_gt0"]:
        avg_coarse = float(sum(mae_coarse_all) / len(mae_coarse_all))
        avg_old = float(sum(mae_old_all) / len(mae_old_all))
        checks["avg_old_not_much_worse"] = avg_old <= avg_coarse + 0.25
        checks["avg_old_lt_1p5px_feat"] = avg_old < 1.5
    else:
        avg_coarse = float("nan")
        avg_old = float("nan")
        checks["avg_old_not_much_worse"] = False
        checks["avg_old_lt_1p5px_feat"] = False

    print(f"\nOrtalama coarse MAE: {avg_coarse:.4f} feat-px")
    print(f"Ortalama old MAE   : {avg_old:.4f} feat-px")
    for k, v in checks.items():
        print(f"  {k:<24}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask_c")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    c_model = run_model_contract(cfg_name=args.cfg, batch=args.batch, seed=args.seed)
    c_syn = run_synthetic(feat_h=10, feat_w=26, num_lanes=int(system_configs.num_queries))
    c_real = {"skipped": True} if args.skip_real else run_real(cfg_name=args.cfg, samples=args.samples)

    ok = all(c_model.values()) and all(c_syn.values()) and (True if args.skip_real else all(c_real.values()))

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 0 — Offset baseline referans: {'✅' if ok else '🚨'}")
    if args.skip_real:
        print("Not: --skip-real kullanıldığı için gerçek veri bölümü atlandı.")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

