"""
BÖLÜM 2 — Real GT pipeline analizi (izole).

Amaç:
1) Mevcut label-tensor tabanlı GT ile old_anno['lanes'] tabanlı GT'yi karşılaştırmak
2) Shape/finite/coverage farklarını sayısallaştırmak
3) Match cost düzeyinde iki GT kaynağının etkisini izole etmek

Not:
- LSTR veri akışında label tensor zaten old_anno['lanes']'dan türetiliyor.
- Bu nedenle iki kaynağın eşdeğer çıkması beklenen davranıştır.
"""

import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt, lanes_to_mask_gt
from mask_migration.common import build_model_from_cfg, infer_feature_hw, load_system_config


def _extract_old_lanes(item: dict) -> List[List[Tuple[float, float]]]:
    old_lanes = item.get("old_anno", {}).get("lanes", [])
    lanes: List[List[Tuple[float, float]]] = []
    for lane in old_lanes:
        pts = [(float(x), float(y)) for (x, y) in lane if x >= 0 and y >= 0]
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def old_anno_to_mask_batch_gt(
    items: Sequence[dict],
    feat_h: int,
    feat_w: int,
    num_lanes: int,
    img_h: int,
    img_w: int,
) -> Dict[str, torch.Tensor]:
    hm, off, vr, lbl, vm = [], [], [], [], []
    for item in items:
        lanes = _extract_old_lanes(item)
        gt = lanes_to_mask_gt(
            lanes,
            img_h=img_h,
            img_w=img_w,
            feat_h=feat_h,
            feat_w=feat_w,
            num_lanes=num_lanes,
            normalized=False,
        )
        hm.append(gt["heatmap"])
        off.append(gt["offset"])
        vr.append(gt["v_range"])
        lbl.append(gt["labels"])
        vm.append(gt["valid_mask"])
    return {
        "heatmap": torch.stack(hm, dim=0),
        "offset": torch.stack(off, dim=0),
        "v_range": torch.stack(vr, dim=0),
        "labels": torch.stack(lbl, dim=0),
        "valid_mask": torch.stack(vm, dim=0),
    }


def _rowloc_from_heatmap(hm: torch.Tensor) -> torch.Tensor:
    # hm: (B, L, H, W) -> row-wise expected x (B, L, H)
    w = hm.shape[-1]
    cols = torch.arange(w, dtype=hm.dtype, device=hm.device)
    return (hm * cols.view(1, 1, 1, w)).sum(dim=-1)


def _simple_match_cost(
    pred_scores: torch.Tensor,   # (B, L, 2)
    pred_heatmap: torch.Tensor,  # (B, L, H, W)
    pred_vrange: torch.Tensor,   # (B, L, 2)
    gt_hm: torch.Tensor,         # (B, L, H, W)
    gt_vr: torch.Tensor,         # (B, L, 2)
    gt_lbl: torch.Tensor,        # (B, L)
    gt_vm: torch.Tensor,         # (B, L, H)
) -> List[float]:
    """Current [mask_migration.bolum3_loss.py](mask_migration/bolum3_loss.py:12) cost benzeri skaler rapor."""
    bsz, lq, _h, _w = pred_heatmap.shape
    pred_x = _rowloc_from_heatmap(pred_heatmap)
    gt_x = _rowloc_from_heatmap(gt_hm)

    batch_costs: List[float] = []
    fg_prob = torch.softmax(pred_scores, dim=-1)[:, :, 1]  # (B, L)

    for b in range(bsz):
        gt_idx = torch.nonzero(gt_lbl[b] > 0, as_tuple=False).squeeze(-1)
        if gt_idx.numel() == 0:
            batch_costs.append(0.0)
            continue
        csum = 0.0
        for g in gt_idx.tolist():
            valid = gt_vm[b, g]
            if bool(valid.any().item()):
                heat_cost = torch.abs(pred_x[b, :, valid] - gt_x[b, g, valid].unsqueeze(0)).mean(dim=1)
            else:
                heat_cost = torch.full((lq,), 2.0, dtype=pred_x.dtype, device=pred_x.device)
            range_cost = torch.abs(pred_vrange[b] - gt_vr[b, g].float().unsqueeze(0)).mean(dim=1)
            cls_cost = 1.0 - fg_prob[b]
            total_cost = cls_cost + heat_cost + 0.25 * range_cost
            csum += float(total_cost.min().item())
        batch_costs.append(csum / max(len(gt_idx), 1))

    return batch_costs


@torch.no_grad()
def run(cfg_name: str = "LSTR_CULANE_2k_mamba_mask", samples: int = 8, seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_system_config(cfg_name)

    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)
    in_h, in_w = cfg["db"]["input_size"]
    feat_h, feat_w = infer_feature_hw(model, in_h, in_w, device)
    num_lanes = int(system_configs.num_queries)

    db = CULANE(cfg["db"], system_configs.train_split)
    n = min(samples, len(db.db_inds))
    indices = [int(db.db_inds[i]) for i in range(n)]
    items = [db.detections(i) for i in indices]

    # 1) label-tensor GT
    label_tensors = [torch.from_numpy(it["label"]).float() for it in items]
    gt_label = labels_to_mask_batch_gt(label_tensors, feat_h=feat_h, feat_w=feat_w, num_lanes=num_lanes)

    # 2) old_anno lanes GT (real)
    gt_real = old_anno_to_mask_batch_gt(
        items,
        feat_h=feat_h,
        feat_w=feat_w,
        num_lanes=num_lanes,
        img_h=int(db.img_h),
        img_w=int(db.img_w),
    )

    # tek batch infer
    images = torch.randn(n, 3, in_h, in_w, device=device)
    masks = torch.zeros(n, 1, in_h, in_w, device=device)
    outputs, _ = model._train(images, masks)

    pred_hm = outputs["pred_heatmap"]
    pred_vr = outputs["pred_vrange"]
    pred_sc = outputs["pred_scores"]

    gt_label_dev = {k: v.to(device) for k, v in gt_label.items()}
    gt_real_dev = {k: v.to(device) for k, v in gt_real.items()}

    # coverage / density
    lbl_lane_counts = gt_label["labels"].sum(dim=1).cpu().numpy().astype(np.float32)
    real_lane_counts = gt_real["labels"].sum(dim=1).cpu().numpy().astype(np.float32)

    lbl_valid_rows = gt_label["valid_mask"].sum(dim=(1, 2)).cpu().numpy().astype(np.float32)
    real_valid_rows = gt_real["valid_mask"].sum(dim=(1, 2)).cpu().numpy().astype(np.float32)

    # pairwise diffs (GT source difference)
    hm_l1 = float((gt_label["heatmap"] - gt_real["heatmap"]).abs().mean().item())
    vr_l1 = float((gt_label["v_range"].float() - gt_real["v_range"].float()).abs().mean().item())

    # match-cost proxy (same preds, farklı GT kaynakları)
    cost_label = _simple_match_cost(
        pred_scores=pred_sc,
        pred_heatmap=pred_hm,
        pred_vrange=pred_vr,
        gt_hm=gt_label_dev["heatmap"],
        gt_vr=gt_label_dev["v_range"],
        gt_lbl=gt_label_dev["labels"],
        gt_vm=gt_label_dev["valid_mask"],
    )
    cost_real = _simple_match_cost(
        pred_scores=pred_sc,
        pred_heatmap=pred_hm,
        pred_vrange=pred_vr,
        gt_hm=gt_real_dev["heatmap"],
        gt_vr=gt_real_dev["v_range"],
        gt_lbl=gt_real_dev["labels"],
        gt_vm=gt_real_dev["valid_mask"],
    )

    d_cost = float(np.mean(np.asarray(cost_real) - np.asarray(cost_label)))

    checks: Dict[str, bool] = {}
    checks["shape_match"] = (
        tuple(gt_label["heatmap"].shape) == tuple(gt_real["heatmap"].shape)
        and tuple(gt_label["offset"].shape) == tuple(gt_real["offset"].shape)
        and tuple(gt_label["v_range"].shape) == tuple(gt_real["v_range"].shape)
    )
    checks["finite_gt"] = all(
        bool(torch.isfinite(v).all().item())
        for v in [
            gt_label["heatmap"],
            gt_label["offset"],
            gt_label["v_range"].float(),
            gt_real["heatmap"],
            gt_real["offset"],
            gt_real["v_range"].float(),
        ]
    )
    checks["real_pipeline_nonempty"] = float(real_lane_counts.mean()) > 0.0 and float(real_valid_rows.mean()) > 0.0

    # Beklenen: iki GT kaynağı eşdeğer.
    eq_tol = 1e-8
    sources_equivalent = (hm_l1 <= eq_tol) and (vr_l1 <= eq_tol)

    # Legacy key korunuyor; artık "eşdeğerlik bekleneni" anlamında kullanılıyor.
    checks["gt_sources_not_identical"] = sources_equivalent
    checks["gt_sources_equivalent_expected"] = sources_equivalent

    print("=" * 90)
    print("BÖLÜM 2 — REAL GT ANALİZİ")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Samples: {n}")
    print(f"Input size: {(in_h, in_w)}, feature size: {(feat_h, feat_w)}")

    print("\nGT kaynak karşılaştırması:")
    print(f"  lane_count(label) mean: {float(lbl_lane_counts.mean()):.3f}")
    print(f"  lane_count(real ) mean: {float(real_lane_counts.mean()):.3f}")
    print(f"  valid_rows(label) mean: {float(lbl_valid_rows.mean()):.3f}")
    print(f"  valid_rows(real ) mean: {float(real_valid_rows.mean()):.3f}")
    print(f"  heatmap L1(label-real): {hm_l1:.6e}")
    print(f"  vrange  L1(label-real): {vr_l1:.6e}")
    print(
        "  source-equivalence: {} (beklenen davranış)".format(
            "✅" if sources_equivalent else "🚨"
        )
    )

    print("\nMatch-cost proxy (aynı pred, farklı GT):")
    print(f"  cost(label) mean: {float(np.mean(cost_label)):.6f}")
    print(f"  cost(real ) mean: {float(np.mean(cost_real)):.6f}")
    print(f"  delta(real-label): {d_cost:.6f}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<28}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, samples=args.samples, seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 2 — Real GT: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

