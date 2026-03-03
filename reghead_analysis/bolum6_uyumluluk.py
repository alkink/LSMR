"""
BÖLÜM 6 — RegHead uyumluluk kontrolü.

Kontroller:
1) Train/Loss kontratı (yeni reg loss) çalışıyor mu?
2) Postprocess çıktısı evaluator beklediği lane-point yapısında mı?
3) Legacy test pipeline için adapter ile geçiş mümkün mü?
4) CULane writer lane-point girişini dosyaya yazabiliyor mu?
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from config import system_configs
from db.culane import CULANE
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt, lanes_to_mask_gt
from mask_migration.common import load_system_config
from reghead_analysis.bolum2_gt_reg import compute_gt_reg
from reghead_analysis.bolum3_loss_reg import compute_new_mask_loss
from reghead_analysis.bolum5_postprocess_reg import new_postprocess


def _is_point(p) -> bool:
    if isinstance(p, torch.Tensor):
        if p.numel() != 2:
            return False
        p = p.detach().cpu().tolist()
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return False
    return all(isinstance(v, (int, float)) for v in p)


def _is_lane(lane) -> bool:
    if not isinstance(lane, (list, tuple)):
        return False
    if len(lane) == 0:
        return True
    return all(_is_point(p) for p in lane)


def _is_batch_lane_points_structure(batch_pred) -> bool:
    """
    batch_pred: [lane_0, lane_1, ...]
    lane_i: [(x,y), ...]
    """
    if not isinstance(batch_pred, (list, tuple)):
        return False
    return all(_is_lane(l) for l in batch_pred)


def _reg_to_legacy_offset(pred_reg: torch.Tensor, feat_w: int) -> torch.Tensor:
    """
    pred_reg: (B,L,H) in normalized residual
    return:   (B,L,H,W) old offset format

    Eski decode: x_old = x_coarse + offset[row, n_col]
    Yeni decode: x_new = x_coarse + reg[row] * W
    => offset[:, :, row, col] = reg[:, :, row] * W
    """
    if pred_reg.dim() != 3:
        raise ValueError(f"pred_reg must be (B,L,H), got {tuple(pred_reg.shape)}")
    return pred_reg.unsqueeze(-1).expand(-1, -1, -1, feat_w) * float(feat_w)


def check_train_loss_contract(seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz, lq, h, w = 2, 7, 10, 26

    hm_logits = torch.randn(bsz, lq, h, w, device=device, requires_grad=True)
    reg_raw = torch.randn(bsz, lq, h, device=device, requires_grad=True)
    vr_raw = torch.randn(bsz, lq, 2, device=device, requires_grad=True)
    sc_raw = torch.randn(bsz, lq, 2, device=device, requires_grad=True)

    pred_hm = F.softmax(hm_logits, dim=-1)
    pred_reg = torch.tanh(reg_raw)

    gt0 = lanes_to_mask_gt(
        [[(150, 250), (160, 200), (170, 150)], [(400, 250), (410, 200), (420, 150)]],
        img_h=295,
        img_w=820,
        feat_h=h,
        feat_w=w,
        num_lanes=lq,
    )
    gt1 = lanes_to_mask_gt(
        [[(120, 250), (130, 200), (140, 150)]],
        img_h=295,
        img_w=820,
        feat_h=h,
        feat_w=w,
        num_lanes=lq,
    )

    gt_hm = torch.stack([gt0["heatmap"], gt1["heatmap"]], dim=0).to(device)
    gt_off = torch.stack([gt0["offset"], gt1["offset"]], dim=0).to(device)
    gt_vr = torch.stack([gt0["v_range"], gt1["v_range"]], dim=0).to(device)
    gt_lbl = torch.stack([gt0["labels"], gt1["labels"]], dim=0).to(device)
    gt_vm = torch.stack([gt0["valid_mask"], gt1["valid_mask"]], dim=0).to(device)

    gt_reg = []
    for b in range(bsz):
        gt_reg.append(compute_gt_reg(gt_hm[b], gt_vm[b], feat_w=w, gt_offset=gt_off[b]))
    gt_reg = torch.stack(gt_reg, dim=0)

    losses = compute_new_mask_loss(
        pred_hm,
        pred_reg,
        vr_raw,
        sc_raw,
        gt_hm,
        gt_reg,
        gt_vr,
        gt_lbl,
        gt_vm,
    )
    losses["total"].backward()

    checks: Dict[str, bool] = {}
    checks["losses_finite"] = all(bool(torch.isfinite(v).item()) for v in losses.values())
    checks["grad_hm"] = hm_logits.grad is not None and float(hm_logits.grad.abs().mean().item()) > 1e-12
    checks["grad_reg"] = reg_raw.grad is not None and float(reg_raw.grad.abs().mean().item()) > 1e-12
    checks["grad_vr"] = vr_raw.grad is not None and float(vr_raw.grad.abs().mean().item()) > 1e-12
    checks["grad_sc"] = sc_raw.grad is not None and float(sc_raw.grad.abs().mean().item()) > 1e-12

    print("=" * 90)
    print("BÖLÜM 6.1 — TRAIN/LOSS KONTRATI")
    print("=" * 90)
    for k, v in losses.items():
        print(f"  {k:<14}: {float(v.item()):.6f}")
    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<16}: {'✅' if v else '🚨'}")

    return checks


def check_postprocess_structure() -> Dict[str, bool]:
    feat_h, feat_w = 10, 26
    img_h, img_w = 295, 820
    num_lanes = 7

    gt = lanes_to_mask_gt(
        [[(160, 250), (170, 200), (180, 150), (190, 100)]],
        img_h=img_h,
        img_w=img_w,
        feat_h=feat_h,
        feat_w=feat_w,
        num_lanes=num_lanes,
    )

    reg = compute_gt_reg(gt["heatmap"], gt["valid_mask"], feat_w=feat_w, gt_offset=gt["offset"])
    scores = torch.full((num_lanes, 2), -8.0)
    scores[0, 1] = 8.0

    lanes = new_postprocess(
        heatmap=gt["heatmap"],
        reg=reg,
        vrange=gt["v_range"].float(),
        scores=scores,
        img_h=img_h,
        img_w=img_w,
        feat_h=feat_h,
        feat_w=feat_w,
        score_thresh=0.5,
    )

    checks: Dict[str, bool] = {}
    checks["structure_ok"] = _is_batch_lane_points_structure(lanes)
    checks["non_empty"] = len(lanes) > 0 and any(len(l) >= 2 for l in lanes)

    in_bound = True
    for lane in lanes:
        for x, y in lane:
            if not (0 <= x < img_w and 0 <= y < img_h):
                in_bound = False
                break
    checks["points_in_image"] = in_bound

    print("\n" + "=" * 90)
    print("BÖLÜM 6.2 — POSTPROCESS YAPISI")
    print("=" * 90)
    print(f"Çıkan lane sayısı: {len(lanes)}")
    if lanes:
        print(f"İlk lane nokta sayısı: {len(lanes[0])}")
    for k, v in checks.items():
        print(f"  {k:<16}: {'✅' if v else '🚨'}")

    return checks


def check_legacy_adapter() -> Dict[str, bool]:
    """
    test/culane.PostProcess için reg->offset adapter kontrolü.
    """
    try:
        from test.culane import PostProcess
    except Exception:
        return {
            "postprocess_import": False,
            "legacy_accepts_adapted": False,
        }

    bsz, lq, h, w = 1, 7, 10, 26
    pred_hm = F.softmax(torch.randn(bsz, lq, h, w), dim=-1)
    pred_reg = torch.tanh(torch.randn(bsz, lq, h))
    pred_vr = torch.randn(bsz, lq, 2)
    pred_sc = torch.randn(bsz, lq, 2)

    outputs_legacy = {
        "pred_heatmap": pred_hm,
        "pred_offset": _reg_to_legacy_offset(pred_reg, feat_w=w),
        "pred_vrange": pred_vr,
        "pred_scores": pred_sc,
    }

    pp = PostProcess()
    target_sizes = torch.tensor([[295, 820]], dtype=torch.long)
    out = pp(outputs_legacy, target_sizes)

    checks: Dict[str, bool] = {}
    checks["postprocess_import"] = True
    checks["legacy_accepts_adapted"] = isinstance(out, list) and len(out) == bsz
    checks["legacy_structure_ok"] = isinstance(out, list) and (len(out) == 0 or _is_batch_lane_points_structure(out[0]))

    print("\n" + "=" * 90)
    print("BÖLÜM 6.3 — LEGACY ADAPTER")
    print("=" * 90)
    for k, v in checks.items():
        print(f"  {k:<22}: {'✅' if v else '🚨'}")

    return checks


def check_culane_writer(cfg_name: str = "LSTR_CULANE_2k_mamba_mask_c", samples: int = 1) -> Dict[str, bool]:
    """
    Lane-point pred'in CULane writer'a verilmesi.
    """
    try:
        cfg = load_system_config(cfg_name)
        db = CULANE(cfg["db"], system_configs.train_split)
    except Exception:
        return {
            "writer_init": False,
            "writer_output_exists": False,
        }

    feat_h, feat_w = 10, 26
    num_lanes = int(system_configs.num_queries)
    exp_dir = os.path.join("cache", "reghead_analysis_compat")
    os.makedirs(exp_dir, exist_ok=True)

    ok_exists = True
    used = 0

    for i in range(min(samples, len(db.db_inds))):
        db_ind = int(db.db_inds[i])
        item = db.detections(db_ind)
        label_t = torch.from_numpy(item["label"]).float()

        gt_batch = labels_to_mask_batch_gt([label_t], feat_h=feat_h, feat_w=feat_w, num_lanes=num_lanes)
        hm = gt_batch["heatmap"][0]
        vm = gt_batch["valid_mask"][0]
        vr = gt_batch["v_range"][0].float()
        reg = compute_gt_reg(hm, vm, feat_w=feat_w, gt_offset=gt_batch["offset"][0])

        scores = torch.full((num_lanes, 2), -8.0)
        fg_idx = torch.nonzero(gt_batch["labels"][0] > 0, as_tuple=False).squeeze(-1)
        if fg_idx.numel() > 0:
            scores[fg_idx, 1] = 8.0

        lanes = new_postprocess(
            hm,
            reg,
            vr,
            scores,
            img_h=db.img_h,
            img_w=db.img_w,
            feat_h=feat_h,
            feat_w=feat_w,
            score_thresh=0.5,
        )

        db.pred2culaneformat(i, lanes, runtime=0.01, exp_dir=exp_dir)

        img_name = db._annotations[i]["old_anno"]["org_path"]
        save_name = os.path.join(exp_dir, img_name[1:-4] + ".lines.txt")
        exists = os.path.exists(save_name)
        ok_exists = ok_exists and exists
        used += 1
        print(f"  Writer sample {i}: {'✅' if exists else '🚨'} -> {save_name}")

    checks: Dict[str, bool] = {}
    checks["writer_init"] = True
    checks["writer_used_samples"] = used > 0
    checks["writer_output_exists"] = ok_exists and used > 0

    print("\n" + "=" * 90)
    print("BÖLÜM 6.4 — CULANE WRITER")
    print("=" * 90)
    for k, v in checks.items():
        print(f"  {k:<20}: {'✅' if v else '🚨'}")

    return checks


def run(cfg_name: str = "LSTR_CULANE_2k_mamba_mask_c", samples: int = 1, seed: int = 0, skip_real: bool = False) -> Dict[str, bool]:
    c1 = check_train_loss_contract(seed=seed)
    c2 = check_postprocess_structure()
    c3 = check_legacy_adapter()
    c4 = {"writer_skipped": True} if skip_real else check_culane_writer(cfg_name=cfg_name, samples=samples)

    checks: Dict[str, bool] = {}
    checks.update({f"loss_{k}": v for k, v in c1.items()})
    checks.update({f"post_{k}": v for k, v in c2.items()})
    checks.update({f"legacy_{k}": v for k, v in c3.items()})
    if skip_real:
        checks["writer_skipped"] = True
    else:
        checks.update({f"writer_{k}": v for k, v in c4.items()})

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask_c")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, samples=args.samples, seed=args.seed, skip_real=args.skip_real)
    ok = all(bool(v) for v in checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 6 — Uyumluluk: {'✅' if ok else '🚨'}")
    if args.skip_real:
        print("Not: --skip-real kullanıldığı için writer gerçek veri kontrolü atlandı.")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

