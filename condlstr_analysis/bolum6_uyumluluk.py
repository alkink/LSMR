"""
BÖLÜM 6 — Uyumluluk kontrolü (pre-integration).

Amaç:
1) [test/culane.py](test/culane.py:44) postprocess kontratı ile önerilen değişikliklerin uyumunu doğrulamak
2) [db/culane.py](db/culane.py:416) lane-point serileştirme beklentisiyle yapı uyumunu doğrulamak
3) Gelecek entegrasyon için risk checklist üretmek
"""

import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np

from config import system_configs
from db.culane import CULANE
from mask_migration.bolum5_postprocess import mask_to_lane_coords
from mask_migration.common import build_model_from_cfg, load_system_config
from test.culane import PostProcess


def _is_point(p) -> bool:
    if isinstance(p, np.ndarray):
        return p.ndim == 1 and p.shape[0] == 2
    return isinstance(p, (list, tuple)) and len(p) == 2


def _is_lane(lane) -> bool:
    # lane: list/tuple/ndarray of points
    if isinstance(lane, np.ndarray):
        if lane.ndim != 2 or lane.shape[1] != 2:
            return False
        return True
    if not isinstance(lane, (list, tuple)):
        return False
    if len(lane) == 0:
        return True
    return all(_is_point(p) for p in lane)


def _is_batch_lane_points_structure(batch_pred) -> bool:
    """
    Beklenen yapı: batch -> lane -> point(x,y)
    - batch_pred: list[B]
    - batch_pred[b]: list[lane]
    - lane: list[(x,y)] or ndarray[N,2]
    """
    if not isinstance(batch_pred, list):
        return False
    if len(batch_pred) == 0:
        return True

    for lanes in batch_pred:
        if not isinstance(lanes, (list, tuple)):
            return False
        for lane in lanes:
            if not _is_lane(lane):
                return False
    return True


@torch.no_grad()
def run(cfg_name: str = "LSTR_CULANE_2k_mamba_mask", batch: int = 2, score_thresh: float = 0.5) -> Dict[str, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_system_config(cfg_name)
    in_h, in_w = cfg["db"]["input_size"]

    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)
    post = PostProcess().to(device)

    # 1) model out -> PostProcess
    images = torch.randn(batch, 3, in_h, in_w, device=device)
    masks = torch.zeros(batch, 1, in_h, in_w, device=device)
    outputs, _ = model._train(images, masks)
    target_sizes = torch.tensor([[in_h, in_w]] * batch, device=device)

    lanes_batch = post(outputs, target_sizes)

    # 2) manual postprocess branch consistency
    manual_batch: List[List[List[Tuple[int, int]]]] = []
    for b in range(batch):
        lanes = mask_to_lane_coords(
            outputs["pred_heatmap"][b],
            outputs["pred_offset"][b],
            outputs["pred_vrange"][b],
            outputs["pred_scores"][b],
            score_thresh=score_thresh,
            img_h=in_h,
            img_w=in_w,
        )
        manual_batch.append(lanes)

    # 3) db serializer compatibility check (single sample)
    db = CULANE(cfg["db"], system_configs.test_split)
    dummy_runtime = 0.01
    serial_ok = False
    try:
        _ = db.pred2culaneformat(0, lanes_batch[0], dummy_runtime, system_configs.result_dir)
        serial_ok = True
    except Exception:
        serial_ok = False

    # structure checks
    checks: Dict[str, bool] = {}
    checks["post_returns_list"] = isinstance(lanes_batch, list) and len(lanes_batch) == batch
    checks["lane_points_structure"] = _is_batch_lane_points_structure(lanes_batch)
    checks["manual_post_structure"] = _is_batch_lane_points_structure(manual_batch)
    checks["serializer_accepts_lane_points"] = serial_ok

    # exact equality beklenmez (threshold/float farkı olabilir), sadece tür ve eleman sayısı bazlı kontrol
    checks["manual_vs_post_batch_size"] = len(manual_batch) == len(lanes_batch)

    print("=" * 90)
    print("BÖLÜM 6 — UYUMLULUK")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Input size: {(in_h, in_w)}")
    print(f"Post batch len: {len(lanes_batch)}")
    print(f"Manual batch len: {len(manual_batch)}")
    if len(lanes_batch) > 0:
        print(f"Post[0] lane count: {len(lanes_batch[0])}")
    if len(manual_batch) > 0:
        print(f"Manual[0] lane count: {len(manual_batch[0])}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<32}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, batch=args.batch, score_thresh=args.score_thresh)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 6 — Uyumluluk: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

