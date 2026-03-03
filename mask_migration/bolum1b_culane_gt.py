"""
BÖLÜM 1.2 — Gerçek CULane label -> mask GT doğrulaması.
"""
import argparse

import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.common import require_cuda, load_system_config, build_model_from_cfg, infer_feature_hw
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt


def main(cfg_name: str = "LSTR_CULANE_2k_mamba", n_samples: int = 5):
    device = require_cuda()
    cfg = load_system_config(cfg_name)

    net = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)
    in_h, in_w = cfg["db"]["input_size"]
    feat_h, feat_w = infer_feature_hw(net, in_h, in_w, device)

    db = CULANE(cfg["db"], system_configs.train_split)

    print("=" * 90)
    print("BÖLÜM 1.2 — GERÇEK CULANE GT TESTİ")
    print("=" * 90)
    print(f"Config input size: {(in_h, in_w)}")
    print(f"Inferred feature size: {(feat_h, feat_w)}")

    success_count = 0
    fail_count = 0

    for i in range(min(n_samples, len(db.db_inds))):
        db_ind = int(db.db_inds[i])
        item = db.detections(db_ind)

        print(f"\nSample {i} keys: {list(item.keys())}")
        print(f"  path: {item['path']}")
        print(f"  label shape: {item['label'].shape}")
        print(f"  categories len: {len(item.get('categories', []))}")
        print(f"  old lanes len: {len(item.get('old_anno', {}).get('lanes', []))}")

        label_tensor = torch.from_numpy(item["label"]).float()
        gt = labels_to_mask_batch_gt(
            [label_tensor],
            feat_h=feat_h,
            feat_w=feat_w,
            num_lanes=system_configs.num_queries,
        )

        valid_rows = gt["valid_mask"][0]
        has_any = bool(valid_rows.any().item())
        row_sums_ok = True

        for lane_id in range(valid_rows.shape[0]):
            rows = valid_rows[lane_id].nonzero(as_tuple=False).squeeze(-1)
            for r in rows.tolist():
                s = gt["heatmap"][0, lane_id, r].sum().item()
                if abs(s - 1.0) > 1e-2:
                    row_sums_ok = False
                    break

        if has_any and row_sums_ok:
            success_count += 1
            print("  ✅ GT üretim PASS (valid row var, heatmap satır toplamı doğru)")
        else:
            fail_count += 1
            print(
                "  🚨 GT üretim FAIL "
                f"(has_any_valid={has_any}, row_sums_ok={row_sums_ok})"
            )

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print("BÖLÜM 1b — CULane GT:         {}".format("✅" if fail_count == 0 else "🚨"))
    print(f"Başarılı sample: {success_count}, Hatalı sample: {fail_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="LSTR_CULANE_2k_mamba", type=str)
    parser.add_argument("--samples", default=5, type=int)
    args = parser.parse_args()
    main(cfg_name=args.cfg, n_samples=args.samples)

