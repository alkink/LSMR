"""
BÖLÜM 4 — Sadece DynamicMaskHead overfit testi.
"""
import argparse
import os

import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.common import require_cuda, load_system_config, build_model_from_cfg, forward_to_decoder
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt
from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.bolum3_loss import compute_mask_loss


def main(cfg_name: str = "LSTR_CULANE_2k_mamba", epochs: int = 300, lr: float = 1e-3):
    device = require_cuda()
    cfg = load_system_config(cfg_name)

    backbone_model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)

    ckpt_path = os.path.join("cache", "nnet", cfg_name, f"{cfg_name}_{system_configs.max_iter}.pkl")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        backbone_model.load_state_dict(state, strict=False)
        print(f"✅ Checkpoint yüklendi: {ckpt_path}")
    else:
        print(f"⚠️  Checkpoint bulunamadı, mevcut ağırlıklar kullanılıyor: {ckpt_path}")

    for p in backbone_model.parameters():
        p.requires_grad_(False)

    db = CULANE(cfg["db"], system_configs.train_split)
    db_ind = int(db.db_inds[0])
    img_tensor, label_np, _ = db.__getitem__(db_ind, transform=True)
    images = img_tensor.unsqueeze(0).to(device)
    masks = torch.zeros((1, 1, images.shape[-2], images.shape[-1]), device=device)

    with torch.no_grad():
        hs, memory, _ = forward_to_decoder(backbone_model, images, masks)
        T_fixed = hs[-1].permute(1, 0, 2).detach()  # (N, B, C)
        M_fixed = memory.detach()                   # (B, C, H, W)

    _, _, H, W = M_fixed.shape
    num_queries = system_configs.num_queries
    gt = labels_to_mask_batch_gt([label_np], feat_h=H, feat_w=W, num_lanes=num_queries)
    gt_hm = gt["heatmap"].to(device)
    gt_off = gt["offset"].to(device)
    gt_vr = gt["v_range"].to(device)
    gt_lbl = gt["labels"].to(device)
    gt_vm = gt["valid_mask"].to(device)

    head = DynamicMaskHead(num_queries=num_queries, feat_dim=T_fixed.shape[-1], feat_h=H, feat_w=W).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    print("\n" + "=" * 90)
    print("BÖLÜM 4 — OVERFİT TESTİ (HEAD ONLY)")
    print("=" * 90)
    print(f"T_fixed: {tuple(T_fixed.shape)}, M_fixed: {tuple(M_fixed.shape)}")
    print(f"GT heatmap: {tuple(gt_hm.shape)}, labels: {gt_lbl[0].tolist()}")

    losses_hist = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = head(T_fixed, M_fixed)
        losses = compute_mask_loss(
            pred["heatmap"],
            pred["offset"],
            pred["v_range"],
            pred["scores"],
            gt_hm,
            gt_off,
            gt_vr,
            gt_lbl,
            gt_vm,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        losses_hist.append(float(losses["total"].item()))
        if epoch % 30 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}: total={losses['total'].item():.4f}  "
                f"heat={losses['heat_loss'].item():.4f}  "
                f"offset={losses['offset_loss'].item():.4f}  "
                f"cls={losses['cls_loss'].item():.4f}"
            )

    init_loss = losses_hist[0]
    final_loss = losses_hist[-1]
    reduction = (init_loss - final_loss) / max(init_loss, 1e-6) * 100.0

    print("\n" + "=" * 90)
    print(f"Başlangıç: {init_loss:.6f}")
    print(f"Final:     {final_loss:.6f}")
    print(f"Azalma:    %{reduction:.2f}")

    if reduction > 90:
        print("✅ Head overfit edebildi — entegrasyona hazır")
    elif reduction > 70:
        print("⚠️  Kısmi öğrenme — loss ağırlıkları/lr ayarlanmalı")
    else:
        print("🚨 Overfit başarısız — loss/head tekrar incelenmeli")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="LSTR_CULANE_2k_mamba", type=str)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()
    main(cfg_name=args.cfg, epochs=args.epochs, lr=args.lr)

