"""
BÖLÜM 4 — Eski (offset) vs yeni (reg) overfit karşılaştırması.

Amaç:
1) Aynı sabit decoder girdisinde iki başlığın öğrenme davranışını ölçmek
2) Eski yol: (B,L,H,W) offset loss
3) Yeni yol: (B,L,H) reg loss
"""

import argparse
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import system_configs
from db.culane import CULANE
from mask_migration.bolum1_gt_mask import labels_to_mask_batch_gt
from mask_migration.bolum2_mask_head import DynamicMaskHead
from mask_migration.bolum3_loss import compute_mask_loss
from mask_migration.common import build_model_from_cfg, forward_to_decoder, load_system_config
from reghead_analysis.bolum2_gt_reg import compute_gt_reg
from reghead_analysis.bolum3_loss_reg import compute_new_mask_loss


class RegVariantHead(nn.Module):
    """
    Yeni başlık:
      - heatmap: (B,L,H,W)
      - reg    : (B,L,H)
      - v_range: (B,L,2)
      - scores : (B,L,2)
    """

    def __init__(
        self,
        num_queries: int,
        feat_dim: int,
        feat_h: int,
        feat_w: int,
        use_coords: bool = False,
        prior_prob: float = None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feat_dim = feat_dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.use_coords = use_coords

        self.kernel_dim = feat_dim + (2 if use_coords else 0)

        self.mlp_heatmap = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.kernel_dim),
        )
        self.mlp_reg = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_h),
            nn.Tanh(),
        )
        self.mlp_vrange = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )
        self.mlp_score = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )

        if prior_prob is not None:
            if not (0.0 < float(prior_prob) < 1.0):
                raise ValueError(f"prior_prob must be in (0,1), got {prior_prob}")
            bias_value = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
            with torch.no_grad():
                self.mlp_score[-1].bias[0] = 0.0
                self.mlp_score[-1].bias[1] = float(bias_value)

    @staticmethod
    def _inject_coords(feat: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = feat.shape
        ys = torch.linspace(0.0, 1.0, steps=h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(0.0, 1.0, steps=w, device=feat.device, dtype=feat.dtype)
        y_plane = ys.view(1, 1, h, 1).expand(b, 1, h, w)
        x_plane = xs.view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([feat, y_plane, x_plane], dim=1)

    def forward(self, t: torch.Tensor, m_prime: torch.Tensor):
        # t: (L,B,C) or (B,L,C)
        bsz, _c, h, w = m_prime.shape
        if t.dim() != 3:
            raise ValueError(f"t must be 3D, got {tuple(t.shape)}")

        if t.shape[1] == bsz:  # (L,B,C)
            t_blc = t.permute(1, 0, 2)
            lanes = t.shape[0]
        elif t.shape[0] == bsz:  # (B,L,C)
            t_blc = t
            lanes = t.shape[1]
        else:
            raise ValueError(f"Cannot infer batch dim from t={tuple(t.shape)} and m_prime={tuple(m_prime.shape)}")

        feat = self._inject_coords(m_prime) if self.use_coords else m_prime
        flat = feat.flatten(2)  # (B, C(+2), HW)

        k_hm = self.mlp_heatmap(t_blc)      # (B,L,C(+2))
        b_raw = torch.bmm(k_hm, flat)       # (B,L,HW)
        hm = F.softmax(b_raw.view(bsz, lanes, h, w), dim=-1)

        reg = self.mlp_reg(t_blc)           # (B,L,H)
        vr = self.mlp_vrange(t_blc)         # (B,L,2)
        sc = self.mlp_score(t_blc)          # (B,L,2)

        return {
            "heatmap": hm,
            "reg": reg,
            "v_range": vr,
            "scores": sc,
        }


@torch.no_grad()
def copy_shared_weights(old_head: DynamicMaskHead, new_head: RegVariantHead) -> None:
    """Isolasyon için heatmap/vrange/score başlangıcını eskiyle eşler."""
    new_head.mlp_heatmap.load_state_dict(old_head.mlp_heatmap.state_dict(), strict=True)
    new_head.mlp_vrange.load_state_dict(old_head.mlp_vrange.state_dict(), strict=True)
    new_head.mlp_score.load_state_dict(old_head.mlp_score.state_dict(), strict=True)


def _extract_one_sample(cfg_name: str, device: torch.device):
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)

    db_ind = int(db.db_inds[0])
    img_tensor, label_np, _ = db.__getitem__(db_ind, transform=True)
    images = img_tensor.unsqueeze(0).to(device)
    masks = torch.zeros((1, 1, images.shape[-2], images.shape[-1]), device=device)
    label_t = torch.from_numpy(label_np).float()
    return images, masks, label_t


def train_old_offset(
    head: DynamicMaskHead,
    t_fixed: torch.Tensor,
    m_fixed: torch.Tensor,
    gt_hm: torch.Tensor,
    gt_off: torch.Tensor,
    gt_vr: torch.Tensor,
    gt_lbl: torch.Tensor,
    gt_vm: torch.Tensor,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    hist = []

    for ep in range(epochs):
        opt.zero_grad()
        pred = head(t_fixed, m_fixed)
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
        total = losses["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        hist.append(float(total.item()))

        if ep % 20 == 0 or ep == epochs - 1:
            print(
                f"[OLD] Epoch {ep:3d}: total={float(total.item()):.4f} "
                f"heat={float(losses['heat_loss'].item()):.4f} "
                f"offset={float(losses['offset_loss'].item()):.4f} "
                f"cls={float(losses['cls_loss'].item()):.4f}"
            )

    init_loss = hist[0]
    final_loss = hist[-1]
    reduction = (init_loss - final_loss) / max(init_loss, 1e-6) * 100.0
    return {
        "init": init_loss,
        "final": final_loss,
        "reduction": reduction,
    }


def train_new_reg(
    head: RegVariantHead,
    t_fixed: torch.Tensor,
    m_fixed: torch.Tensor,
    gt_hm: torch.Tensor,
    gt_reg: torch.Tensor,
    gt_vr: torch.Tensor,
    gt_lbl: torch.Tensor,
    gt_vm: torch.Tensor,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    hist = []

    for ep in range(epochs):
        opt.zero_grad()
        pred = head(t_fixed, m_fixed)
        losses = compute_new_mask_loss(
            pred["heatmap"],
            pred["reg"],
            pred["v_range"],
            pred["scores"],
            gt_hm,
            gt_reg,
            gt_vr,
            gt_lbl,
            gt_vm,
        )
        total = losses["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        hist.append(float(total.item()))

        if ep % 20 == 0 or ep == epochs - 1:
            print(
                f"[NEW] Epoch {ep:3d}: total={float(total.item()):.4f} "
                f"heat={float(losses['heat_loss'].item()):.4f} "
                f"reg={float(losses['reg_loss'].item()):.4f} "
                f"cls={float(losses['cls_loss'].item()):.4f}"
            )

    init_loss = hist[0]
    final_loss = hist[-1]
    reduction = (init_loss - final_loss) / max(init_loss, 1e-6) * 100.0
    return {
        "init": init_loss,
        "final": final_loss,
        "reduction": reduction,
    }


def run(cfg_name: str = "LSTR_CULANE_2k_mamba_mask_c", epochs: int = 120, lr: float = 1e-3, seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, masks, label_t = _extract_one_sample(cfg_name=cfg_name, device=device)
    backbone = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)
    for p in backbone.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        hs, memory, _ = forward_to_decoder(backbone, images, masks)
        t_fixed = hs[-1].detach()   # (B,L,C)
        m_fixed = memory.detach()   # (B,C,H,W)

    _bsz, _c, h, w = m_fixed.shape
    lq = int(t_fixed.shape[1])
    use_coords = bool(getattr(backbone.mask_head, "use_coords", False))
    # C varyantı için prior bias var; karşılaştırmada confound olmaması için iki başlığa da aynı ayar veriyoruz.
    prior_prob = 0.01 if use_coords else None

    gt = labels_to_mask_batch_gt([label_t], feat_h=h, feat_w=w, num_lanes=lq)
    gt_hm = gt["heatmap"].to(device)
    gt_off = gt["offset"].to(device)
    gt_vr = gt["v_range"].to(device)
    gt_lbl = gt["labels"].to(device)
    gt_vm = gt["valid_mask"].to(device)
    gt_reg = compute_gt_reg(gt_hm[0], gt_vm[0], feat_w=w, gt_offset=gt_off[0]).unsqueeze(0)

    # Eski başlık (offset)
    old_head = DynamicMaskHead(
        num_queries=lq,
        feat_dim=t_fixed.shape[-1],
        feat_h=h,
        feat_w=w,
        use_coords=use_coords,
        prior_prob=prior_prob,
    ).to(device)

    # Yeni başlık (reg)
    new_head = RegVariantHead(
        num_queries=lq,
        feat_dim=t_fixed.shape[-1],
        feat_h=h,
        feat_w=w,
        use_coords=use_coords,
        prior_prob=prior_prob,
    ).to(device)

    copy_shared_weights(old_head, new_head)

    print("=" * 90)
    print("BÖLÜM 4 — OVERFİT KARŞILAŞTIRMA (OLD OFFSET vs NEW REG)")
    print("=" * 90)
    print(f"Device      : {device}")
    print(f"Config      : {cfg_name}")
    print(f"T_fixed     : {tuple(t_fixed.shape)}")
    print(f"M_fixed     : {tuple(m_fixed.shape)}")
    print(f"GT heatmap  : {tuple(gt_hm.shape)}")
    print(f"GT reg      : {tuple(gt_reg.shape)}")

    old_stat = train_old_offset(
        head=old_head,
        t_fixed=t_fixed,
        m_fixed=m_fixed,
        gt_hm=gt_hm,
        gt_off=gt_off,
        gt_vr=gt_vr,
        gt_lbl=gt_lbl,
        gt_vm=gt_vm,
        epochs=epochs,
        lr=lr,
    )

    new_stat = train_new_reg(
        head=new_head,
        t_fixed=t_fixed,
        m_fixed=m_fixed,
        gt_hm=gt_hm,
        gt_reg=gt_reg,
        gt_vr=gt_vr,
        gt_lbl=gt_lbl,
        gt_vm=gt_vm,
        epochs=epochs,
        lr=lr,
    )

    checks: Dict[str, bool] = {}
    checks["old_reduced"] = old_stat["final"] < old_stat["init"]
    checks["new_reduced"] = new_stat["final"] < new_stat["init"]
    checks["old_over_30pct"] = old_stat["reduction"] > 30.0
    checks["new_over_30pct"] = new_stat["reduction"] > 30.0
    checks["new_not_catastrophic"] = new_stat["reduction"] >= old_stat["reduction"] - 35.0

    print("\n" + "=" * 90)
    print("SONUÇ KARŞILAŞTIRMASI")
    print("=" * 90)
    print(
        f"OLD offset: init={old_stat['init']:.6f}, final={old_stat['final']:.6f}, "
        f"reduction={old_stat['reduction']:.2f}%"
    )
    print(
        f"NEW reg   : init={new_stat['init']:.6f}, final={new_stat['final']:.6f}, "
        f"reduction={new_stat['reduction']:.2f}%"
    )
    print(f"Delta(new-old): {new_stat['reduction'] - old_stat['reduction']:+.2f} puan")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<22}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask_c")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, epochs=args.epochs, lr=args.lr, seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 4 — Overfit old vs new: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

