"""
BÖLÜM 0 — Baseline analiz.

Amaç:
1) Mevcut LSTR_CULANE_2k_mamba_mask çıkış kontratını doğrulamak
2) Tensor shape/finite kontrolü yapmak
3) Decoder memory -> M_prime aktarımının gerçekten doğrudan olduğunu doğrulamak
"""

import argparse
from typing import Dict

import torch

from mask_migration.common import build_model_from_cfg, forward_to_decoder, load_system_config


def _shape(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    return type(x).__name__


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


@torch.no_grad()
def run(cfg_name: str = "LSTR_CULANE_2k_mamba_mask", batch: int = 2, seed: int = 0) -> Dict[str, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_system_config(cfg_name)
    in_h, in_w = cfg["db"]["input_size"]

    net = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)

    images = torch.randn(batch, 3, in_h, in_w, device=device)
    masks = torch.zeros(batch, 1, in_h, in_w, device=device)

    hs, memory, weights = forward_to_decoder(net, images, masks)
    t_blc = hs[-1]  # (B, L, C)

    manual_out = net.mask_head(t_blc, memory)
    train_out, train_weights = net._train(images, masks)

    required = ["pred_heatmap", "pred_offset", "pred_vrange", "pred_scores"]
    checks: Dict[str, bool] = {}
    checks["required_keys"] = all(k in train_out for k in required)

    if checks["required_keys"]:
        phm = train_out["pred_heatmap"]
        pof = train_out["pred_offset"]
        pvr = train_out["pred_vrange"]
        psc = train_out["pred_scores"]

        checks["shape_heatmap_offset_match"] = tuple(phm.shape) == tuple(pof.shape)
        checks["shape_vrange_scores_ok"] = (pvr.shape[-1] == 2) and (psc.shape[-1] == 2)
        checks["memory_hw_match"] = tuple(memory.shape[-2:]) == tuple(phm.shape[-2:])

        row_sums = phm.sum(dim=-1)
        checks["heatmap_row_softmax_ok"] = float((row_sums - 1.0).abs().mean().item()) < 1e-3

        checks["finite_outputs"] = all(
            _is_finite(v)
            for v in [
                phm,
                pof,
                pvr,
                psc,
                manual_out["heatmap"],
                manual_out["offset"],
                manual_out["v_range"],
                manual_out["scores"],
            ]
        )

        d_hm = float((manual_out["heatmap"] - phm).abs().max().item())
        d_of = float((manual_out["offset"] - pof).abs().max().item())
        d_vr = float((manual_out["v_range"] - pvr).abs().max().item())
        d_sc = float((manual_out["scores"] - psc).abs().max().item())
        checks["memory_to_mprime_direct"] = max(d_hm, d_of, d_vr, d_sc) < 1e-6
    else:
        d_hm = d_of = d_vr = d_sc = float("inf")

    print("=" * 90)
    print("BÖLÜM 0 — BASELINE")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Input size: {(in_h, in_w)}")
    print(f"hs shape: {_shape(hs)}")
    print(f"memory shape: {tuple(memory.shape)}")
    print(f"encoder weights shape: {_shape(weights)}")
    print(f"train weights shape: {_shape(train_weights)}")
    print(f"train_out shapes: {_shape(train_out)}")

    print("\nManual vs _train max |diff|:")
    print(f"  heatmap: {d_hm:.3e}")
    print(f"  offset : {d_of:.3e}")
    print(f"  vrange : {d_vr:.3e}")
    print(f"  scores : {d_sc:.3e}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<30}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_mask")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, batch=args.batch, seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print(f"BÖLÜM 0 — Baseline: {'✅' if ok else '🚨'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

