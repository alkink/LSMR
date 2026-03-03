"""
DENEY 1 BÖLÜM 4 — Overfit karşılaştırma (BASE / B / D).
"""

import argparse
from typing import Dict

from mamba_decoder_analysis.bolum6_overfit import run as run_overfit


def run(cfg_name: str = "LSTR_CULANE_2k_mamba_dec_b", epochs: int = 80, lr: float = 1e-4, d_state: int = 64, d_conv: int = 4, seed: int = 0) -> Dict[str, bool]:
    return run_overfit(
        cfg_name=cfg_name,
        epochs=epochs,
        lr=lr,
        d_state=d_state,
        d_conv=d_conv,
        variants=["BASE", "B", "D"],
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_dec_b")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        cfg_name=args.cfg,
        epochs=args.epochs,
        lr=args.lr,
        d_state=args.d_state,
        d_conv=args.d_conv,
        seed=args.seed,
    )
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY1 BÖLÜM4 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

