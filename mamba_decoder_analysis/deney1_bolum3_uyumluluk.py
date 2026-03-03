"""
DENEY 1 BÖLÜM 3 — Pipeline uyumluluk (BASE / B / D).
"""

import argparse
from typing import Dict

from mamba_decoder_analysis.bolum5_uyumluluk import run as run_pipeline


def run(cfg_name: str = "LSTR_CULANE_2k_mamba_dec_b", d_state: int = 64, d_conv: int = 4, seed: int = 0) -> Dict[str, bool]:
    return run_pipeline(
        cfg_name=cfg_name,
        d_state=d_state,
        d_conv=d_conv,
        variants=["BASE", "B", "D"],
        headdim=32,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_dec_b")
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, d_state=args.d_state, d_conv=args.d_conv, seed=args.seed)
    ok = bool(checks.get("all_output_contract", False)) and bool(checks.get("all_loss_finite", False)) and bool(checks.get("all_postprocess_ok", False))

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY1 BÖLÜM3 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

