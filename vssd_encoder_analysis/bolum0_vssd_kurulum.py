"""
BÖLÜM 0 — VSSD kurulum ve import kontrolü.
"""

import argparse
from typing import Dict

import torch

from mamba_decoder_analysis.decoder_variants import _build_mamba2, _ensure_stride8_channel_last


class NonCausalMamba2Approx(torch.nn.Module):
    def __init__(self, d_model: int = 32, d_state: int = 64, headdim: int = 32):
        super().__init__()
        self.fwd, self.fwd_kw = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=4, headdim=headdim)
        self.bwd, self.bwd_kw = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=4, headdim=headdim)
        self.merge = torch.nn.Linear(d_model * 2, d_model)

    def forward(self, x_bsc: torch.Tensor) -> torch.Tensor:
        x_bsc = _ensure_stride8_channel_last(x_bsc.contiguous())
        f = self.fwd(x_bsc)
        b = self.bwd(torch.flip(x_bsc, dims=[1]).contiguous())
        b = torch.flip(b, dims=[1])
        return self.merge(torch.cat([f, b], dim=-1))


def run(batch: int = 2, seq: int = 260, d_model: int = 32, d_state: int = 64, headdim: int = 32) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks: Dict[str, object] = {}

    try:
        import vssd  # type: ignore
        checks["vssd_available"] = True
        checks["vssd_module"] = str(vssd)
    except Exception:
        checks["vssd_available"] = False

    approx_ok = False
    out_shape_ok = False
    try:
        approx = NonCausalMamba2Approx(d_model=d_model, d_state=d_state, headdim=headdim).to(device)
        x = torch.randn(batch, seq, d_model, device=device)
        y = approx(x)
        approx_ok = True
        out_shape_ok = tuple(y.shape) == (batch, seq, d_model)
    except Exception as e:
        checks["approx_error"] = str(e)

    checks["approx_build_ok"] = approx_ok
    checks["forward_shape_ok"] = out_shape_ok

    print("=" * 96)
    print("DENEY 3 BÖLÜM 0 — VSSD KURULUM")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"vssd_available : {'✅' if checks['vssd_available'] else '⚠️'}")
    print(f"approx_build_ok : {'✅' if checks['approx_build_ok'] else '🚨'}")
    print(f"forward_shape_ok: {'✅' if checks['forward_shape_ok'] else '🚨'}")
    if "approx_error" in checks:
        print(f"approx_error    : {checks['approx_error']}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=260)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=32)
    args = parser.parse_args()

    checks = run(batch=args.batch, seq=args.seq, d_model=args.d_model, d_state=args.d_state, headdim=args.headdim)
    ok = bool(checks.get("approx_build_ok", False)) and bool(checks.get("forward_shape_ok", False))
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY3 BÖLÜM0 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

