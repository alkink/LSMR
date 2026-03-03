"""
BÖLÜM 2 — VSSD / non-causal SSM izole test.
"""

import argparse
import random
import time
from typing import Dict

import numpy as np
import torch

from mamba_decoder_analysis.decoder_variants import _build_mamba2, _ensure_stride8_channel_last


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NonCausalMambaEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 32, d_state: int = 64, headdim: int = 32, num_layers: int = 2):
        super().__init__()
        self.fwd = torch.nn.ModuleList()
        self.bwd = torch.nn.ModuleList()
        self.kw = []
        for _ in range(num_layers):
            fwd, kw_f = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=4, headdim=headdim)
            bwd, kw_b = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=4, headdim=headdim)
            self.fwd.append(fwd)
            self.bwd.append(bwd)
            self.kw.append((kw_f, kw_b))
        self.merge = torch.nn.ModuleList([torch.nn.Linear(d_model * 2, d_model) for _ in range(num_layers)])
        self.norm = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x_bsc: torch.Tensor) -> torch.Tensor:
        x = _ensure_stride8_channel_last(x_bsc.contiguous())
        for f, b, m, n in zip(self.fwd, self.bwd, self.merge, self.norm):
            xf = f(x)
            xb = torch.flip(b(torch.flip(x, dims=[1]).contiguous()), dims=[1])
            x = n(x + m(torch.cat([xf, xb], dim=-1)))
        return x


def _bench(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def run(batch: int = 2, seq: int = 260, d_model: int = 32, d_state: int = 64, headdim: int = 32, seed: int = 0) -> Dict[str, object]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = _ensure_stride8_channel_last(torch.randn(batch, seq, d_model, device=device).contiguous())
    enc = NonCausalMambaEncoder(d_model=d_model, d_state=d_state, headdim=headdim, num_layers=2).to(device)
    y = enc(x)

    checks: Dict[str, object] = {}
    checks["output_shape_ok"] = tuple(y.shape) == (batch, seq, d_model)
    checks["no_nan"] = bool(torch.isfinite(y).all().item())

    # grad
    enc.zero_grad(set_to_none=True)
    xg = _ensure_stride8_channel_last(x.detach().clone().contiguous()).requires_grad_(True)
    yg = enc(xg)
    loss = yg.pow(2).mean()
    loss.backward()
    grad_ok = any((p.grad is not None) and torch.isfinite(p.grad).all() and float(p.grad.abs().mean().item()) > 1e-12 for p in enc.parameters())
    checks["gradient_flow"] = grad_ok and (xg.grad is not None) and float(xg.grad.abs().mean().item()) > 1e-12

    # baseline bidir mamba1 encoder (izole basit hız karşılaştırma)
    from mamba_ssm import Mamba
    m1f = Mamba(d_model=d_model, d_state=16, d_conv=4).to(device)
    m1b = Mamba(d_model=d_model, d_state=16, d_conv=4).to(device)

    def _bidir_m1(inp):
        yf = m1f(inp)
        yb = torch.flip(m1b(torch.flip(inp, dims=[1])), dims=[1])
        return yf + yb

    t_bidir_m1 = _bench(lambda: _bidir_m1(x))
    t_vssd = _bench(lambda: enc(x))
    checks["speedup_vs_bidir"] = float(t_bidir_m1 / max(t_vssd, 1e-12))

    # kalite proxy: sample içi token cosine çeşitliliği
    yy = y[0]
    yy_n = torch.nn.functional.normalize(yy, dim=-1)
    sim = yy_n @ yy_n.t()
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    quality = float(off.mean().item())
    checks["quality_vs_bidir"] = quality

    print("=" * 96)
    print("DENEY 3 BÖLÜM 2 — VSSD İZOLE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"speedup_vs_bidir={checks['speedup_vs_bidir']:.3f}x")
    print(f"quality_proxy(avg_off_diag_cos)={quality:.4f}")

    print("\nKontroller:")
    for k in ["output_shape_ok", "no_nan", "gradient_flow"]:
        print(f"  {k:<18}: {'✅' if checks[k] else '🚨'}")
    print(f"  {'speedup_vs_bidir':<18}: {checks['speedup_vs_bidir']:.3f}")
    print(f"  {'quality_vs_bidir':<18}: {quality:.4f}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=260)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        seq=args.seq,
        d_model=args.d_model,
        d_state=args.d_state,
        headdim=args.headdim,
        seed=args.seed,
    )
    ok = bool(checks.get("output_shape_ok", False)) and bool(checks.get("no_nan", False)) and bool(checks.get("gradient_flow", False))
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY3 BÖLÜM2 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

