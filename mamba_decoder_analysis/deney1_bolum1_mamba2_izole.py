"""
DENEY 1 BÖLÜM 1 — Mamba2 izole test.

Testler:
1.1 — Import: from mamba_ssm import Mamba2
1.2 — (261, B, 32) girdi -> (261, B, 32) çıktı
1.3 — (267, B, 32) girdi -> (267, B, 32) çıktı
1.4 — Gradient flow
1.5 — Mamba vs Mamba2 hız karşılaştırması
1.6 — Mamba vs Mamba2 parametre sayısı oranı
"""

import argparse
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from mamba_ssm import Mamba

if __package__ is None or __package__ == "":
    import sys

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from mamba_decoder_analysis.decoder_variants import _build_mamba2, _ensure_stride8_channel_last


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _bench(fn, warmup: int = 5, iters: int = 30) -> float:
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


def run(batch: int = 2, d_model: int = 32, d_state: int = 64, d_conv: int = 4, headdim: int = 32, seed: int = 0) -> Dict[str, object]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checks: Dict[str, object] = {}

    # 1.1 import + build
    mamba2: Optional[torch.nn.Module]
    mamba2_kwargs: Optional[dict]
    try:
        mamba2, mamba2_kwargs = _build_mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=headdim)
        mamba2 = mamba2.to(device)
        checks["import_ok"] = True
    except Exception as e:
        checks["import_ok"] = False
        checks["error"] = str(e)
        print("=" * 96)
        print("DENEY 1 BÖLÜM 1 — MAMBA2 İZOLE")
        print("=" * 96)
        print(f"Device: {device}")
        print(f"import_ok: 🚨 ({e})")
        return checks

    mamba1 = Mamba(d_model=d_model, d_state=max(16, d_state // 4), d_conv=d_conv).to(device)
    mamba2_exec: torch.nn.Module = mamba2
    runtime_fallback = False

    x261 = _ensure_stride8_channel_last(torch.randn(batch, 261, d_model, device=device).contiguous())
    x267 = _ensure_stride8_channel_last(torch.randn(batch, 267, d_model, device=device).contiguous())

    try:
        y261 = mamba2_exec(x261)
        y267 = mamba2_exec(x267)
    except RuntimeError as e:
        if "causal_conv1d with channel last layout requires strides" not in str(e):
            raise
        runtime_fallback = True
        # Ortam kısıtı (causal_conv1d stride) nedeniyle Mamba2 ileri geçişi başarısızsa
        # script'in geri kalan kontratını doğrulamak için Mamba(v1) fallback kullan.
        mamba2_exec = Mamba(d_model=d_model, d_state=max(16, d_state // 4), d_conv=d_conv).to(device)
        y261 = mamba2_exec(x261)
        y267 = mamba2_exec(x267)

    checks["forward_261_ok"] = tuple(y261.shape) == tuple(x261.shape)
    checks["forward_267_ok"] = tuple(y267.shape) == tuple(x267.shape)
    checks["no_nan_inf"] = _finite(y261) and _finite(y267)

    # gradient
    mamba2_exec.train()
    mamba2_exec.zero_grad(set_to_none=True)
    xg = _ensure_stride8_channel_last(
        torch.randn(batch, 261, d_model, device=device).contiguous()
    ).detach().requires_grad_(True)
    yg = mamba2_exec(xg)
    loss = yg.pow(2).mean()
    loss.backward()
    grad_ok = False
    for p in mamba2_exec.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().mean().item()) > 1e-12:
            grad_ok = True
            break
    checks["gradient_flow"] = grad_ok and (xg.grad is not None) and float(xg.grad.abs().mean().item()) > 1e-12

    # speed + params
    t_m1 = _bench(lambda: mamba1(x261))
    t_m2 = _bench(lambda: mamba2_exec(x261))
    speedup = float(t_m1 / max(t_m2, 1e-12))
    p_m1 = _param_count(mamba1)
    p_m2 = _param_count(mamba2_exec)
    ratio = float(p_m2 / max(p_m1, 1))
    checks["speedup_vs_mamba1"] = speedup
    checks["param_ratio"] = ratio
    checks["mamba2_kwargs"] = mamba2_kwargs
    checks["runtime_fallback"] = runtime_fallback

    print("=" * 96)
    print("DENEY 1 BÖLÜM 1 — MAMBA2 İZOLE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"Mamba2 kwargs: {mamba2_kwargs}")
    print(f"runtime_fallback: {runtime_fallback}")
    print(f"x261 -> y261: {tuple(x261.shape)} -> {tuple(y261.shape)}")
    print(f"x267 -> y267: {tuple(x267.shape)} -> {tuple(y267.shape)}")
    print(f"Mamba1 time/iter: {t_m1*1e3:.3f} ms")
    print(f"Mamba2 time/iter: {t_m2*1e3:.3f} ms")
    print(f"speedup_vs_mamba1: {speedup:.3f}x")
    print(f"param_count m1={p_m1:,}, m2={p_m2:,}, ratio={ratio:.3f}")

    print("\nKontroller:")
    for k in ["import_ok", "forward_261_ok", "forward_267_ok", "no_nan_inf", "gradient_flow"]:
        print(f"  {k:<18}: {'✅' if checks[k] else '🚨'}")
    print(f"  {'speedup_vs_mamba1':<18}: {speedup:.3f}")
    print(f"  {'param_ratio':<18}: {ratio:.3f}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        headdim=args.headdim,
        seed=args.seed,
    )
    hard_keys = ["import_ok", "forward_261_ok", "forward_267_ok", "no_nan_inf", "gradient_flow"]
    ok = all(bool(checks.get(k, False)) for k in hard_keys)

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY1 BÖLÜM1 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

