"""
BÖLÜM 1 — Proje Mamba implementasyonunu izole test et.

Testler:
1.1 — (261, B, 32) girdi
1.2 — (267, B, 32) girdi
1.3 — Causal özellik (son token perturbasyonu ilk token'ı etkilememeli)
1.4 — Gradient flow
1.5 — Determinizm (eval)
"""

import argparse
import random
from typing import Dict

import numpy as np
import torch
from mamba_ssm import Mamba


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def run(batch: int = 2, d_model: int = 32, d_state: int = 16, d_conv: int = 4, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv).to(device)

    # 1.1 / 1.2
    x261 = torch.randn(batch, 261, d_model, device=device)
    x267 = torch.randn(batch, 267, d_model, device=device)

    y261 = mamba(x261)
    y267 = mamba(x267)

    checks: Dict[str, bool] = {}
    checks["forward_261_ok"] = tuple(y261.shape) == tuple(x261.shape)
    checks["forward_267_ok"] = tuple(y267.shape) == tuple(x267.shape)
    checks["no_nan_inf"] = _finite(y261) and _finite(y267)

    # 1.3 causal
    with torch.no_grad():
        x = torch.randn(batch, 261, d_model, device=device)
        y1 = mamba(x)
        x2 = x.clone()
        x2[:, -1, :] = x2[:, -1, :] + 10.0
        y2 = mamba(x2)
        first_diff = float((y1[:, 0, :] - y2[:, 0, :]).abs().mean().item())
        checks["causal_ok"] = first_diff < 1e-6

    # 1.4 gradient
    mamba.train()
    mamba.zero_grad(set_to_none=True)
    xg = torch.randn(batch, 261, d_model, device=device, requires_grad=True)
    yg = mamba(xg)
    loss = yg.pow(2).mean()
    loss.backward()

    grad_flow = False
    for p in mamba.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().mean().item()) > 1e-12:
            grad_flow = True
            break
    checks["gradient_flow"] = grad_flow and xg.grad is not None and float(xg.grad.abs().mean().item()) > 1e-12

    # 1.5 deterministic in eval
    mamba.eval()
    with torch.no_grad():
        xd = torch.randn(batch, 261, d_model, device=device)
        yd1 = mamba(xd)
        yd2 = mamba(xd)
        det_diff = float((yd1 - yd2).abs().max().item())
    checks["deterministic"] = det_diff < 1e-8

    print("=" * 96)
    print("BÖLÜM 1 — MAMBA İZOLE TEST")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"d_model={d_model}, d_state={d_state}, d_conv={d_conv}, batch={batch}")
    print(f"x261 -> y261: {tuple(x261.shape)} -> {tuple(y261.shape)}")
    print(f"x267 -> y267: {tuple(x267.shape)} -> {tuple(y267.shape)}")
    print(f"causal first-token diff: {first_diff:.3e}")
    print(f"deterministic max diff : {det_diff:.3e}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<16}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        seed=args.seed,
    )
    ok = all(checks.values())
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"BÖLÜM 1 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

