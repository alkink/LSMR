"""
DENEY 1 BÖLÜM 2 — Varyant D (Query-aware Mamba2) izole test.
"""

import argparse
import os
import random
import time
from typing import Dict

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from mamba_decoder_analysis.decoder_variants import MambaDecoderLayerB_Batch, MambaDecoderLayerD_Mamba2


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _has_grad(module: torch.nn.Module, eps: float = 1e-12) -> bool:
    for p in module.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().mean().item()) > eps:
            return True
    return False


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


def _toy_overfit_step(layer: torch.nn.Module, memory: torch.Tensor, query: torch.Tensor, steps: int = 60, lr: float = 1e-3):
    target = torch.zeros_like(query)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = layer(memory, query)
        loss = (out - target).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.item()))
    init, final = hist[0], hist[-1]
    red = (init - final) / max(init, 1e-12) * 100.0
    return init, final, red


def run(batch: int = 2, d_model: int = 32, d_state: int = 64, d_conv: int = 4, headdim: int = 32, num_queries: int = 7, seed: int = 0) -> Dict[str, object]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = 260
    lq = num_queries
    memory = torch.randn(s, batch, d_model, device=device)
    query = torch.randn(lq, batch, d_model, device=device)

    layer_b = MambaDecoderLayerB_Batch(d_model=d_model, d_state=16, d_conv=d_conv).to(device)
    layer_d = MambaDecoderLayerD_Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=headdim).to(device)

    out_b = layer_b(memory, query)
    out_d = layer_d(memory, query)

    max_diff = float((out_b - out_d).abs().max().item())
    mean_diff = float((out_b - out_d).abs().mean().item())

    checks: Dict[str, object] = {}
    checks["output_shape_ok"] = tuple(out_d.shape) == (lq, batch, d_model)
    checks["no_nan"] = bool(torch.isfinite(out_d).all().item())
    checks["pipeline_compat"] = tuple(out_d.shape) == tuple(out_b.shape)
    checks["mean_diff_vs_B"] = mean_diff
    checks["max_diff_vs_B"] = max_diff

    # gradient
    layer_d.train()
    layer_d.zero_grad(set_to_none=True)
    mem_g = memory.detach().clone().requires_grad_(True)
    qry_g = query.detach().clone().requires_grad_(True)
    loss = layer_d(mem_g, qry_g).pow(2).mean()
    loss.backward()
    checks["gradient_flow"] = _has_grad(layer_d) and mem_g.grad is not None and qry_g.grad is not None

    # speed
    t_b = _bench(lambda: layer_b(memory, query))
    t_d = _bench(lambda: layer_d(memory, query))
    checks["speedup_vs_B"] = float(t_b / max(t_d, 1e-12))

    # toy overfit compare
    b_init, b_final, b_red = _toy_overfit_step(
        MambaDecoderLayerB_Batch(d_model=d_model, d_state=16, d_conv=d_conv).to(device),
        memory,
        query,
        steps=60,
        lr=1e-3,
    )
    d_init, d_final, d_red = _toy_overfit_step(
        MambaDecoderLayerD_Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=headdim).to(device),
        memory,
        query,
        steps=60,
        lr=1e-3,
    )
    checks["overfit_reduction"] = float(d_red)
    checks["overfit_vs_B"] = float(d_red - b_red)

    print("=" * 96)
    print("DENEY 1 BÖLÜM 2 — VARYANT D (MAMBA2)")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"memory={tuple(memory.shape)} query={tuple(query.shape)}")
    print(f"out_B={tuple(out_b.shape)} out_D={tuple(out_d.shape)}")
    print(f"mean_diff_vs_B={mean_diff:.3e}, max_diff_vs_B={max_diff:.3e}")
    print(f"time_B={t_b*1e3:.3f} ms, time_D={t_d*1e3:.3f} ms, speedup_vs_B={checks['speedup_vs_B']:.3f}x")
    print(f"toy_overfit B: {b_init:.5f}->{b_final:.5f} ({b_red:.2f}%)")
    print(f"toy_overfit D: {d_init:.5f}->{d_final:.5f} ({d_red:.2f}%)")

    print("\nKontroller:")
    for k in ["output_shape_ok", "no_nan", "gradient_flow", "pipeline_compat"]:
        print(f"  {k:<18}: {'✅' if checks[k] else '🚨'}")
    print(f"  {'overfit_reduction':<18}: {checks['overfit_reduction']:.2f}")
    print(f"  {'speedup_vs_B':<18}: {checks['speedup_vs_B']:.3f}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        headdim=args.headdim,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    hard_keys = ["output_shape_ok", "no_nan", "gradient_flow", "pipeline_compat"]
    ok = all(bool(checks.get(k, False)) for k in hard_keys)
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY1 BÖLÜM2 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

