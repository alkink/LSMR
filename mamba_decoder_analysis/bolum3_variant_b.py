"""
BÖLÜM 3 — Varyant B: Query-aware Mamba izole test.
"""

import argparse
import random
import time
from typing import Dict

import numpy as np
import torch

from mamba_decoder_analysis.decoder_variants import MambaDecoderLayerB_Batch, MambaDecoderLayerB_Naive


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


def run(batch: int = 2, d_model: int = 32, d_state: int = 16, d_conv: int = 4, num_queries: int = 7, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = 260
    lq = num_queries
    memory = torch.randn(s, batch, d_model, device=device)
    query = torch.randn(lq, batch, d_model, device=device)

    naive = MambaDecoderLayerB_Naive(d_model=d_model, d_state=d_state, d_conv=d_conv).to(device)
    batch_m = MambaDecoderLayerB_Batch(d_model=d_model, d_state=d_state, d_conv=d_conv).to(device)

    # ağırlık eşitle
    batch_m.load_state_dict(naive.state_dict(), strict=True)

    out_naive = naive(memory, query)
    out_batch = batch_m(memory, query)

    max_diff = float((out_naive - out_batch).abs().max().item())

    checks: Dict[str, bool] = {}
    checks["naive_output_shape_ok"] = tuple(out_naive.shape) == (lq, batch, d_model)
    checks["batch_output_shape_ok"] = tuple(out_batch.shape) == (lq, batch, d_model)
    checks["naive_batch_equiv"] = max_diff < 1e-5
    checks["no_nan"] = bool(torch.isfinite(out_naive).all().item() and torch.isfinite(out_batch).all().item())

    # hız
    t_naive = _bench(lambda: naive(memory, query))
    t_batch = _bench(lambda: batch_m(memory, query))
    speedup = float(t_naive / max(t_batch, 1e-12))

    # gradient naive
    naive.train()
    naive.zero_grad(set_to_none=True)
    mem_n = memory.detach().clone().requires_grad_(True)
    qry_n = query.detach().clone().requires_grad_(True)
    loss_n = naive(mem_n, qry_n).pow(2).mean()
    loss_n.backward()
    checks["gradient_naive"] = _has_grad(naive) and mem_n.grad is not None and qry_n.grad is not None

    # gradient batch
    batch_m.train()
    batch_m.zero_grad(set_to_none=True)
    mem_b = memory.detach().clone().requires_grad_(True)
    qry_b = query.detach().clone().requires_grad_(True)
    loss_b = batch_m(mem_b, qry_b).pow(2).mean()
    loss_b.backward()
    checks["gradient_batch"] = _has_grad(batch_m) and mem_b.grad is not None and qry_b.grad is not None

    print("=" * 96)
    print("BÖLÜM 3 — VARYANT B İZOLE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"memory={tuple(memory.shape)} query={tuple(query.shape)}")
    print(f"out_naive={tuple(out_naive.shape)} out_batch={tuple(out_batch.shape)}")
    print(f"naive_batch max_diff : {max_diff:.3e}")
    print(f"naive time/iter      : {t_naive*1e3:.3f} ms")
    print(f"batch time/iter      : {t_batch*1e3:.3f} ms")
    print(f"speedup_ratio        : {speedup:.3f}x")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<22}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"BÖLÜM 3 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

