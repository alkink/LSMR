"""
BÖLÜM 4 — Varyant C: Bidirectional Query-aware Mamba izole test.
"""

import argparse
import random
from typing import Dict

import numpy as np
import torch

from mamba_decoder_analysis.decoder_variants import MambaDecoderLayerB_Batch, MambaDecoderLayerC


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


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def run(batch: int = 2, d_model: int = 32, d_state: int = 16, d_conv: int = 4, num_queries: int = 7, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = 260
    lq = num_queries
    memory = torch.randn(s, batch, d_model, device=device)
    query = torch.randn(lq, batch, d_model, device=device)

    layer_c = MambaDecoderLayerC(d_model=d_model, d_state=d_state, d_conv=d_conv).to(device)
    out, fwd_last, bwd_last = layer_c.forward_branches(memory, query)

    checks: Dict[str, bool] = {}
    checks["output_shape_ok"] = tuple(out.shape) == (lq, batch, d_model)
    checks["fwd_bwd_differ"] = float((fwd_last - bwd_last).abs().mean().item()) > 1e-6
    checks["no_nan"] = bool(torch.isfinite(out).all().item())

    # gradient checks
    layer_c.train()
    layer_c.zero_grad(set_to_none=True)
    mem_g = memory.detach().clone().requires_grad_(True)
    qry_g = query.detach().clone().requires_grad_(True)
    out_g, _f, _b = layer_c.forward_branches(mem_g, qry_g)
    loss = out_g.pow(2).mean()
    loss.backward()

    checks["gradient_fwd_mamba"] = _has_grad(layer_c.mamba_fwd)
    checks["gradient_bwd_mamba"] = _has_grad(layer_c.mamba_bwd)
    checks["gradient_linear"] = _has_grad(layer_c.linear_merge)

    # param count ratio C vs B (tek katman)
    layer_b = MambaDecoderLayerB_Batch(d_model=d_model, d_state=d_state, d_conv=d_conv).to(device)
    p_c = _count_params(layer_c)
    p_b = _count_params(layer_b)
    ratio = float(p_c / max(p_b, 1))
    checks["param_count_vs_B"] = ratio > 1.8

    print("=" * 96)
    print("BÖLÜM 4 — VARYANT C İZOLE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"memory={tuple(memory.shape)} query={tuple(query.shape)} out={tuple(out.shape)}")
    print(f"fwd_bwd mean diff : {float((fwd_last - bwd_last).abs().mean().item()):.6e}")
    print(f"param_count B={p_b:,}  C={p_c:,}  ratio={ratio:.3f}")

    print("\nKontroller:")
    for k, v in checks.items():
        if k == "param_count_vs_B":
            print(f"  {k:<20}: {'✅' if v else '🚨'} (ratio={ratio:.3f})")
        else:
            print(f"  {k:<20}: {'✅' if v else '🚨'}")

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
    print(f"BÖLÜM 4 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

