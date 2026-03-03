"""
BÖLÜM 2 — Varyant A: Prefix Concat izole test.
"""

import argparse
import random
from typing import Dict

import numpy as np
import torch

from mamba_decoder_analysis.decoder_variants import MambaDecoderA, MambaDecoderLayerA


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _has_grad(t: torch.Tensor, eps: float = 1e-12) -> bool:
    return t.grad is not None and bool(torch.isfinite(t.grad).all().item()) and float(t.grad.abs().mean().item()) > eps


def _param_has_grad(module: torch.nn.Module, eps: float = 1e-12) -> bool:
    for p in module.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().mean().item()) > eps:
            return True
    return False


def run(batch: int = 2, d_model: int = 32, d_state: int = 16, d_conv: int = 4, num_queries: int = 7, layers: int = 4, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = 260
    lq = num_queries
    memory = torch.randn(s, batch, d_model, device=device)
    query = torch.randn(lq, batch, d_model, device=device)

    # 2.1 single layer
    layer = MambaDecoderLayerA(d_model=d_model, d_state=d_state, d_conv=d_conv).to(device)
    out1 = layer(memory, query)

    # 2.2 multi-layer
    stack = MambaDecoderA(
        num_layers=layers,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        return_intermediate=True,
    ).to(device)
    out_all = stack(query, memory)

    checks: Dict[str, bool] = {}
    checks["output_shape_ok"] = tuple(out1.shape) == (lq, batch, d_model)
    checks["4layer_ok"] = out_all.dim() == 4 and tuple(out_all[-1].shape) == (lq, batch, d_model)
    checks["no_nan"] = bool(torch.isfinite(out1).all().item() and torch.isfinite(out_all).all().item())

    # 2.3 residual effect
    with torch.no_grad():
        combined = torch.cat([memory, query], dim=0)
        raw = layer.mamba(combined.permute(1, 0, 2)).permute(1, 0, 2)[-lq:]
        no_res = layer.norm(raw)
        res = layer(memory, query)
        residual_delta = float((res - no_res).abs().mean().item())
    checks["residual_effective"] = residual_delta > 1e-6

    # 2.4 gradient to memory/query
    layer.train()
    layer.zero_grad(set_to_none=True)
    mem_g = memory.detach().clone().requires_grad_(True)
    qry_g = query.detach().clone().requires_grad_(True)
    out_g = layer(mem_g, qry_g)
    loss = out_g.pow(2).mean()
    loss.backward()

    checks["memory_gradient"] = _has_grad(mem_g)
    checks["query_gradient"] = _has_grad(qry_g)
    checks["param_gradient"] = _param_has_grad(layer)

    # 2.5 cross-query coupling
    with torch.no_grad():
        base = layer(memory, query)
        q2 = query.clone()
        q2[0] = q2[0] + 1.0
        pert = layer(memory, q2)
        other = (pert[1:] - base[1:]).abs().mean().item()
        self_change = (pert[0] - base[0]).abs().mean().item()
        coupling = float(other / (self_change + 1e-12))

    print("=" * 96)
    print("BÖLÜM 2 — VARYANT A İZOLE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"memory={tuple(memory.shape)} query={tuple(query.shape)} out1={tuple(out1.shape)}")
    print(f"stack output={tuple(out_all.shape)}")
    print(f"residual_delta={residual_delta:.6e}")
    print(f"cross_query_coupling={coupling:.6f}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<20}: {'✅' if v else '🚨'}")

    print(f"  {'cross_query_coupling':<20}: {coupling:.6f}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=7)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        num_queries=args.num_queries,
        layers=args.layers,
        seed=args.seed,
    )
    ok = all(checks.values())
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"BÖLÜM 2 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

