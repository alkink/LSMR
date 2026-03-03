"""
DENEY 2 BÖLÜM 1 — Hybrid decoder izole test.

Katman sırası:
  layer0: MambaDecoderLayerB_Batch
  layer1: TransformerDecoderLayer (cross-attn)
"""

import argparse
import random
from typing import Dict

import numpy as np
import torch

from mamba_decoder_analysis.decoder_variants import MambaDecoderE_Hybrid, _is_attn_like_layer


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


def run(batch: int = 2, d_model: int = 32, d_state: int = 16, d_conv: int = 4, num_heads: int = 2, dim_feedforward: int = 128, num_queries: int = 7, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = 260
    lq = num_queries
    memory = torch.randn(s, batch, d_model, device=device)
    query = torch.randn(lq, batch, d_model, device=device)
    pos = torch.randn(s, batch, d_model, device=device)
    query_pos = torch.randn(lq, batch, d_model, device=device)

    dec = MambaDecoderE_Hybrid(
        num_layers=2,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        return_intermediate=True,
        dropout=0.1,
    ).to(device)

    layer0 = dec.layers[0]
    layer1 = dec.layers[1]

    out0 = layer0(memory, query)
    out1 = layer1(out0, memory, pos=pos, query_pos=query_pos)
    out_stack = dec(query, memory, pos=pos, query_pos=query_pos)

    checks: Dict[str, bool] = {}
    checks["mamba_layer_ok"] = tuple(out0.shape) == (lq, batch, d_model)
    checks["crossattn_layer_ok"] = _is_attn_like_layer(layer1) and tuple(out1.shape) == (lq, batch, d_model)
    checks["stack_output_shape_ok"] = out_stack.dim() == 4 and tuple(out_stack[-1].shape) == (lq, batch, d_model)
    checks["no_nan"] = bool(torch.isfinite(out_stack).all().item())

    # gradient check
    dec.train()
    dec.zero_grad(set_to_none=True)
    mem_g = memory.detach().clone().requires_grad_(True)
    qry_g = query.detach().clone().requires_grad_(True)
    out_g = dec(qry_g, mem_g, pos=pos, query_pos=query_pos)
    loss = out_g[-1].pow(2).mean()
    loss.backward()

    checks["gradient_mamba_layer"] = _has_grad(layer0)
    checks["gradient_attn_layer"] = _has_grad(layer1)
    checks["memory_gradient"] = mem_g.grad is not None and float(mem_g.grad.abs().mean().item()) > 1e-12

    print("=" * 96)
    print("DENEY 2 BÖLÜM 1 — HYBRID İZOLE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"memory={tuple(memory.shape)} query={tuple(query.shape)}")
    print(f"layer0_out={tuple(out0.shape)} layer1_out={tuple(out1.shape)} stack_out={tuple(out_stack.shape)}")

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
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(
        batch=args.batch,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    ok = all(checks.values())
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY2 BÖLÜM1 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

