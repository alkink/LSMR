"""
BÖLÜM 0 — Mevcut Transformer Decoder davranışını kaydet.

Testler:
0.1 — Shape kontratı
0.2 — Cross-attention entropy
0.3 — Gradient akışı
0.4 — Decoder katmanları arası output drift
"""

import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mask_migration.common import build_model_from_cfg, load_system_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_to_transformer(model, images: torch.Tensor, masks: torch.Tensor):
    p = model.conv1(images)
    p = model.bn1(p)
    p = model.relu(p)
    p = model.maxpool(p)
    p = model.layer1(p)
    p = model.layer2(p)
    p = model.layer3(p)
    p = model.layer4(p)

    pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
    pos = model.position_embedding(p, pmasks)
    src = model.input_proj(p)
    hs, memory, weights = model.transformer(src, pmasks, model.query_embed.weight, pos)
    return hs, memory, weights


def _collect_cross_attn_weights(decoder) -> Tuple[List[Tuple[int, torch.Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    cache: List[Tuple[int, torch.Tensor]] = []
    hooks = []

    for idx, layer in enumerate(decoder.layers):
        if not hasattr(layer, "multihead_attn"):
            continue

        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, out):
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    cache.append((layer_idx, out[1].detach()))
            return _hook

        hooks.append(layer.multihead_attn.register_forward_hook(_make_hook(idx)))

    return cache, hooks


def _attn_entropy(attn_w: torch.Tensor) -> float:
    # expected: (B, L, S) or (B, H, L, S)
    w = attn_w.float()
    if w.dim() == 4:
        w = w.mean(dim=1)
    if w.dim() != 3:
        return float("nan")

    p = w.clamp(min=1e-12)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    ent = -(p * torch.log(p)).sum(dim=-1)  # (B, L)
    return float(ent.mean().item())


def _has_grad(module: torch.nn.Module, eps: float = 1e-12) -> bool:
    for p in module.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            continue
        if float(p.grad.abs().mean().item()) > eps:
            return True
    return False


def run(cfg_name: str = "LSTR_CULANE_2k_mamba", batch: int = 2, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_system_config(cfg_name)
    in_h, in_w = cfg["db"]["input_size"]

    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=False)

    images = torch.randn(batch, 3, in_h, in_w, device=device)
    masks = torch.zeros(batch, 1, in_h, in_w, device=device)

    # 0.1 + 0.2 + 0.4 (eval pass)
    model.eval()
    attn_cache, hooks = _collect_cross_attn_weights(model.transformer.decoder)
    with torch.no_grad():
        hs_eval, memory_eval, _ = _forward_to_transformer(model, images, masks)
    for h in hooks:
        h.remove()

    memory_seq = memory_eval.flatten(2).permute(2, 0, 1)  # (S,B,C)
    query_seq = model.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)  # (L,B,C)
    out_seq = hs_eval[-1].permute(1, 0, 2)  # (L,B,C)

    expected_s = 260
    expected_l = 7
    expected_c = 32

    checks: Dict[str, bool] = {}
    checks["memory_shape_ok"] = tuple(memory_seq.shape) == (expected_s, batch, expected_c)
    checks["query_shape_ok"] = tuple(query_seq.shape) == (expected_l, batch, expected_c)
    checks["output_shape_ok"] = tuple(out_seq.shape) == (expected_l, batch, expected_c)
    checks["output_finite"] = bool(torch.isfinite(out_seq).all().item() and torch.isfinite(memory_seq).all().item())
    checks["decoder_layer_count_is4"] = len(model.transformer.decoder.layers) == 4

    # entropy
    entropies = []
    for layer_idx, w in sorted(attn_cache, key=lambda x: x[0]):
        ent = _attn_entropy(w)
        entropies.append((layer_idx, ent))

    max_entropy = math.log(float(max(memory_seq.shape[0], 1)))
    if len(entropies) > 0:
        checks["attn_not_uniform"] = all(np.isfinite(e) and (e < max_entropy - 1e-3) for _, e in entropies)
    else:
        checks["attn_not_uniform"] = False

    # drift
    drifts = []
    if hs_eval.dim() == 4 and hs_eval.shape[0] > 1:
        for i in range(hs_eval.shape[0] - 1):
            drifts.append(float((hs_eval[i + 1] - hs_eval[i]).abs().mean().item()))
        checks["layers_change_output"] = all(d > 1e-6 for d in drifts)
    else:
        checks["layers_change_output"] = True

    # 0.3 gradient pass
    model.train()
    model.zero_grad(set_to_none=True)
    hs_train, _memory_train, _ = _forward_to_transformer(model, images, masks)
    loss = hs_train[-1].pow(2).mean()
    loss.backward()

    dec_layers = len(model.transformer.decoder.layers)
    checks["gradient_decoder_layer0"] = _has_grad(model.transformer.decoder.layers[0]) if dec_layers > 0 else False

    # Promptta layer3 istendi; modelde katman sayısı daha azsa son katmanı kullan.
    idx_l3 = min(3, max(dec_layers - 1, 0))
    checks["gradient_decoder_layer3"] = _has_grad(model.transformer.decoder.layers[idx_l3]) if dec_layers > 0 else False
    checks["gradient_reaches_backbone"] = (
        model.conv1.weight.grad is not None
        and bool(torch.isfinite(model.conv1.weight.grad).all().item())
        and float(model.conv1.weight.grad.abs().mean().item()) > 1e-12
    )

    print("=" * 96)
    print("BÖLÜM 0 — BASELINE DECODER KONTRATI")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Input: {(batch, 3, in_h, in_w)}")
    print(f"memory_seq: {tuple(memory_seq.shape)}")
    print(f"query_seq : {tuple(query_seq.shape)}")
    print(f"out_seq   : {tuple(out_seq.shape)}")

    if len(entropies) > 0:
        print("\nCross-attention entropy (layer bazlı):")
        for idx, ent in entropies:
            print(f"  layer{idx}: {ent:.6f}  (uniform upper-bound ~ {max_entropy:.6f})")
    else:
        print("\nCross-attention entropy: ölçülemedi (hook çıktısı yok)")

    if len(drifts) > 0:
        print("\nKatmanlar arası output drift:")
        for i, d in enumerate(drifts):
            print(f"  layer{i}->{i+1}: {d:.6e}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<26}: {'✅' if v else '🚨'}")

    print(f"\nNot: gradient_decoder_layer3 kontrolü mevcut katman sayısına göre layer{idx_l3} üzerinde yapıldı.")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, batch=args.batch, seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"BÖLÜM 0 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

