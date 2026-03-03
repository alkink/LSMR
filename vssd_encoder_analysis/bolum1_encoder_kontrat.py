"""
BÖLÜM 1 — Mevcut BidirectionalMambaEncoder kontratını kaydet.
"""

import argparse
import random
from typing import Dict

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


def _forward_to_encoder(model, images, masks):
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

    src = model.input_proj(p).flatten(2).permute(2, 0, 1)
    pos_s = pos.flatten(2).permute(2, 0, 1)
    memory, _w = model.transformer.encoder(src, src_key_padding_mask=pmasks.flatten(1), pos=pos_s)
    return src, memory


def run(cfg_name: str = "LSTR_CULANE_2k_mamba_dec_b", batch: int = 2, seed: int = 0) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_system_config(cfg_name)
    in_h, in_w = cfg["db"]["input_size"]
    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=False)

    images = torch.randn(batch, 3, in_h, in_w, device=device)
    masks = torch.zeros(batch, 1, in_h, in_w, device=device)

    src, memory = _forward_to_encoder(model, images, masks)

    checks: Dict[str, bool] = {}
    checks["input_shape_ok"] = tuple(src.shape) == (260, batch, 32)
    checks["output_shape_ok"] = tuple(memory.shape) == (260, batch, 32)

    # basit feature diversity
    flat = memory.permute(1, 0, 2).reshape(batch, -1)
    norm = torch.nn.functional.normalize(flat, dim=-1)
    sim = norm @ norm.t()
    off = sim[~torch.eye(batch, dtype=torch.bool, device=sim.device)]
    avg_off = float(off.mean().item()) if off.numel() > 0 else 0.0
    checks["feature_diversity"] = avg_off < 0.85

    # gradient path
    model.zero_grad(set_to_none=True)
    loss = memory.pow(2).mean()
    loss.backward()
    checks["gradient_encoder"] = any(
        (p.grad is not None) and torch.isfinite(p.grad).all() and float(p.grad.abs().mean().item()) > 1e-12
        for p in model.transformer.encoder.parameters()
    )
    # Not: Bu backward yalnızca encoder çıktısı (memory) üstünden tanımlı.
    # Bu nedenle query_embed gradyanı beklenmez; bu bir hata değil, beklenen davranıştır.
    checks["decoder_grad_expected_false"] = model.query_embed.weight.grad is None

    print("=" * 96)
    print("DENEY 3 BÖLÜM 1 — ENCODER KONTRAT")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"src={tuple(src.shape)} memory={tuple(memory.shape)}")
    print(f"avg_off_diag_cos={avg_off:.4f}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<26}: {'✅' if v else '🚨'}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba_dec_b")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(cfg_name=args.cfg, batch=args.batch, seed=args.seed)
    ok = all(checks.values())
    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY3 BÖLÜM1 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

