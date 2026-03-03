"""
BÖLÜM 0 — Mevcut encoder/decoder shape analizi.
"""
import argparse
from collections import OrderedDict

import torch

from mask_migration.common import require_cuda, load_system_config, build_model_from_cfg


def _shape_of(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_of(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}
    return type(x).__name__


def main(cfg_name: str = "LSTR_CULANE_2k_mamba", batch: int = 2):
    device = require_cuda()
    cfg = load_system_config(cfg_name)
    inp_h, inp_w = cfg["db"]["input_size"]

    net = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=True)

    shapes = OrderedDict()

    def make_hook(name):
        def _hook(_m, _inp, out):
            shapes[name] = _shape_of(out)

        return _hook

    hooks = []
    key_names = (
        "mamba",
        "encoder",
        "decoder",
        "input_proj",
        "class_embed",
        "specific_embed",
        "shared_embed",
    )
    for name, mod in net.named_modules():
        if any(k in name for k in key_names):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    dummy_img = torch.randn(batch, 3, inp_h, inp_w, device=device)
    dummy_mask = torch.zeros(batch, 1, inp_h, inp_w, device=device)

    with torch.no_grad():
        out, weights = net._train(dummy_img, dummy_mask)

    for h in hooks:
        h.remove()

    print("=" * 90)
    print("BÖLÜM 0 — ENCODER/DECODER ARA BOYUTLARI")
    print("=" * 90)
    for name, shp in shapes.items():
        print(f"  {name:<60} -> {shp}")

    print("\n" + "=" * 90)
    print("MODEL ÇIKTILARI")
    print("=" * 90)
    for k, v in out.items():
        print(f"  {k:<20}: {_shape_of(v)}")
    print(f"  {'encoder_weights':<20}: {_shape_of(weights)}")

    if "pred_curves" in out:
        print("\nNot: Mevcut model polinom tabanlı (pred_curves) çalışıyor.")

    print("\n" + "=" * 90)
    print("RAPOR")
    print("=" * 90)
    print("BÖLÜM 0 — Shape analizi:      ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="LSTR_CULANE_2k_mamba", type=str)
    parser.add_argument("--batch", default=2, type=int)
    args = parser.parse_args()
    main(cfg_name=args.cfg, batch=args.batch)

