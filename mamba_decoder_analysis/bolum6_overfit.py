"""
BÖLÜM 6 — Overfit karşılaştırması (BASE vs Decoder A/B/C).

Amaç:
1) Aynı tek örnek üzerinde farklı decoder tasarımlarının öğrenme kabiliyetini ölçmek
2) Decoder değişiminden sonra train-loss hattının stabil kaldığını doğrulamak
3) En iyi/uygun varyantı pre-integration aşaması için seçmek
"""

import argparse
import importlib
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.common import build_model_from_cfg, load_system_config
from mamba_decoder_analysis.decoder_variants import build_variant_decoder


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_variant_int_map(spec: str) -> Dict[str, int]:
    """
    Örn: "D:64,E:32" -> {"D": 64, "E": 32}
    """
    out: Dict[str, int] = {}
    raw = str(spec).strip()
    if not raw:
        return out

    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --d-state-map token: '{item}'. Beklenen biçim: VARIANT:INT")
        key, val = item.split(":", 1)
        out[key.strip().upper()] = int(val.strip())
    return out


def _variant_d_state(variant: str, default_d_state: int, d_state_map: Optional[Dict[str, int]]) -> int:
    if d_state_map is None:
        return int(default_d_state)
    return int(d_state_map.get(str(variant).upper(), int(default_d_state)))


def _replace_decoder(model: torch.nn.Module, variant: str, d_state: int, d_conv: int) -> None:
    if variant.upper() == "BASE":
        return

    old_decoder = model.transformer.decoder
    num_layers = len(old_decoder.layers)
    d_model = int(getattr(model.transformer, "d_model", system_configs.attn_dim))
    return_intermediate = bool(getattr(old_decoder, "return_intermediate", True))

    new_decoder = build_variant_decoder(
        variant=variant,
        num_layers=num_layers,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        return_intermediate=return_intermediate,
        headdim=32,
        num_heads=int(system_configs.num_heads),
        dim_feedforward=int(system_configs.dim_feedforward),
    )
    model.transformer.decoder = new_decoder.to(next(model.parameters()).device)


def _prepare_single_sample(cfg_name: str, device: torch.device):
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)
    db_ind = int(db.db_inds[0])
    image_t, label_np, _ = db.__getitem__(db_ind, transform=True)

    images = image_t.unsqueeze(0).to(device)
    masks = torch.zeros((1, 1, images.shape[-2], images.shape[-1]), device=device)
    # AELoss beklentisi: targets[1:] elemanları (B, L, D) olmalı.
    # B=1 olduğu için tek bir label tensor yeterli.
    label_batch = torch.from_numpy(label_np.astype(np.float32)).unsqueeze(0).to(device)
    targets = [images, label_batch]

    return db_ind, images, masks, targets


def _freeze_non_decoder_parts(model: torch.nn.Module) -> None:
    """
    Decoder etkisini daha net görmek için backbone + encoder dondur.
    Query/head katmanları açık kalır ki model tek örneği gerçekten öğrenebilsin.
    """
    freeze_prefixes = [
        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "input_proj",
        "transformer.encoder",
    ]
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in freeze_prefixes):
            p.requires_grad_(False)


def _collect_trainable_stats(model: torch.nn.Module):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _train_one_variant(
    cfg_name: str,
    variant: str,
    images: torch.Tensor,
    masks: torch.Tensor,
    targets: List[torch.Tensor],
    d_state: int,
    d_conv: int,
    epochs: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    _set_seed(seed)
    device = images.device

    model_module = importlib.import_module(f"models.{cfg_name}")
    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=False)
    _replace_decoder(model=model, variant=variant, d_state=d_state, d_conv=d_conv)
    _freeze_non_decoder_parts(model)

    criterion = model_module.loss().to(device)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(optim_params, lr=lr)

    total_params, trainable_params = _collect_trainable_stats(model)

    losses_hist: List[float] = []
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        outputs, _weights = model._train(images, masks)
        loss_pack = criterion(
            iteration=0,
            save=False,
            viz_split="train",
            outputs=outputs,
            targets=targets,
        )
        total_loss = loss_pack[0]

        if not torch.isfinite(total_loss).all():
            raise RuntimeError(f"Non-finite loss at variant={variant}, epoch={ep}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
        optimizer.step()

        loss_v = float(total_loss.item())
        losses_hist.append(loss_v)

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[{variant}] epoch={ep:03d} total_loss={loss_v:.6f}")

        # erken durdurma: tek örnekte aşırı iyi fit
        if ep > 10 and loss_v < 0.03:
            break

    init_loss = float(losses_hist[0])
    final_loss = float(losses_hist[-1])
    reduction = float((init_loss - final_loss) / max(init_loss, 1e-12) * 100.0)

    return {
        "d_state": float(d_state),
        "init": init_loss,
        "final": final_loss,
        "delta": float(final_loss - init_loss),
        "reduction": reduction,
        "epochs_ran": float(len(losses_hist)),
        "total_params": float(total_params),
        "trainable_params": float(trainable_params),
    }


def run(
    cfg_name: str = "LSTR_CULANE_2k_mamba",
    epochs: int = 80,
    lr: float = 1e-4,
    d_state: int = 16,
    d_conv: int = 4,
    variants: Optional[List[str]] = None,
    d_state_map: Optional[Dict[str, int]] = None,
    seed: int = 0,
) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_ind, images, masks, targets = _prepare_single_sample(cfg_name=cfg_name, device=device)

    variants_run = variants if variants is not None else ["BASE", "A", "B", "C"]
    rows = []
    for i, variant in enumerate(variants_run):
        variant_u = str(variant).upper()
        d_state_eff = _variant_d_state(variant_u, d_state, d_state_map)
        print("\n" + "-" * 96)
        print(f"Training variant {variant_u} ({i + 1}/{len(variants_run)}) [d_state={d_state_eff}]")
        print("-" * 96)
        stat = _train_one_variant(
            cfg_name=cfg_name,
            variant=variant_u,
            images=images,
            masks=masks,
            targets=targets,
            d_state=d_state_eff,
            d_conv=d_conv,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        row = {"variant": variant_u, **stat}
        rows.append(row)

    best = max(rows, key=lambda x: x["reduction"])
    best_final = min(rows, key=lambda x: x["final"])
    baseline = next((r for r in rows if str(r["variant"]).upper() == "BASE"), None)
    variant_set = {str(v).upper() for v in variants_run}

    checks: Dict[str, bool] = {}
    checks["all_variants_reduce"] = all(r["final"] < r["init"] for r in rows)
    checks["base_over_20pct"] = True if baseline is None else bool(baseline["reduction"] > 20.0)
    checks["at_least_one_mamba_over_20pct"] = any(
        r["reduction"] > 20.0 for r in rows if str(r["variant"]).upper() != "BASE"
    )
    checks["best_variant_exists"] = best["variant"] in variant_set
    checks["best_final_variant_exists"] = best_final["variant"] in variant_set
    checks["no_variant_catastrophic"] = all(r["reduction"] > -5.0 for r in rows)

    print("\n" + "=" * 96)
    print("BÖLÜM 6 — OVERFİT KARŞILAŞTIRMASI")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Sample db_ind: {db_ind}")
    print(f"Input image shape: {tuple(images.shape)}")
    print(f"epochs(max)={epochs}, lr={lr}, d_state(default)={d_state}, d_conv={d_conv}")
    if d_state_map:
        print(f"d_state_map={d_state_map}")

    print("\nVaryant sonuçları:")
    for r in rows:
        print(
            f"  [{r['variant']}] "
            f"d_state={int(r['d_state'])} "
            f"init={r['init']:.6f} "
            f"final={r['final']:.6f} "
            f"delta={r['delta']:+.6f} "
            f"reduction={r['reduction']:+.2f}% "
            f"epochs={int(r['epochs_ran'])} "
            f"trainable={int(r['trainable_params']):,}/{int(r['total_params']):,}"
        )

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<30}: {'✅' if v else '🚨'}")

    print("\nKarar özeti:")
    print(f"  En iyi reduction : {best['variant']} ({best['reduction']:+.2f}%)")
    print(f"  En iyi final_loss: {best_final['variant']} ({best_final['final']:.6f})")
    if baseline is not None:
        print(f"  REDUCTION farkı  : {best['reduction'] - baseline['reduction']:+.2f} puan (best_vs_BASE)")
        print(f"  FINAL_LOSS farkı : {best_final['final'] - baseline['final']:+.6f} (best_final_vs_BASE)")
    else:
        print("  BASE ile farkı   : N/A (bu çalıştırmada BASE varyantı dahil edilmedi)")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-state-map", type=str, default="")
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--variants", type=str, default="BASE,A,B,C")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    d_state_map = _parse_variant_int_map(args.d_state_map)

    checks = run(
        cfg_name=args.cfg,
        epochs=args.epochs,
        lr=args.lr,
        d_state=args.d_state,
        d_conv=args.d_conv,
        variants=variants,
        d_state_map=d_state_map,
        seed=args.seed,
    )
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"BÖLÜM 6 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

