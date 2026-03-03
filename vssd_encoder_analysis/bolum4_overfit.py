"""
DENEY 3 BÖLÜM 4 — Overfit karşılaştırma (BASE encoder vs VSSD encoder).

Amaç:
1) Aynı tek örnekte encoder swap'in öğrenilebilirliğe etkisini ölçmek
2) VSSD encoder'ın gradient ve loss stabilitesini doğrulamak
"""

from __future__ import annotations

import argparse
import importlib
import random
from typing import Dict, List

import numpy as np
import torch

from config import system_configs
from db.culane import CULANE
from mask_migration.common import build_model_from_cfg, load_system_config
from models.LSTR_CULANE_2k_mamba_vssd import NonCausalMambaEncoder


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_single_sample(cfg_name: str, device: torch.device):
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)
    db_ind = int(db.db_inds[0])
    image_t, label_np, _ = db.__getitem__(db_ind, transform=True)

    images = image_t.unsqueeze(0).to(device)
    masks = torch.zeros((1, 1, images.shape[-2], images.shape[-1]), device=device)
    label_batch = torch.from_numpy(label_np.astype(np.float32)).unsqueeze(0).to(device)
    targets = [images, label_batch]

    return db_ind, images, masks, targets


def _freeze_parts_for_stage(model: torch.nn.Module, stage: str = "encoder_only") -> None:
    """
    stage='encoder_only' : sadece encoder trainable
    stage='encoder_decoder_head' : encoder + decoder + query/class/curve head trainable
    """
    for p in model.parameters():
        p.requires_grad_(False)

    if stage == "encoder_only":
        for p in model.transformer.encoder.parameters():
            p.requires_grad_(True)
        return

    if stage == "encoder_decoder_head":
        for p in model.transformer.encoder.parameters():
            p.requires_grad_(True)
        for p in model.transformer.decoder.parameters():
            p.requires_grad_(True)
        for p in model.query_embed.parameters():
            p.requires_grad_(True)
        for p in model.class_embed.parameters():
            p.requires_grad_(True)
        for p in model.specific_embed.parameters():
            p.requires_grad_(True)
        for p in model.shared_embed.parameters():
            p.requires_grad_(True)
        return

    raise ValueError(f"Unknown stage: {stage}")


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
    images: torch.Tensor,
    masks: torch.Tensor,
    targets: List[torch.Tensor],
    epochs: int,
    lr: float,
    seed: int,
    stage: str,
) -> Dict[str, float | str]:
    _set_seed(seed)
    device = images.device

    model_module = importlib.import_module(f"models.{cfg_name}")
    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=False)
    _freeze_parts_for_stage(model, stage=stage)

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
            raise RuntimeError(f"Non-finite loss at cfg={cfg_name}, epoch={ep}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
        optimizer.step()

        loss_v = float(total_loss.item())
        losses_hist.append(loss_v)

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[{cfg_name}] epoch={ep:03d} total_loss={loss_v:.6f}")

        if ep > 10 and loss_v < 0.05:
            break

    init_loss = float(losses_hist[0])
    final_loss = float(losses_hist[-1])
    reduction = float((init_loss - final_loss) / max(init_loss, 1e-12) * 100.0)

    return {
        "cfg": cfg_name,
        "stage": stage,
        "encoder_type": type(model.transformer.encoder).__name__,
        "mamba2_runtime_fallback": bool(
            isinstance(model.transformer.encoder, NonCausalMambaEncoder)
            and bool(getattr(model.transformer.encoder, "runtime_fallback_used", False))
        ),
        "init": init_loss,
        "final": final_loss,
        "reduction": reduction,
        "epochs_ran": float(len(losses_hist)),
        "total_params": float(total_params),
        "trainable_params": float(trainable_params),
    }


def run(
    base_cfg: str = "LSTR_CULANE_2k_mamba",
    vssd_cfg: str = "LSTR_CULANE_2k_mamba_vssd",
    epochs: int = 80,
    lr: float = 1e-4,
    seed: int = 0,
    stage: str = "encoder_decoder_head",
) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_ind, images, masks, targets = _prepare_single_sample(cfg_name=base_cfg, device=device)

    rows = []
    for cfg_name in [base_cfg, vssd_cfg]:
        print("\n" + "-" * 96)
        print(f"Training encoder variant: {cfg_name}")
        print("-" * 96)
        rows.append(
            _train_one_variant(
                cfg_name=cfg_name,
                images=images,
                masks=masks,
                targets=targets,
                epochs=epochs,
                lr=lr,
                seed=seed,
                stage=stage,
            )
        )

    base_row = next(r for r in rows if r["cfg"] == base_cfg)
    vssd_row = next(r for r in rows if r["cfg"] == vssd_cfg)

    checks: Dict[str, bool] = {}
    checks["base_reduces"] = bool(base_row["final"] < base_row["init"])
    checks["vssd_reduces"] = bool(vssd_row["final"] < vssd_row["init"])
    checks["base_over_20pct"] = bool(base_row["reduction"] > 20.0)
    checks["vssd_over_20pct"] = bool(vssd_row["reduction"] > 20.0)
    checks["vssd_not_catastrophic"] = bool(vssd_row["reduction"] > -5.0)
    checks["vssd_encoder_present"] = "NonCausalMambaEncoder" in str(vssd_row["encoder_type"])
    checks["vssd_mamba2_native_path"] = not bool(vssd_row.get("mamba2_runtime_fallback", False))

    print("\n" + "=" * 96)
    print("DENEY 3 BÖLÜM 4 — OVERFİT KARŞILAŞTIRMA")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"base_cfg={base_cfg}, vssd_cfg={vssd_cfg}")
    print(f"sample_db_ind={db_ind}, input={tuple(images.shape)}")
    print(f"epochs(max)={epochs}, lr={lr}, stage={stage}")

    print("\nSonuçlar:")
    for r in rows:
        print(
            f"  [{r['cfg']}] encoder={r['encoder_type']} "
            f"init={r['init']:.6f} final={r['final']:.6f} "
            f"reduction={r['reduction']:+.2f}% epochs={int(r['epochs_ran'])} "
            f"trainable={int(r['trainable_params']):,}/{int(r['total_params']):,} "
            f"m2_fallback={'yes' if bool(r.get('mamba2_runtime_fallback', False)) else 'no'}"
        )

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<22}: {'✅' if v else '🚨'}")

    print("\nKarar özeti:")
    print(f"  VSSD - BASE reduction farkı: {float(vssd_row['reduction']) - float(base_row['reduction']):+.2f} puan")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cfg", type=str, default="LSTR_CULANE_2k_mamba")
    parser.add_argument("--vssd-cfg", type=str, default="LSTR_CULANE_2k_mamba_vssd")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=str, default="encoder_decoder_head", choices=["encoder_only", "encoder_decoder_head"])
    args = parser.parse_args()

    checks = run(
        base_cfg=args.base_cfg,
        vssd_cfg=args.vssd_cfg,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        stage=args.stage,
    )
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY3 BÖLÜM4 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

