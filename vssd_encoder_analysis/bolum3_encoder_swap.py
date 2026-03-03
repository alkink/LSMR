"""
DENEY 3 BÖLÜM 3 — Encoder swap pipeline testi (BASE vs VSSD).

Amaç:
1) Encoder değişiminden sonra model output contract korunuyor mu?
2) AELoss hattı finite kalıyor mu?
3) PostProcess -> Evaluator -> CULane writer zinciri bozulmadan çalışıyor mu?
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
from db.utils.evaluator import Evaluator
from mask_migration.common import build_model_from_cfg, load_system_config
from models.LSTR_CULANE_2k_mamba_vssd import NonCausalMambaEncoder
from sample.culane import sample_data
from test.culane import PostProcess


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_single_sample(cfg_name: str, device: torch.device):
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)

    original_batch_size = int(system_configs.batch_size)
    system_configs.update_config({"batch_size": 1})
    try:
        sampled, _ = sample_data(db, 0)
    finally:
        system_configs.update_config({"batch_size": original_batch_size})

    images = sampled["xs"][0].to(device)
    masks = sampled["xs"][1].to(device)
    targets = [y.to(device) for y in sampled["ys"]]

    db_ind = int(db.db_inds[0])
    target_sizes = torch.tensor(
        [[int(images.shape[-2]), int(images.shape[-1])]] * int(images.shape[0]),
        device=device,
        dtype=torch.long,
    )
    return db, db_ind, images, masks, targets, target_sizes


def _run_one_cfg(
    cfg_name: str,
    db: CULANE,
    db_ind: int,
    images: torch.Tensor,
    masks: torch.Tensor,
    targets: List[torch.Tensor],
    target_sizes: torch.Tensor,
    post: PostProcess,
    device: torch.device,
) -> Dict[str, object]:
    model_module = importlib.import_module(f"models.{cfg_name}")
    model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=False)

    model.train()
    outputs, enc_weights = model._train(images, masks)

    key_ok = all(k in outputs for k in ["pred_logits", "pred_curves"])
    shape_ok = (
        key_ok
        and outputs["pred_logits"].dim() == 3
        and outputs["pred_curves"].dim() == 3
        and int(outputs["pred_logits"].shape[0]) == 1
        and int(outputs["pred_curves"].shape[0]) == 1
    )

    weights_ok = (
        isinstance(enc_weights, torch.Tensor)
        and enc_weights.dim() == 3
        and int(enc_weights.shape[-1]) == int(enc_weights.shape[-2])
        and bool(torch.isfinite(enc_weights).all().item())
    )

    loss_finite = False
    loss_value = float("nan")
    try:
        criterion = model_module.loss().to(device)
        loss_pack = criterion(
            iteration=0,
            save=False,
            viz_split="train",
            outputs=outputs,
            targets=targets,
        )
        loss_total = loss_pack[0]
        loss_finite = bool(torch.isfinite(loss_total).all().item())
        loss_value = float(loss_total.item())
    except Exception:
        loss_finite = False

    postprocess_ok = False
    evaluator_ok = False
    writer_ok = False
    post_shape = None
    try:
        model.eval()
        with torch.no_grad():
            out_eval, _ = model._train(images, masks)
            post_out = post(out_eval, target_sizes)

        postprocess_ok = isinstance(post_out, torch.Tensor) and post_out.dim() == 3 and int(post_out.shape[0]) == 1
        if postprocess_ok:
            post_shape = tuple(post_out.shape)

            evaluator = Evaluator(dataset=db, exp_dir=system_configs.result_dir)
            evaluator.add_prediction(0, post_out[0:1].detach().cpu().numpy(), runtime=0.01)
            evaluator_ok = evaluator.predictions_mode == "poly" and evaluator.predictions is not None

            _ = db.pred2culaneformat(
                db_ind,
                post_out[0].detach().cpu().numpy(),
                runtime=0.01,
                exp_dir=system_configs.result_dir,
            )
            writer_ok = True
    except Exception:
        postprocess_ok = False
        evaluator_ok = False
        writer_ok = False

    return {
        "cfg": cfg_name,
        "encoder_type": type(model.transformer.encoder).__name__,
        "output_contract": bool(key_ok and shape_ok),
        "weights_contract": bool(weights_ok),
        "loss_finite": bool(loss_finite),
        "loss_value": float(loss_value),
        "postprocess_ok": bool(postprocess_ok),
        "evaluator_ok": bool(evaluator_ok),
        "writer_ok": bool(writer_ok),
        "post_shape": post_shape,
        "mamba2_runtime_fallback": bool(
            isinstance(model.transformer.encoder, NonCausalMambaEncoder)
            and bool(getattr(model.transformer.encoder, "runtime_fallback_used", False))
        ),
    }


def run(
    base_cfg: str = "LSTR_CULANE_2k_mamba",
    vssd_cfg: str = "LSTR_CULANE_2k_mamba_vssd",
    seed: int = 0,
) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db, db_ind, images, masks, targets, target_sizes = _prepare_single_sample(cfg_name=base_cfg, device=device)
    post = PostProcess().to(device)

    rows = []
    for cfg_name in [base_cfg, vssd_cfg]:
        rows.append(
            _run_one_cfg(
                cfg_name=cfg_name,
                db=db,
                db_ind=db_ind,
                images=images,
                masks=masks,
                targets=targets,
                target_sizes=target_sizes,
                post=post,
                device=device,
            )
        )

    checks: Dict[str, bool] = {}
    checks["all_output_contract"] = all(r["output_contract"] for r in rows)
    checks["all_weights_contract"] = all(r["weights_contract"] for r in rows)
    checks["all_loss_finite"] = all(r["loss_finite"] for r in rows)
    checks["all_postprocess_ok"] = all(r["postprocess_ok"] for r in rows)
    checks["all_evaluator_ok"] = all(r["evaluator_ok"] for r in rows)
    checks["all_writer_ok"] = all(r["writer_ok"] for r in rows)
    checks["vssd_encoder_present"] = any(
        (r["cfg"] == vssd_cfg) and ("NonCausalMambaEncoder" in str(r["encoder_type"]))
        for r in rows
    )
    checks["vssd_mamba2_native_path"] = all(
        not bool(r.get("mamba2_runtime_fallback", False))
        for r in rows
        if r["cfg"] == vssd_cfg
    )

    print("=" * 96)
    print("DENEY 3 BÖLÜM 3 — ENCODER SWAP PIPELINE")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"base_cfg={base_cfg}, vssd_cfg={vssd_cfg}")
    print(f"sample_db_ind={db_ind}, input={tuple(images.shape)}")

    print("\nModel bazlı sonuçlar:")
    for r in rows:
        print(
            f"  [{r['cfg']}] encoder={r['encoder_type']} "
            f"contract={'✅' if r['output_contract'] else '🚨'} "
            f"weights={'✅' if r['weights_contract'] else '🚨'} "
            f"loss={'✅' if r['loss_finite'] else '🚨'}({r['loss_value']:.4f}) "
            f"post={'✅' if r['postprocess_ok'] else '🚨'} "
            f"eval={'✅' if r['evaluator_ok'] else '🚨'} "
            f"writer={'✅' if r['writer_ok'] else '🚨'} "
            f"m2_fallback={'yes' if bool(r.get('mamba2_runtime_fallback', False)) else 'no'}"
        )
        if r["post_shape"] is not None:
            print(f"      post_out shape: {r['post_shape']}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<24}: {'✅' if v else '🚨'}")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cfg", type=str, default="LSTR_CULANE_2k_mamba")
    parser.add_argument("--vssd-cfg", type=str, default="LSTR_CULANE_2k_mamba_vssd")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checks = run(base_cfg=args.base_cfg, vssd_cfg=args.vssd_cfg, seed=args.seed)
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"DENEY3 BÖLÜM3 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

