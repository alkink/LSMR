"""
BÖLÜM 5 — Pipeline uyumluluk kontrolü (decoder swap sonrası).

Amaç:
1) Decoder değiştirildiğinde model forward kontratı bozuluyor mu?
2) Eğitim loss hattı (AELoss) finite kalıyor mu?
3) Test hattı (PostProcess -> Evaluator -> CULane writer) çalışıyor mu?
"""

import argparse
import importlib
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from config import system_configs
from db.culane import CULANE
from db.utils.evaluator import Evaluator
from mask_migration.common import build_model_from_cfg, load_system_config
from mamba_decoder_analysis.decoder_variants import MambaDecoderLayerD_Mamba2, build_variant_decoder
from sample.culane import sample_data
from test.culane import PostProcess


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    )
    model.transformer.decoder = new_decoder.to(next(model.parameters()).device)


def _prepare_single_sample(cfg_name: str, device: torch.device):
    cfg = load_system_config(cfg_name)
    db = CULANE(cfg["db"], system_configs.train_split)

    # AELoss ile birebir uyumlu target formatı için doğrudan örnekleyiciyi kullan.
    original_batch_size = int(system_configs.batch_size)
    system_configs.update_config({"batch_size": 1})
    try:
        sampled, _next_ind = sample_data(db, 0)
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

    return cfg, db, db_ind, images, masks, targets, target_sizes


def run(
    cfg_name: str = "LSTR_CULANE_2k_mamba",
    d_state: int = 16,
    d_conv: int = 4,
    variants: Optional[List[str]] = None,
    headdim: int = 32,
    seed: int = 0,
) -> Dict[str, bool]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _cfg, db, db_ind, images, masks, targets, target_sizes = _prepare_single_sample(cfg_name=cfg_name, device=device)
    post = PostProcess().to(device)
    model_module = importlib.import_module(f"models.{cfg_name}")

    run_variants = variants if variants is not None else ["BASE", "A", "B", "C"]
    variant_rows: List[dict] = []

    for variant in run_variants:
        model = build_model_from_cfg(cfg_name, flag=True, device=device, eval_mode=False)
        old_decoder = model.transformer.decoder
        if variant.upper() != "BASE":
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
                headdim=headdim,
                num_heads=int(system_configs.num_heads),
                dim_feedforward=int(system_configs.dim_feedforward),
            )
            model.transformer.decoder = new_decoder.to(next(model.parameters()).device)

        # 5.1 output contract
        model.train()
        outputs, _weights = model._train(images, masks)
        key_ok = all(k in outputs for k in ["pred_logits", "pred_curves"])
        shape_ok = (
            key_ok
            and outputs["pred_logits"].dim() == 3
            and outputs["pred_curves"].dim() == 3
            and int(outputs["pred_logits"].shape[0]) == 1
            and int(outputs["pred_curves"].shape[0]) == 1
        )

        # 5.2 training-loss path
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

        # 5.3 test/postprocess/evaluator/writer path
        postprocess_ok = False
        evaluator_ok = False
        writer_ok = False
        post_shape = None

        try:
            model.eval()
            with torch.no_grad():
                out_eval, _w_eval = model._train(images, masks)
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

        has_multihead_attn = hasattr(model.transformer.decoder.layers[-1], "multihead_attn")

        variant_rows.append(
            {
                "variant": variant,
                "output_contract": bool(key_ok and shape_ok),
                "loss_finite": bool(loss_finite),
                "loss_value": loss_value,
                "postprocess_ok": bool(postprocess_ok),
                "evaluator_ok": bool(evaluator_ok),
                "writer_ok": bool(writer_ok),
                "post_shape": post_shape,
                "has_multihead_attn": bool(has_multihead_attn),
                "mamba2_runtime_fallback": bool(
                    variant.upper() == "D"
                    and isinstance(model.transformer.decoder.layers[-1], MambaDecoderLayerD_Mamba2)
                    and bool(getattr(model.transformer.decoder.layers[-1], "runtime_fallback_used", False))
                ),
            }
        )

    checks: Dict[str, bool] = {}
    checks["all_output_contract"] = all(r["output_contract"] for r in variant_rows)
    checks["all_loss_finite"] = all(r["loss_finite"] for r in variant_rows)
    checks["all_postprocess_ok"] = all(r["postprocess_ok"] for r in variant_rows)
    checks["all_evaluator_ok"] = all(r["evaluator_ok"] for r in variant_rows)
    checks["all_writer_ok"] = all(r["writer_ok"] for r in variant_rows)

    print("=" * 96)
    print("BÖLÜM 5 — PIPELINE UYUMLULUK")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"Config: {cfg_name}")
    print(f"Sample db_ind: {db_ind}")
    print(f"Input image: {tuple(images.shape)}")

    print("\nVaryant bazlı sonuçlar:")
    for row in variant_rows:
        print(
            f"  [{row['variant']}] "
            f"contract={'✅' if row['output_contract'] else '🚨'} "
            f"loss={'✅' if row['loss_finite'] else '🚨'}({row['loss_value']:.4f}) "
            f"post={'✅' if row['postprocess_ok'] else '🚨'} "
            f"eval={'✅' if row['evaluator_ok'] else '🚨'} "
            f"writer={'✅' if row['writer_ok'] else '🚨'} "
            f"attn_hook={'yes' if row['has_multihead_attn'] else 'no'} "
            f"m2_fallback={'yes' if row['mamba2_runtime_fallback'] else 'no'}"
        )
        if row["post_shape"] is not None:
            print(f"      post_out shape: {row['post_shape']}")

    print("\nKontroller:")
    for k, v in checks.items():
        print(f"  {k:<24}: {'✅' if v else '🚨'}")

    print("\nNot: Varyant A/B/C decoder katmanlarında `multihead_attn` bulunmaz;")
    print("     `test.py --debugDec` attention görselleştirme hook'u baseline'a özeldir.")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="LSTR_CULANE_2k_mamba")
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--variants", type=str, default="BASE,A,B,C")
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]

    checks = run(
        cfg_name=args.cfg,
        d_state=args.d_state,
        d_conv=args.d_conv,
        variants=variants,
        headdim=args.headdim,
        seed=args.seed,
    )
    ok = all(checks.values())

    print("\n" + "=" * 96)
    print("RAPOR")
    print("=" * 96)
    print(f"BÖLÜM 5 sonucu: {'✅' if ok else '🚨'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

