import argparse
import copy
import importlib
import json
import random

import cv2
import numpy as np
import torch

from config import system_configs
from db.datasets import datasets
from models.py_utils.mamba_encoder import BidirectionalMambaEncoder
from sample.culane import sample_data


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_name):
    cfg_path = f"config/{cfg_name}.json" if not cfg_name.endswith(".json") else cfg_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["system"]["snapshot_name"] = cfg_name.replace(".json", "").split("/")[-1]
    return cfg


def apply_system_config(base_system, **overrides):
    system_cfg = copy.deepcopy(base_system)
    system_cfg.update(overrides)
    system_configs.update_config(system_cfg)
    return system_cfg


def grad_norm(parameters):
    total = 0.0
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach().float().norm(2).item()
        total += value * value
        found = True
    if not found:
        return 0.0
    return total ** 0.5


def check_dataset_and_sample_batch(base_system, db_cfg, batch_size, seed):
    apply_system_config(base_system, batch_size=batch_size, scan_order="none")
    set_all_seeds(seed)

    dataset_name = system_configs.dataset
    train_split = system_configs.train_split
    db = datasets[dataset_name](db_cfg, train_split)

    first_valid = None
    first_shape = None
    first_pos = None
    scan_limit = min(int(db.db_inds.size), 512)

    for pos in range(scan_limit):
        db_ind = int(db.db_inds[pos])
        item = db.detections(db_ind)
        img = cv2.imread(item["path"])
        if img is None:
            continue
        first_valid = item["path"]
        first_shape = tuple(int(x) for x in img.shape)
        first_pos = pos
        break

    if first_valid is None:
        raise RuntimeError("Ilk 512 ornekte okunabilir goruntu bulunamadi.")

    print(f"[dataset] first_valid_pos={first_pos}")
    print(f"[dataset] first_valid_path={first_valid}")
    print(f"[dataset] first_valid_shape={first_shape}")

    db.shuffle_inds = lambda: None
    batch, _ = sample_data(db, first_pos)
    xs_shapes = [tuple(x.shape) for x in batch["xs"]]
    ys_shapes = [tuple(y.shape) for y in batch["ys"]]
    print(f"[dataset] xs_shapes={xs_shapes}")
    print(f"[dataset] ys_shapes={ys_shapes}")
    return batch


def run_permutation_debug(base_system, order, seq_len, device, seed):
    print("")
    print("=" * 80)
    print(f"[perm-debug] scan_order={order}")
    print("=" * 80)

    apply_system_config(base_system, scan_order=order)
    set_all_seeds(seed)
    enc = BidirectionalMambaEncoder().to(device)

    resolved_hw = enc._resolve_hw(seq_len)
    print(f"resolved_hw={resolved_hw}")

    perm, inv = enc._build_permutation(seq_len, torch.device("cpu"))
    if perm is None:
        print("perm=None (scan disabled or invalid hw)")
    else:
        print(f"perm[0]={perm[0].item()}")
        print(f"perm[:12]={perm[:12].tolist()}")
        x = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
        x_back = x.index_select(1, perm).index_select(1, inv)
        print(f"roundtrip_ok={torch.equal(x, x_back)}")

        h, w = resolved_hw
        coords = torch.stack([perm // w, perm % w], dim=1)
        jumps = (coords[1:] - coords[:-1]).abs().sum(dim=1)
        print(f"jump_mean={jumps.float().mean().item():.4f}")
        print(f"jump_max={int(jumps.max().item())}")
        print(f"jump_gt1={(jumps > 1).sum().item()}")
        print(f"jump_gt5={(jumps > 5).sum().item()}")

        xg = torch.randn(2, seq_len, system_configs.attn_dim, requires_grad=True)
        yg = xg.index_select(1, perm).index_select(1, inv)
        loss = yg.pow(2).mean()
        loss.backward()
        print(f"perm_grad_finite={torch.isfinite(xg.grad).all().item()}")
        print(f"perm_grad_norm={xg.grad.norm().item():.6f}")

    src = torch.randn(seq_len, 2, system_configs.attn_dim, device=device, requires_grad=True)
    out, _ = enc(src)
    loss = out.pow(2).mean()
    loss.backward()
    print(f"out_shape={tuple(out.shape)}")
    print(f"out_finite={torch.isfinite(out).all().item()}")
    print(f"out_std={out.std().item():.6f}")
    print(f"src_grad_norm={src.grad.norm().item():.6f}")


def build_template_state(base_system, snapshot_name, seed):
    apply_system_config(base_system, scan_order="none")
    set_all_seeds(seed)
    module = importlib.import_module(f"models.{snapshot_name}")
    model = module.model(flag=True)
    state = copy.deepcopy(model.state_dict())
    del model
    return state, module


def run_overfit_compare(base_system, batch, orders, device, steps, lr, seed):
    print("")
    print("=" * 80)
    print("[overfit-debug] same batch short-run comparison")
    print("=" * 80)

    snapshot_name = base_system["snapshot_name"]
    template_state, module = build_template_state(base_system, snapshot_name, seed)
    xs_cpu = [x.clone() for x in batch["xs"]]
    ys_cpu = [y.clone() for y in batch["ys"]]
    summaries = []

    for order in orders:
        apply_system_config(base_system, scan_order=order)
        set_all_seeds(seed)

        model = module.model(flag=True).to(device)
        model.load_state_dict(template_state)
        criterion = module.loss().to(device)
        model.train()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )

        xs = [x.to(device) for x in xs_cpu]
        ys = [y.to(device) for y in ys_cpu]
        losses = []
        enc_grads = []

        print("")
        print(f"[overfit-debug] order={order}")

        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            preds, _ = model(*xs)
            loss_tuple = criterion(step, False, "scan_debug", preds, ys)
            loss = loss_tuple[0].mean()
            weighted_loss = float(loss_tuple[-1])
            loss.backward()

            enc_grad = grad_norm(model.transformer.encoder.parameters())
            total_grad = grad_norm(model.parameters())
            optimizer.step()

            losses.append(weighted_loss)
            enc_grads.append(enc_grad)

            if step == 1 or step == steps // 2 or step == steps:
                print(
                    f"  step={step:03d} "
                    f"loss={weighted_loss:.6f} "
                    f"enc_grad={enc_grad:.6f} "
                    f"total_grad={total_grad:.6f}"
                )

        summary = {
            "order": order,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_drop": losses[0] - losses[-1],
            "last_enc_grad": enc_grads[-1],
        }
        summaries.append(summary)

        print(
            f"[summary] order={order} "
            f"initial={summary['initial_loss']:.6f} "
            f"final={summary['final_loss']:.6f} "
            f"best={summary['best_loss']:.6f} "
            f"drop={summary['loss_drop']:.6f} "
            f"last_enc_grad={summary['last_enc_grad']:.6f}"
        )

        del model
        del criterion
        torch.cuda.empty_cache()

    print("")
    print("=" * 80)
    print("[overfit-debug] final table")
    print("=" * 80)
    for summary in summaries:
        print(
            f"{summary['order']:>8} | "
            f"initial={summary['initial_loss']:.6f} | "
            f"final={summary['final_loss']:.6f} | "
            f"best={summary['best_loss']:.6f} | "
            f"drop={summary['loss_drop']:.6f} | "
            f"enc_grad={summary['last_enc_grad']:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Scan-order debug utility")
    parser.add_argument("--cfg", default="LSTR_CULANE_2k_mamba_scan")
    parser.add_argument("--orders", default="none,col-bu")
    parser.add_argument("--seq-len", default=260, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    base_system = copy.deepcopy(cfg["system"])
    db_cfg = copy.deepcopy(cfg["db"])
    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    device = torch.device(args.device)

    print(f"[config] cfg={args.cfg}")
    print(f"[config] snapshot_name={base_system['snapshot_name']}")
    print(f"[config] device={device}")
    print(f"[config] orders={orders}")

    batch = check_dataset_and_sample_batch(
        base_system=base_system,
        db_cfg=db_cfg,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    for order in orders:
        run_permutation_debug(
            base_system=base_system,
            order=order,
            seq_len=args.seq_len,
            device=device,
            seed=args.seed,
        )

    run_overfit_compare(
        base_system=base_system,
        batch=batch,
        orders=orders,
        device=device,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
