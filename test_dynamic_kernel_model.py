"""
Smoke checks for LSTR_CULANE_2k_mamba_dynamic (contract-safe version).

This test intentionally validates compatibility with the existing LSTR training/eval
contracts, not a custom pred_xcoords API.
"""

import json
import torch

from config import system_configs


def _load_cfg(cfg_name="LSTR_CULANE_2k_mamba_dynamic"):
    with open(f"config/{cfg_name}.json", "r") as f:
        cfg = json.load(f)
    cfg["system"]["snapshot_name"] = cfg_name
    system_configs.update_config(cfg["system"])
    return cfg


def test_model_and_loss_contract():
    _load_cfg()

    from models.LSTR_CULANE_2k_mamba_dynamic import model, loss

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Mamba encoder kernels in this project. "
            "Run this test on a CUDA-enabled environment."
        )

    device = torch.device("cuda")

    net = model(flag=True).to(device)
    criterion = loss().to(device)

    # --- forward contract
    B, H, W = 2, 295, 820
    images = torch.randn(B, 3, H, W, device=device)
    masks = torch.zeros(B, 1, H, W, device=device)

    outputs, _ = net(images, masks)

    assert "pred_logits" in outputs, "missing pred_logits"
    assert "pred_curves" in outputs, "missing pred_curves"
    assert outputs["pred_logits"].shape[0] == B
    assert outputs["pred_curves"].shape[0] == B
    assert outputs["pred_curves"].shape[-1] == system_configs.lsp_dim

    if system_configs.aux_loss:
        assert "aux_outputs" in outputs, "missing aux_outputs"
        assert len(outputs["aux_outputs"]) == (system_configs.dec_layers - 1)

    # --- loss contract
    # Build targets in current pipeline format: list[tensor], each tensor (L, 1+2+2*max_points)
    # Minimal synthetic valid labels for matcher/criterion.
    max_points = 18
    lanes_per_img = 4
    tgt_dim = 1 + 2 + 2 * max_points

    gt_list = []
    for _ in range(B):
        # IMPORTANT: mimic real sampler contract.
        # In sample/culane.py each GT entry in ys[1:] has shape (B, L, D),
        # and AELoss takes tgt[0] -> (L, D).
        t = torch.ones((lanes_per_img, tgt_dim), dtype=torch.float32, device=device) * -1e5
        t[:, 0] = 1.0  # class id lane
        t[:, 1] = 0.1  # lower
        t[:, 2] = 0.9  # upper

        xs = torch.linspace(0.2, 0.8, max_points, device=device)
        ys = torch.linspace(0.1, 0.9, max_points, device=device)
        t[:, 3:3 + max_points] = xs
        t[:, 3 + max_points:3 + 2 * max_points] = ys

        # (B, L, D) to match training pipeline
        gt_list.append(t.unsqueeze(0).repeat(B, 1, 1))

    for g in gt_list:
        assert g.ndim == 3 and g.shape[0] == B, f"invalid GT shape: {g.shape}"

    # AELoss forward signature:
    # loss(iteration, save, viz_split, outputs, targets)
    # where targets = [images, *gt_list]
    losses = criterion(0, False, "train", outputs, [images, *gt_list])
    total_loss = losses[0]
    assert torch.isfinite(total_loss), "loss is NaN/Inf"

    print("PASS | dynamic model contract and loss smoke test")


if __name__ == "__main__":
    test_model_and_loss_contract()

