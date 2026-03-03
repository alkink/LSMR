import json
import importlib
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from config import system_configs


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mamba-based scripts. Please run on a CUDA-enabled environment.")
    return torch.device("cuda")


def load_system_config(cfg_name: str = "LSTR_CULANE_2k_mamba") -> dict:
    cfg_path = Path(system_configs.config_dir) / f"{cfg_name}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("system", {})
    cfg["system"]["snapshot_name"] = cfg_name
    system_configs.update_config(cfg["system"])
    return cfg


def build_model_from_cfg(
    cfg_name: str = "LSTR_CULANE_2k_mamba",
    flag: bool = True,
    device: torch.device = torch.device("cuda"),
    eval_mode: bool = True,
):
    load_system_config(cfg_name)
    module = importlib.import_module(f"models.{cfg_name}")
    net = module.model(flag=flag).to(device)
    if eval_mode:
        net.eval()
    return net


@torch.no_grad()
def infer_feature_hw(model, input_h: int, input_w: int, device: torch.device) -> Tuple[int, int]:
    x = torch.randn(1, 3, input_h, input_w, device=device)
    p = model.conv1(x)
    p = model.bn1(p)
    p = model.relu(p)
    p = model.maxpool(p)
    p = model.layer1(p)
    p = model.layer2(p)
    p = model.layer3(p)
    p = model.layer4(p)
    return int(p.shape[-2]), int(p.shape[-1])


def forward_to_decoder(model, images: torch.Tensor, masks: torch.Tensor):
    """
    Returns:
        hs:      (num_layers, B, num_queries, C)
        memory:  (B, C, H, W)
        weights: encoder attention-like weights
    """
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
    hs, memory, weights = model.transformer(model.input_proj(p), pmasks, model.query_embed.weight, pos)
    return hs, memory, weights

