#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from imgaug.augmentables.lines import LineStringsOnImage

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory
from utils import normalize_
from utils.condlstr_parity_targets import build_parity_targets_from_legacy_targets


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(cfg_name: str) -> dict:
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    system_configs.update_config(configs["system"])
    return configs


def _build_batch(db, indices: List[int]):
    input_h, input_w = db.configs["input_size"]
    batch_size = len(indices)
    images = np.zeros((batch_size, 3, input_h, input_w), dtype=np.float32)
    masks = np.zeros((batch_size, 1, input_h, input_w), dtype=np.float32)
    gt_lanes = []
    image_keys = []

    for b_ind, db_ind in enumerate(indices):
        item = db.detections(int(db_ind))
        img = cv2.imread(item["path"])
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {item['path']}")

        seg_mask = np.ones((1, img.shape[0], img.shape[1], 1), dtype=bool)
        line_strings = db.lane_to_linestrings(item["old_anno"]["lanes"])
        line_strings = LineStringsOnImage(line_strings, shape=img.shape)
        img, line_strings, seg_mask = db.transform(
            image=img,
            line_strings=line_strings,
            segmentation_maps=seg_mask,
        )
        line_strings.clip_out_of_image_()

        new_anno = {
            "path": item["path"],
            "lanes": db.linestrings_to_lanes(line_strings),
            "categories": item["categories"],
        }
        label = db._transform_annotation(new_anno, img_wh=(input_w, input_h))["label"]

        image_keys.append(item["old_anno"].get("org_path", item["path"]))
        gt_lanes.append(torch.from_numpy(label[None].astype(np.float32)))

        img = (img / 255.0).astype(np.float32)
        normalize_(img, db.mean, db.std)
        images[b_ind] = img.transpose((2, 0, 1))
        masks[b_ind] = np.logical_not(seg_mask[:, :, :, 0])

    image_tensor = torch.from_numpy(images)
    mask_tensor = torch.from_numpy(masks)
    xs = [image_tensor, mask_tensor]
    ys = [image_tensor.clone()] + gt_lanes
    return xs, ys, image_keys


def main():
    parser = argparse.ArgumentParser(description="Fixed-panel DN instability probe")
    parser.add_argument("cfg_file", type=str, help="Config/model alias name")
    parser.add_argument("--split", type=str, default="train_2k", help="Dataset split for panel selection")
    parser.add_argument("--iterations", nargs="+", type=int, required=True, help="Checkpoint iterations to probe")
    parser.add_argument("--panel-size", type=int, default=64, help="Number of fixed images to probe")
    parser.add_argument("--panel-seed", type=int, default=123, help="Seed for fixed panel selection")
    parser.add_argument("--batch-size", type=int, default=4, help="Probe batch size")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    configs = _load_config(args.cfg_file)
    _set_global_seed(int(system_configs.seed))

    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], args.split)
    db.aug_chance = 0.0
    if hasattr(db, "_data_rng"):
        db._data_rng = np.random.RandomState(args.panel_seed)

    panel_rng = np.random.RandomState(args.panel_seed)
    db_indices = np.array(db.db_inds, copy=True)
    if args.panel_size > len(db_indices):
        raise ValueError(f"panel_size={args.panel_size} exceeds split size={len(db_indices)}")
    panel = panel_rng.choice(db_indices, size=args.panel_size, replace=False).tolist()

    nnet = NetworkFactory(flag=True)
    nnet.model.eval()
    nnet.loss.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    for iteration in args.iterations:
        nnet.load_params(int(iteration))
        for start in range(0, len(panel), args.batch_size):
            batch_indices = panel[start : start + args.batch_size]
            xs, ys, image_keys = _build_batch(db, batch_indices)
            xs = nnet._move_batch(xs)
            ys = nnet._move_batch(ys)

            with torch.no_grad():
                outputs, _ = nnet.model(*xs, targets=ys, image_keys=image_keys, force_dn=True)
                spatial_size = (
                    int(outputs["pred_dense_mask"].shape[-2]),
                    int(outputs["pred_dense_mask"].shape[-1]),
                )
                dense_targets = build_parity_targets_from_legacy_targets(
                    targets=ys,
                    target_size=spatial_size,
                    device=outputs["pred_object_logits"].device,
                    line_width=nnet.loss.line_width,
                    min_valid_rows=nnet.loss.min_valid_rows,
                )
                _, _, diagnostics = nnet.loss.criterion(
                    outputs,
                    dense_targets,
                    image_keys=image_keys,
                    collect_diagnostics=True,
                )

            with args.out.open("a", encoding="utf-8") as handle:
                for db_ind, payload in zip(batch_indices, diagnostics):
                    record = {
                        "iteration": int(iteration),
                        "db_index": int(db_ind),
                        "cfg_file": args.cfg_file,
                        "split": args.split,
                    }
                    record.update(payload)
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"wrote diagnostics -> {args.out}")


if __name__ == "__main__":
    main()
