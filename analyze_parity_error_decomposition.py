#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import system_configs
from db.datasets import datasets
from models.condlstr_parity_postprocess import parity_outputs_to_lane_coords_all_queries
from nnet.py_factory import NetworkFactory
from utils import normalize_


LanePoints = Sequence[Tuple[float, float]]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(cfg_name: str, suffix: str | None) -> dict:
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    if suffix is not None:
        suffixed = os.path.join(system_configs.config_dir, f"{cfg_name}-{suffix}.json")
        if os.path.exists(suffixed):
            cfg_file = suffixed
    with open(cfg_file, "r", encoding="utf-8") as handle:
        configs = json.load(handle)
    configs["system"]["snapshot_name"] = cfg_name
    system_configs.update_config(configs["system"])
    return configs


def _resolve_split(split_name: str) -> str:
    return {
        "training": system_configs.train_split,
        "validation": system_configs.val_split,
        "testing": system_configs.test_split,
    }.get(split_name, split_name)


def _prepare_input(image: np.ndarray, input_size: Tuple[int, int], mean: np.ndarray, std: np.ndarray, device: torch.device):
    input_h, input_w = input_size
    resized = cv2.resize(image, (input_w, input_h))
    resized = (resized / 255.0).astype(np.float32)
    normalize_(resized, mean, std)
    image_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    mask_tensor = torch.zeros((1, 1, input_h, input_w), dtype=torch.float32, device=device)
    return image_tensor, mask_tensor


def _lane_to_mask(points: LanePoints, image_size: Tuple[int, int], line_width: int) -> np.ndarray:
    image_h, image_w = image_size
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    if len(points) < 2:
        return mask
    polyline = np.asarray(points, dtype=np.float32)
    polyline[:, 0] = np.clip(polyline[:, 0], 0, image_w - 1)
    polyline[:, 1] = np.clip(polyline[:, 1], 0, image_h - 1)
    polyline = np.round(polyline).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [polyline], isClosed=False, color=1, thickness=int(line_width))
    return mask


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _normalize_image_key(path: str) -> str:
    path = path.strip()
    if not path:
        return path
    if not path.startswith("/"):
        path = "/" + path
    return path


def _load_culane_scenario_map() -> Dict[str, str]:
    scenario_map: Dict[str, str] = {}
    split_dir = Path(system_configs.data_dir) / "CULane" / "list" / "test_split"
    if not split_dir.exists():
        return scenario_map

    for path in sorted(split_dir.glob("test*.txt")):
        name = path.stem
        scenario = name.split("_", 1)[1] if "_" in name else name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key = _normalize_image_key(line.strip())
                if key:
                    scenario_map[key] = scenario
    return scenario_map


def _iou_bucket(best_iou: float) -> str:
    best_iou = float(best_iou)
    if best_iou < 0.1:
        return "0.0-0.1"
    if best_iou < 0.3:
        return "0.1-0.3"
    if best_iou < 0.5:
        return "0.3-0.5"
    return "0.5+"


def _oracle_match_flags(
    iou_matrix: np.ndarray,
    eligible_query_mask: np.ndarray,
    iou_thresh: float,
) -> Tuple[List[bool], List[int | None]]:
    num_gt = int(iou_matrix.shape[0])
    if num_gt == 0:
        return [], []

    eligible_indices = np.nonzero(eligible_query_mask)[0]
    if eligible_indices.size == 0 or iou_matrix.shape[1] == 0:
        return [False] * num_gt, [None] * num_gt

    sub_iou = iou_matrix[:, eligible_indices]
    candidate = (sub_iou >= float(iou_thresh)).astype(np.float32)
    benefit = candidate * 1000.0 + sub_iou
    row_ind, col_ind = linear_sum_assignment(-benefit)

    matched = [False] * num_gt
    matched_query_slots: List[int | None] = [None] * num_gt
    for gt_idx, sub_query_idx in zip(row_ind.tolist(), col_ind.tolist()):
        if candidate[gt_idx, sub_query_idx] > 0.0:
            matched[gt_idx] = True
            matched_query_slots[gt_idx] = int(eligible_indices[sub_query_idx])
    return matched, matched_query_slots


def _summarize_lane_records(lane_records: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(lane_records)
    pre = sum(1 for rec in lane_records if rec["has_candidate_pre"])
    post = sum(1 for rec in lane_records if rec["has_candidate_post"])
    oracle_pre = sum(1 for rec in lane_records if rec["matched_pre_oracle"])
    oracle_post = sum(1 for rec in lane_records if rec["matched_post_oracle"])
    score_blocked = sum(1 for rec in lane_records if rec["has_candidate_pre"] and not rec["has_candidate_post"])
    representation_miss = sum(1 for rec in lane_records if not rec["has_candidate_pre"])
    duplicate = sum(1 for rec in lane_records if int(rec["num_good_queries"]) > 1)
    conflict_pre = sum(1 for rec in lane_records if rec["has_candidate_pre"] and not rec["matched_pre_oracle"])
    conflict_post = sum(1 for rec in lane_records if rec["has_candidate_post"] and not rec["matched_post_oracle"])
    best_ious = [float(rec["best_iou"]) for rec in lane_records]
    best_scores = [float(rec["best_query_score"]) for rec in lane_records if rec["best_query_score"] is not None]
    iou_buckets = {
        "0.0-0.1": 0,
        "0.1-0.3": 0,
        "0.3-0.5": 0,
        "0.5+": 0,
    }
    for rec in lane_records:
        iou_buckets[_iou_bucket(float(rec["best_iou"]))] += 1

    return {
        "num_gt_lanes": total,
        "prethreshold_candidate_upper_bound": (pre / total) if total else None,
        "postthreshold_candidate_upper_bound": (post / total) if total else None,
        "prethreshold_oracle_upper_bound": (oracle_pre / total) if total else None,
        "postthreshold_oracle_upper_bound": (oracle_post / total) if total else None,
        "score_blocked_ratio": (score_blocked / total) if total else None,
        "representation_miss_ratio": (representation_miss / total) if total else None,
        "duplicate_candidate_ratio": (duplicate / total) if total else None,
        "shared_query_conflict_pre_ratio": (conflict_pre / total) if total else None,
        "shared_query_conflict_post_ratio": (conflict_post / total) if total else None,
        "mean_best_iou": (sum(best_ious) / len(best_ious)) if best_ious else None,
        "mean_best_query_score": (sum(best_scores) / len(best_scores)) if best_scores else None,
        "best_iou_buckets": iou_buckets,
        "counts": {
            "prethreshold_candidate": pre,
            "postthreshold_candidate": post,
            "prethreshold_oracle": oracle_pre,
            "postthreshold_oracle": oracle_post,
            "score_blocked": score_blocked,
            "representation_miss": representation_miss,
            "duplicate_candidate": duplicate,
            "shared_query_conflict_pre": conflict_pre,
            "shared_query_conflict_post": conflict_post,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Parity lane error decomposition")
    parser.add_argument("cfg_file", type=str, help="Model/config alias")
    parser.add_argument("--testiter", type=int, required=True, help="Checkpoint iteration")
    parser.add_argument("--split", type=str, default="testing", help="training/validation/testing or raw dataset split")
    parser.add_argument("--suffix", type=str, default="thr04", help="Optional config suffix")
    parser.add_argument("--score-thresh", type=float, default=None, help="Score threshold used in deployed predictions")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="Lane IoU threshold")
    parser.add_argument("--line-width", type=int, default=30, help="Rasterization width in pixels")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to analyze")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N images")
    parser.add_argument("--out", type=Path, required=True, help="Summary JSON output path")
    parser.add_argument("--details-jsonl", type=Path, default=None, help="Optional per-GT-lane JSONL output path")
    args = parser.parse_args()

    configs = _load_config(args.cfg_file, args.suffix)
    _set_global_seed(int(system_configs.seed))

    dataset_name = system_configs.dataset
    split = _resolve_split(args.split)
    db = datasets[dataset_name](configs["db"], split)
    scenario_map = _load_culane_scenario_map()

    nnet = NetworkFactory()
    nnet.load_params(int(args.testiter))
    nnet.cuda()
    nnet.eval_mode()
    device = getattr(nnet, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    score_thresh = args.score_thresh
    if score_thresh is None:
        score_thresh = float(system_configs.full.get("condlstr_score_thresh", 0.7))

    db_indices = db.db_inds
    if args.limit is not None:
        db_indices = db_indices[: int(args.limit)]
    total_images = int(len(db_indices))

    per_lane_records: List[Dict[str, object]] = []
    per_image_summary: List[Dict[str, object]] = []
    slot_stats = {
        "decoded_all_counts": defaultdict(int),
        "decoded_post_counts": defaultdict(int),
        "best_query_counts": defaultdict(int),
        "oracle_pre_query_counts": defaultdict(int),
        "oracle_post_query_counts": defaultdict(int),
    }

    if args.details_jsonl is not None:
        args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.details_jsonl.exists():
            args.details_jsonl.unlink()

    start_time = time.time()

    def _format_seconds(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    print(
        f"[error-decomp] cfg={args.cfg_file} split={split} images={total_images} "
        f"iter={args.testiter} score_thresh={score_thresh:.3f} iou_thresh={args.iou_thresh:.3f}"
    )

    for image_counter, db_ind in enumerate(db_indices, start=1):
        item = db.detections(int(db_ind))
        image = cv2.imread(item["path"])
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {item['path']}")

        image_h, image_w = image.shape[:2]
        images, masks = _prepare_input(image, tuple(db.configs["input_size"]), db.mean, db.std, device)
        target_sizes = torch.tensor([[image_h, image_w]], dtype=torch.long, device=device)

        outputs, _ = nnet.test([images, masks])
        visibility_thresh = float(outputs.get("visibility_thresh", system_configs.full.get("condlstr_visibility_thresh", 0.5)))
        decoded_queries = parity_outputs_to_lane_coords_all_queries(
            outputs=outputs,
            target_sizes=target_sizes,
            min_points=2,
            visibility_thresh=visibility_thresh,
        )[0]
        image_key = _normalize_image_key(item["old_anno"].get("org_path", item["path"]))
        scenario = scenario_map.get(image_key, "unknown")

        gt_lanes = item["old_anno"]["lanes"]
        gt_masks = [_lane_to_mask(lane, (image_h, image_w), args.line_width) for lane in gt_lanes]
        pred_masks = [_lane_to_mask(query_info["points"], (image_h, image_w), args.line_width) for query_info in decoded_queries]
        pred_scores = np.asarray([float(query_info["score"]) for query_info in decoded_queries], dtype=np.float32)
        iou_matrix = np.zeros((len(gt_lanes), len(decoded_queries)), dtype=np.float32)
        for gt_index, gt_mask in enumerate(gt_masks):
            for query_index, pred_mask in enumerate(pred_masks):
                iou_matrix[gt_index, query_index] = _compute_iou(pred_mask, gt_mask)

        pre_match_flags, pre_match_slots = _oracle_match_flags(
            iou_matrix=iou_matrix,
            eligible_query_mask=np.ones((len(decoded_queries),), dtype=bool),
            iou_thresh=float(args.iou_thresh),
        )
        post_match_flags, post_match_slots = _oracle_match_flags(
            iou_matrix=iou_matrix,
            eligible_query_mask=(pred_scores >= float(score_thresh)) if pred_scores.size else np.zeros((0,), dtype=bool),
            iou_thresh=float(args.iou_thresh),
        )

        for query_info in decoded_queries:
            slot_stats["decoded_all_counts"][int(query_info["query_index"])] += 1
            if float(query_info["score"]) >= float(score_thresh):
                slot_stats["decoded_post_counts"][int(query_info["query_index"])] += 1

        image_lane_records: List[Dict[str, object]] = []
        for gt_index, (gt_lane, gt_mask) in enumerate(zip(gt_lanes, gt_masks)):
            query_ious = iou_matrix[gt_index].tolist() if iou_matrix.size else []
            query_scores = pred_scores.tolist() if pred_scores.size else []

            if query_ious:
                best_query_idx = int(np.argmax(np.asarray(query_ious, dtype=np.float32)))
                best_iou = float(query_ious[best_query_idx])
                best_score = float(query_scores[best_query_idx])
                best_query_id = int(decoded_queries[best_query_idx]["query_index"])
                good_query_indices = [idx for idx, value in enumerate(query_ious) if value >= float(args.iou_thresh)]
                post_query_indices = [
                    idx for idx in good_query_indices if query_scores[idx] >= float(score_thresh)
                ]
            else:
                best_query_idx = -1
                best_iou = 0.0
                best_score = None
                best_query_id = None
                good_query_indices = []
                post_query_indices = []

            record = {
                "image_key": image_key,
                "scenario": scenario,
                "db_index": int(db_ind),
                "gt_index": int(gt_index),
                "gt_num_points": int(len(gt_lane)),
                "best_query_slot": best_query_id,
                "best_query_rank_in_decoded": best_query_idx,
                "best_iou": best_iou,
                "best_query_score": best_score,
                "num_decoded_queries": int(len(decoded_queries)),
                "num_good_queries": int(len(good_query_indices)),
                "has_candidate_pre": bool(best_iou >= float(args.iou_thresh)),
                "has_candidate_post": bool(len(post_query_indices) > 0),
                "matched_pre_oracle": bool(pre_match_flags[gt_index]) if gt_index < len(pre_match_flags) else False,
                "matched_post_oracle": bool(post_match_flags[gt_index]) if gt_index < len(post_match_flags) else False,
                "oracle_pre_query_slot": pre_match_slots[gt_index] if gt_index < len(pre_match_slots) else None,
                "oracle_post_query_slot": post_match_slots[gt_index] if gt_index < len(post_match_slots) else None,
                "score_blocked": bool(best_iou >= float(args.iou_thresh) and len(post_query_indices) == 0),
                "representation_miss": bool(best_iou < float(args.iou_thresh)),
                "best_iou_bucket": _iou_bucket(best_iou),
            }
            image_lane_records.append(record)
            per_lane_records.append(record)
            if best_query_id is not None:
                slot_stats["best_query_counts"][int(best_query_id)] += 1
            if record["oracle_pre_query_slot"] is not None:
                slot_stats["oracle_pre_query_counts"][int(record["oracle_pre_query_slot"])] += 1
            if record["oracle_post_query_slot"] is not None:
                slot_stats["oracle_post_query_counts"][int(record["oracle_post_query_slot"])] += 1

        image_summary = _summarize_lane_records(image_lane_records)
        image_summary.update(
            {
                "image_key": image_key,
                "scenario": scenario,
                "db_index": int(db_ind),
                "num_gt_lanes": len(gt_lanes),
                "num_decoded_queries": len(decoded_queries),
            }
        )
        per_image_summary.append(image_summary)

        if args.details_jsonl is not None:
            with args.details_jsonl.open("a", encoding="utf-8") as handle:
                for record in image_lane_records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        if int(args.log_every) > 0 and (
            image_counter % int(args.log_every) == 0 or image_counter == total_images
        ):
            elapsed = time.time() - start_time
            per_image = elapsed / float(image_counter)
            remaining = per_image * float(max(total_images - image_counter, 0))
            print(
                f"[error-decomp] {image_counter}/{total_images} "
                f"elapsed={_format_seconds(elapsed)} "
                f"eta={_format_seconds(remaining)} "
                f"sec_per_image={per_image:.3f}"
            )

    global_summary = _summarize_lane_records(per_lane_records)
    scenario_summary = {}
    lane_records_by_scenario: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in per_lane_records:
        lane_records_by_scenario[str(record.get("scenario", "unknown"))].append(record)
    for scenario, records in sorted(lane_records_by_scenario.items()):
        scenario_summary[scenario] = _summarize_lane_records(records)

    top_score_blocked = sorted(
        [rec for rec in per_lane_records if rec["score_blocked"]],
        key=lambda x: (-float(x["best_iou"]), 0.0 if x["best_query_score"] is None else float(x["best_query_score"]), x["image_key"], int(x["gt_index"])),
    )[:20]
    top_representation_miss = sorted(
        [rec for rec in per_lane_records if rec["representation_miss"]],
        key=lambda x: (float(x["best_iou"]), x["image_key"], int(x["gt_index"])),
    )[:20]

    summary = {
        "cfg_file": args.cfg_file,
        "testiter": int(args.testiter),
        "split": split,
        "suffix": args.suffix,
        "score_thresh": float(score_thresh),
        "iou_thresh": float(args.iou_thresh),
        "line_width": int(args.line_width),
        "num_images": int(len(per_image_summary)),
        "global_summary": global_summary,
        "scenario_summary": scenario_summary,
        "slot_stats": {
            key: {str(slot): int(count) for slot, count in sorted(value.items())}
            for key, value in slot_stats.items()
        },
        "top_score_blocked_lanes": top_score_blocked,
        "top_representation_miss_lanes": top_representation_miss,
        "per_image_summary_top20_low_recall": sorted(
            per_image_summary,
            key=lambda x: (
                1.0 if x["postthreshold_oracle_upper_bound"] is None else float(x["postthreshold_oracle_upper_bound"]),
                x["image_key"],
            ),
        )[:20],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["global_summary"], indent=2))
    print(f"wrote summary -> {args.out}")
    if args.details_jsonl is not None:
        print(f"wrote details -> {args.details_jsonl}")


if __name__ == "__main__":
    main()
