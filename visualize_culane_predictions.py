#!/usr/bin/env python
import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CULane predictions from .lines.txt files")
    parser.add_argument("--data-root", required=True, help="Path to dataset root that contains CULane/")
    parser.add_argument("--pred-dir", required=True, help="Path to prediction directory, e.g. results/<cfg>/<iter>/testing")
    parser.add_argument("--output-dir", required=True, help="Directory to save overlay images")
    parser.add_argument("--list", default=None, help="CULane list file, defaults to <data-root>/CULane/list/test.txt")
    parser.add_argument("--limit", type=int, default=20, help="How many images to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--draw-gt", action="store_true", help="Overlay ground-truth lanes too")
    parser.add_argument("--line-thickness", type=int, default=6, help="Lane thickness")
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    return parser.parse_args()


def read_list_file(list_path: Path) -> List[str]:
    with list_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def read_lane_file(path: Path) -> List[List[Tuple[int, int]]]:
    if not path.exists():
        return []

    lanes: List[List[Tuple[int, int]]] = []
    with path.open("r") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) < 4 or len(parts) % 2 != 0:
                continue
            coords = list(map(float, parts))
            lane = []
            for i in range(0, len(coords), 2):
                x = int(round(coords[i]))
                y = int(round(coords[i + 1]))
                lane.append((x, y))
            if len(lane) >= 2:
                lanes.append(lane)
    return lanes


def draw_lanes(image, lanes: List[List[Tuple[int, int]]], color, thickness: int):
    overlay = image.copy()
    for lane in lanes:
        for p1, p2 in zip(lane[:-1], lane[1:]):
            cv2.line(overlay, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return overlay


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    culane_root = data_root / "CULane"
    list_path = Path(args.list) if args.list is not None else culane_root / "list" / "test.txt"
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rel_images = read_list_file(list_path)
    if args.sample_mode == "random":
        random.seed(args.seed)
        random.shuffle(rel_images)

    selected = rel_images[: max(int(args.limit), 0)]

    for idx, rel_image in enumerate(selected):
        rel_image = rel_image[1:] if rel_image.startswith("/") else rel_image
        image_path = culane_root / rel_image
        pred_path = pred_dir / rel_image.replace(".jpg", ".lines.txt")
        gt_path = culane_root / rel_image.replace(".jpg", ".lines.txt")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[skip] image missing: {image_path}")
            continue

        pred_lanes = read_lane_file(pred_path)
        gt_lanes = read_lane_file(gt_path) if args.draw_gt else []

        vis = image.copy()
        if gt_lanes:
            vis = draw_lanes(vis, gt_lanes, color=(255, 0, 255), thickness=max(args.line_thickness - 2, 2))
        if pred_lanes:
            vis = draw_lanes(vis, pred_lanes, color=(0, 255, 0), thickness=args.line_thickness)

        cv2.putText(
            vis,
            f"pred={len(pred_lanes)} gt={len(gt_lanes)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_name = f"{idx:03d}_{Path(rel_image).stem}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), vis)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
