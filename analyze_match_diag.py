#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _safe_mean(values):
    return float(mean(values)) if values else None


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_iterations(records):
    by_iter = defaultdict(list)
    for rec in records:
        by_iter[int(rec["iteration"])].append(rec)

    summaries = []
    for iteration in sorted(by_iter):
        batch = by_iter[iteration]
        summaries.append(
            {
                "iteration": iteration,
                "records": len(batch),
                "mean_query_margin": _safe_mean(
                    [r["mean_query_margin"] for r in batch if r.get("mean_query_margin") is not None]
                ),
                "mean_target_margin": _safe_mean(
                    [r["mean_target_margin"] for r in batch if r.get("mean_target_margin") is not None]
                ),
                "mean_low_query_margin_ratio": _safe_mean(
                    [r["low_query_margin_ratio"] for r in batch if r.get("low_query_margin_ratio") is not None]
                ),
                "mean_low_target_margin_ratio": _safe_mean(
                    [r["low_target_margin_ratio"] for r in batch if r.get("low_target_margin_ratio") is not None]
                ),
                "main_fg_mean": _safe_mean(
                    [r["main_object_fg_stats"]["mean"] for r in batch if r.get("main_object_fg_stats")]
                ),
                "dn_fg_valid_mean": _safe_mean(
                    [
                        r["dn_object_fg_stats_valid"]["mean"]
                        for r in batch
                        if r.get("dn_object_fg_stats_valid") and r.get("dn_valid_count", 0) > 0
                    ]
                ),
            }
        )
    return summaries


def summarize_flips(records):
    by_image = defaultdict(list)
    for rec in records:
        image_key = rec.get("image_key")
        if image_key is None:
            continue
        by_image[image_key].append(rec)

    comparable = 0
    flips = 0
    image_stats = []

    for image_key, seq in by_image.items():
        seq = sorted(seq, key=lambda x: int(x["iteration"]))
        image_comparable = 0
        image_flips = 0
        cardinality_changes = 0

        for prev, curr in zip(seq, seq[1:]):
            prev_map = {int(a["target"]): int(a["query"]) for a in prev.get("assignments", [])}
            curr_map = {int(a["target"]): int(a["query"]) for a in curr.get("assignments", [])}
            shared_targets = sorted(set(prev_map) & set(curr_map))
            if prev.get("num_matches") != curr.get("num_matches"):
                cardinality_changes += 1
            for target_id in shared_targets:
                comparable += 1
                image_comparable += 1
                if prev_map[target_id] != curr_map[target_id]:
                    flips += 1
                    image_flips += 1

        image_stats.append(
            {
                "image_key": image_key,
                "num_records": len(seq),
                "flip_rate": (image_flips / image_comparable) if image_comparable else None,
                "num_flips": image_flips,
                "comparable_pairs": image_comparable,
                "cardinality_changes": cardinality_changes,
            }
        )

    image_stats.sort(
        key=lambda x: (
            -1 if x["flip_rate"] is None else -x["flip_rate"],
            -x["num_flips"],
            x["image_key"],
        )
    )
    return {
        "num_images": len(by_image),
        "comparable_pairs": comparable,
        "num_flips": flips,
        "flip_rate": (flips / comparable) if comparable else None,
        "top_unstable_images": image_stats[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze CondLSTR parity match diagnostics JSONL")
    parser.add_argument("jsonl", type=Path, help="Path to match_diag_train.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    records = load_records(args.jsonl)
    if not records:
        raise SystemExit("No records found")

    iteration_summaries = summarize_iterations(records)
    flip_summary = summarize_flips(records)

    summary = {
        "path": str(args.jsonl),
        "num_records": len(records),
        "num_iterations": len({int(r["iteration"]) for r in records}),
        "iteration_summaries": iteration_summaries,
        "flip_summary": flip_summary,
    }

    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
