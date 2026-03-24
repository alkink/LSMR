from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def _slice_target(target: Dict[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    return {
        key: value[:count].clone()
        for key, value in target.items()
    }


def _lane_summary_from_target(target: Dict[str, torch.Tensor], feature_width: int) -> torch.Tensor:
    row_ranges = target['gt_row_rng']
    row_locations = target['gt_row_loc']
    row_masks = target['gt_row_loc_mask']

    if row_ranges.numel() == 0:
        return row_ranges.new_zeros((0, 8))

    denom = float(max(feature_width - 1, 1))
    summaries = []
    for lane_index in range(row_ranges.size(0)):
        row_mask = row_masks[lane_index] > 0.5
        valid_rows = torch.nonzero(row_mask, as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            continue

        if valid_rows.numel() == 1:
            sample_rows = valid_rows.repeat(4)
        else:
            sample_rows = torch.linspace(
                float(valid_rows[0].item()),
                float(valid_rows[-1].item()),
                steps=4,
                device=row_locations.device,
            ).round().long()

        sampled_x = row_locations[lane_index, sample_rows] / denom
        visible_x = row_locations[lane_index, row_mask] / denom
        center_x = visible_x.mean()
        visible_fraction = row_mask.float().mean()
        summary = torch.stack(
            [
                row_ranges[lane_index, 0],
                row_ranges[lane_index, 1],
                sampled_x[0].clamp(0.0, 1.0),
                sampled_x[1].clamp(0.0, 1.0),
                sampled_x[2].clamp(0.0, 1.0),
                sampled_x[3].clamp(0.0, 1.0),
                center_x.clamp(0.0, 1.0),
                visible_fraction.clamp(0.0, 1.0),
            ]
        )
        summaries.append(summary)

    if not summaries:
        return row_ranges.new_zeros((0, 8))
    return torch.stack(summaries, dim=0)


def build_dn_lane_queries(
    targets: Sequence[Dict[str, torch.Tensor]],
    num_dn_queries: int,
    feature_width: int,
    x_noise_scale: float = 0.05,
    range_noise_scale: float = 0.03,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
    if num_dn_queries <= 0:
        device = targets[0]['gt_row_loc'].device if targets else torch.device('cpu')
        return torch.zeros((len(targets), 0, 8), device=device), list(targets), torch.zeros((len(targets),), dtype=torch.long, device=device)

    batch_size = len(targets)
    device = targets[0]['gt_row_loc'].device if targets else torch.device('cpu')
    dn_queries = torch.zeros((batch_size, num_dn_queries, 8), dtype=torch.float32, device=device)
    dn_valid_counts = torch.zeros((batch_size,), dtype=torch.long, device=device)
    dn_targets: List[Dict[str, torch.Tensor]] = []

    for batch_index, target in enumerate(targets):
        lane_count = int(target['gt_row_rng'].size(0))
        if lane_count == 0:
            dn_targets.append(_slice_target(target, 0))
            continue

        summary = _lane_summary_from_target(target, feature_width=feature_width)
        lane_count = min(int(summary.size(0)), int(num_dn_queries))
        summary = summary[:lane_count].clone()
        if summary.numel() > 0:
            summary[:, :2] = (summary[:, :2] + torch.randn_like(summary[:, :2]) * float(range_noise_scale)).clamp(0.0, 1.0)
            summary[:, 2:7] = (summary[:, 2:7] + torch.randn_like(summary[:, 2:7]) * float(x_noise_scale)).clamp(0.0, 1.0)
            start = torch.minimum(summary[:, 0], summary[:, 1])
            end = torch.maximum(summary[:, 0], summary[:, 1])
            summary[:, 0] = start
            summary[:, 1] = end
            dn_queries[batch_index, :lane_count] = summary

        dn_valid_counts[batch_index] = lane_count
        dn_targets.append(_slice_target(target, lane_count))

    return dn_queries, dn_targets, dn_valid_counts
