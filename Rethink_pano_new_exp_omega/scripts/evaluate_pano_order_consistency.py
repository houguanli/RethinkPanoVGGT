#!/usr/bin/env python3
"""Measure multi-pano prediction sensitivity to anchor and input ordering."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_depth_checkpoint import (
    apply_checkpoint_eval_defaults,
    build_eval_model,
    load_checkpoint_payload,
    normalize_args_for_eval,
    select_eval_indices,
)
from training.train_pano_omega import (
    build_dataset,
    move_batch_to_device,
    parse_args as parse_training_args,
    resolve_device,
    set_seed,
)
from vggt_omega.utils.rotation import quat_to_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset", default="panocity")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pano-count", type=int, default=0)
    parser.add_argument("--num-yaw", type=int, default=0, help="Override windows per pano; 0 keeps config.")
    parser.add_argument("--random-permutations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    set_seed(cli.seed)
    device = resolve_device(cli.device, {"distributed": False, "local_rank": 0})
    args = parse_training_args(["--config", str(cli.config)])
    args.device = cli.device
    args.distributed = "none"
    args.dataset_format = "pano_minimal"
    args.minimal_datasets = str(cli.dataset)
    args.dataset_split = cli.split
    args.dataset_max_samples = None
    args.num_workers = 0
    args.randomize_pano_order = False
    if cli.dataset_root is not None:
        args.dataset_root = cli.dataset_root
    if cli.pano_count > 0:
        args.pano_sample_mode = "fixed_neighborhood"
        args.pano_min_count = int(cli.pano_count)
        args.pano_max_count = int(cli.pano_count)
    if cli.num_yaw > 0:
        args.num_yaw = int(cli.num_yaw)
    normalize_args_for_eval(args)
    payload = load_checkpoint_payload(cli.checkpoint)
    apply_checkpoint_eval_defaults(args, payload)
    model = build_eval_model(args, cli.checkpoint, payload, device)
    model.eval()

    pano_size = (args.pano_height, args.pano_width) if args.pano_height > 0 and args.pano_width > 0 else None
    dataset = build_dataset(args, pano_size)
    indices, sample_info = select_eval_indices(dataset, cli.limit, cli.seed, "anchor")
    generator = torch.Generator().manual_seed(cli.seed)
    rows = []
    with torch.no_grad():
        for dataset_index in indices:
            batch = move_batch_to_device(default_collate([dataset[int(dataset_index)]]), device)
            pano_images = batch["pano_image"]
            pano_count = int(pano_images.shape[1])
            permutations = [torch.arange(pano_count - 1, -1, -1, device=device)]
            for _ in range(max(int(cli.random_permutations), 0)):
                permutations.append(torch.randperm(pano_count, generator=generator).to(device))
            base = predict(model, pano_images)
            for permutation_index, permutation in enumerate(permutations):
                permuted = predict(model, pano_images[:, permutation])
                inverse = torch.argsort(permutation)
                reordered = reorder_prediction(permuted, inverse, pano_count)
                rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "permutation_index": int(permutation_index),
                        "permutation": [int(value) for value in permutation.cpu().tolist()],
                        **compare_predictions(base, reordered, pano_count),
                    }
                )

    result = {
        "config": str(cli.config),
        "checkpoint": str(cli.checkpoint),
        "dataset": str(cli.dataset),
        "split": cli.split,
        "sample_policy": sample_info,
        "evaluated_samples": len(indices),
        "comparisons": len(rows),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


def predict(model: torch.nn.Module, pano_images: torch.Tensor) -> dict[str, torch.Tensor]:
    output = model(pano_images=pano_images, return_sampler_output=True)
    return {
        "depth": output["depth"].detach().float(),
        "center": output["pano_camera_center"].detach().float(),
        "rotation": quat_to_mat(output["pano_rotation_quat_w2c"].detach().float()),
    }


def reorder_prediction(prediction: dict[str, torch.Tensor], inverse: torch.Tensor, pano_count: int) -> dict[str, torch.Tensor]:
    depth = prediction["depth"]
    views = int(depth.shape[1])
    if views % pano_count != 0:
        raise ValueError(f"Depth views {views} are not divisible by pano count {pano_count}")
    depth = depth.reshape(depth.shape[0], pano_count, views // pano_count, *depth.shape[2:])
    return {
        "depth": depth[:, inverse].reshape(prediction["depth"].shape),
        "center": prediction["center"][:, inverse],
        "rotation": prediction["rotation"][:, inverse],
    }


def compare_predictions(base: dict[str, torch.Tensor], other: dict[str, torch.Tensor], pano_count: int) -> dict[str, float]:
    base_depth = base["depth"].clamp_min(1e-6)
    other_depth = other["depth"].clamp_min(1e-6)
    valid = torch.isfinite(base_depth) & torch.isfinite(other_depth)
    log_delta = torch.log(other_depth[valid]) - torch.log(base_depth[valid])
    log_delta = log_delta - log_delta.median()

    r_errors = []
    t_errors = []
    for i in range(pano_count):
        for j in range(i + 1, pano_count):
            base_rel_r = base["rotation"][:, j] @ base["rotation"][:, i].transpose(-1, -2)
            other_rel_r = other["rotation"][:, j] @ other["rotation"][:, i].transpose(-1, -2)
            delta_r = other_rel_r @ base_rel_r.transpose(-1, -2)
            trace = delta_r.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            r_errors.append(torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)) * (180.0 / math.pi))

            base_t = base["rotation"][:, i] @ (base["center"][:, j] - base["center"][:, i]).unsqueeze(-1)
            other_t = other["rotation"][:, i] @ (other["center"][:, j] - other["center"][:, i]).unsqueeze(-1)
            base_t = base_t.squeeze(-1)
            other_t = other_t.squeeze(-1)
            denom = torch.linalg.vector_norm(base_t, dim=-1) * torch.linalg.vector_norm(other_t, dim=-1)
            cosine = (base_t * other_t).sum(dim=-1) / denom.clamp_min(1e-8)
            t_errors.append(torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi))
    r_all = torch.cat(r_errors) if r_errors else base_depth.new_zeros(0)
    t_all = torch.cat(t_errors) if t_errors else base_depth.new_zeros(0)
    return {
        "depth_scale_aligned_abs_log_mean": float(log_delta.abs().mean().cpu()),
        "depth_scale_aligned_abs_log_median": float(log_delta.abs().median().cpu()),
        "pair_rotation_consistency_deg_mean": float(r_all.mean().cpu()) if r_all.numel() else 0.0,
        "pair_rotation_consistency_deg_median": float(r_all.median().cpu()) if r_all.numel() else 0.0,
        "pair_translation_consistency_deg_mean": float(t_all.mean().cpu()) if t_all.numel() else 0.0,
        "pair_translation_consistency_deg_median": float(t_all.median().cpu()) if t_all.numel() else 0.0,
    }


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [key for key in rows[0] if key.endswith(("_mean", "_median"))] if rows else []
    result = {}
    for key in keys:
        values = torch.tensor([float(row[key]) for row in rows])
        result[key] = {"mean": float(values.mean()), "median": float(values.median()), "max": float(values.max())}
    return result


if __name__ == "__main__":
    main()
