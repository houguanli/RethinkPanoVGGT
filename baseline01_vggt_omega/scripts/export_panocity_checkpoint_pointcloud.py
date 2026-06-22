#!/usr/bin/env python3
"""Export GT and predicted PanoCity window point clouds for a baseline01 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
for path in (PROJECT_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.dataset_util import depth_to_world_coords_points  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="panocity_paired_4xrtx5000_full_warmup_3h_for_luna")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    cfg.model.activation_checkpointing = False
    dataset = instantiate(cfg.data.train.dataset, common_config=cfg.data.train.common_config, _recursive_=False)
    model = instantiate(cfg.model, _recursive_=False).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint={args.checkpoint}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    pred_depth_scale = float(cfg.loss.depth.get("pred_depth_scale", 1.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "pred_depth_scale": pred_depth_scale,
        "samples": [],
    }

    with torch.no_grad():
        for offset in range(max(0, args.num_samples)):
            sample_index = args.start_index + offset
            sample = dataset[(sample_index, int(cfg.data.train.common_config.fix_img_num), 1.0)]
            images = sample["images"].unsqueeze(0).to(device)
            amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                pred = model(images=images)["depth"].float() * pred_depth_scale
            pred_depths = pred[0, ..., 0].detach().cpu().numpy()

            sample_name = safe_name(str(sample["seq_name"]))
            out_dir = args.output_dir / f"{sample_index:04d}_{sample_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            gt_points, gt_colors = collect_points(sample, sample["depths"].numpy(), args.stride)
            pred_points, pred_colors = collect_points(sample, pred_depths, args.stride)
            write_ply(out_dir / "gt_window_points.ply", gt_points, gt_colors)
            write_ply(out_dir / "pred_window_points.ply", pred_points, pred_colors)
            write_preview(out_dir / "preview_depths.npz", sample, pred_depths)
            row = {
                "sample_index": sample_index,
                "seq_name": str(sample["seq_name"]),
                "gt_points": int(len(gt_points)),
                "pred_points": int(len(pred_points)),
                "output_dir": str(out_dir),
            }
            summary["samples"].append(row)
            print(json.dumps(row, ensure_ascii=False))

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={args.output_dir / 'summary.json'}")


def load_config(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(TRAINING_ROOT / "config")):
        return compose(config_name=name)


def collect_points(sample: dict, depths: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    points_all = []
    colors_all = []
    images = sample["images"].permute(0, 2, 3, 1).numpy()
    masks = sample["point_masks"].numpy().astype(bool)
    extrinsics = sample["extrinsics"].numpy()
    intrinsics = sample["intrinsics"].numpy()
    stride = max(1, int(stride))
    sparse = np.zeros_like(masks[0], dtype=bool)
    sparse[::stride, ::stride] = True

    for view_idx in range(depths.shape[0]):
        depth = depths[view_idx].astype(np.float32)
        valid = masks[view_idx] & sparse & np.isfinite(depth) & (depth > 0)
        world, _, point_mask = depth_to_world_coords_points(depth, extrinsics[view_idx], intrinsics[view_idx])
        valid = valid & point_mask & np.isfinite(world).all(axis=-1)
        if not valid.any():
            continue
        points_all.append(world[valid].astype(np.float32))
        colors_all.append(np.clip(images[view_idx][valid] * 255.0, 0, 255).astype(np.uint8))

    if not points_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(points_all, axis=0), np.concatenate(colors_all, axis=0)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_preview(path: Path, sample: dict, pred_depths: np.ndarray) -> None:
    np.savez_compressed(
        path,
        gt_depths=sample["depths"].numpy().astype(np.float32),
        pred_depths=pred_depths.astype(np.float32),
        masks=sample["point_masks"].numpy().astype(bool),
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:96]


if __name__ == "__main__":
    main()
