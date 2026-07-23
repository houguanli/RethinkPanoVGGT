#!/usr/bin/env python3
"""Export GT radial point clouds directly from the original 12-view UE depth captures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.reconstruct_pano_omega import erp_rays_np, save_depth_image, write_ply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-shot-dir", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--processed-depth", type=Path, default=None)
    parser.add_argument("--merge-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-depth-scale", type=float, default=0.01, help="UE centimeters to meters.")
    parser.add_argument("--processed-depth-scale", type=float, default=100.0)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--max-range-relative-disagreement", type=float, default=0.02)
    parser.add_argument("--blend-gamma", type=float, default=1.0)
    parser.add_argument("--back-yaw-deg", type=float, default=45.0)
    parser.add_argument("--max-points", type=int, default=400000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merger = import_python_module(args.merge_module)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    height, width = rgb.shape[:2]

    front_cube = merger.load_cubemap_from_folder(
        str(args.raw_shot_dir), "depth", "exr", merger.SET_FRONT, prefer_single_channel=True
    )
    back_cube = merger.load_cubemap_from_folder(
        str(args.raw_shot_dir), "depth", "exr", merger.SET_BACK, prefer_single_channel=True
    )
    dx, dy, dz = merger.make_equirect_dirs(width, height)
    dx_back, dy_back, dz_back = merger.rotate_dirs_around_z(dx, dy, dz, -args.back_yaw_deg)
    front_z, front_weight, _ = merger.sample_cubemap(front_cube, dx, dy, dz, blend_gamma=args.blend_gamma)
    back_z, back_weight, _ = merger.sample_cubemap(back_cube, dx_back, dy_back, dz_back, blend_gamma=args.blend_gamma)

    front_factor = cube_projection_factor(dx, dy, dz)
    back_factor = cube_projection_factor(dx_back, dy_back, dz_back)
    front_range = front_z[..., 0] * args.input_depth_scale / front_factor
    back_range = back_z[..., 0] * args.input_depth_scale / back_factor
    front_weight = front_weight[..., 0]
    back_weight = back_weight[..., 0]
    weight_sum = np.maximum(front_weight + back_weight, 1e-8)
    blended_range = (front_range * front_weight + back_range * back_weight) / weight_sum

    finite_pair = (
        np.isfinite(front_range)
        & np.isfinite(back_range)
        & (front_range > 0)
        & (back_range > 0)
    )
    relative_disagreement = np.full_like(blended_range, np.inf, dtype=np.float32)
    relative_disagreement[finite_pair] = np.abs(front_range[finite_pair] - back_range[finite_pair]) / np.maximum(
        np.minimum(front_range[finite_pair], back_range[finite_pair]), 1e-6
    )
    range_valid = finite_pair & (blended_range <= args.max_depth_m)
    consistent_valid = range_valid & (relative_disagreement <= args.max_range_relative_disagreement)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rays = erp_rays_np(height, width)
    write_ply(
        args.output_dir / "target_raw12_range_blended.ply",
        rays[range_valid] * blended_range[range_valid, None],
        rgb[range_valid],
        args.max_points,
    )
    write_ply(
        args.output_dir / "target_raw12_range_consistent.ply",
        rays[consistent_valid] * blended_range[consistent_valid, None],
        rgb[consistent_valid],
        args.max_points,
    )
    save_depth_image(blended_range, range_valid, args.output_dir / "target_raw12_range.png", args.max_depth_m)
    Image.fromarray((consistent_valid.astype(np.uint8) * 255)).save(args.output_dir / "target_raw12_consistent_mask.png")
    save_projection_comparison(
        args.output_dir / "target_raw12_projection_comparison.png",
        rays * blended_range[..., None],
        range_valid,
        consistent_valid,
    )

    summary = {
        "raw_shot_dir": str(args.raw_shot_dir),
        "source_depth_definition": "UE SceneCaptureSource.SCS_SCENE_DEPTH perspective Z in Unreal centimeters",
        "conversion": "Each source depth is converted to radial range before weighted double-cubemap fusion.",
        "max_depth_m": args.max_depth_m,
        "max_range_relative_disagreement": args.max_range_relative_disagreement,
        "range_valid_ratio": float(range_valid.mean()),
        "consistent_valid_ratio": float(consistent_valid.mean()),
        "discarded_boundary_ratio": float((range_valid & ~consistent_valid).mean()),
        "front_back_relative_disagreement_p50_p90_p95": percentiles(relative_disagreement[range_valid], [50, 90, 95]),
        "range_m_p10_p50_p90": percentiles(blended_range[consistent_valid], [10, 50, 90]),
    }
    if args.processed_depth is not None:
        processed = cv2.imread(str(args.processed_depth), cv2.IMREAD_UNCHANGED)
        if processed is None:
            raise FileNotFoundError(args.processed_depth)
        processed_z = processed.astype(np.float32) / args.processed_depth_scale
        effective_factor = (front_weight * front_factor + back_weight * back_factor) / weight_sum
        recovered_range = processed_z / np.maximum(effective_factor, 1e-6)
        overlap = consistent_valid & np.isfinite(recovered_range) & (recovered_range > 0)
        relative_error = np.abs(recovered_range[overlap] - blended_range[overlap]) / np.maximum(blended_range[overlap], 1e-6)
        summary["processed_double_cubemap_recovery_relative_error_p50_p90_p95"] = percentiles(relative_error, [50, 90, 95])

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def import_python_module(path: Path):
    spec = importlib.util.spec_from_file_location("merge_double_cube_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import merge module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cube_projection_factor(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([np.abs(dx), np.abs(dy), np.abs(dz)]).astype(np.float32)


def percentiles(values: np.ndarray, levels: list[int]) -> list[float] | None:
    if values.size == 0:
        return None
    return [float(value) for value in np.percentile(values, levels)]


def save_projection_comparison(
    path: Path,
    points: np.ndarray,
    range_valid: np.ndarray,
    consistent_valid: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = {
        "raw12 range blend": points[range_valid][::40],
        "raw12 clean mask": points[consistent_valid][::40],
    }
    reference = np.concatenate(list(views.values()), axis=0)
    xlim = np.percentile(reference[:, 0], [1, 99])
    zlim = np.percentile(reference[:, 2], [1, 99])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=160)
    for axis, (title, visible) in zip(axes, views.items()):
        axis.scatter(visible[:, 0], visible[:, 2], c=visible[:, 1], s=0.12, cmap="coolwarm", rasterized=True)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Z (top)")
        axis.set_xlim(xlim)
        axis.set_ylim(zlim)
        axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
