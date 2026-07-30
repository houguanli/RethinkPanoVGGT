#!/usr/bin/env python3
"""Export ERP GT point clouds under plausible depth-buffer semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.reconstruct_pano_omega import erp_range_depth_np, erp_rays_np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=100.0)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--max-points", type=int, default=400000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    raw = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(args.depth)
    depth = raw.astype(np.float32) / args.depth_scale
    rays = erp_rays_np(*depth.shape)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cubemap_range = erp_range_depth_np(depth, rays, "cubemap_z")
    double_cubemap_range = erp_range_depth_np(depth, rays, "double_cubemap_z")
    point_clouds = {
        "legacy_as_radial_range": rays * depth[..., None],
        "single_cubemap_z_approx": rays * cubemap_range[..., None],
        "double_cubemap_z": rays * double_cubemap_range[..., None],
    }
    valid = np.isfinite(double_cubemap_range) & (double_cubemap_range > 0) & (double_cubemap_range <= args.max_depth_m)
    summaries = {}
    for name, points in point_clouds.items():
        write_ply(args.output_dir / f"gt_{name}.ply", points[valid], rgb[valid], args.max_points)
        summaries[name] = point_summary(points[valid])
    make_comparison(args.output_dir / "gt_depth_semantics_projection.png", point_clouds, valid)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[INFO] exported GT depth hypotheses = {args.output_dir}")


def point_summary(points: np.ndarray) -> dict:
    radius = np.linalg.norm(points, axis=-1)
    return {
        "points": int(points.shape[0]),
        "radius_p10_p50_p90": [float(value) for value in np.percentile(radius, [10, 50, 90])],
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int) -> None:
    if points.shape[0] > max_points:
        keep = np.random.default_rng(42).choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def make_comparison(path: Path, point_clouds: dict[str, np.ndarray], valid: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visible = {name: points[valid][::40] for name, points in point_clouds.items()}
    reference = np.concatenate(list(visible.values()), axis=0)
    xlim = np.percentile(reference[:, 0], [1, 99])
    ylim = np.percentile(reference[:, 1], [1, 99])
    zlim = np.percentile(reference[:, 2], [1, 99])
    fig, axes = plt.subplots(2, len(visible), figsize=(10, 8), dpi=160)
    for column, (name, points) in enumerate(visible.items()):
        axes[0, column].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.12, cmap="viridis", rasterized=True)
        axes[1, column].scatter(points[:, 0], points[:, 2], c=points[:, 1], s=0.12, cmap="coolwarm", rasterized=True)
        axes[0, column].set_title(name)
        axes[0, column].set_xlim(xlim)
        axes[0, column].set_ylim(ylim)
        axes[1, column].set_xlim(xlim)
        axes[1, column].set_ylim(zlim)
    axes[0, 0].set_ylabel("Y up (front)")
    axes[1, 0].set_ylabel("Z (top)")
    for axis in axes.reshape(-1):
        axis.set_xlabel("X")
        axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
