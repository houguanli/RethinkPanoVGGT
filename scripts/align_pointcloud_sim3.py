#!/usr/bin/env python3
"""Post-align reconstructed point clouds to a GT point cloud with Sim3.

The alignment is intended for evaluation/visualization of monocular or pano
reconstruction outputs where global scale, rotation, and translation are not
intrinsically observable. It uses a symmetric nearest-neighbor ICP loop and an
optional robust-span scale refinement so dense near-camera regions do not
dominate the final scale.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--target-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--max-fit-source", type=int, default=180000)
    parser.add_argument("--max-fit-target", type=int, default=180000)
    parser.add_argument("--max-output-points", type=int, default=400000)
    parser.add_argument("--icp-iters", type=int, default=30)
    parser.add_argument("--trim-fraction", type=float, default=0.85)
    parser.add_argument("--span-refine", action="store_true", default=True)
    parser.add_argument("--no-span-refine", dest="span_refine", action="store_false")
    parser.add_argument("--span-low", type=float, default=1.0)
    parser.add_argument("--span-high", type=float, default=99.0)
    parser.add_argument("--allow-reflection", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    source_points, source_colors = read_ply(args.source_ply)
    target_points, target_colors = read_ply(args.target_ply)
    source_fit = sample_points(source_points, args.max_fit_source, rng)
    target_fit = sample_points(target_points, args.max_fit_target, rng)

    initial = bbox_initial_transform(source_fit, target_fit)
    result, history = sim3_icp(
        source_fit,
        target_fit,
        initial,
        iters=args.icp_iters,
        trim_fraction=args.trim_fraction,
        allow_reflection=args.allow_reflection,
    )
    if args.span_refine:
        result = refine_scale_by_robust_span(
            source_fit,
            target_fit,
            result,
            low=args.span_low,
            high=args.span_high,
        )

    aligned_points = apply_transform(source_points, result["scale"], result["orthogonal"], result["translation"])
    metrics = compute_metrics(aligned_points, target_points, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / f"{args.name}_sim3_aligned_to_gt.ply"
    target_path = args.output_dir / "gt_reference.ply"
    overlay_path = args.output_dir / f"{args.name}_sim3_overlay_xyz.png"
    write_ply(aligned_path, *sample_points_and_colors(aligned_points, source_colors, args.max_output_points, rng))
    write_ply(target_path, *sample_points_and_colors(target_points, target_colors, args.max_output_points, rng))
    write_overlay(
        overlay_path,
        sample_points(aligned_points, 70000, rng),
        sample_points(target_points, 70000, rng),
        args.name,
    )

    summary = {
        "method": "symmetric_sim3_icp_with_robust_span_scale" if args.span_refine else "symmetric_sim3_icp",
        "source_ply": str(args.source_ply),
        "target_ply": str(args.target_ply),
        "aligned_ply": str(aligned_path),
        "target_reference_ply": str(target_path),
        "overlay": str(overlay_path),
        "source_points": int(source_points.shape[0]),
        "target_points": int(target_points.shape[0]),
        "fit_source_points": int(source_fit.shape[0]),
        "fit_target_points": int(target_fit.shape[0]),
        "icp_trim_fraction": float(args.trim_fraction),
        "scale": float(result["scale"]),
        "orthogonal_determinant": float(np.linalg.det(result["orthogonal"])),
        "orthogonal": result["orthogonal"].tolist(),
        "translation": result["translation"].tolist(),
        "matrix": sim_matrix(result["scale"], result["orthogonal"], result["translation"]).tolist(),
        "span_refine": result.get("span_refine", None),
        "history": history,
        "metrics": metrics,
    }
    summary_path = args.output_dir / f"{args.name}_sim3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def read_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertex_count = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                header_lines = idx + 1
                break
    if vertex_count is None:
        raise ValueError(f"Missing vertex count in {path}")
    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    points = data[:, :3]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if data.shape[1] >= 6:
        colors = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)[valid]
    else:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    return points, colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep]


def sample_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, colors
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep], colors[keep]


def bbox_initial_transform(src: np.ndarray, tgt: np.ndarray) -> dict:
    src_q = np.percentile(src, [1.0, 99.0], axis=0)
    tgt_q = np.percentile(tgt, [1.0, 99.0], axis=0)
    src_span = np.maximum(src_q[1] - src_q[0], 1e-6)
    tgt_span = np.maximum(tgt_q[1] - tgt_q[0], 1e-6)
    scale = float(np.median(tgt_span / src_span))
    orthogonal = np.eye(3, dtype=np.float64)
    src_center = np.median(src, axis=0).astype(np.float64)
    tgt_center = np.median(tgt, axis=0).astype(np.float64)
    translation = tgt_center - scale * src_center
    return {"scale": scale, "orthogonal": orthogonal, "translation": translation}


def sim3_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: dict,
    iters: int,
    trim_fraction: float,
    allow_reflection: bool,
) -> tuple[dict, list[dict]]:
    current = initial
    target_tree = cKDTree(target)
    history = []
    trim_fraction = float(np.clip(trim_fraction, 0.05, 1.0))
    for iteration in range(iters):
        transformed = apply_transform(source, current["scale"], current["orthogonal"], current["translation"])
        source_tree = cKDTree(transformed)
        dist_st, idx_st = target_tree.query(transformed, k=1, workers=-1)
        dist_ts, idx_ts = source_tree.query(target, k=1, workers=-1)
        paired_src = np.concatenate([source, source[idx_ts]], axis=0)
        paired_tgt = np.concatenate([target[idx_st], target], axis=0)
        paired_dist = np.concatenate([dist_st, dist_ts], axis=0)
        keep_count = max(32, int(paired_dist.shape[0] * trim_fraction))
        keep = np.argpartition(paired_dist, keep_count - 1)[:keep_count]
        try:
            refined = estimate_similarity(paired_src[keep], paired_tgt[keep], allow_reflection=allow_reflection)
        except np.linalg.LinAlgError:
            break
        transformed_refined = apply_transform(source, refined["scale"], refined["orthogonal"], refined["translation"])
        refined_dist, _ = target_tree.query(transformed_refined, k=1, workers=-1)
        metric = {
            "iter": iteration,
            "source_to_target_median": float(np.median(refined_dist)),
            "source_to_target_p90": float(np.percentile(refined_dist, 90.0)),
            "scale": float(refined["scale"]),
        }
        history.append(metric)
        if (
            abs(refined["scale"] - current["scale"]) < 1e-5
            and np.linalg.norm(refined["translation"] - current["translation"]) < 1e-4
        ):
            current = refined
            break
        current = refined
    return current, history


def estimate_similarity(src: np.ndarray, tgt: np.ndarray, allow_reflection: bool) -> dict:
    src64 = src.astype(np.float64)
    tgt64 = tgt.astype(np.float64)
    src_mean = src64.mean(axis=0)
    tgt_mean = tgt64.mean(axis=0)
    src_c = src64 - src_mean
    tgt_c = tgt64 - tgt_mean
    cov = (tgt_c.T @ src_c) / max(src64.shape[0], 1)
    u, singular, vt = np.linalg.svd(cov)
    signs = np.ones(3)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        signs[-1] = -1.0
    orthogonal = u @ np.diag(signs) @ vt
    var_src = np.mean(np.sum(src_c**2, axis=1))
    scale = float(np.sum(singular * signs) / max(var_src, 1e-12))
    if scale <= 0 or not math.isfinite(scale):
        raise np.linalg.LinAlgError("invalid scale")
    translation = tgt_mean - scale * (src_mean @ orthogonal.T)
    return {"scale": scale, "orthogonal": orthogonal, "translation": translation}


def refine_scale_by_robust_span(
    source: np.ndarray,
    target: np.ndarray,
    transform: dict,
    low: float,
    high: float,
) -> dict:
    aligned = apply_transform(source, transform["scale"], transform["orthogonal"], transform["translation"])
    src_q = np.percentile(aligned, [low, high], axis=0)
    tgt_q = np.percentile(target, [low, high], axis=0)
    src_span = np.maximum(src_q[1] - src_q[0], 1e-6)
    tgt_span = np.maximum(tgt_q[1] - tgt_q[0], 1e-6)
    axis_ratio = tgt_span / src_span
    scale_adjust = float(np.median(np.clip(axis_ratio, 0.25, 4.0)))
    source_center = np.median(aligned, axis=0).astype(np.float64)
    target_center = np.median(target, axis=0).astype(np.float64)
    refined = {
        "scale": float(transform["scale"] * scale_adjust),
        "orthogonal": transform["orthogonal"],
        "translation": target_center + scale_adjust * (transform["translation"] - source_center),
        "span_refine": {
            "low_percentile": low,
            "high_percentile": high,
            "axis_ratio": axis_ratio.tolist(),
            "scale_adjust": scale_adjust,
        },
    }
    return refined


def apply_transform(points: np.ndarray, scale: float, orthogonal: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (points.astype(np.float64) @ orthogonal.T) + translation[None]).astype(np.float32)


def compute_metrics(points: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> dict:
    points_fit = sample_points(points, 180000, rng)
    target_fit = sample_points(target, 180000, rng)
    target_tree = cKDTree(target_fit)
    source_tree = cKDTree(points_fit)
    dist_st, _ = target_tree.query(points_fit, k=1, workers=-1)
    dist_ts, _ = source_tree.query(target_fit, k=1, workers=-1)
    src_q = np.percentile(points, [1.0, 50.0, 99.0], axis=0)
    tgt_q = np.percentile(target, [1.0, 50.0, 99.0], axis=0)
    return {
        "source_to_target_median": float(np.median(dist_st)),
        "source_to_target_p90": float(np.percentile(dist_st, 90.0)),
        "source_to_target_mean": float(np.mean(dist_st)),
        "target_to_source_median": float(np.median(dist_ts)),
        "target_to_source_p90": float(np.percentile(dist_ts, 90.0)),
        "target_to_source_mean": float(np.mean(dist_ts)),
        "symmetric_nn_mean": float(0.5 * (np.mean(dist_st) + np.mean(dist_ts))),
        "aligned_p01": src_q[0].round(5).tolist(),
        "aligned_median": src_q[1].round(5).tolist(),
        "aligned_p99": src_q[2].round(5).tolist(),
        "aligned_robust_span_p01_p99": (src_q[2] - src_q[0]).round(5).tolist(),
        "target_p01": tgt_q[0].round(5).tolist(),
        "target_median": tgt_q[1].round(5).tolist(),
        "target_p99": tgt_q[2].round(5).tolist(),
        "target_robust_span_p01_p99": (tgt_q[2] - tgt_q[0]).round(5).tolist(),
    }


def write_overlay(path: Path, source: np.ndarray, target: np.ndarray, name: str) -> None:
    combined = np.concatenate([source, target], axis=0)
    views = [("XY top", 0, 1), ("XZ side/up", 0, 2), ("YZ side/up", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160)
    for ax, (title, i, j) in zip(axes, views):
        low = np.percentile(combined[:, [i, j]], 1.0, axis=0)
        high = np.percentile(combined[:, [i, j]], 99.0, axis=0)
        pad = (high - low) * 0.05 + 1e-6
        ax.scatter(target[:, i], target[:, j], s=0.25, c="#111111", alpha=0.18, linewidths=0, label="GT")
        ax.scatter(source[:, i], source[:, j], s=0.25, c="#d9480f", alpha=0.25, linewidths=0, label=name)
        ax.set_title(title)
        ax.set_xlabel("xyz"[i])
        ax.set_ylabel("xyz"[j])
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.25, alpha=0.3)
    axes[0].legend(loc="best", markerscale=8)
    fig.suptitle(f"{name} Sim3 aligned to GT")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sim_matrix(scale: float, orthogonal: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * orthogonal
    matrix[:3, 3] = translation
    return matrix


if __name__ == "__main__":
    main()
