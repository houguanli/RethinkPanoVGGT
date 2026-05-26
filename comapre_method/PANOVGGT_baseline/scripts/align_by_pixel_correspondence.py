#!/usr/bin/env python3
"""Align PanoVGGT output to GT using ERP pixel correspondences.

For the same input panorama, each predicted point and each GT range-depth point
share an ERP pixel. This gives automatic point pairs, which is a better coarse
alignment source than nearest-neighbor ICP when there is no reliable initial
pose. The solver estimates a robust 7-parameter similarity transform
(scale, rotation/reflection convention, translation) and writes aligned PLYs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHOD_ROOT.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SAMPLE_ROOT = PROJECT_ROOT / "dataset" / "converted_pano4vggt_omega_range_v2" / "pano_x_-1370_y_-23000"
PANO_OUTPUT = OUTPUT_ROOT / "panovggt_baseline_sample0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-space", choices=["world", "local"], default="world")
    parser.add_argument("--panovggt-output", type=Path, default=PANO_OUTPUT)
    parser.add_argument("--gt-depth", type=Path, default=SAMPLE_ROOT / "clone/frames/depth/Camera_0/depth_00000.png")
    parser.add_argument("--gt-valid", type=Path, default=SAMPLE_ROOT / "clone/frames/depth_valid/Camera_0/valid_00000.png")
    parser.add_argument("--gt-meta", type=Path, default=SAMPLE_ROOT / "clone/pano_meta.json")
    parser.add_argument("--rgb", type=Path, default=PROJECT_ROOT / "dataset/panovggt_compare_sample/images/000000.jpg")
    parser.add_argument("--output-dir", type=Path, default=PANO_OUTPUT / "pixel_aligned_to_gt")
    parser.add_argument("--max-fit-pairs", type=int, default=120000)
    parser.add_argument("--ransac-iters", type=int, default=500)
    parser.add_argument("--ransac-sample-size", type=int, default=8)
    parser.add_argument("--trim-fraction", type=float, default=0.70)
    parser.add_argument("--refine-iters", type=int, default=20)
    parser.add_argument("--allow-reflection", dest="allow_reflection", action="store_true", default=True)
    parser.add_argument("--no-allow-reflection", dest="allow_reflection", action="store_false")
    parser.add_argument("--max-output-points", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    source_points = np.load(args.panovggt_output / "arrays" / f"000000_{args.source_space}_points.npy").astype(np.float32)
    source_mask = read_mask(args.panovggt_output / "arrays" / f"000000_{args.source_space}_valid_mask.png", source_points.shape[:2])
    gt_depth = read_depth_m(args.gt_depth, args.gt_meta, source_points.shape[:2])
    gt_mask = read_mask(args.gt_valid, source_points.shape[:2])
    gt_points = erp_points_from_range_depth(gt_depth)
    rgb = read_rgb(args.rgb, source_points.shape[:2])

    valid = source_mask & gt_mask & np.isfinite(source_points).all(axis=-1) & np.isfinite(gt_points).all(axis=-1)
    valid &= np.linalg.norm(source_points, axis=-1) > 0
    valid &= np.linalg.norm(gt_points, axis=-1) > 0
    source_pairs = source_points[valid]
    target_pairs = gt_points[valid]
    colors = rgb[valid]
    if source_pairs.shape[0] < 16:
        raise RuntimeError(f"Too few valid paired points: {source_pairs.shape[0]}")

    fit_src, fit_tgt = sample_pairs(source_pairs, target_pairs, args.max_fit_pairs, rng)
    result = robust_similarity_from_pairs(
        fit_src,
        fit_tgt,
        ransac_iters=args.ransac_iters,
        sample_size=args.ransac_sample_size,
        trim_fraction=args.trim_fraction,
        refine_iters=args.refine_iters,
        allow_reflection=args.allow_reflection,
        rng=rng,
    )
    aligned = apply_transform(source_pairs, result["scale"], result["orthogonal"], result["translation"])
    residual = np.linalg.norm(aligned - target_pairs, axis=1)

    inlier_threshold = float(np.percentile(residual, args.trim_fraction * 100.0))
    inliers = residual <= inlier_threshold
    output_points, output_colors = sample_points_and_colors(aligned, colors, args.max_output_points, rng)
    inlier_points, inlier_colors = sample_points_and_colors(aligned[inliers], colors[inliers], args.max_output_points, rng)
    target_output, target_colors = sample_points_and_colors(target_pairs, colors, args.max_output_points, rng)
    target_inlier_output, target_inlier_colors = sample_points_and_colors(target_pairs[inliers], colors[inliers], args.max_output_points, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / f"panovggt_{args.source_space}_pixel_aligned_to_gt.ply"
    aligned_inliers_path = args.output_dir / f"panovggt_{args.source_space}_pixel_aligned_inliers_to_gt.ply"
    target_path = args.output_dir / "gt_pixel_pairs_target.ply"
    target_inliers_path = args.output_dir / "gt_pixel_pairs_target_inliers.ply"
    write_ply(aligned_path, output_points, output_colors)
    write_ply(aligned_inliers_path, inlier_points, inlier_colors)
    write_ply(target_path, target_output, target_colors)
    write_ply(target_inliers_path, target_inlier_output, target_inlier_colors)

    summary = {
        "method": "robust_pixel_correspondence_similarity",
        "source_space": args.source_space,
        "source_points_npy": str(args.panovggt_output / "arrays" / f"000000_{args.source_space}_points.npy"),
        "gt_depth": str(args.gt_depth),
        "valid_pixel_pairs": int(source_pairs.shape[0]),
        "fit_pairs": int(fit_src.shape[0]),
        "aligned_ply": str(aligned_path),
        "aligned_inliers_ply": str(aligned_inliers_path),
        "target_ply": str(target_path),
        "target_inliers_ply": str(target_inliers_path),
        "inlier_pairs": int(inliers.sum()),
        "inlier_percentile": float(args.trim_fraction * 100.0),
        "inlier_threshold_m": inlier_threshold,
        "scale": float(result["scale"]),
        "orthogonal_determinant": float(np.linalg.det(result["orthogonal"])),
        "orthogonal": result["orthogonal"].tolist(),
        "translation": result["translation"].tolist(),
        "matrix": sim_matrix(result["scale"], result["orthogonal"], result["translation"]).tolist(),
        "fit_trim_rmse": float(result["trim_rmse"]),
        "fit_trim_median": float(result["trim_median"]),
        "paired_residual_median": float(np.median(residual)),
        "paired_residual_p90": float(np.percentile(residual, 90)),
        "paired_residual_p95": float(np.percentile(residual, 95)),
        "paired_residual_mean": float(np.mean(residual)),
    }
    (args.output_dir / f"alignment_summary_{args.source_space}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def read_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def read_depth_m(path: Path, meta_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    scale = 100.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scale = float((meta.get("depth_policy") or {}).get("output_depth_scale", scale))
    depth_m = depth.astype(np.float32) / scale
    return cv2.resize(depth_m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def read_rgb(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.full((shape[0], shape[1], 3), 220, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def erp_points_from_range_depth(depth_m: np.ndarray) -> np.ndarray:
    height, width = depth_m.shape
    y = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    x = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    vv, uu = np.meshgrid(y, x, indexing="ij")
    longitude = (uu * 2.0 - 1.0) * np.pi
    latitude = (0.5 - vv) * np.pi
    cos_lat = np.cos(latitude)
    rays = np.stack(
        [cos_lat * np.cos(longitude), cos_lat * np.sin(longitude), np.sin(latitude)],
        axis=-1,
    )
    return rays.astype(np.float32) * depth_m[..., None]


def sample_pairs(src: np.ndarray, tgt: np.ndarray, max_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if src.shape[0] <= max_pairs:
        return src, tgt
    keep = rng.choice(src.shape[0], size=max_pairs, replace=False)
    return src[keep], tgt[keep]


def robust_similarity_from_pairs(
    src: np.ndarray,
    tgt: np.ndarray,
    ransac_iters: int,
    sample_size: int,
    trim_fraction: float,
    refine_iters: int,
    allow_reflection: bool,
    rng: np.random.Generator,
) -> dict:
    eval_count = min(src.shape[0], 40000)
    eval_idx = rng.choice(src.shape[0], size=eval_count, replace=False) if src.shape[0] > eval_count else np.arange(src.shape[0])
    src_eval = src[eval_idx]
    tgt_eval = tgt[eval_idx]

    candidates = []
    for reflection in ([False, True] if allow_reflection else [False]):
        candidates.append(estimate_similarity(src_eval, tgt_eval, allow_reflection=reflection))
    for _ in range(ransac_iters):
        idx = rng.choice(src.shape[0], size=sample_size, replace=False)
        for reflection in ([False, True] if allow_reflection else [False]):
            try:
                candidates.append(estimate_similarity(src[idx], tgt[idx], allow_reflection=reflection))
            except np.linalg.LinAlgError:
                continue

    best = None
    trim_count = max(16, int(eval_count * trim_fraction))
    for candidate in candidates:
        transformed = apply_transform(src_eval, candidate["scale"], candidate["orthogonal"], candidate["translation"])
        residual = np.linalg.norm(transformed - tgt_eval, axis=1)
        keep = np.argpartition(residual, trim_count - 1)[:trim_count]
        rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
        if not math.isfinite(rmse):
            continue
        if best is None or rmse < best["trim_rmse"]:
            candidate["trim_rmse"] = rmse
            candidate["trim_median"] = float(np.median(residual[keep]))
            best = candidate

    if best is None:
        raise RuntimeError("No valid similarity candidate found.")

    current = best
    for _ in range(refine_iters):
        transformed = apply_transform(src, current["scale"], current["orthogonal"], current["translation"])
        residual = np.linalg.norm(transformed - tgt, axis=1)
        trim_count = max(32, int(src.shape[0] * trim_fraction))
        keep = np.argpartition(residual, trim_count - 1)[:trim_count]
        allow_reflect_this = np.linalg.det(current["orthogonal"]) < 0
        refined = estimate_similarity(src[keep], tgt[keep], allow_reflection=allow_reflect_this)
        transformed_keep = apply_transform(src[keep], refined["scale"], refined["orthogonal"], refined["translation"])
        residual_keep = np.linalg.norm(transformed_keep - tgt[keep], axis=1)
        refined["trim_rmse"] = float(np.sqrt(np.mean(residual_keep**2)))
        refined["trim_median"] = float(np.median(residual_keep))
        if abs(current["trim_rmse"] - refined["trim_rmse"]) < 1e-5:
            current = refined
            break
        current = refined
    return current


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


def apply_transform(points: np.ndarray, scale: float, orthogonal: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (points.astype(np.float64) @ orthogonal.T) + translation[None]).astype(np.float32)


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


def sim_matrix(scale: float, orthogonal: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * orthogonal
    matrix[:3, 3] = translation
    return matrix


if __name__ == "__main__":
    main()
