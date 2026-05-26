#!/usr/bin/env python3
"""Robustly align a PanoVGGT baseline point cloud to the GT point cloud.

The alignment is a Sim(3): uniform scale, proper rotation, and translation.
It uses multiple PCA/axis-permutation initializations followed by trimmed
nearest-neighbor ICP with Umeyama similarity updates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


METHOD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHOD_ROOT.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Rethink_pano_new_exp_omega" / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "panovggt_baseline_sample0" / "pointclouds" / "merged.ply",
        help="PanoVGGT source PLY to be aligned.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "luna_range_v2_single4090_1h_delta_20260526"
        / "validation_sample0_gtmask_trained"
        / "target_erp_points.ply",
        help="GT target PLY.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "panovggt_baseline_sample0" / "aligned_to_gt",
    )
    parser.add_argument("--max-source-points", type=int, default=70000)
    parser.add_argument("--max-target-points", type=int, default=90000)
    parser.add_argument("--max-output-points", type=int, default=400000)
    parser.add_argument("--trim-fraction", type=float, default=0.65)
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    source_points, source_colors = read_ply(args.source)
    target_points, _ = read_ply(args.target)

    source_fit = random_sample(source_points, args.max_source_points, rng)
    target_fit = random_sample(target_points, args.max_target_points, rng)
    init_transforms = build_initial_transforms(source_fit, target_fit)

    best = None
    for idx, init in enumerate(init_transforms):
        result = trimmed_similarity_icp(
            source_fit,
            target_fit,
            init,
            trim_fraction=args.trim_fraction,
            iterations=args.iterations,
        )
        result["candidate_index"] = idx
        if best is None or result["trim_rmse"] < best["trim_rmse"]:
            best = result

    if best is None:
        raise RuntimeError("No alignment candidate was evaluated.")

    aligned_points = apply_transform(source_points, best["scale"], best["rotation"], best["translation"])
    output_points, output_colors = sample_points_and_colors(
        aligned_points,
        source_colors,
        args.max_output_points,
        rng,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "panovggt_merged_aligned_to_gt.ply"
    write_ply(aligned_path, output_points, output_colors)

    metrics = evaluate_alignment(aligned_points, target_points, rng)
    summary = {
        "source": str(args.source),
        "target": str(args.target),
        "aligned_ply": str(aligned_path),
        "source_points": int(source_points.shape[0]),
        "target_points": int(target_points.shape[0]),
        "output_points": int(output_points.shape[0]),
        "fit_source_points": int(source_fit.shape[0]),
        "fit_target_points": int(target_fit.shape[0]),
        "trim_fraction": args.trim_fraction,
        "iterations": args.iterations,
        "best_candidate_index": int(best["candidate_index"]),
        "scale": float(best["scale"]),
        "rotation": best["rotation"].tolist(),
        "translation": best["translation"].tolist(),
        "sim3_matrix": sim3_matrix(best["scale"], best["rotation"], best["translation"]).tolist(),
        "fit_trim_rmse": float(best["trim_rmse"]),
        "fit_trim_median": float(best["trim_median"]),
        **metrics,
    }
    (args.output_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        vertex_count = None
        for line in f:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Could not read vertex count from {path}")
        data = np.loadtxt(f, dtype=np.float32, max_rows=vertex_count)
    if data.ndim == 1:
        data = data[None]
    points = data[:, :3].astype(np.float32, copy=False)
    if data.shape[1] >= 6:
        colors = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


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


def random_sample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.copy()
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


def build_initial_transforms(source: np.ndarray, target: np.ndarray) -> list[dict]:
    src_center, src_scale = robust_center_scale(source)
    tgt_center, tgt_scale = robust_center_scale(target)
    src_basis = pca_basis(source, src_center)
    tgt_basis = pca_basis(target, tgt_center)

    rotations = []
    rotations.extend(signed_permutation_rotations())
    for perm in signed_permutation_rotations():
        rotations.append(tgt_basis @ perm @ src_basis.T)

    unique = []
    seen = set()
    for rotation in rotations:
        rotation = project_to_rotation(rotation)
        key = tuple(np.round(rotation, 4).reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rotation)

    scale = tgt_scale / max(src_scale, 1e-6)
    transforms = []
    for rotation in unique:
        translation = tgt_center - scale * (src_center @ rotation.T)
        transforms.append({"scale": scale, "rotation": rotation, "translation": translation})
    return transforms


def robust_center_scale(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    scale = float(np.percentile(radius, 90))
    return center.astype(np.float64), max(scale, 1e-6)


def pca_basis(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = points.astype(np.float64) - center[None]
    cov = np.cov(centered, rowvar=False)
    _, vecs = np.linalg.eigh(cov)
    basis = vecs[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    return basis


def signed_permutation_rotations() -> Iterable[np.ndarray]:
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for row, col in enumerate(perm):
            base[row, col] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.diag(signs) @ base
            if np.linalg.det(matrix) > 0:
                yield matrix


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def trimmed_similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    init: dict,
    trim_fraction: float,
    iterations: int,
) -> dict:
    tree = cKDTree(target)
    scale = float(init["scale"])
    rotation = init["rotation"].astype(np.float64)
    translation = init["translation"].astype(np.float64)
    trim_count = max(32, int(source.shape[0] * trim_fraction))

    best = None
    prev_rmse = math.inf
    for _ in range(iterations):
        transformed = apply_transform(source, scale, rotation, translation)
        distances, indices = tree.query(transformed, k=1, workers=-1)
        keep = np.argpartition(distances, trim_count - 1)[:trim_count]
        matched = target[indices[keep]]
        scale_new, rotation_new, translation_new = estimate_similarity_umeyama(source[keep], matched)
        transformed_keep = apply_transform(source[keep], scale_new, rotation_new, translation_new)
        residual = np.linalg.norm(transformed_keep - matched, axis=1)
        rmse = float(np.sqrt(np.mean(residual**2)))
        median = float(np.median(residual))
        current = {
            "scale": scale_new,
            "rotation": rotation_new,
            "translation": translation_new,
            "trim_rmse": rmse,
            "trim_median": median,
        }
        if best is None or rmse < best["trim_rmse"]:
            best = current
        scale, rotation, translation = scale_new, rotation_new, translation_new
        if abs(prev_rmse - rmse) < 1e-5:
            break
        prev_rmse = rmse
    return best


def estimate_similarity_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    cov = (tgt_centered.T @ src_centered) / max(src.shape[0], 1)
    u, singular, vt = np.linalg.svd(cov)
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1.0
    rotation = u @ np.diag(signs) @ vt
    var_src = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.sum(singular * signs) / max(var_src, 1e-12))
    translation = tgt_mean - scale * (src_mean @ rotation.T)
    return scale, rotation, translation


def apply_transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (points.astype(np.float64) @ rotation.T) + translation[None]).astype(np.float32)


def evaluate_alignment(
    aligned_source: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    max_eval_points: int = 80000,
) -> dict:
    src_eval = random_sample(aligned_source, max_eval_points, rng)
    tgt_eval = random_sample(target, max_eval_points, rng)
    target_tree = cKDTree(tgt_eval)
    source_tree = cKDTree(src_eval)
    src_to_tgt, _ = target_tree.query(src_eval, k=1, workers=-1)
    tgt_to_src, _ = source_tree.query(tgt_eval, k=1, workers=-1)
    return {
        "eval_source_to_target_median": float(np.median(src_to_tgt)),
        "eval_source_to_target_p90": float(np.percentile(src_to_tgt, 90)),
        "eval_target_to_source_median": float(np.median(tgt_to_src)),
        "eval_target_to_source_p90": float(np.percentile(tgt_to_src, 90)),
        "eval_symmetric_median": float(0.5 * (np.median(src_to_tgt) + np.median(tgt_to_src))),
        "eval_symmetric_p90": float(0.5 * (np.percentile(src_to_tgt, 90) + np.percentile(tgt_to_src, 90))),
    }


def sim3_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


if __name__ == "__main__":
    main()
