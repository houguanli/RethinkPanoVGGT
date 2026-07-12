#!/usr/bin/env python3
"""Calibrate native panorama camera conventions with GT depth reprojection."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.pano_minimal import (  # noqa: E402
    PanoMinimalDataset,
    _read_depth_tensor,
    _scene_group_key,
)
from training.train_pano_omega import erp_depth_to_range_depth  # noqa: E402


DATASET_NAMES = ("panocity", "matterport3d", "stanford2d3ds", "structured3d")
Y_FORWARD_Z_UP_WORLD_TO_OPENCV = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
WORLD_NATIVE_TO_CANONICAL = {
    "panocity": np.eye(3, dtype=np.float64),
    "matterport3d": Y_FORWARD_Z_UP_WORLD_TO_OPENCV,
    "stanford2d3ds": Y_FORWARD_Z_UP_WORLD_TO_OPENCV,
    "structured3d": Y_FORWARD_Z_UP_WORLD_TO_OPENCV,
}
DOCUMENTED_PANOVGGT_CAMERA_BASIS_CANONICAL_TO_NATIVE = {
    "panocity": np.eye(3, dtype=np.float64),
    "matterport3d": np.diag([1.0, -1.0, -1.0]),
    "stanford2d3ds": np.eye(3, dtype=np.float64),
    "structured3d": Y_FORWARD_Z_UP_WORLD_TO_OPENCV.T,
}


@dataclass
class PanoRecord:
    scene: str
    name: str
    depth: np.ndarray
    center: np.ndarray
    rotation_c2w: np.ndarray
    rotation_valid: bool


@dataclass
class PairRecord:
    scene: str
    first: PanoRecord
    second: PanoRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--calibration-pairs", type=int, default=12)
    parser.add_argument("--validation-pairs", type=int, default=8)
    parser.add_argument("--pairs-per-scene", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--points-per-direction", type=int, default=1024)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--depth-semantics", choices=["range", "cubemap_z", "double_cubemap_z"], default="range")
    parser.add_argument("--position-scales", default="0.01,0.03,0.1,0.3,1,3,10,30,100")
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)
    parser.add_argument("--refine-top", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = parse_datasets(args.datasets)
    scales = parse_positive_floats(args.position_scales)
    if args.height < 8 or args.width < 16:
        raise ValueError("Calibration resolution is too small.")
    if args.points_per_direction < 64:
        raise ValueError("points-per-direction must be >= 64.")

    result: dict[str, Any] = {
        "root": str(args.root.resolve()),
        "split": args.split,
        "depth_semantics": args.depth_semantics,
        "canonical_erp_axes": {
            "center_forward": [0.0, 0.0, 1.0],
            "image_right": [1.0, 0.0, 0.0],
            "image_up": [0.0, -1.0, 0.0],
        },
        "datasets": {},
    }
    for offset, dataset_name in enumerate(selected):
        print(f"[calibrate-camera] dataset={dataset_name}")
        dataset = PanoMinimalDataset(
            root=args.root,
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
            split=args.split,
            split_seed=args.seed,
            datasets=dataset_name,
            strict=True,
        )
        pairs = load_pairs(
            dataset,
            count=args.calibration_pairs + args.validation_pairs,
            pairs_per_scene=args.pairs_per_scene,
            height=args.height,
            width=args.width,
            depth_semantics=args.depth_semantics,
            seed=args.seed + offset * 1009,
        )
        calibration_pairs, validation_pairs = split_pairs_by_scene(
            pairs,
            calibration_count=args.calibration_pairs,
            validation_count=args.validation_pairs,
            seed=args.seed + offset * 2027,
        )
        if not calibration_pairs:
            raise RuntimeError(f"No calibration pairs found for {dataset_name}.")

        source_rays = erp_rays(args.height, args.width)
        sampled_calibration = prepare_pair_samples(
            calibration_pairs,
            source_rays,
            points_per_direction=args.points_per_direction,
            max_depth_m=args.max_depth_m,
            seed=args.seed + offset * 3001,
        )
        sampled_validation = prepare_pair_samples(
            validation_pairs,
            source_rays,
            points_per_direction=args.points_per_direction,
            max_depth_m=args.max_depth_m,
            seed=args.seed + offset * 4001,
        )

        identity = np.eye(3, dtype=np.float64)
        identity_cal = score_candidate(sampled_calibration, identity, 1.0)
        identity_val = score_candidate(sampled_validation, identity, 1.0)
        documented_matrix = DOCUMENTED_PANOVGGT_CAMERA_BASIS_CANONICAL_TO_NATIVE[dataset_name]
        documented_cal = score_candidate(sampled_calibration, documented_matrix, 1.0)
        documented_val = score_candidate(sampled_validation, documented_matrix, 1.0)
        coarse = search_coarse(sampled_calibration, scales)
        refined = search_refined(
            sampled_calibration,
            coarse[: max(1, args.refine_top)],
            yaw_step_deg=args.yaw_step_deg,
        )
        best = best_supported_candidate([*coarse, *refined])
        best_matrix = np.asarray(best["camera_basis_canonical_to_native"], dtype=np.float64)
        best_scale = float(best["position_scale_to_m"])
        best_validation = score_candidate(sampled_validation, best_matrix, best_scale)
        best_calibration = score_candidate(sampled_calibration, best_matrix, best_scale)
        fixed_coarse = search_coarse(sampled_calibration, [1.0])
        fixed_refined = search_refined(
            sampled_calibration,
            fixed_coarse[: max(1, args.refine_top)],
            yaw_step_deg=args.yaw_step_deg,
            refine_scale=False,
        )
        fixed_best = best_supported_candidate([*fixed_coarse, *fixed_refined])
        fixed_matrix = np.asarray(fixed_best["camera_basis_canonical_to_native"], dtype=np.float64)
        fixed_calibration = score_candidate(sampled_calibration, fixed_matrix, 1.0)
        fixed_validation = score_candidate(sampled_validation, fixed_matrix, 1.0)
        accepted = calibration_is_stable(identity_cal, fixed_calibration, identity_val, fixed_validation)

        dataset_result = {
            "items": len(dataset.items),
            "groups": len(dataset.groups),
            "calibration_pairs": pair_descriptions(calibration_pairs),
            "validation_pairs": pair_descriptions(validation_pairs),
            "rotation_gt_available": bool(
                any(pair.first.rotation_valid and pair.second.rotation_valid for pair in pairs)
            ),
            "identity": {
                "camera_basis_canonical_to_native": identity.tolist(),
                "position_scale_to_m": 1.0,
                "calibration": identity_cal,
                "validation": identity_val,
            },
            "official_panovggt": {
                "camera_basis_canonical_to_native": documented_matrix.tolist(),
                "position_scale_to_m": 1.0,
                "calibration": documented_cal,
                "validation": documented_val,
            },
            "best": {
                **{key: value for key, value in best.items() if key != "metrics"},
                "diagnostic_only": True,
                "calibration": best_calibration,
                "validation": best_validation,
            },
            "best_fixed_official_scale": {
                **{key: value for key, value in fixed_best.items() if key != "metrics"},
                "position_scale_to_m": 1.0,
                "calibration": fixed_calibration,
                "validation": fixed_validation,
            },
            "accepted": accepted,
            "acceptance_note": (
                "Fixed-scale camera basis improves held-out visible-surface reprojection."
                if accepted
                else "Do not infer a free camera transform from these pairs; use the documented loader convention."
            ),
            "top_candidates": coarse[:5],
        }
        result["datasets"][dataset_name] = dataset_result
        print(
            "[calibrate-camera] "
            f"dataset={dataset_name} accepted={accepted} "
            f"identity_val={identity_val['score']:.6f} "
            f"official_val={documented_val['score']:.6f} "
            f"fixed_val={fixed_validation['score']:.6f} "
            f"best_val={best_validation['score']:.6f} "
            f"scale={best_scale:.6g} axes={best['axis_map']} yaw={best['yaw_offset_deg']:.1f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[calibrate-camera] wrote {args.output.resolve()}")


def parse_datasets(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_NAMES)
    names = [value.strip().lower() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(names) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return names


def parse_positive_floats(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("position-scales must contain finite positive values.")
    return values


def canonicalize_camera_pose(
    dataset_name: str,
    center_native: np.ndarray,
    rotation_native: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_basis = WORLD_NATIVE_TO_CANONICAL[dataset_name]
    camera_basis = DOCUMENTED_PANOVGGT_CAMERA_BASIS_CANONICAL_TO_NATIVE[dataset_name]
    return world_basis @ center_native, world_basis @ rotation_native @ camera_basis


def load_pairs(
    dataset: PanoMinimalDataset,
    count: int,
    pairs_per_scene: int,
    height: int,
    width: int,
    depth_semantics: str,
    seed: int,
) -> list[PairRecord]:
    rng = random.Random(seed)
    groups_by_scene: dict[str, list[tuple[int, int]]] = {}
    seen: set[tuple[int, int]] = set()
    for group in dataset.groups:
        if len(group) < 2:
            continue
        pair = tuple(sorted((int(group[0]), int(group[1]))))
        if pair in seen:
            continue
        seen.add(pair)
        first_item, second_item = dataset.items[pair[0]], dataset.items[pair[1]]
        if not bool(first_item.get("pano_position_valid", True)):
            continue
        if not bool(second_item.get("pano_position_valid", True)):
            continue
        first_center = np.asarray(first_item.get("pano_position_m"), dtype=np.float64)
        second_center = np.asarray(second_item.get("pano_position_m"), dtype=np.float64)
        if first_center.shape != (3,) or second_center.shape != (3,):
            continue
        if np.linalg.norm(first_center - second_center) <= 1e-6:
            continue
        scene = _scene_group_key(first_item)
        groups_by_scene.setdefault(scene, []).append(pair)

    scenes = sorted(groups_by_scene)
    rng.shuffle(scenes)
    selected: list[PairRecord] = []
    for scene in scenes:
        scene_pairs = groups_by_scene[scene]
        rng.shuffle(scene_pairs)
        for first_index, second_index in scene_pairs[: max(1, pairs_per_scene)]:
            try:
                first = read_record(dataset.items[first_index], height, width, depth_semantics)
                second = read_record(dataset.items[second_index], height, width, depth_semantics)
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"[calibrate-camera][WARN] skipping pair {scene}: {exc}")
                continue
            selected.append(PairRecord(scene=scene, first=first, second=second))
            if len(selected) >= count:
                return selected
    return selected


def split_pairs_by_scene(
    pairs: list[PairRecord],
    calibration_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord]]:
    by_scene: dict[str, list[PairRecord]] = {}
    for pair in pairs:
        by_scene.setdefault(pair.scene, []).append(pair)
    scenes = sorted(by_scene)
    random.Random(seed).shuffle(scenes)
    validation_scene_count = max(1, round(len(scenes) * 0.35)) if len(scenes) > 1 else 0
    validation_scenes = set(scenes[:validation_scene_count])
    validation = [pair for pair in pairs if pair.scene in validation_scenes][:validation_count]
    calibration = [pair for pair in pairs if pair.scene not in validation_scenes][:calibration_count]
    if not calibration:
        calibration = pairs[:calibration_count]
    if not validation and len(pairs) > len(calibration):
        validation = pairs[len(calibration) : len(calibration) + validation_count]
    return calibration, validation


def read_record(item: dict[str, Any], height: int, width: int, depth_semantics: str) -> PanoRecord:
    depth = _read_depth_tensor(
        Path(item["depth_path"]),
        output_depth_scale=float(item["output_depth_scale"]),
        invalid_depth_value=65535.0,
    )
    if depth_semantics != "range":
        depth = erp_depth_to_range_depth(depth[None], depth_semantics)[0]
    depth_np = depth[0].numpy().astype(np.float32)
    depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_np[~np.isfinite(depth_np)] = np.nan
    rotation = np.asarray(item.get("pano_rotation_c2w"), dtype=np.float64)
    if rotation.shape != (3, 3):
        rotation = np.eye(3, dtype=np.float64)
    return PanoRecord(
        scene=_scene_group_key(item),
        name=str(item.get("scene_name") or Path(item["depth_path"]).stem),
        depth=depth_np,
        center=np.asarray(item.get("pano_position_m"), dtype=np.float64),
        rotation_c2w=orthonormalize(rotation),
        rotation_valid=bool(item.get("pano_rotation_valid", False)),
    )


def erp_rays(height: int, width: int) -> np.ndarray:
    v = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    u = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    theta = (uu - 0.5) * (2.0 * math.pi)
    phi = (vv - 0.5) * math.pi
    cos_phi = np.cos(phi)
    return np.stack(
        [cos_phi * np.sin(theta), np.sin(phi), cos_phi * np.cos(theta)],
        axis=-1,
    )


def prepare_pair_samples(
    pairs: Iterable[PairRecord],
    rays: np.ndarray,
    points_per_direction: int,
    max_depth_m: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    prepared: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        for direction_index, (source, target) in enumerate(
            ((pair.first, pair.second), (pair.second, pair.first))
        ):
            valid = np.isfinite(source.depth) & (source.depth > 0)
            if max_depth_m > 0:
                valid &= source.depth <= max_depth_m
            indices = np.flatnonzero(valid.reshape(-1))
            if len(indices) < 64:
                continue
            take = min(points_per_direction, len(indices))
            selected = rng.choice(indices, size=take, replace=False)
            prepared.append(
                {
                    "scene": pair.scene,
                    "pair_index": pair_index,
                    "direction_index": direction_index,
                    "source_rays": rays.reshape(-1, 3)[selected],
                    "source_depth": source.depth.reshape(-1)[selected].astype(np.float64),
                    "source_center": source.center,
                    "source_rotation": source.rotation_c2w,
                    "target_center": target.center,
                    "target_rotation": target.rotation_c2w,
                    "target_depth": target.depth,
                }
            )
    return prepared


def search_coarse(samples: list[dict[str, Any]], scales: list[float]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for matrix, axis_map in proper_axis_matrices():
        for scale in scales:
            metrics = score_candidate(samples, matrix, scale)
            candidates.append(
                {
                    "camera_basis_canonical_to_native": matrix.tolist(),
                    "axis_map": axis_map,
                    "yaw_offset_deg": 0.0,
                    "position_scale_to_m": float(scale),
                    "metrics": metrics,
                }
            )
    candidates.sort(key=candidate_rank)
    return candidates


def search_refined(
    samples: list[dict[str, Any]],
    coarse_candidates: list[dict[str, Any]],
    yaw_step_deg: float,
    refine_scale: bool = True,
) -> list[dict[str, Any]]:
    if yaw_step_deg <= 0:
        return []
    refined: list[dict[str, Any]] = []
    yaw_values = np.arange(-180.0, 180.0, yaw_step_deg, dtype=np.float64)
    for coarse in coarse_candidates:
        base = np.asarray(coarse["camera_basis_canonical_to_native"], dtype=np.float64)
        base_scale = float(coarse["position_scale_to_m"])
        scale_values = (
            sorted({base_scale * factor for factor in (0.5, 0.75, 1.0, 1.5, 2.0)})
            if refine_scale
            else [base_scale]
        )
        for yaw_deg in yaw_values:
            matrix = base @ rotation_about_y(math.radians(float(yaw_deg)))
            for scale in scale_values:
                metrics = score_candidate(samples, matrix, scale)
                refined.append(
                    {
                        "camera_basis_canonical_to_native": matrix.tolist(),
                        "axis_map": coarse["axis_map"],
                        "yaw_offset_deg": float(yaw_deg),
                        "position_scale_to_m": float(scale),
                        "metrics": metrics,
                    }
                )
    refined.sort(key=candidate_rank)
    return refined


def candidate_rank(candidate: dict[str, Any]) -> float:
    metrics = candidate["metrics"]
    score = float(metrics["score"])
    if not math.isfinite(score) or int(metrics["valid_points"]) < 64:
        return float("inf")
    if float(metrics["coverage"]) < 0.05:
        return float("inf")
    return score


def best_supported_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [candidate for candidate in candidates if math.isfinite(candidate_rank(candidate))]
    if supported:
        return min(supported, key=candidate_rank)
    return min(candidates, key=lambda candidate: float(candidate["metrics"]["score"]))


def score_candidate(
    samples: list[dict[str, Any]],
    camera_basis: np.ndarray,
    position_scale: float,
) -> dict[str, Any]:
    all_errors: list[np.ndarray] = []
    total_requested = 0
    total_projected = 0
    pair_scores: list[float] = []
    for sample in samples:
        reprojection = reproject_sample_depth(sample, camera_basis, position_scale)
        comparable = reprojection["comparable_mask"].reshape(-1)
        overlap = reprojection["overlap_mask"].reshape(-1)
        total_requested += int(comparable.sum())
        total_projected += int(reprojection["projected_mask"].sum())
        valid = overlap
        if not np.any(valid):
            continue
        errors = reprojection["abs_log_error"].reshape(-1)[valid]
        all_errors.append(errors)
        pair_scores.append(float(np.mean(np.minimum(errors, 0.5))))

    if not all_errors:
        return {
            "score": float("inf"),
            "valid_points": 0,
            "coverage": 0.0,
            "overlap_ratio": 0.0,
            "projected_valid_points": int(total_requested),
            "projected_points": int(total_projected),
            "median_abs_log_error": float("inf"),
            "p90_abs_log_error": float("inf"),
            "inlier_5pct": 0.0,
            "inlier_10pct": 0.0,
            "inlier_20pct": 0.0,
            "pair_score_std": float("inf"),
        }
    errors = np.concatenate(all_errors)
    coverage = float(len(errors) / max(total_requested, 1))
    clipped_mean = float(np.mean(np.minimum(errors, 0.5)))
    # Keep ranking quality separate from the overlap-only error. A candidate
    # cannot win by producing a tiny number of accidental depth matches.
    score = clipped_mean + 0.5 * (1.0 - coverage)
    return {
        "score": score,
        "valid_points": int(len(errors)),
        "coverage": coverage,
        "overlap_ratio": coverage,
        "projected_valid_points": int(total_requested),
        "projected_points": int(total_projected),
        "median_abs_log_error": float(np.median(errors)),
        "p90_abs_log_error": float(np.quantile(errors, 0.9)),
        "inlier_5pct": float(np.mean(errors < math.log(1.05))),
        "inlier_10pct": float(np.mean(errors < math.log(1.10))),
        "inlier_20pct": float(np.mean(errors < math.log(1.20))),
        "pair_score_mean": float(np.mean(pair_scores)),
        "pair_score_std": float(np.std(pair_scores)),
    }


def reproject_sample_depth(
    sample: dict[str, Any],
    camera_basis: np.ndarray,
    position_scale: float,
) -> dict[str, np.ndarray]:
    """Reproject one sampled source depth map into the target ERP."""
    source_rays = sample["source_rays"]
    source_depth = sample["source_depth"]
    native_rays = (camera_basis @ source_rays.T).T
    world_rays = (sample["source_rotation"] @ native_rays.T).T
    world_points = (
        float(position_scale) * sample["source_center"][None, :]
        + source_depth[:, None] * world_rays
    )
    target_native = (
        sample["target_rotation"].T
        @ (
            world_points
            - float(position_scale) * sample["target_center"][None, :]
        ).T
    ).T
    target_canonical = (camera_basis.T @ target_native.T).T
    predicted_range = np.linalg.norm(target_canonical, axis=-1)
    valid_direction = np.isfinite(predicted_range) & (predicted_range > 1e-6)
    directions = target_canonical / np.maximum(predicted_range[:, None], 1e-8)
    u, v = rays_to_erp(directions)
    height, width = sample["target_depth"].shape
    x = np.mod(np.floor(u * width).astype(np.int64), width)
    y = np.clip(np.floor(v * height).astype(np.int64), 0, height - 1)
    pixel_index = y * width + x
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, pixel_index[valid_direction], predicted_range[valid_direction])
    z_buffer = z_buffer.reshape(height, width)
    target_depth = sample["target_depth"].astype(np.float64)
    projected = np.isfinite(z_buffer)
    target_valid = np.isfinite(target_depth) & (target_depth > 0)
    comparable = projected & target_valid
    error = np.full((height, width), np.nan, dtype=np.float64)
    error[comparable] = np.abs(
        np.log(np.maximum(z_buffer[comparable], 1e-6))
        - np.log(np.maximum(target_depth[comparable], 1e-6))
    )
    # A projected point belongs to the same visible surface only when both
    # views agree in depth. Twenty percent is used to identify overlap; the
    # reported 5/10-percent inlier metrics remain stricter diagnostics.
    overlap = comparable & (error <= math.log(1.20))
    return {
        "target_depth": target_depth,
        "reprojected_depth": z_buffer,
        "projected_mask": projected,
        "comparable_mask": comparable,
        "overlap_mask": overlap,
        "abs_log_error": error,
    }


def rays_to_erp(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)
    theta = np.arctan2(normalized[:, 0], normalized[:, 2])
    phi = np.arcsin(np.clip(normalized[:, 1], -1.0, 1.0))
    u = np.mod(theta / (2.0 * math.pi) + 0.5, 1.0)
    v = np.clip(0.5 + phi / math.pi, 0.0, 1.0)
    return u, v


def sample_target_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    x = np.mod(np.floor(u * width).astype(np.int64), width)
    y = np.clip(np.floor(v * height).astype(np.int64), 0, height - 1)
    return depth[y, x].astype(np.float64)


def proper_axis_matrices() -> Iterable[tuple[np.ndarray, str]]:
    axes = ("X", "Y", "Z")
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            labels = []
            for canonical_axis, native_axis in enumerate(permutation):
                matrix[native_axis, canonical_axis] = signs[canonical_axis]
                prefix = "+" if signs[canonical_axis] > 0 else "-"
                labels.append(prefix + axes[native_axis])
            if np.linalg.det(matrix) > 0.5:
                yield matrix, ",".join(labels)


def rotation_about_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def calibration_is_stable(
    identity_calibration: dict[str, Any],
    best_calibration: dict[str, Any],
    identity_validation: dict[str, Any],
    best_validation: dict[str, Any],
) -> bool:
    if best_validation["valid_points"] < 512:
        return False
    if best_validation["coverage"] < 0.25:
        return False
    calibration_gain = identity_calibration["score"] - best_calibration["score"]
    validation_gain = identity_validation["score"] - best_validation["score"]
    return bool(calibration_gain > 0.01 and validation_gain > 0.005)


def pair_descriptions(pairs: Iterable[PairRecord]) -> list[dict[str, str]]:
    return [
        {"scene": pair.scene, "first": pair.first.name, "second": pair.second.name}
        for pair in pairs
    ]


if __name__ == "__main__":
    main()
