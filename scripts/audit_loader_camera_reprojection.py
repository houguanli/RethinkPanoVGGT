#!/usr/bin/env python3
"""Audit loader camera poses by GT-depth reprojection.

This script checks the exact camera tensors produced by PanoMinimalDataset,
instead of fitting an extra camera basis in a diagnostic script. It answers the
question needed before camera supervision is trusted:

  dataset files -> PanoMinimalDataset pose/depth -> reprojection

If this path does not agree with official GT depth inside a scene, the training
camera targets are not meaningful yet.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.pano_minimal import PanoMinimalDataset, _scene_group_key  # noqa: E402


DATASET_NAMES = ("panocity", "matterport3d", "stanford2d3ds", "structured3d")
RAY_CONVENTIONS = ("training_y_up", "official_y_down")
TILE_GAP_PX = 8
ROW_GAP_PX = 10
DIRECTION_GAP_PX = 18
GAP_COLOR = (52, 52, 52)


@dataclass
class LoaderRecord:
    scene: str
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    center: np.ndarray
    rotation_c2w: np.ndarray
    position_valid: bool
    rotation_valid: bool
    rgb_path: str
    depth_path: str


@dataclass
class LoaderPair:
    dataset: str
    scene: str
    first: LoaderRecord
    second: LoaderRecord


def main() -> None:
    args = build_parser().parse_args()
    selected = parse_datasets(args.datasets)
    conventions = parse_ray_conventions(args.ray_conventions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "root": str(args.root.resolve()),
        "split": args.split,
        "height": args.height,
        "width": args.width,
        "max_depth_m": args.max_depth_m,
        "ray_conventions": list(conventions),
        "note": (
            "All poses/depths are read through PanoMinimalDataset. "
            "training_y_up matches vggt_omega.models.layers.pano_position; "
            "official_y_down matches the Matterport official OpenCV-style "
            "GT reprojection diagnostic."
        ),
        "datasets": {},
    }

    for offset, dataset_name in enumerate(selected):
        print(f"[loader-camera-audit] dataset={dataset_name}")
        dataset = PanoMinimalDataset(
            root=args.root,
            pano_size=(args.height, args.width),
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
            split=args.split,
            split_seed=args.seed + offset * 1009,
            datasets=dataset_name,
            bad_sample_list=args.bad_sample_list,
            strict=True,
        )
        pairs = select_pairs(
            dataset=dataset,
            dataset_name=dataset_name,
            count=args.pairs_per_dataset,
            seed=args.seed + offset * 2027,
            max_depth_m=args.max_depth_m,
        )
        if not pairs:
            raise RuntimeError(f"No valid same-scene camera pairs found for {dataset_name}.")

        dataset_summary: dict[str, Any] = {
            "items": len(dataset.items),
            "groups": len(dataset.groups),
            "pairs": [],
            "aggregate": {},
        }
        aggregate_values: dict[str, list[dict[str, float]]] = {name: [] for name in conventions}
        for pair_index, pair in enumerate(pairs):
            pair_entry: dict[str, Any] = {
                "scene": pair.scene,
                "first": pair.first.name,
                "second": pair.second.name,
                "first_rgb_path": pair.first.rgb_path,
                "second_rgb_path": pair.second.rgb_path,
                "first_depth_path": pair.first.depth_path,
                "second_depth_path": pair.second.depth_path,
                "first_center_m": pair.first.center.tolist(),
                "second_center_m": pair.second.center.tolist(),
                "baseline_m": float(np.linalg.norm(pair.first.center - pair.second.center)),
                "first_rotation_valid": bool(pair.first.rotation_valid),
                "second_rotation_valid": bool(pair.second.rotation_valid),
                "identity_checks": {},
                "directions": {},
            }
            panels: list[np.ndarray] = []
            for convention in conventions:
                rays = erp_rays(args.height, args.width, convention)
                pair_entry["identity_checks"][convention] = {
                    label: reproject_record(
                        source=record,
                        target=record,
                        rays=rays,
                        convention=convention,
                        max_depth_m=args.max_depth_m,
                    )["metrics"]
                    for label, record in (("first_to_self", pair.first), ("second_to_self", pair.second))
                }
                direction_entries = []
                for direction_index, (source, target) in enumerate(
                    ((pair.first, pair.second), (pair.second, pair.first))
                ):
                    result = reproject_record(
                        source=source,
                        target=target,
                        rays=rays,
                        convention=convention,
                        max_depth_m=args.max_depth_m,
                    )
                    direction_entries.append(
                        {
                            "direction": "A_to_B" if direction_index == 0 else "B_to_A",
                            "source": source.name,
                            "target": target.name,
                            "source_rgb_path": source.rgb_path,
                            "target_rgb_path": target.rgb_path,
                            "source_depth_path": source.depth_path,
                            "target_depth_path": target.depth_path,
                            "metrics": result["metrics"],
                        }
                    )
                    aggregate_values[convention].append(result["metrics"])
                    if pair_index < args.visual_pairs:
                        panels.append(
                            make_direction_panel(
                                dataset_name=dataset_name,
                                convention=convention,
                                source=source,
                                target=target,
                                result=result,
                                direction="A -> B" if direction_index == 0 else "B -> A",
                            )
                        )
                pair_entry["directions"][convention] = direction_entries

            dataset_summary["pairs"].append(pair_entry)
            if panels:
                image_path = args.output_dir / f"{dataset_name}_pair{pair_index:02d}_loader_camera_reprojection.png"
                cv2.imwrite(str(image_path), stack_panels(panels))
                pair_entry["visualization"] = str(image_path.resolve())
                print(f"[loader-camera-audit] wrote {image_path.resolve()}")

        for convention, values in aggregate_values.items():
            dataset_summary["aggregate"][convention] = aggregate_metrics(values)
        summary["datasets"][dataset_name] = dataset_summary
        for convention, metrics in dataset_summary["aggregate"].items():
            print(
                "[loader-camera-audit] "
                f"{dataset_name} {convention} "
                f"coverage={metrics['target_coverage_mean']:.4f} "
                f"median_abs_log={metrics['median_abs_log_error_median']:.4f} "
                f"inlier10={metrics['inlier_10pct_mean']:.4f} "
                f"inlier20={metrics['inlier_20pct_mean']:.4f}"
            )

    summary_path = args.output_dir / "loader_camera_reprojection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[loader-camera-audit] wrote {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Minimal PanoVGGT dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pairs-per-dataset", type=int, default=4)
    parser.add_argument("--visual-pairs", type=int, default=2)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ray-conventions", default="training_y_up,official_y_down")
    parser.add_argument("--bad-sample-list", type=Path, default=None)
    return parser


def parse_datasets(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_NAMES)
    aliases = {
        "mp3d": "matterport3d",
        "matterport": "matterport3d",
        "stanford": "stanford2d3ds",
        "2d3ds": "stanford2d3ds",
        "s3d": "structured3d",
        "pano_city": "panocity",
    }
    selected = [aliases.get(value.strip().lower(), value.strip().lower()) for value in raw.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return selected


def parse_ray_conventions(raw: str) -> tuple[str, ...]:
    selected = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(selected) - set(RAY_CONVENTIONS))
    if unknown:
        raise ValueError(f"Unknown ray conventions: {unknown}")
    if not selected:
        raise ValueError("At least one ray convention is required.")
    return selected


def select_pairs(
    dataset: PanoMinimalDataset,
    dataset_name: str,
    count: int,
    seed: int,
    max_depth_m: float,
) -> list[LoaderPair]:
    group_indices = list(range(len(dataset.groups)))
    random.Random(seed).shuffle(group_indices)
    pairs: list[LoaderPair] = []
    seen_scenes: set[str] = set()
    for group_index in group_indices:
        group = dataset.groups[group_index]
        if len(group) < 2:
            continue
        scene_keys = {_scene_group_key(dataset.items[item_index]) for item_index in group[:2]}
        if len(scene_keys) != 1:
            raise RuntimeError(f"Dataset group crosses scenes before reading: {sorted(scene_keys)}")
        scene = next(iter(scene_keys))
        # Spread diagnostics across scenes first. If a dataset has fewer scenes
        # than requested pairs, a later pass below can still add extra pairs.
        if scene in seen_scenes and len(seen_scenes) >= count:
            continue
        try:
            samples = dataset._read_group_with_scene_fallback(group[:2])  # noqa: SLF001
        except (FileNotFoundError, OSError, ValueError, RuntimeError, SyntaxError) as exc:
            print(f"[loader-camera-audit][WARN] skipping group {scene}: {exc}")
            continue
        records = [record_from_sample(sample, scene) for sample in samples[:2]]
        if not all(record.position_valid for record in records):
            continue
        if np.linalg.norm(records[0].center - records[1].center) <= 1e-6:
            continue
        if min(valid_depth_count(record.depth, max_depth_m) for record in records) < 64:
            continue
        pairs.append(LoaderPair(dataset=dataset_name, scene=scene, first=records[0], second=records[1]))
        seen_scenes.add(scene)
        if len(pairs) >= count:
            return pairs
    return pairs


def record_from_sample(sample: dict[str, Any], scene: str) -> LoaderRecord:
    rgb = sample["pano_image"].detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)[..., ::-1].copy()
    depth = sample["pano_depth"][0].detach().cpu().numpy().astype(np.float32)
    depth[(~np.isfinite(depth)) | (depth <= 0.0)] = np.nan
    rotation = sample["pano_rotation_c2w"].detach().cpu().numpy().astype(np.float64)
    return LoaderRecord(
        scene=scene,
        name=str(sample["scene_name"]),
        rgb=rgb,
        depth=depth,
        center=sample["pano_position_m"].detach().cpu().numpy().astype(np.float64),
        rotation_c2w=orthonormalize(rotation),
        position_valid=bool(sample["pano_position_valid"].detach().cpu().item()),
        rotation_valid=bool(sample["pano_rotation_valid"].detach().cpu().item()),
        rgb_path=str(sample["rgb_path"]),
        depth_path=str(sample["depth_path"]),
    )


def valid_depth_count(depth: np.ndarray, max_depth_m: float) -> int:
    valid = np.isfinite(depth) & (depth > 0.0)
    if max_depth_m > 0:
        valid &= depth <= max_depth_m
    return int(np.count_nonzero(valid))


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        return np.eye(3, dtype=np.float64)
    u, _s, vh = np.linalg.svd(rotation.astype(np.float64))
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def erp_rays(height: int, width: int, convention: str) -> np.ndarray:
    if height < 2 or width < 2:
        raise ValueError("height and width must be >= 2")
    v = np.arange(height, dtype=np.float64) / float(height - 1)
    u = np.arange(width, dtype=np.float64) / float(width - 1)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    theta = (uu - 0.5) * (2.0 * math.pi)
    if convention == "training_y_up":
        phi = (0.5 - vv) * math.pi
        y = np.sin(phi)
    elif convention == "official_y_down":
        phi = -(vv - 0.5) * math.pi
        y = -np.sin(phi)
    else:
        raise ValueError(f"Unknown ray convention: {convention}")
    cos_phi = np.cos(phi)
    rays = np.stack([cos_phi * np.sin(theta), y, cos_phi * np.cos(theta)], axis=-1)
    return rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)


def rays_to_erp(directions: np.ndarray, convention: str, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    directions = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    theta = np.arctan2(directions[:, 0], directions[:, 2])
    if convention == "training_y_up":
        phi = np.arcsin(np.clip(directions[:, 1], -1.0, 1.0))
        u = np.rint((theta / (2.0 * math.pi) + 0.5) * (width - 1)).astype(np.int64)
        v = np.rint((0.5 - phi / math.pi) * (height - 1)).astype(np.int64)
    elif convention == "official_y_down":
        phi = np.arcsin(np.clip(directions[:, 1], -1.0, 1.0))
        u = np.rint((theta / (2.0 * math.pi) + 0.5) * (width - 1)).astype(np.int64)
        v = np.rint((0.5 + phi / math.pi) * (height - 1)).astype(np.int64)
    else:
        raise ValueError(f"Unknown ray convention: {convention}")
    return np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)


def reproject_record(
    source: LoaderRecord,
    target: LoaderRecord,
    rays: np.ndarray,
    convention: str,
    max_depth_m: float,
) -> dict[str, Any]:
    height, width = source.depth.shape
    valid_source = np.isfinite(source.depth) & (source.depth > 0.0)
    if max_depth_m > 0:
        valid_source &= source.depth <= max_depth_m
    source_depth = source.depth[valid_source].astype(np.float64)
    source_rays = rays[valid_source].astype(np.float64)
    points_world = source.center[None, :] + source_depth[:, None] * (source.rotation_c2w @ source_rays.T).T
    points_target = (target.rotation_c2w.T @ (points_world - target.center[None, :]).T).T
    target_range = np.linalg.norm(points_target, axis=-1)
    valid_target = np.isfinite(target_range) & (target_range > 1e-6)
    points_target = points_target[valid_target]
    target_range = target_range[valid_target]
    directions = points_target / target_range[:, None]
    u, v = rays_to_erp(directions, convention, height, width)

    linear = v * width + u
    order = np.lexsort((target_range, linear))
    linear_sorted = linear[order]
    first = np.concatenate(([True], linear_sorted[1:] != linear_sorted[:-1]))
    projected = np.full(height * width, np.nan, dtype=np.float32)
    projected[linear_sorted[first]] = target_range[order[first]].astype(np.float32)
    projected = projected.reshape(height, width)

    target_valid = np.isfinite(target.depth) & (target.depth > 0.0)
    if max_depth_m > 0:
        target_valid &= target.depth <= max_depth_m
    comparable = np.isfinite(projected) & target_valid
    abs_log = np.full((height, width), np.nan, dtype=np.float32)
    abs_rel = np.full((height, width), np.nan, dtype=np.float32)
    abs_log[comparable] = np.abs(
        np.log(np.maximum(projected[comparable], 1e-6))
        - np.log(np.maximum(target.depth[comparable], 1e-6))
    )
    abs_rel[comparable] = np.abs(projected[comparable] - target.depth[comparable]) / np.maximum(
        target.depth[comparable],
        1e-6,
    )
    inlier20 = comparable & (abs_rel <= 0.20)
    target_on_comparable = np.full((height, width), np.nan, dtype=np.float32)
    target_on_comparable[comparable] = target.depth[comparable]
    consistent_projected = np.full((height, width), np.nan, dtype=np.float32)
    consistent_projected[inlier20] = projected[inlier20]
    metrics = build_metrics(valid_source, projected, target_valid, comparable, abs_log, abs_rel)
    return {
        "projected_depth": projected,
        "target_on_comparable": target_on_comparable,
        "consistent_projected_depth": consistent_projected,
        "comparable_mask": comparable,
        "abs_log_error": abs_log,
        "inlier20_mask": inlier20,
        "metrics": metrics,
    }


def build_metrics(
    valid_source: np.ndarray,
    projected: np.ndarray,
    target_valid: np.ndarray,
    comparable: np.ndarray,
    abs_log: np.ndarray,
    abs_rel: np.ndarray,
) -> dict[str, float]:
    log_values = abs_log[comparable & np.isfinite(abs_log)]
    rel_values = abs_rel[comparable & np.isfinite(abs_rel)]
    comparable_count = int(np.count_nonzero(comparable))
    target_valid_count = int(np.count_nonzero(target_valid))
    return {
        "source_valid_pixels": float(np.count_nonzero(valid_source)),
        "projected_pixels": float(np.count_nonzero(np.isfinite(projected))),
        "target_valid_pixels": float(target_valid_count),
        "comparable_pixels": float(comparable_count),
        "target_coverage": float(comparable_count / max(1, target_valid_count)),
        "median_abs_log_error": finite_stat(log_values, np.median),
        "mean_abs_log_error": finite_stat(log_values, np.mean),
        "p90_abs_log_error": finite_stat(log_values, lambda values: np.quantile(values, 0.9)),
        "inlier_05pct": finite_ratio(rel_values, 0.05),
        "inlier_10pct": finite_ratio(rel_values, 0.10),
        "inlier_20pct": finite_ratio(rel_values, 0.20),
    }


def finite_stat(values: np.ndarray, fn) -> float:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else float("nan")


def finite_ratio(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite <= threshold)) if finite.size else 0.0


def aggregate_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = [
        "target_coverage",
        "median_abs_log_error",
        "mean_abs_log_error",
        "p90_abs_log_error",
        "inlier_05pct",
        "inlier_10pct",
        "inlier_20pct",
    ]
    output = {"num_directions": float(len(values))}
    for key in keys:
        series = np.asarray([value[key] for value in values], dtype=np.float64)
        series = series[np.isfinite(series)]
        output[f"{key}_mean"] = float(series.mean()) if series.size else float("nan")
        output[f"{key}_median"] = float(np.median(series)) if series.size else float("nan")
    return output


def make_direction_panel(
    dataset_name: str,
    convention: str,
    source: LoaderRecord,
    target: LoaderRecord,
    result: dict[str, Any],
    direction: str,
) -> np.ndarray:
    metrics = result["metrics"]
    finite_depth = np.concatenate(
        [source.depth[np.isfinite(source.depth)], target.depth[np.isfinite(target.depth)]]
    )
    depth_max = float(np.quantile(finite_depth, 0.98)) if finite_depth.size else 10.0
    top_tiles = [
        rgb_tile(source.rgb, f"Source RGB: {source.name}"),
        rgb_tile(target.rgb, f"Target RGB: {target.name}"),
        depth_tile(source.depth, depth_max, f"Source GT depth: {source.name}"),
        depth_tile(target.depth, depth_max, f"Target GT depth: {target.name}"),
    ]
    bottom_tiles = [
        depth_tile(result["projected_depth"], depth_max, "Raw source splat in target ERP"),
        depth_tile(result["target_on_comparable"], depth_max, "Target GT at raw-splat pixels"),
        depth_tile(
            result["consistent_projected_depth"],
            depth_max,
            "GT-consistent reprojection (relative error <= 20%)",
        ),
        error_tile(result["abs_log_error"], result["comparable_mask"], "Abs-log error at raw-splat pixels"),
    ]
    panel = join_images(
        [
            join_images(top_tiles, axis=1, gap=TILE_GAP_PX),
            join_images(bottom_tiles, axis=1, gap=TILE_GAP_PX),
        ],
        axis=0,
        gap=ROW_GAP_PX,
    )
    title = (
        f"{dataset_name} {convention} | direction {direction} | {source.name} -> {target.name} | "
        f"coverage={metrics['target_coverage']:.3f} "
        f"med_log={metrics['median_abs_log_error']:.3f} "
        f"in10={metrics['inlier_10pct']:.3f} in20={metrics['inlier_20pct']:.3f}"
    )
    return add_header(panel, title)


def rgb_tile(rgb: np.ndarray, label: str) -> np.ndarray:
    return add_label(rgb.copy(), label)


def depth_tile(depth: np.ndarray, maximum: float, label: str) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = np.log1p(np.clip(depth[valid], 0.0, maximum)) / np.log1p(maximum)
        normalized[valid] = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (24, 24, 24)
    return add_label(colored, label)


def error_tile(error: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    valid = mask & np.isfinite(error)
    normalized = np.zeros(error.shape, dtype=np.uint8)
    normalized[valid] = np.clip(error[valid] / 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    colored[~valid] = (24, 24, 24)
    return add_label(colored, label)


def mask_tile(mask: np.ndarray, label: str) -> np.ndarray:
    colored = np.full((*mask.shape, 3), 24, dtype=np.uint8)
    colored[mask] = (80, 210, 100)
    return add_label(colored, label)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (18, 18, 18), thickness=-1)
    cv2.putText(output, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return output


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((34, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, title[:220], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    max_width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[1] < max_width:
            pad = np.full((panel.shape[0], max_width - panel.shape[1], 3), 18, dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=1)
        padded.append(panel)
    return join_images(padded, axis=0, gap=DIRECTION_GAP_PX)


def join_images(images: list[np.ndarray], axis: int, gap: int) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required.")
    if len(images) == 1 or gap <= 0:
        return images[0].copy() if len(images) == 1 else np.concatenate(images, axis=axis)
    joined: list[np.ndarray] = []
    for index, image in enumerate(images):
        if index:
            if axis == 0:
                separator_shape = (gap, image.shape[1], image.shape[2])
            elif axis == 1:
                separator_shape = (image.shape[0], gap, image.shape[2])
            else:
                raise ValueError(f"Unsupported axis: {axis}")
            separator = np.full(separator_shape, GAP_COLOR, dtype=np.uint8)
            joined.append(separator)
        joined.append(image)
    return np.concatenate(joined, axis=axis)


if __name__ == "__main__":
    main()
