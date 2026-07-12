#!/usr/bin/env python3
"""Reproject one Matterport3D GT-depth pair with PanoVGGT's official convention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


CAMERA_MP3D_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
WORLD_MP3D_TO_OPENCV = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--scan", default="mJXqzFtmKg4")
    parser.add_argument("--source", default="6a552fe5ac3943989afe912054c54c49")
    parser.add_argument("--target", default="ca048420b46b4994af7aca4cba7015e7")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scan_root = args.matterport_root / args.scan
    room_id, room_name = verify_same_room(
        args.matterport_root / "parsed_json" / f"{args.scan}.json", args.source, args.target
    )
    records = {
        pano_id: load_record(scan_root, pano_id, args.height, args.width)
        for pano_id in (args.source, args.target)
    }
    baseline_m = float(
        np.linalg.norm(records[args.source]["pose_cv"][:3, 3] - records[args.target]["pose_cv"][:3, 3])
    )
    directions = []
    for source_id, target_id in ((args.source, args.target), (args.target, args.source)):
        result = reproject(records[source_id], records[target_id])
        result.update(source_id=source_id, target_id=target_id)
        directions.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "matterport3d_official_gt_depth_reprojection.png"
    summary_path = args.output_dir / "matterport3d_official_gt_depth_reprojection.json"
    cv2.imwrite(
        str(image_path),
        make_panel(directions, args.scan, room_id, room_name, baseline_m),
    )
    summary = {
        "scope": "Matterport3D official GT depth and official PanoVGGT coordinate conversion only",
        "scan": args.scan,
        "room_id": room_id,
        "room_name": room_name,
        "source": args.source,
        "target": args.target,
        "baseline_m": baseline_m,
        "depth": {
            "source_format": "Matterport3D pano_depth uint16 PNG",
            "meters_per_unit": 1.0 / 4000.0,
            "semantics": "radial range, matching PanoVGGT unproject_pano_depth_to_camera_coords",
        },
        "erp_camera_axes": {
            "x": "right",
            "y": "down",
            "z": "forward",
            "theta": "(u/(W-1)-0.5)*2*pi",
            "phi": "-(v/(H-1)-0.5)*pi",
            "ray": "[cos(phi)*sin(theta), -sin(phi), cos(phi)*cos(theta)]",
        },
        "pose_conversion": {
            "formula": "c2w_opencv = world_mp3d_to_opencv @ c2w_mp3d @ inv(camera_mp3d_to_opencv)",
            "camera_mp3d_to_opencv": CAMERA_MP3D_TO_OPENCV.tolist(),
            "world_mp3d_to_opencv": WORLD_MP3D_TO_OPENCV.tolist(),
        },
        "cameras": {
            pano_id: {
                "pose_c2w_mp3d": records[pano_id]["pose_mp3d"].tolist(),
                "pose_c2w_opencv": records[pano_id]["pose_cv"].tolist(),
            }
            for pano_id in (args.source, args.target)
        },
        "directions": [
            {
                "source": result["source_id"],
                "target": result["target_id"],
                "metrics": result["metrics"],
            }
            for result in directions
        ],
        "visualization": str(image_path.resolve()),
        "note": "The splatted tile is display-only; metrics use the raw nearest-pixel z-buffer.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[matterport-official] scan={args.scan} room={room_id}:{room_name} baseline={baseline_m:.4f}m")
    for result in directions:
        metrics = result["metrics"]
        print(
            "[matterport-official] "
            f"{result['source_id']} -> {result['target_id']} "
            f"coverage={metrics['target_coverage']:.4f} "
            f"median_abs_log={metrics['median_abs_log_error']:.4f} "
            f"inlier10={metrics['inlier_10pct']:.4f} inlier20={metrics['inlier_20pct']:.4f}"
        )
    print(f"[matterport-official] wrote {image_path.resolve()}")
    print(f"[matterport-official] wrote {summary_path.resolve()}")


def verify_same_room(parsed_json_path: Path, first_id: str, second_id: str) -> tuple[str, str]:
    payload = json.loads(parsed_json_path.read_text(encoding="utf-8"))
    matches = []
    for room_id, room in payload.items():
        panoramas = set(room.get("panoramas", []))
        if first_id in panoramas and second_id in panoramas:
            matches.append((str(room_id), str(room.get("room_name", ""))))
    if len(matches) != 1:
        raise ValueError(f"Pair is not uniquely assigned to one room: {matches}")
    return matches[0]


def load_record(scan_root: Path, pano_id: str, height: int, width: int) -> dict[str, np.ndarray]:
    depth_path = scan_root / "pano_depth" / f"{pano_id}.png"
    pose_path = scan_root / "pano_poses" / f"{pano_id}.txt"
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise OSError(f"Cannot read {depth_path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    depth_m = depth_raw.astype(np.float32) / 4000.0
    depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m[(~np.isfinite(depth_m)) | (depth_m <= 0.0)] = np.nan

    pose_mp3d = np.loadtxt(pose_path, dtype=np.float64)
    if pose_mp3d.shape != (4, 4):
        raise ValueError(f"Invalid pose shape {pose_mp3d.shape}: {pose_path}")
    pose_cv = WORLD_MP3D_TO_OPENCV @ pose_mp3d @ np.linalg.inv(CAMERA_MP3D_TO_OPENCV)
    rotation_error = float(np.max(np.abs(pose_cv[:3, :3].T @ pose_cv[:3, :3] - np.eye(3))))
    if rotation_error > 1e-3 or np.linalg.det(pose_cv[:3, :3]) < 0.0:
        raise ValueError(f"Invalid converted rotation for {pano_id}: error={rotation_error}")
    return {"depth": depth_m, "pose_mp3d": pose_mp3d, "pose_cv": pose_cv}


def official_erp_rays(height: int, width: int) -> np.ndarray:
    # Match PanoVGGT's endpoint-inclusive torch.arange/(size-1) grid exactly.
    v = np.arange(height, dtype=np.float64) / (height - 1)
    u = np.arange(width, dtype=np.float64) / (width - 1)
    v_grid, u_grid = np.meshgrid(v, u, indexing="ij")
    theta = (u_grid - 0.5) * (2.0 * math.pi)
    phi = -(v_grid - 0.5) * math.pi
    return np.stack(
        [np.cos(phi) * np.sin(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)],
        axis=-1,
    )


def reproject(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, object]:
    depth_source = source["depth"]
    depth_target = target["depth"]
    height, width = depth_source.shape
    rays = official_erp_rays(height, width)
    valid_source = np.isfinite(depth_source) & (depth_source > 0.0)
    points_source = rays[valid_source] * depth_source[valid_source, None]

    source_pose = source["pose_cv"]
    target_pose = target["pose_cv"]
    points_world = points_source @ source_pose[:3, :3].T + source_pose[:3, 3]
    points_target = (points_world - target_pose[:3, 3]) @ target_pose[:3, :3]
    ranges = np.linalg.norm(points_target, axis=1)
    valid_points = np.isfinite(ranges) & (ranges > 1e-6)
    points_target = points_target[valid_points]
    ranges = ranges[valid_points]
    unit = points_target / ranges[:, None]

    theta = np.arctan2(unit[:, 0], unit[:, 2])
    u = np.rint((theta / (2.0 * math.pi) + 0.5) * (width - 1)).astype(np.int64)
    v = np.rint((0.5 + np.arcsin(np.clip(unit[:, 1], -1.0, 1.0)) / math.pi) * (height - 1)).astype(np.int64)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    linear = v * width + u
    order = np.lexsort((ranges, linear))
    linear_sorted = linear[order]
    first = np.concatenate(([True], linear_sorted[1:] != linear_sorted[:-1]))
    projected = np.full(height * width, np.nan, dtype=np.float32)
    projected[linear_sorted[first]] = ranges[order[first]].astype(np.float32)
    projected = projected.reshape(height, width)

    comparable = np.isfinite(projected) & np.isfinite(depth_target) & (depth_target > 0.0)
    abs_log = np.full((height, width), np.nan, dtype=np.float32)
    abs_relative = np.full((height, width), np.nan, dtype=np.float32)
    abs_log[comparable] = np.abs(np.log(projected[comparable]) - np.log(depth_target[comparable]))
    abs_relative[comparable] = np.abs(projected[comparable] - depth_target[comparable]) / depth_target[comparable]
    log_values = abs_log[comparable]
    relative_values = abs_relative[comparable]
    target_valid_count = int(np.count_nonzero(np.isfinite(depth_target) & (depth_target > 0.0)))
    comparable_count = int(np.count_nonzero(comparable))
    metrics = {
        "source_valid_pixels": int(np.count_nonzero(valid_source)),
        "projected_zbuffer_pixels": int(np.count_nonzero(np.isfinite(projected))),
        "comparable_pixels": comparable_count,
        "target_coverage": float(comparable_count / max(1, target_valid_count)),
        "median_abs_log_error": finite_stat(log_values, np.median),
        "mean_abs_log_error": finite_stat(log_values, np.mean),
        "inlier_05pct": finite_ratio(relative_values, 0.05),
        "inlier_10pct": finite_ratio(relative_values, 0.10),
        "inlier_20pct": finite_ratio(relative_values, 0.20),
    }
    return {
        "source_depth": depth_source,
        "target_depth": depth_target,
        "projected_depth": projected,
        "projected_depth_splat": splat_for_display(projected),
        "abs_log_error": abs_log,
        "inlier_20_mask": comparable & (abs_relative <= 0.20),
        "comparable_mask": comparable,
        "metrics": metrics,
    }


def finite_stat(values: np.ndarray, fn) -> float | None:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else None


def finite_ratio(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite <= threshold)) if finite.size else 0.0


def splat_for_display(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    inverse = np.zeros(depth.shape, dtype=np.float32)
    inverse[valid] = 1.0 / depth[valid]
    inverse = cv2.dilate(inverse, np.ones((3, 3), dtype=np.uint8), iterations=1)
    output = np.full(depth.shape, np.nan, dtype=np.float32)
    output[inverse > 0.0] = 1.0 / inverse[inverse > 0.0]
    return output


def make_panel(
    directions: list[dict[str, object]], scan: str, room_id: str, room_name: str, baseline_m: float
) -> np.ndarray:
    all_depth = np.concatenate(
        [result[key][np.isfinite(result[key])] for result in directions for key in ("source_depth", "target_depth")]
    )
    depth_max = float(np.quantile(all_depth, 0.98))
    rows = []
    for result in directions:
        metrics = result["metrics"]
        tiles = [
            depth_tile(result["source_depth"], depth_max, "Source official GT depth"),
            depth_tile(result["target_depth"], depth_max, "Target official GT depth"),
            depth_tile(result["projected_depth"], depth_max, "Official reprojection (raw z-buffer)"),
            depth_tile(result["projected_depth_splat"], depth_max, "Official reprojection (1px display splat)"),
            error_tile(result["abs_log_error"], result["comparable_mask"], "Abs-log error (all comparable pixels)"),
            mask_tile(result["inlier_20_mask"], "Depth agreement within 20%"),
        ]
        row = np.concatenate(tiles, axis=1)
        label = (
            f"{str(result['source_id'])[:8]} -> {str(result['target_id'])[:8]} | "
            f"coverage={metrics['target_coverage']:.3f} | median abs-log={metrics['median_abs_log_error']:.3f} | "
            f"inlier 10%={metrics['inlier_10pct']:.3f} 20%={metrics['inlier_20pct']:.3f}"
        )
        rows.append(add_header(row, label, 34))
    panel = np.concatenate(rows, axis=0)
    title = (
        f"Matterport3D official GT reprojection | scan={scan} room={room_id}:{room_name} | "
        f"baseline={baseline_m:.3f} m | no fitted yaw/basis"
    )
    return add_header(panel, title, 42)


def depth_tile(depth: np.ndarray, maximum: float, label: str) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        np.log1p(np.clip(depth[valid], 0.0, maximum)) / np.log1p(maximum) * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (20, 20, 20)
    return add_label(colored, label)


def error_tile(error: np.ndarray, valid: np.ndarray, label: str) -> np.ndarray:
    shown = np.zeros(error.shape, dtype=np.uint8)
    shown[valid] = np.clip(error[valid] / 0.7 * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(shown, cv2.COLORMAP_INFERNO)
    colored[~valid] = (20, 20, 20)
    return add_label(colored, label)


def mask_tile(mask: np.ndarray, label: str) -> np.ndarray:
    colored = np.full((*mask.shape, 3), 20, dtype=np.uint8)
    colored[mask] = (80, 210, 100)
    return add_label(colored, label)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 25), (16, 16, 16), thickness=-1)
    cv2.putText(output, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (244, 244, 244), 1, cv2.LINE_AA)
    return output


def add_header(image: np.ndarray, text: str, height: int) -> np.ndarray:
    header = np.full((height, image.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(header, text, (9, int(height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


if __name__ == "__main__":
    main()
