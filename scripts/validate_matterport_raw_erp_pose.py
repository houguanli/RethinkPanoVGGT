#!/usr/bin/env python3
"""Validate the Matterport processed ERP pose from official raw camera files.

The processed Matterport bundle stores ERP color/depth panoramas, but the
existing ``pano_poses`` files come from a single raw camera pose.  This script
reconstructs the ERP image-frame pose from official raw files and checks it by
reprojecting processed GT depth between same-room panoramas.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


MATTERPORT_SKYBOX_TRANSFORMS = (
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    np.eye(3, dtype=np.float64),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
)


@dataclass(frozen=True)
class Candidate:
    reference_key: str
    fixed_name: str
    fixed_image_to_camera: np.ndarray


@dataclass
class PanoRecord:
    scan: str
    room_id: str
    room_name: str
    pano_id: str
    depth_path: Path
    raw_c2w_by_key: dict[str, np.ndarray]


@dataclass
class PairRecord:
    first: PanoRecord
    second: PanoRecord
    baseline_m: float


def main() -> None:
    args = build_parser().parse_args()
    processed_root = args.processed_root.resolve()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_status = find_raw_scan_dirs(raw_root)
    if not raw_status:
        raise SystemExit(
            "No Matterport raw scan folders with matterport_camera_poses.zip, "
            "undistorted_camera_parameters.zip, and matterport_skybox_images.zip "
            f"were found under {raw_root}."
        )

    pairs = select_same_room_pairs(
        processed_root=processed_root,
        raw_scan_dirs=raw_status,
        split=args.split,
        max_pairs=args.pairs,
        max_pairs_per_room=args.max_pairs_per_room,
        seed=args.seed,
    )
    if not pairs:
        raise SystemExit("No same-room Matterport pairs with raw camera parameters and processed GT depth were found.")

    rays = converter_erp_rays(args.height, args.width).reshape(-1, 3)
    pair_samples = prepare_pair_samples(
        pairs,
        rays,
        height=args.height,
        width=args.width,
        points_per_direction=args.points_per_direction,
        max_depth_m=args.max_depth_m,
        seed=args.seed + 17,
    )
    if not pair_samples:
        raise SystemExit("No valid depth samples were available for reprojection.")

    candidates = build_candidates(args.reference_keys)
    scored = []
    for candidate in candidates:
        metrics = score_candidate(pair_samples, candidate)
        scored.append(
            {
                "reference_key": candidate.reference_key,
                "fixed_name": candidate.fixed_name,
                "fixed_image_to_camera": candidate.fixed_image_to_camera.tolist(),
                "metrics": metrics,
            }
        )
    scored.sort(key=lambda row: candidate_rank(row["metrics"]))
    best = scored[0]

    high_overlap_pairs = [
        row
        for row in best["metrics"]["pairs"]
        if row["coverage"] >= args.high_overlap_coverage
    ]
    high_overlap_medians = [
        row["median_abs_log_error"]
        for row in high_overlap_pairs
        if row["median_abs_log_error"] is not None and math.isfinite(row["median_abs_log_error"])
    ]
    accepted = bool(
        len(high_overlap_medians) >= args.min_high_overlap_pairs
        and float(np.median(high_overlap_medians)) <= args.accept_median_abs_log
    )

    visualizations = []
    if args.visualize_pairs > 0:
        best_candidate = Candidate(
            reference_key=str(best["reference_key"]),
            fixed_name=str(best["fixed_name"]),
            fixed_image_to_camera=np.asarray(best["fixed_image_to_camera"], dtype=np.float64),
        )
        visualizations = write_visualizations(
            output_dir=output_dir,
            pairs=pairs[: args.visualize_pairs],
            candidate=best_candidate,
            height=args.visual_height,
            width=args.visual_width,
            max_depth_m=args.max_depth_m,
        )

    summary = {
        "processed_root": str(processed_root),
        "raw_root": str(raw_root),
        "split": args.split,
        "pairs_requested": args.pairs,
        "pairs_loaded": len(pairs),
        "samples_loaded": len(pair_samples),
        "height": args.height,
        "width": args.width,
        "points_per_direction": args.points_per_direction,
        "acceptance": {
            "accepted": accepted,
            "criterion": {
                "min_high_overlap_pairs": args.min_high_overlap_pairs,
                "high_overlap_coverage": args.high_overlap_coverage,
                "median_abs_log_threshold": args.accept_median_abs_log,
            },
            "high_overlap_pairs": len(high_overlap_medians),
            "high_overlap_median_abs_log": finite_stat(np.asarray(high_overlap_medians, dtype=np.float64), np.median),
        },
        "best": best,
        "top_candidates": scored[: min(args.top_k, len(scored))],
        "pair_selection": [
            {
                "scan": pair.first.scan,
                "room_id": pair.first.room_id,
                "room_name": pair.first.room_name,
                "source": pair.first.pano_id,
                "target": pair.second.pano_id,
                "baseline_m": pair.baseline_m,
            }
            for pair in pairs
        ],
        "visualizations": visualizations,
        "note": (
            "The pose tested here is c2w_erp_raw = c2w_reference_camera @ fixed_image_to_camera. "
            "Depth rays use the ERP convention implemented by the raw Matterport converter "
            "that generated the processed equirectangular depth maps."
        ),
    }
    summary_path = output_dir / "matterport_raw_erp_pose_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[matterport-raw-erp] raw_scan_dirs={len(raw_status)} pairs={len(pairs)} samples={len(pair_samples)}")
    print(
        "[matterport-raw-erp] "
        f"best reference={best['reference_key']} fixed={best['fixed_name']} "
        f"score={best['metrics']['score']:.6f} coverage={best['metrics']['coverage']:.4f} "
        f"median_abs_log={best['metrics']['median_abs_log_error']:.4f}"
    )
    print(
        "[matterport-raw-erp] "
        f"accepted={accepted} high_overlap_pairs={len(high_overlap_medians)} "
        f"high_overlap_median_abs_log={summary['acceptance']['high_overlap_median_abs_log']}"
    )
    print(f"[matterport-raw-erp] wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True, help="Processed Matterport3D root.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Raw Matterport root containing per-scan folders with the official zip files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--pairs", type=int, default=80)
    parser.add_argument("--max-pairs-per-room", type=int, default=4)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--visual-height", type=int, default=160)
    parser.add_argument("--visual-width", type=int, default=320)
    parser.add_argument("--points-per-direction", type=int, default=2048)
    parser.add_argument("--max-depth-m", type=float, default=15.0)
    parser.add_argument("--reference-keys", default="1_5,0_0,1_0,1_1,1_2,1_3,1_4")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--visualize-pairs", type=int, default=8)
    parser.add_argument("--high-overlap-coverage", type=float, default=0.35)
    parser.add_argument("--min-high-overlap-pairs", type=int, default=10)
    parser.add_argument("--accept-median-abs-log", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def find_raw_scan_dirs(raw_root: Path) -> dict[str, Path]:
    required = {
        "matterport_camera_poses.zip",
        "undistorted_camera_parameters.zip",
        "matterport_skybox_images.zip",
    }
    out: dict[str, Path] = {}
    if not raw_root.exists():
        return out
    for scan_dir in sorted(raw_root.iterdir()):
        if not scan_dir.is_dir():
            continue
        if all((scan_dir / name).is_file() for name in required):
            out[scan_dir.name] = scan_dir
    return out


def select_same_room_pairs(
    processed_root: Path,
    raw_scan_dirs: dict[str, Path],
    split: str,
    max_pairs: int,
    max_pairs_per_room: int,
    seed: int,
) -> list[PairRecord]:
    allowed = load_split_panos(processed_root, split)
    parsed_dir = processed_root / "parsed_json"
    candidates: list[PairRecord] = []
    raw_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for json_path in sorted(parsed_dir.glob("*.json")):
        scan = json_path.stem
        if scan not in raw_scan_dirs:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        raw_cache[scan] = load_raw_camera_extrinsics(raw_scan_dirs[scan], scan)
        for room_id, room in sorted(payload.items(), key=lambda item: str(item[0])):
            pano_ids = [str(value) for value in room.get("panoramas", []) or []]
            if allowed is not None:
                pano_ids = [pano_id for pano_id in pano_ids if (scan, pano_id) in allowed]
            records = [
                make_pano_record(
                    processed_root=processed_root,
                    raw_cameras=raw_cache[scan],
                    scan=scan,
                    room_id=str(room_id),
                    room_name=str(room.get("room_name", "")),
                    pano_id=pano_id,
                )
                for pano_id in pano_ids
            ]
            records = [record for record in records if record is not None]
            if len(records) < 2:
                continue
            room_pairs = []
            for first, second in itertools.combinations(records, 2):
                baseline = reference_baseline(first, second, preferred_key="1_5")
                if baseline is None or baseline <= 1e-6:
                    continue
                room_pairs.append(PairRecord(first=first, second=second, baseline_m=baseline))
            room_pairs.sort(key=lambda pair: pair.baseline_m)
            candidates.extend(room_pairs[: max(1, max_pairs_per_room)])
    rng = random.Random(seed)
    # Prefer nearest same-room pairs, but shuffle within small baseline bands to
    # avoid evaluating only one scan if that scan has many rooms.
    candidates.sort(key=lambda pair: pair.baseline_m)
    banded: list[PairRecord] = []
    for start in range(0, len(candidates), 32):
        band = candidates[start : start + 32]
        rng.shuffle(band)
        banded.extend(band)
    return banded[:max_pairs]


def load_split_panos(processed_root: Path, split: str) -> set[tuple[str, str]] | None:
    if split == "all":
        return None
    index_path = processed_root / "cache" / f"matterport3d_{split}_index.json"
    if not index_path.is_file():
        return None
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    allowed: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        scan = str(row[0])
        for pano_id in row[3] or []:
            allowed.add((scan, str(pano_id)))
    return allowed


def make_pano_record(
    processed_root: Path,
    raw_cameras: dict[str, dict[str, np.ndarray]],
    scan: str,
    room_id: str,
    room_name: str,
    pano_id: str,
) -> PanoRecord | None:
    depth_path = processed_root / scan / "pano_depth" / f"{pano_id}.png"
    color_path = processed_root / scan / "pano_skybox_color" / f"{pano_id}.jpg"
    if not depth_path.is_file() or not color_path.is_file():
        return None
    raw = raw_cameras.get(pano_id)
    if not raw:
        return None
    return PanoRecord(
        scan=scan,
        room_id=room_id,
        room_name=room_name,
        pano_id=pano_id,
        depth_path=depth_path,
        raw_c2w_by_key=raw,
    )


def load_raw_camera_extrinsics(scan_dir: Path, scan: str) -> dict[str, dict[str, np.ndarray]]:
    with zipfile.ZipFile(scan_dir / "matterport_camera_poses.zip") as z_pose:
        pose_members = zip_member_lookup(z_pose)
        pose_panos = {
            name.rsplit("_pose_", 1)[0]
            for name in pose_members
            if "_pose_" in name and name.endswith(".txt")
        }
    with zipfile.ZipFile(scan_dir / "matterport_skybox_images.zip") as z_skybox:
        skybox_members = zip_member_lookup(z_skybox)
        skybox_panos = {
            name.split("_skybox", 1)[0]
            for name in skybox_members
            if "_skybox" in name and name.endswith((".jpg", ".jpeg", ".png"))
        }
    with zipfile.ZipFile(scan_dir / "undistorted_camera_parameters.zip") as z_params:
        conf_member = f"{scan}/undistorted_camera_parameters/{scan}.conf"
        text = z_params.read(conf_member).decode("utf-8", errors="strict")
    _intrinsics, extrinsics = parse_undistorted_camera_parameters(text)
    return {
        pano: {key: pose for key, (pose, _w2c) in by_key.items()}
        for pano, by_key in extrinsics.items()
        if pano in pose_panos and pano in skybox_panos
    }


def zip_member_lookup(zf: zipfile.ZipFile) -> dict[str, str]:
    return {Path(name).name: name for name in zf.namelist() if not name.endswith("/")}


def parse_undistorted_camera_parameters(
    text: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    intrinsics: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    extrinsics: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    current_k = None
    for line in text.splitlines():
        if line.startswith("intrinsics_matrix"):
            vals = [float(value) for value in line.split()[1:]]
            current_k = np.asarray(vals, dtype=np.float64).reshape(3, 3)
            continue
        if not line.startswith("scan "):
            continue
        if current_k is None:
            raise RuntimeError(f"Camera scan entry before intrinsics: {line}")
        parts = line.split()
        depth_name = parts[1]
        color_name = parts[2]
        pano = depth_name.rsplit("_d", 1)[0]
        camera_key = color_name.rsplit(".", 1)[0].rsplit("_i", 1)[1]
        camera_group = camera_key.split("_", 1)[0]
        pose_c2w = np.asarray([float(value) for value in parts[3:19]], dtype=np.float64).reshape(4, 4)
        intrinsics[pano][camera_group] = current_k.copy()
        extrinsics[pano][camera_key] = (pose_c2w, np.linalg.inv(pose_c2w))
    return intrinsics, extrinsics


def reference_baseline(first: PanoRecord, second: PanoRecord, preferred_key: str) -> float | None:
    first_pose = first.raw_c2w_by_key.get(preferred_key)
    if first_pose is None:
        first_pose = next(iter(first.raw_c2w_by_key.values()), None)
    second_pose = second.raw_c2w_by_key.get(preferred_key)
    if second_pose is None:
        second_pose = next(iter(second.raw_c2w_by_key.values()), None)
    if first_pose is None or second_pose is None:
        return None
    return float(np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3]))


def build_candidates(reference_keys_raw: str) -> list[Candidate]:
    reference_keys = [value.strip() for value in reference_keys_raw.split(",") if value.strip()]
    fixed = proper_axis_matrices()
    candidates = []
    for reference_key in reference_keys:
        for fixed_name, matrix in fixed:
            candidates.append(
                Candidate(
                    reference_key=reference_key,
                    fixed_name=fixed_name,
                    fixed_image_to_camera=matrix,
                )
            )
    return candidates


def proper_axis_matrices() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    axes = np.eye(3, dtype=np.float64)
    for perm in itertools.permutations(range(3)):
        base = axes[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                parts = []
                for col in range(3):
                    axis = int(np.argmax(np.abs(matrix[:, col])))
                    sign = "+" if matrix[axis, col] > 0 else "-"
                    parts.append(f"{sign}{'xyz'[axis]}")
                out.append((",".join(parts), matrix.astype(np.float64)))
    # Put converter-derived likely transforms first for easier reading.
    named = [(f"skybox_transform_{idx}", matrix.copy()) for idx, matrix in enumerate(MATTERPORT_SKYBOX_TRANSFORMS)]
    seen = {tuple(np.round(matrix.reshape(-1), 6)) for _name, matrix in named}
    named.extend((name, matrix) for name, matrix in out if tuple(np.round(matrix.reshape(-1), 6)) not in seen)
    return named


def converter_erp_rays(height: int, width: int) -> np.ndarray:
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    lon = (xs / float(width) - 0.5) * (2.0 * math.pi)
    lat = (0.5 - ys / float(height)) * math.pi
    return np.stack(
        [np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)],
        axis=-1,
    )


def prepare_pair_samples(
    pairs: Iterable[PairRecord],
    rays_flat: np.ndarray,
    height: int,
    width: int,
    points_per_direction: int,
    max_depth_m: float,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    out = []
    for pair_index, pair in enumerate(pairs):
        first_depth = read_depth(pair.first.depth_path, height, width)
        second_depth = read_depth(pair.second.depth_path, height, width)
        for direction_index, (source, target, source_depth, target_depth) in enumerate(
            (
                (pair.first, pair.second, first_depth, second_depth),
                (pair.second, pair.first, second_depth, first_depth),
            )
        ):
            valid = np.isfinite(source_depth) & (source_depth > 0.0)
            if max_depth_m > 0:
                valid &= source_depth <= max_depth_m
            indices = np.flatnonzero(valid.reshape(-1))
            if indices.size < 64:
                continue
            take = min(int(points_per_direction), int(indices.size))
            selected = rng.choice(indices, size=take, replace=False)
            out.append(
                {
                    "pair_index": pair_index,
                    "direction_index": direction_index,
                    "scan": source.scan,
                    "room_id": source.room_id,
                    "source": source.pano_id,
                    "target": target.pano_id,
                    "baseline_m": pair.baseline_m,
                    "source_rays": rays_flat[selected],
                    "source_depth": source_depth.reshape(-1)[selected].astype(np.float64),
                    "source_raw_c2w_by_key": source.raw_c2w_by_key,
                    "target_raw_c2w_by_key": target.raw_c2w_by_key,
                    "target_depth": target_depth,
                }
            )
    return out


def read_depth(path: Path, height: int, width: int) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"Cannot read {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32) / 4000.0
    depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m[(~np.isfinite(depth_m)) | (depth_m <= 0.0)] = np.nan
    return depth_m


def score_candidate(samples: list[dict[str, object]], candidate: Candidate) -> dict[str, object]:
    errors_all: list[np.ndarray] = []
    pair_rows = []
    total_requested = 0
    total_comparable = 0
    total_projected = 0
    for sample in samples:
        result = reproject_sample(sample, candidate)
        comparable = result["comparable_mask"].reshape(-1)
        total_requested += int(sample["target_valid_count"])
        total_projected += int(result["projected_count"])
        total_comparable += int(comparable.sum())
        values = result["abs_log_error"].reshape(-1)[comparable]
        if values.size:
            errors_all.append(values)
        pair_rows.append(
            {
                "pair_index": int(sample["pair_index"]),
                "direction_index": int(sample["direction_index"]),
                "scan": str(sample["scan"]),
                "room_id": str(sample["room_id"]),
                "source": str(sample["source"]),
                "target": str(sample["target"]),
                "baseline_m": float(sample["baseline_m"]),
                "coverage": float(comparable.sum() / max(1, int(sample["target_valid_count"]))),
                "projected_pixels": int(result["projected_count"]),
                "comparable_pixels": int(comparable.sum()),
                "median_abs_log_error": finite_stat(values, np.median),
                "p90_abs_log_error": finite_stat(values, lambda x: np.quantile(x, 0.9)),
                "inlier_05pct": finite_ratio_from_log(values, 0.05),
                "inlier_10pct": finite_ratio_from_log(values, 0.10),
                "inlier_20pct": finite_ratio_from_log(values, 0.20),
            }
        )
    if not errors_all:
        return {
            "score": float("inf"),
            "coverage": 0.0,
            "valid_points": 0,
            "projected_points": total_projected,
            "median_abs_log_error": float("inf"),
            "p90_abs_log_error": float("inf"),
            "inlier_05pct": 0.0,
            "inlier_10pct": 0.0,
            "inlier_20pct": 0.0,
            "pairs": pair_rows,
        }
    errors = np.concatenate(errors_all)
    coverage = float(total_comparable / max(1, total_requested))
    clipped_mean = float(np.mean(np.minimum(errors, 0.7)))
    return {
        "score": clipped_mean + 0.5 * (1.0 - coverage),
        "coverage": coverage,
        "valid_points": int(errors.size),
        "projected_points": total_projected,
        "median_abs_log_error": float(np.median(errors)),
        "p90_abs_log_error": float(np.quantile(errors, 0.9)),
        "inlier_05pct": finite_ratio_from_log(errors, 0.05),
        "inlier_10pct": finite_ratio_from_log(errors, 0.10),
        "inlier_20pct": finite_ratio_from_log(errors, 0.20),
        "pairs": pair_rows,
    }


def candidate_rank(metrics: dict[str, object]) -> float:
    score = float(metrics["score"])
    coverage = float(metrics["coverage"])
    if not math.isfinite(score) or coverage < 0.03:
        return float("inf")
    return score


def reproject_sample(sample: dict[str, object], candidate: Candidate) -> dict[str, object]:
    source_pose = make_erp_c2w(sample["source_raw_c2w_by_key"], candidate)
    target_pose = make_erp_c2w(sample["target_raw_c2w_by_key"], candidate)
    source_rays = np.asarray(sample["source_rays"], dtype=np.float64)
    source_depth = np.asarray(sample["source_depth"], dtype=np.float64)
    target_depth = np.asarray(sample["target_depth"], dtype=np.float64)
    height, width = target_depth.shape

    points_source = source_rays * source_depth[:, None]
    points_world = points_source @ source_pose[:3, :3].T + source_pose[:3, 3]
    points_target = (points_world - target_pose[:3, 3]) @ target_pose[:3, :3]
    ranges = np.linalg.norm(points_target, axis=1)
    valid = np.isfinite(ranges) & (ranges > 1e-6)
    points_target = points_target[valid]
    ranges = ranges[valid]
    unit = points_target / ranges[:, None]
    u, v = converter_rays_to_erp(unit)
    x = np.mod(np.rint(u * width).astype(np.int64), width)
    y = np.clip(np.rint(v * height).astype(np.int64), 0, height - 1)
    linear = y * width + x
    order = np.lexsort((ranges, linear))
    linear_sorted = linear[order]
    first = np.concatenate(([True], linear_sorted[1:] != linear_sorted[:-1]))
    projected = np.full(height * width, np.inf, dtype=np.float64)
    projected[linear_sorted[first]] = ranges[order[first]]
    projected = projected.reshape(height, width)

    target_valid = np.isfinite(target_depth) & (target_depth > 0.0)
    comparable = np.isfinite(projected) & target_valid
    error = np.full((height, width), np.nan, dtype=np.float64)
    error[comparable] = np.abs(
        np.log(np.maximum(projected[comparable], 1e-6))
        - np.log(np.maximum(target_depth[comparable], 1e-6))
    )
    sample["target_valid_count"] = int(target_valid.sum())
    return {
        "projected": projected,
        "comparable_mask": comparable,
        "projected_count": int(np.isfinite(projected).sum()),
        "abs_log_error": error,
    }


def make_erp_c2w(raw_c2w_by_key: dict[str, np.ndarray], candidate: Candidate) -> np.ndarray:
    ref = raw_c2w_by_key.get(candidate.reference_key)
    if ref is None:
        raise KeyError(f"Missing reference camera {candidate.reference_key}")
    c2w = ref.copy()
    c2w[:3, :3] = ref[:3, :3] @ candidate.fixed_image_to_camera
    return c2w


def converter_rays_to_erp(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)
    lon = np.arctan2(unit[:, 0], unit[:, 2])
    lat = np.arcsin(np.clip(unit[:, 1], -1.0, 1.0))
    u = np.mod(lon / (2.0 * math.pi) + 0.5, 1.0)
    v = np.clip(0.5 - lat / math.pi, 0.0, 1.0)
    return u, v


def finite_stat(values: np.ndarray, fn) -> float | None:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else None


def finite_ratio_from_log(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite <= math.log1p(threshold)))


def write_visualizations(
    output_dir: Path,
    pairs: list[PairRecord],
    candidate: Candidate,
    height: int,
    width: int,
    max_depth_m: float,
) -> list[str]:
    rays = converter_erp_rays(height, width).reshape(-1, 3)
    out_dir = output_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, pair in enumerate(pairs):
        samples = prepare_pair_samples(
            [pair],
            rays,
            height=height,
            width=width,
            points_per_direction=height * width,
            max_depth_m=max_depth_m,
            seed=index,
        )
        rows = []
        for sample in samples:
            result = reproject_sample(sample, candidate)
            rows.append(make_visual_row(sample, result))
        if not rows:
            continue
        panel = np.concatenate(rows, axis=0)
        title = (
            f"{pair.first.scan} room={pair.first.room_id} "
            f"{pair.first.pano_id[:8]} <-> {pair.second.pano_id[:8]} "
            f"baseline={pair.baseline_m:.3f}m ref={candidate.reference_key} fixed={candidate.fixed_name}"
        )
        panel = add_header(panel, title, 34)
        path = out_dir / f"{index:03d}_{pair.first.scan}_{pair.first.pano_id[:8]}_{pair.second.pano_id[:8]}.png"
        cv2.imwrite(str(path), panel)
        written.append(str(path))
    return written


def make_visual_row(sample: dict[str, object], result: dict[str, object]) -> np.ndarray:
    target_depth = np.asarray(sample["target_depth"], dtype=np.float64)
    projected = np.asarray(result["projected"], dtype=np.float64)
    comparable = np.asarray(result["comparable_mask"], dtype=bool)
    error = np.asarray(result["abs_log_error"], dtype=np.float64)
    valid_values = target_depth[np.isfinite(target_depth) & (target_depth > 0.0)]
    maximum = float(np.quantile(valid_values, 0.98)) if valid_values.size else 10.0
    metrics_values = error[comparable]
    coverage = float(comparable.sum() / max(1, np.isfinite(target_depth).sum()))
    median = finite_stat(metrics_values, np.median)
    label = (
        f"{sample['source'][:8]} -> {sample['target'][:8]} "
        f"coverage={coverage:.3f} median={median if median is not None else float('nan'):.3f}"
    )
    tiles = [
        depth_tile(target_depth, maximum, "Target GT depth"),
        depth_tile(projected, maximum, "Raw ERP pose reprojection"),
        error_tile(error, comparable, "Abs-log error"),
        mask_tile(comparable, "Comparable pixels"),
    ]
    return add_header(np.concatenate(tiles, axis=1), label, 28)


def depth_tile(depth: np.ndarray, maximum: float, label: str) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        normalized[valid] = np.clip(
            np.log1p(np.clip(depth[valid], 0.0, maximum)) / np.log1p(max(maximum, 1e-6)) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (22, 22, 22)
    return add_label(colored, label)


def error_tile(error: np.ndarray, valid: np.ndarray, label: str) -> np.ndarray:
    shown = np.zeros(error.shape, dtype=np.uint8)
    shown[valid] = np.clip(error[valid] / 0.7 * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(shown, cv2.COLORMAP_INFERNO)
    colored[~valid] = (22, 22, 22)
    return add_label(colored, label)


def mask_tile(mask: np.ndarray, label: str) -> np.ndarray:
    colored = np.full((*mask.shape, 3), 22, dtype=np.uint8)
    colored[mask] = (80, 210, 100)
    return add_label(colored, label)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 22), (15, 15, 15), thickness=-1)
    cv2.putText(output, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    return output


def add_header(image: np.ndarray, text: str, height: int) -> np.ndarray:
    header = np.full((height, image.shape[1], 3), 15, dtype=np.uint8)
    cv2.putText(header, text, (8, int(height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
