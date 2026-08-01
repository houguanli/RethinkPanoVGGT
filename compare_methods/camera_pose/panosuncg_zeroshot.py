#!/usr/bin/env python3
"""Evaluate BiFuse++ or Pi3 on the official DA-2 PanoSUNCG split."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_DATASET_ROOT = Path(
    "/mnt/e/PanoVGGT_minimal_datasets/datasets/PanoSUNCG_zeroshot"
)
DEFAULT_BIFUSE_DEPTH_CKPT = (
    REPO_ROOT / "ckpt" / "BiFusePlusPlus" / "pretrain" / "supervised_pretrain.pkl"
)
DEFAULT_BIFUSE_CAMERA_CKPT = (
    REPO_ROOT / "ckpt" / "BiFusePlusPlus" / "pretrain" / "selfsupervised_pretrain.pkl"
)
DEFAULT_PI3_CKPT = REPO_ROOT / "ckpt" / "Pi3" / "model.safetensors"
SPLIT_SHA256 = "f7ac96361b88315293801b98aec2373f4f57845f5531e43a0fc7602592a71d7a"

DEPTH_FIELDS = (
    "sample_index",
    "rgb_relative_path",
    "depth_relative_path",
    "valid_pixels",
    "scale",
    "shift",
    "abs_rel",
    "rmse",
    "delta1",
    "delta2",
    "abs_rel_sum",
    "sq_error_sum",
    "delta1_count",
    "delta2_count",
    "coverage",
    "seconds",
)
TRAJECTORY_FIELDS = (
    "scene",
    "trajectory",
    "num_available_frames",
    "num_evaluated_frames",
    "selected_indices",
    "pair_count",
    "sim3_scale",
    "ate_rmse",
    "ate_mean",
    "ate_median",
    "ate_normalized_rmse",
    "direction_deg_mean",
    "direction_deg_median",
    "direction_auc3",
    "direction_auc5",
    "direction_auc15",
    "direction_auc30",
    "relative_length_error_mean",
    "relative_length_error_median",
    "inference_seconds",
    "trajectory_seconds",
)
FRAME_FIELDS = (
    "scene",
    "trajectory",
    "frame_index",
    "gt_x",
    "gt_y",
    "gt_z",
    "pred_aligned_x",
    "pred_aligned_y",
    "pred_aligned_z",
    "ate",
)
PAIR_FIELDS = (
    "scene",
    "trajectory",
    "frame_i",
    "frame_j",
    "gt_baseline",
    "pred_aligned_baseline",
    "direction_error_deg",
    "relative_length_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bifusepp", "pi3"), required=True)
    parser.add_argument("--task", choices=("depth", "camera"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=0, help="Depth samples; 0 is the complete split")
    parser.add_argument("--max-trajectories", type=int, default=0, help="Camera trajectories; 0 is all")
    parser.add_argument("--frames-per-trajectory", type=int, default=5)
    parser.add_argument("--face-size", type=int, default=196, help="Pi3 perspective face size")
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--latitude-min", type=float, default=-15.0)
    parser.add_argument("--latitude-max", type=float, default=60.0)
    parser.add_argument("--irls-iterations", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def resolve_dataset(root: Path) -> tuple[Path, Path, Path]:
    root = root.expanduser().resolve()
    for candidate in (root, root / "PanoSUNCG_zeroshot"):
        split = candidate / "panosuncg_da2_split.txt"
        rotated = candidate / "PanoSUNCG" / "rotated"
        labels = candidate / "PanoSUNCG" / "labels"
        if split.is_file() and rotated.is_dir() and labels.is_dir():
            digest = hashlib.sha256(split.read_bytes()).hexdigest()
            if digest != SPLIT_SHA256:
                raise RuntimeError(f"Unexpected PanoSUNCG split hash: {digest}")
            return rotated, labels, split
    raise FileNotFoundError(
        f"Cannot resolve PanoSUNCG_zeroshot under {root}; expected the DA-2 split, "
        "PanoSUNCG/rotated, and PanoSUNCG/labels"
    )


def read_split(split_file: Path, rotated_root: Path) -> list[tuple[int, str, str]]:
    samples = []
    for line_index, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines()):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"Malformed split line {line_index + 1}: {raw!r}")
        rgb_rel, depth_rel = fields
        if not (rotated_root / rgb_rel).is_file() or not (rotated_root / depth_rel).is_file():
            raise FileNotFoundError(f"Missing split files at line {line_index + 1}: {raw}")
        samples.append((line_index, rgb_rel, depth_rel))
    return samples


def build_trajectories(split_file: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        fields = raw.strip().split()
        if not fields:
            continue
        parts = Path(fields[0]).parts
        if len(parts) != 3:
            raise ValueError(f"Unexpected PanoSUNCG path: {fields[0]}")
        scene, trajectory, filename = parts
        groups[(scene, trajectory)].append(int(filename.split("_", 1)[0]))
    return [
        {"scene": scene, "trajectory": trajectory, "indices": sorted(set(indices))}
        for (scene, trajectory), indices in sorted(groups.items())
    ]


def evenly_spaced_indices(indices: Sequence[int], count: int) -> list[int]:
    if len(indices) <= count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=count)
    return list(dict.fromkeys(indices[int(round(position))] for position in positions))


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32) / 20.0


def image_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(device)


def load_bifuse_model(checkpoint: Path, mode: str, device: torch.device):
    from bifusepp_inference import (
        DEFAULT_UPSTREAM,
        _build_model,
        _is_runtime_projection_buffer,
        _load_bifuse_module,
        _prepare_released_state_dict,
    )

    module = _load_bifuse_module(DEFAULT_UPSTREAM)
    model = _build_model(module, mode)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, dropped = _prepare_released_state_dict(state)
    incompatible = model.load_state_dict(state, strict=False)
    missing_learned = [
        key for key in incompatible.missing_keys if not _is_runtime_projection_buffer(key)
    ]
    if missing_learned or incompatible.unexpected_keys:
        raise RuntimeError(
            f"BiFuse++ checkpoint mismatch: missing={missing_learned}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval(), {
        "missing_learned_keys": missing_learned,
        "regenerated_projection_buffers": len(incompatible.missing_keys),
        "dropped_untrained_legacy_fusion_keys": len(dropped),
    }


def load_pi3_model(checkpoint: Path, device: torch.device):
    upstream = THIS_DIR / "Pi3"
    if not (upstream / "pi3" / "models" / "pi3.py").is_file():
        raise FileNotFoundError(f"Pi3 submodule is not initialized: {upstream}")
    sys.path.insert(0, str(upstream))
    from pi3.models.pi3 import Pi3
    from safetensors.torch import load_file

    model = Pi3()
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval(), {"strict_checkpoint_load": True}


def amp_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def camera_axes(yaw_degrees: float, pitch_degrees: float, device: torch.device):
    yaw = torch.tensor(math.radians(yaw_degrees), device=device, dtype=torch.float32)
    pitch = torch.tensor(math.radians(pitch_degrees), device=device, dtype=torch.float32)
    forward = torch.stack(
        [torch.sin(yaw) * torch.cos(pitch), torch.sin(pitch), torch.cos(yaw) * torch.cos(pitch)]
    )
    right = torch.stack([torch.cos(yaw), torch.zeros_like(yaw), -torch.sin(yaw)])
    up = torch.linalg.cross(forward, right)
    return forward, right, up


def erp_to_perspectives(
    erp: torch.Tensor,
    orientations: Sequence[tuple[float, float]],
    face_size: int,
    fov_degrees: float = 90.0,
) -> torch.Tensor:
    device = erp.device
    coordinate = (torch.arange(face_size, device=device, dtype=torch.float32) + 0.5)
    coordinate = coordinate * (2.0 / face_size) - 1.0
    grid_y, grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    tangent = math.tan(math.radians(fov_degrees) * 0.5)
    faces = []
    for yaw, pitch in orientations:
        forward, right, up = camera_axes(yaw, pitch, device)
        rays = (
            forward[:, None, None]
            + grid_x[None] * tangent * right[:, None, None]
            - grid_y[None] * tangent * up[:, None, None]
        )
        rays = F.normalize(rays, dim=0)
        longitude = torch.atan2(rays[0], rays[2])
        latitude = torch.asin(rays[1].clamp(-1.0, 1.0))
        sample_grid = torch.stack([longitude / math.pi, -2.0 * latitude / math.pi], dim=-1)
        face = F.grid_sample(
            erp[None],
            sample_grid[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]
        faces.append(face)
    return torch.stack(faces)


def perspective_depths_to_erp(
    depths_z: torch.Tensor,
    orientations: Sequence[tuple[float, float]],
    pano_hw: tuple[int, int],
    fov_degrees: float = 90.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pano_h, pano_w = pano_hw
    device = depths_z.device
    theta = (torch.arange(pano_w, device=device, dtype=torch.float32) + 0.5) / pano_w
    theta = (theta - 0.5) * (2.0 * math.pi)
    phi = 0.5 - (torch.arange(pano_h, device=device, dtype=torch.float32) + 0.5) / pano_h
    phi = phi * math.pi
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")
    cos_phi = torch.cos(phi_grid)
    rays = torch.stack(
        [cos_phi * torch.sin(theta_grid), torch.sin(phi_grid), cos_phi * torch.cos(theta_grid)],
        dim=-1,
    )
    tangent = math.tan(math.radians(fov_degrees) * 0.5)
    weighted_sum = torch.zeros((pano_h, pano_w), device=device, dtype=torch.float32)
    weight_sum = torch.zeros_like(weighted_sum)
    for index, (yaw, pitch) in enumerate(orientations):
        forward, right, up = camera_axes(yaw, pitch, device)
        z_factor = torch.einsum("hwc,c->hw", rays, forward)
        safe_z = z_factor.clamp_min(1e-8)
        grid_x = torch.einsum("hwc,c->hw", rays, right) / (safe_z * tangent)
        grid_y = -torch.einsum("hwc,c->hw", rays, up) / (safe_z * tangent)
        grid = torch.stack([grid_x, grid_y], dim=-1)[None]
        sampled_z = F.grid_sample(
            depths_z[index][None, None].float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0]
        valid = (
            (z_factor > 1e-6)
            & (grid_x.abs() <= 1.0)
            & (grid_y.abs() <= 1.0)
            & torch.isfinite(sampled_z)
            & (sampled_z > 0)
        )
        radial = sampled_z / safe_z
        weight = torch.where(valid, z_factor.square(), torch.zeros_like(z_factor))
        weighted_sum += torch.where(valid, radial, torch.zeros_like(radial)) * weight
        weight_sum += weight
    valid = weight_sum > 0
    return weighted_sum / weight_sum.clamp_min(1e-8), valid


class DepthRunner:
    CUBE_ORIENTATIONS = ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (-90.0, 0.0), (0.0, 90.0), (0.0, -90.0))

    def __init__(self, args: argparse.Namespace, checkpoint: Path, device: torch.device):
        self.args = args
        self.device = device
        if args.method == "bifusepp":
            self.model, self.load_info = load_bifuse_model(checkpoint, "supervised", device)
        else:
            self.model, self.load_info = load_pi3_model(checkpoint, device)

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
        started = time.perf_counter()
        image = image_tensor(rgb, self.device)
        if self.args.method == "bifusepp":
            with torch.inference_mode():
                # Supervised training/validation calls the combined forward,
                # including ImageNet normalization. Clipping matches the
                # released visualization script's output post-processing.
                depth = self.model(image[None])[0][0, 0].float().clamp(0.0, 10.0)
            coverage = 1.0
        else:
            faces = erp_to_perspectives(image, self.CUBE_ORIENTATIONS, self.args.face_size)
            with torch.inference_mode(), amp_context(self.device, self.args.amp_dtype):
                result = self.model(faces[None])
            depths_z = result["local_points"][0, ..., 2].float()
            depth, valid = perspective_depths_to_erp(
                depths_z, self.CUBE_ORIENTATIONS, rgb.shape[:2]
            )
            coverage = float(valid.float().mean().item())
            del faces, result, depths_z, valid
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        result_np = depth.detach().cpu().numpy().astype(np.float32)
        del image, depth
        return result_np, coverage, seconds


def fit_irls_absrel_scale_shift(
    prediction: np.ndarray,
    target: np.ndarray,
    iterations: int,
    device: torch.device,
) -> tuple[float, float]:
    pred = torch.from_numpy(np.ascontiguousarray(prediction, dtype=np.float32)).to(device)
    gt = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).to(device)
    with torch.inference_mode():
        scale = torch.median(gt) / torch.median(pred).clamp_min(1e-8)
        shift = torch.zeros((), device=device, dtype=torch.float32)
        gt_weight = 1.0 / gt.clamp_min(1e-8)
        for _ in range(iterations):
            residual = (scale * pred + shift - gt).abs().clamp_min(1e-8)
            weights = gt_weight / residual
            a00 = (weights * pred * pred).sum()
            a01 = (weights * pred).sum()
            a11 = weights.sum()
            b0 = (weights * pred * gt).sum()
            b1 = (weights * gt).sum()
            determinant = a00 * a11 - a01 * a01
            if determinant.abs() < 1e-12:
                scale = (weights * pred * gt).sum() / (weights * pred * pred).sum().clamp_min(1e-12)
                shift = torch.zeros_like(shift)
            else:
                scale = (b0 * a11 - b1 * a01) / determinant
                shift = (a00 * b1 - a01 * b0) / determinant
    return float(scale.item()), float(shift.item())


def latitude_mask(height: int, width: int, minimum: float, maximum: float) -> np.ndarray:
    latitude = 90.0 - (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)
    return np.broadcast_to(((latitude >= minimum) & (latitude <= maximum))[:, None], (height, width))


def evaluate_depth_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int]:
    if prediction.shape != target.shape:
        tensor = torch.from_numpy(prediction)[None, None]
        prediction = F.interpolate(tensor, size=target.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    full_valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & (target >= args.min_depth)
        & (target < args.max_depth)
    )
    if not np.any(full_valid):
        raise RuntimeError("No valid full-ERP depth pixels")
    scale, shift = fit_irls_absrel_scale_shift(
        prediction[full_valid], target[full_valid], args.irls_iterations, device
    )
    aligned = prediction * np.float32(scale) + np.float32(shift)
    valid = full_valid & latitude_mask(
        target.shape[0], target.shape[1], args.latitude_min, args.latitude_max
    )
    pred_values = aligned[valid].astype(np.float64)
    target_values = target[valid].astype(np.float64)
    difference = pred_values - target_values
    ratio = np.maximum(pred_values / target_values, target_values / pred_values)
    abs_rel_values = np.abs(difference) / target_values
    squared = difference * difference
    valid_pixels = int(valid.sum())
    return {
        "valid_pixels": valid_pixels,
        "scale": scale,
        "shift": shift,
        "abs_rel": float(abs_rel_values.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "abs_rel_sum": float(abs_rel_values.sum()),
        "sq_error_sum": float(squared.sum()),
        "delta1_count": float(np.sum(ratio < 1.25)),
        "delta2_count": float(np.sum(ratio < 1.25**2)),
    }


def initialize_csv(path: Path, fields: Sequence[str], overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def append_rows(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def read_csv(path: Path, resume: bool) -> list[dict[str, str]]:
    if not resume or not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_depth(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "pixel_micro": {}, "image_macro": {}}
    pixels = sum(int(float(row["valid_pixels"])) for row in rows)
    abs_rel_sum = sum(float(row["abs_rel_sum"]) for row in rows)
    sq_error_sum = sum(float(row["sq_error_sum"]) for row in rows)
    delta1_count = sum(float(row["delta1_count"]) for row in rows)
    delta2_count = sum(float(row["delta2_count"]) for row in rows)
    return {
        "count": len(rows),
        "pixel_micro": {
            "absrel": abs_rel_sum / pixels,
            "rmse": math.sqrt(sq_error_sum / pixels),
            "delta1": delta1_count / pixels,
            "delta2": delta2_count / pixels,
            "valid_pixels": pixels,
        },
        "image_macro": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in ("abs_rel", "rmse", "delta1", "delta2", "scale", "shift", "coverage", "seconds")
        },
    }


def run_depth(args: argparse.Namespace, checkpoint: Path, rotated_root: Path, split_file: Path) -> None:
    samples = read_split(split_file, rotated_root)
    expected_samples = len(samples)
    if args.limit > 0:
        samples = samples[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_sample_metrics.csv"
    rows = read_csv(csv_path, args.resume)
    completed = {row["rgb_relative_path"] for row in rows}
    initialize_csv(csv_path, DEPTH_FIELDS, overwrite=not args.resume)
    device = torch.device(args.device)
    runner = DepthRunner(args, checkpoint, device)
    run_config = {
        "dataset": "PanoSUNCG",
        "split": "DA-2 official evaluation split",
        "split_sha256": SPLIT_SHA256,
        "method": args.method,
        "task": "depth",
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "rotated_root": str(rotated_root),
        "expected_full_split_samples": expected_samples,
        "samples_in_run": len(samples),
        "depth_decode": "OpenCV channel 0 divided by 20, matching official DA-2",
        "alignment": (
            f"per-image {args.irls_iterations}-step IRLS scale+shift fitted on full valid ERP"
        ),
        "metric_region_latitude_degrees": [args.latitude_min, args.latitude_max],
        "aggregation": "pixel micro",
        "depth_range": [args.min_depth, args.max_depth],
        "pi3_projection": (
            f"six {args.face_size}x{args.face_size} 90-degree cube faces reconstructed to ERP"
            if args.method == "pi3"
            else None
        ),
        "bifusepp_preprocess": (
            "SupervisedCombinedModel.forward ImageNet normalization, then released [0,10] clipping"
            if args.method == "bifusepp"
            else None
        ),
        "validity": "GT-valid full ERP for alignment; GT-valid latitude band for metrics",
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "checkpoint_load": runner.load_info,
    }
    atomic_write_json(args.output_dir / "run_config.json", run_config)
    started = time.time()
    processed_this_session = 0
    for sample_index, rgb_rel, depth_rel in samples:
        if rgb_rel in completed:
            continue
        sample_started = time.perf_counter()
        try:
            rgb = read_rgb(rotated_root / rgb_rel)
            target = read_depth(rotated_root / depth_rel)
            prediction, coverage, inference_seconds = runner.infer(rgb)
            metrics = evaluate_depth_prediction(prediction, target, args, device)
            row = {
                "sample_index": sample_index,
                "rgb_relative_path": rgb_rel,
                "depth_relative_path": depth_rel,
                **metrics,
                "coverage": coverage,
                "seconds": time.perf_counter() - sample_started,
            }
            append_rows(csv_path, DEPTH_FIELDS, [row])
            rows.append({key: str(value) for key, value in row.items()})
            completed.add(rgb_rel)
            processed_this_session += 1
            if processed_this_session == 1 or processed_this_session % max(args.progress_every, 1) == 0:
                summary = summarize_depth(rows)
                elapsed = time.time() - started
                rate = processed_this_session / max(elapsed, 1e-9)
                remaining = len(samples) - len(completed)
                atomic_write_json(
                    args.output_dir / "progress.json",
                    {
                        "status": "running" if remaining else "completed",
                        "completed_samples": len(completed),
                        "total_samples": len(samples),
                        "eta_seconds": remaining / rate if rate else None,
                        "last_rgb": rgb_rel,
                        "last_inference_seconds": inference_seconds,
                        "running_summary": summary,
                    },
                )
                print(
                    f"[depth:{args.method}] {len(completed)}/{len(samples)} "
                    f"AbsRel={summary['pixel_micro']['absrel']:.5f} "
                    f"RMSE={summary['pixel_micro']['rmse']:.5f} "
                    f"d1={summary['pixel_micro']['delta1']:.5f} "
                    f"coverage={coverage:.5f}",
                    flush=True,
                )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            with (args.output_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"rgb": rgb_rel, "error": str(exc)}) + "\n")
    result = {
        "status": "completed",
        **run_config,
        "completed_samples": len(rows),
        "metrics": summarize_depth(rows),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(args.output_dir / "metrics_summary.json", result)
    atomic_write_json(
        args.output_dir / "progress.json",
        {"status": "completed", "completed_samples": len(rows), "total_samples": len(samples)},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector.astype(np.float64) / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def pose_vector_to_transform(vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axis_angle_to_matrix(vector[:3])
    transform[:3, 3] = vector[3:]
    return transform


class CameraRunner:
    def __init__(self, args: argparse.Namespace, checkpoint: Path, device: torch.device):
        self.args = args
        self.device = device
        if args.method == "bifusepp":
            self.model, self.load_info = load_bifuse_model(checkpoint, "selfsupervised", device)
        else:
            self.model, self.load_info = load_pi3_model(checkpoint, device)

    def infer(self, rgbs: Sequence[np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        images = torch.stack([image_tensor(rgb, self.device) for rgb in rgbs])
        if self.args.method == "bifusepp":
            if len(images) != 5:
                raise ValueError("BiFuse++ trajectory composition requires exactly five frames")
            with torch.inference_mode():
                first = self.model.preprocess(images[[1]])
                first_targets = [self.model.preprocess(images[[0]]), self.model.preprocess(images[[2]])]
                _, pose_first = self.model.pnet(first, first_targets)
                second = self.model.preprocess(images[[3]])
                second_targets = [self.model.preprocess(images[[2]]), self.model.preprocess(images[[4]])]
                _, pose_second = self.model.pnet(second, second_targets)
            pose_first = pose_first[0].float().cpu().numpy()
            pose_second = pose_second[0].float().cpu().numpy()
            transform_0 = pose_vector_to_transform(pose_first[0])
            transform_1 = np.eye(4, dtype=np.float64)
            transform_2 = pose_vector_to_transform(pose_first[1])
            transform_2_from_3 = pose_vector_to_transform(pose_second[0])
            transform_4_from_3 = pose_vector_to_transform(pose_second[1])
            transform_3 = np.linalg.inv(transform_2_from_3) @ transform_2
            transform_4 = transform_4_from_3 @ transform_3
            world_to_camera = [transform_0, transform_1, transform_2, transform_3, transform_4]
            centers = np.stack([np.linalg.inv(transform)[:3, 3] for transform in world_to_camera])
        else:
            front_views = torch.stack(
                [erp_to_perspectives(image, ((0.0, 0.0),), self.args.face_size)[0] for image in images]
            )
            with torch.inference_mode(), amp_context(self.device, self.args.amp_dtype):
                result = self.model(front_views[None])
            poses = result["camera_poses"][0].float().cpu().numpy().astype(np.float64)
            centers = poses[:, :3, 3]
            del front_views, result, poses
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        del images
        return centers, time.perf_counter() - started


def load_gt_centers(labels_path: Path, indices: Sequence[int]) -> np.ndarray:
    centers = []
    for line_number, raw in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if len(fields) != 3:
            raise ValueError(f"{labels_path}:{line_number}: expected XYZ")
        centers.append([float(value) for value in fields])
    array = np.asarray(centers, dtype=np.float64)
    if max(indices) >= len(array):
        raise IndexError(f"Frame index exceeds labels in {labels_path}")
    return array[np.asarray(indices, dtype=np.int64)]


def umeyama_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance < 1e-12:
        raise ValueError("Predicted camera centers are degenerate")
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def direction_auc(errors: np.ndarray, threshold: int) -> float:
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(np.float64) / max(len(errors), 1))))


def evaluate_centers(
    predicted: np.ndarray, target: np.ndarray, selected_indices: Sequence[int]
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    scale, rotation, translation = umeyama_sim3(predicted, target)
    aligned = (scale * (rotation @ predicted.T)).T + translation
    frame_errors = np.linalg.norm(aligned - target, axis=1)
    target_centered = target - target.mean(axis=0)
    trajectory_scale = float(np.sqrt(np.mean(np.sum(target_centered**2, axis=1))))
    frame_rows = [
        {
            "frame_index": int(frame_index),
            "gt_x": float(target[local_index, 0]),
            "gt_y": float(target[local_index, 1]),
            "gt_z": float(target[local_index, 2]),
            "pred_aligned_x": float(aligned[local_index, 0]),
            "pred_aligned_y": float(aligned[local_index, 1]),
            "pred_aligned_z": float(aligned[local_index, 2]),
            "ate": float(frame_errors[local_index]),
        }
        for local_index, frame_index in enumerate(selected_indices)
    ]
    pair_rows = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            gt_delta = target[j] - target[i]
            pred_delta = aligned[j] - aligned[i]
            gt_norm = float(np.linalg.norm(gt_delta))
            pred_norm = float(np.linalg.norm(pred_delta))
            if gt_norm < 1e-8 or pred_norm < 1e-8:
                continue
            cosine = float(np.clip(np.dot(gt_delta, pred_delta) / (gt_norm * pred_norm), -1.0, 1.0))
            pair_rows.append(
                {
                    "frame_i": int(selected_indices[i]),
                    "frame_j": int(selected_indices[j]),
                    "gt_baseline": gt_norm,
                    "pred_aligned_baseline": pred_norm,
                    "direction_error_deg": float(np.degrees(np.arccos(cosine))),
                    "relative_length_error": float(abs(pred_norm - gt_norm) / gt_norm),
                }
            )
    directions = np.asarray([row["direction_error_deg"] for row in pair_rows])
    lengths = np.asarray([row["relative_length_error"] for row in pair_rows])
    metrics = {
        "pair_count": float(len(pair_rows)),
        "sim3_scale": scale,
        "ate_rmse": float(np.sqrt(np.mean(frame_errors**2))),
        "ate_mean": float(np.mean(frame_errors)),
        "ate_median": float(np.median(frame_errors)),
        "ate_normalized_rmse": float(np.sqrt(np.mean(frame_errors**2)) / max(trajectory_scale, 1e-12)),
        "direction_deg_mean": float(np.mean(directions)),
        "direction_deg_median": float(np.median(directions)),
        "direction_auc3": direction_auc(directions, 3),
        "direction_auc5": direction_auc(directions, 5),
        "direction_auc15": direction_auc(directions, 15),
        "direction_auc30": direction_auc(directions, 30),
        "relative_length_error_mean": float(np.mean(lengths)),
        "relative_length_error_median": float(np.median(lengths)),
    }
    return metrics, frame_rows, pair_rows


def finite_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def summarize_camera(
    trajectory_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not trajectory_rows:
        return {"trajectory_count": 0, "frame_count": 0, "pair_count": 0, "macro_by_trajectory": {}, "micro": {}}
    frame_errors = np.asarray([float(row["ate"]) for row in frame_rows], dtype=np.float64)
    directions = np.asarray([float(row["direction_error_deg"]) for row in pair_rows], dtype=np.float64)
    lengths = np.asarray([float(row["relative_length_error"]) for row in pair_rows], dtype=np.float64)
    macro_keys = (
        "ate_rmse", "ate_mean", "ate_median", "ate_normalized_rmse",
        "direction_deg_mean", "direction_deg_median", "direction_auc3", "direction_auc5",
        "direction_auc15", "direction_auc30", "relative_length_error_mean",
        "relative_length_error_median",
    )
    return {
        "trajectory_count": len(trajectory_rows),
        "frame_count": len(frame_rows),
        "pair_count": len(pair_rows),
        "macro_by_trajectory": {key: finite_mean(trajectory_rows, key) for key in macro_keys},
        "micro": {
            "ate_rmse": float(np.sqrt(np.mean(frame_errors**2))),
            "ate_mean": float(np.mean(frame_errors)),
            "ate_median": float(np.median(frame_errors)),
            "direction_deg_mean": float(np.mean(directions)),
            "direction_deg_median": float(np.median(directions)),
            "direction_auc3": direction_auc(directions, 3),
            "direction_auc5": direction_auc(directions, 5),
            "direction_auc15": direction_auc(directions, 15),
            "direction_auc30": direction_auc(directions, 30),
            "relative_length_error_mean": float(np.mean(lengths)),
            "relative_length_error_median": float(np.median(lengths)),
        },
    }


def run_camera(
    args: argparse.Namespace,
    checkpoint: Path,
    rotated_root: Path,
    labels_root: Path,
    split_file: Path,
) -> None:
    if args.frames_per_trajectory != 5:
        raise ValueError("The table protocol requires exactly five frames per trajectory")
    trajectories = build_trajectories(split_file)
    expected_trajectories = len(trajectories)
    if args.max_trajectories > 0:
        trajectories = trajectories[: args.max_trajectories]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_csv = args.output_dir / "trajectory_metrics.csv"
    frame_csv = args.output_dir / "frame_metrics.csv"
    pair_csv = args.output_dir / "pair_metrics.csv"
    trajectory_rows = read_csv(trajectory_csv, args.resume)
    frame_rows = read_csv(frame_csv, args.resume)
    pair_rows = read_csv(pair_csv, args.resume)
    completed = {(row["scene"], row["trajectory"]) for row in trajectory_rows}
    if not args.resume:
        for path, fields in ((trajectory_csv, TRAJECTORY_FIELDS), (frame_csv, FRAME_FIELDS), (pair_csv, PAIR_FIELDS)):
            initialize_csv(path, fields, overwrite=True)
    device = torch.device(args.device)
    runner = CameraRunner(args, checkpoint, device)
    caveat = (
        "The released self-supervised BiFuse++ camera checkpoint was trained on PanoSUNCG; "
        "this is an in-domain evaluation, not strict zero-shot."
        if args.method == "bifusepp"
        else None
    )
    run_config = {
        "dataset": "PanoSUNCG",
        "split": "DA-2 official evaluation split grouped into complete trajectories",
        "split_sha256": SPLIT_SHA256,
        "method": args.method,
        "task": "camera",
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "expected_full_split_trajectories": expected_trajectories,
        "trajectories_in_run": len(trajectories),
        "frames_per_trajectory": 5,
        "frame_selection": "deterministic evenly spaced, including endpoints",
        "alignment": "per-trajectory Umeyama Sim(3)",
        "rotation_ground_truth_available": False,
        "pi3_input": (
            f"one canonical front-facing {args.face_size}x{args.face_size} 90-degree perspective per ERP"
            if args.method == "pi3"
            else None
        ),
        "bifusepp_pose_composition": (
            "two overlapping native three-frame PoseNet calls: [0,1,2] and [2,3,4]"
            if args.method == "bifusepp"
            else None
        ),
        "strict_zero_shot_caveat": caveat,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "checkpoint_load": runner.load_info,
    }
    atomic_write_json(args.output_dir / "run_config.json", run_config)
    started = time.time()
    processed_this_session = 0
    for item in trajectories:
        scene, trajectory = item["scene"], item["trajectory"]
        if (scene, trajectory) in completed:
            continue
        trajectory_started = time.perf_counter()
        try:
            selected = evenly_spaced_indices(item["indices"], 5)
            if len(selected) != 5:
                raise ValueError(f"Trajectory has fewer than five split frames: {scene}/{trajectory}")
            target = load_gt_centers(labels_root / scene / trajectory, selected)
            rgbs = [read_rgb(rotated_root / scene / trajectory / f"{index}_color.png") for index in selected]
            predicted, inference_seconds = runner.infer(rgbs)
            metrics, new_frame_rows, new_pair_rows = evaluate_centers(predicted, target, selected)
            trajectory_row = {
                "scene": scene,
                "trajectory": trajectory,
                "num_available_frames": len(item["indices"]),
                "num_evaluated_frames": len(selected),
                "selected_indices": ",".join(str(index) for index in selected),
                **metrics,
                "pair_count": int(metrics["pair_count"]),
                "inference_seconds": inference_seconds,
                "trajectory_seconds": time.perf_counter() - trajectory_started,
            }
            for row in new_frame_rows:
                row.update({"scene": scene, "trajectory": trajectory})
            for row in new_pair_rows:
                row.update({"scene": scene, "trajectory": trajectory})
            append_rows(trajectory_csv, TRAJECTORY_FIELDS, [trajectory_row])
            append_rows(frame_csv, FRAME_FIELDS, new_frame_rows)
            append_rows(pair_csv, PAIR_FIELDS, new_pair_rows)
            trajectory_rows.append({key: str(value) for key, value in trajectory_row.items()})
            frame_rows.extend({key: str(value) for key, value in row.items()} for row in new_frame_rows)
            pair_rows.extend({key: str(value) for key, value in row.items()} for row in new_pair_rows)
            completed.add((scene, trajectory))
            processed_this_session += 1
            if processed_this_session == 1 or processed_this_session % max(args.progress_every, 1) == 0:
                summary = summarize_camera(trajectory_rows, frame_rows, pair_rows)
                elapsed = time.time() - started
                rate = processed_this_session / max(elapsed, 1e-9)
                remaining = len(trajectories) - len(completed)
                atomic_write_json(
                    args.output_dir / "progress.json",
                    {
                        "status": "running" if remaining else "completed",
                        "completed_trajectories": len(completed),
                        "total_trajectories": len(trajectories),
                        "eta_seconds": remaining / rate if rate else None,
                        "running_summary": summary,
                    },
                )
                print(
                    f"[camera:{args.method}] {len(completed)}/{len(trajectories)} "
                    f"ATE={summary['micro']['ate_rmse']:.4f} "
                    f"T={summary['micro']['direction_deg_mean']:.3f}deg",
                    flush=True,
                )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            with (args.output_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"scene": scene, "trajectory": trajectory, "error": str(exc)}) + "\n")
    summary = summarize_camera(trajectory_rows, frame_rows, pair_rows)
    result = {
        "status": "completed",
        "dataset": "PanoSUNCG",
        "method": args.method,
        "protocol": "camera-center trajectory estimate on DA-2 complete trajectories",
        "limitations": {
            "rotation_metrics": "unavailable: distributed labels contain XYZ centers only",
            "alignment": "Sim(3) removes global translation, rotation, and scale gauge",
            "strict_zero_shot": caveat,
        },
        **summary,
        "run_config": run_config,
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(args.output_dir / "camera_center_summary.json", result)
    atomic_write_json(
        args.output_dir / "progress.json",
        {"status": "completed", "completed_trajectories": len(trajectory_rows), "total_trajectories": len(trajectories)},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.face_size <= 0 or args.face_size % 14 != 0:
        raise ValueError("--face-size must be a positive multiple of Pi3's 14-pixel patch size")
    rotated_root, labels_root, split_file = resolve_dataset(args.dataset_root)
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.method == "bifusepp":
            checkpoint = DEFAULT_BIFUSE_DEPTH_CKPT if args.task == "depth" else DEFAULT_BIFUSE_CAMERA_CKPT
        else:
            checkpoint = DEFAULT_PI3_CKPT
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_dir / ".eval.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another evaluation is writing {args.output_dir}") from exc
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.task == "depth":
        run_depth(args, checkpoint, rotated_root, split_file)
    else:
        run_camera(args, checkpoint, rotated_root, labels_root, split_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
