#!/usr/bin/env python3
"""Evaluate a trained full-pipeline checkpoint on the official DA2 PanoSUNCG split."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_ROOT = PROJECT_ROOT.parent
for root in (PROJECT_ROOT, GIT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.evaluate_depth_checkpoint import (  # noqa: E402
    apply_checkpoint_eval_defaults,
    build_eval_model,
    load_checkpoint_payload,
    normalize_args_for_eval,
)
from training.train_pano_omega import (  # noqa: E402
    parse_args as parse_training_args,
    resolve_device,
    unwrap_model,
)
from vggt_omega.models.layers.pano_position import yaw_pitch_to_axes  # noqa: E402


METRIC_NAMES = (
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
)
CSV_FIELDS = (
    "sample_index",
    "rgb_relative_path",
    "depth_relative_path",
    "valid_pixels",
    "coverage",
    "median_scale",
    *METRIC_NAMES,
    "seconds",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="datasets root, PanoSUNCG_zeroshot root, or PanoSUNCG/rotated root",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--window-size", type=int, default=384)
    parser.add_argument("--num-yaw", type=int, default=6)
    parser.add_argument("--pitch-degrees", default="-55,-15,55")
    parser.add_argument("--fov-degrees", type=float, default=75.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete official split")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir, split_file = resolve_dataset(args.dataset_root)
    samples = read_split(split_file, dataset_dir)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError(f"No PanoSUNCG samples found in {split_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_dir / ".eval.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another evaluation is already writing {args.output_dir}") from exc
    csv_path = args.output_dir / "per_sample_metrics.csv"
    progress_path = args.output_dir / "progress.json"
    summary_path = args.output_dir / "metrics_summary.json"
    config_path = args.output_dir / "run_config.json"
    completed, rows = load_completed(csv_path) if args.resume else (set(), [])
    initialize_csv(csv_path, overwrite=not args.resume)
    if args.resume and rows:
        write_rows_csv(csv_path, rows)

    device = resolve_device(args.device, {"distributed": False, "local_rank": 0})
    train_args = parse_training_args(["--config", str(args.config)])
    train_args.device = args.device
    train_args.distributed = "none"
    train_args.checkpoint = args.checkpoint
    train_args.amp_dtype = args.amp_dtype
    normalize_args_for_eval(train_args)
    payload = load_checkpoint_payload(args.checkpoint)
    apply_checkpoint_eval_defaults(train_args, payload)
    train_args.window_size = int(args.window_size)
    train_args.num_yaw = int(args.num_yaw)
    train_args.pitch_degrees = str(args.pitch_degrees)
    train_args.fov_degrees = float(args.fov_degrees)
    model = build_eval_model(train_args, args.checkpoint, payload, device)
    model.eval()

    pitches = tuple(float(value.strip()) for value in args.pitch_degrees.split(",") if value.strip())
    views = int(args.num_yaw) * len(pitches)
    run_config = {
        "protocol": "DA2 official PanoSUNCG evaluation",
        "official_repository": "https://github.com/EnVision-Research/DA-2",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "dataset_root": str(dataset_dir),
        "split_file": str(split_file),
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "samples": len(samples),
        "sampler": {
            "window_size": int(args.window_size),
            "num_yaw": int(args.num_yaw),
            "pitch_degrees": pitches,
            "fov_degrees": float(args.fov_degrees),
            "views_per_panorama": views,
        },
        "erp_reconstruction": "inverse projection; window Z-depth to radial depth; z_factor^2 weighting",
        "depth_decode": "OpenCV channel 0 divided by 20, matching official DA2 PanoSUNCGDataset",
        "alignment": "per-image median scale, matching official DA2 evaluation",
        "depth_range": [float(args.min_depth), float(args.max_depth)],
        "device": str(device),
        "amp_dtype": str(args.amp_dtype),
        "resume": bool(args.resume),
    }
    atomic_write_json(config_path, run_config)
    print(
        f"[data] {len(samples)} split samples; {len(completed)} completed; "
        f"split_sha256={run_config['split_sha256']}"
    )
    print(f"[tiling] {args.num_yaw} yaw x {len(pitches)} pitch = {views} windows, pitch={pitches}")

    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float32
    started = time.time()
    for sample_index, rgb_rel, depth_rel in samples:
        if rgb_rel in completed:
            continue
        sample_started = time.time()
        rgb = read_rgb(dataset_dir / rgb_rel).to(device)
        gt = read_depth(dataset_dir / depth_rel)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            predictions = model(pano_images=rgb[None, None], return_sampler_output=True)
        pred_z = predictions["depth"][0, ..., 0].detach().float()
        sampler_model = unwrap_model(model)
        camera_meta = predictions.get("pano_camera_meta")
        if camera_meta is None:
            camera_meta = sampler_model.pano_sampler(rgb[None]).camera_meta
        camera_meta = {key: value[0].detach().float() for key, value in camera_meta.items()}
        pred_erp, pred_valid = inverse_project_windows_to_erp(
            pred_z,
            camera_meta,
            gt.shape,
        )

        gt_t = torch.from_numpy(gt).to(device=pred_erp.device, dtype=torch.float32)
        valid = (
            pred_valid
            & torch.isfinite(pred_erp)
            & (pred_erp > 0)
            & torch.isfinite(gt_t)
            & (gt_t >= float(args.min_depth))
            & (gt_t <= float(args.max_depth))
        )
        if int(valid.sum()) == 0:
            raise RuntimeError(f"No valid evaluation pixels for {rgb_rel}")
        pred_median = torch.median(pred_erp[valid]).clamp_min(1e-8)
        median_scale = torch.median(gt_t[valid]) / pred_median
        aligned = (pred_erp * median_scale).clamp(float(args.min_depth), float(args.max_depth))
        metrics = compute_metrics(aligned, gt_t, valid)
        row = {
            "sample_index": sample_index,
            "rgb_relative_path": rgb_rel,
            "depth_relative_path": depth_rel,
            "valid_pixels": int(valid.sum().item()),
            "coverage": float(pred_valid.float().mean().item()),
            "median_scale": float(median_scale.item()),
            **metrics,
            "seconds": time.time() - sample_started,
        }
        append_csv(csv_path, row)
        rows.append(row)
        completed.add(rgb_rel)
        processed = len(rows)
        if processed == 1 or processed % max(1, args.progress_every) == 0 or len(completed) == len(samples):
            summary = summarize(rows)
            state = {
                "state": "running",
                "processed_samples": len(completed),
                "total_samples": len(samples),
                "last_rgb": rgb_rel,
                "last_metrics": metrics,
                "running_mean": summary.get("mean", {}),
                "elapsed_seconds": time.time() - started,
                "updated_at": time.time(),
            }
            atomic_write_json(progress_path, state)
            print(
                f"[progress] {len(completed)}/{len(samples)} "
                f"({100.0 * len(completed) / len(samples):.2f}%) "
                f"AbsRel={summary['mean']['abs_relative_difference']:.6f} "
                f"delta1={summary['mean']['delta1_acc']:.6f} "
                f"coverage={summary['mean']['coverage']:.6f} "
                f"last={row['seconds']:.2f}s",
                flush=True,
            )

    final = {
        **run_config,
        "evaluated_samples": len(rows),
        "metrics": summarize(rows),
    }
    atomic_write_json(summary_path, final)
    atomic_write_json(
        progress_path,
        {
            "state": "done",
            "processed_samples": len(completed),
            "total_samples": len(samples),
            "summary": final["metrics"],
            "updated_at": time.time(),
        },
    )
    print(json.dumps(final, indent=2, ensure_ascii=False))


def resolve_dataset(root: Path) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    candidates = (root, root / "PanoSUNCG_zeroshot")
    for candidate in candidates:
        split = candidate / "panosuncg_da2_split.txt"
        rotated = candidate / "PanoSUNCG" / "rotated"
        if split.is_file() and rotated.is_dir():
            return rotated, split
    if root.name == "rotated" and root.is_dir():
        zero_root = root.parents[1]
        split = zero_root / "panosuncg_da2_split.txt"
        if split.is_file():
            return root, split
    raise FileNotFoundError(
        f"Cannot resolve PanoSUNCG_zeroshot under {root}; expected "
        "panosuncg_da2_split.txt and PanoSUNCG/rotated"
    )


def read_split(split_file: Path, dataset_dir: Path) -> list[tuple[int, str, str]]:
    samples: list[tuple[int, str, str]] = []
    for index, line in enumerate(split_file.read_text(encoding="utf-8").splitlines()):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"Malformed split line {index + 1}: {line!r}")
        rgb_rel, depth_rel = fields
        if not (dataset_dir / rgb_rel).is_file() or not (dataset_dir / depth_rel).is_file():
            raise FileNotFoundError(f"Missing split files at line {index + 1}: {line}")
        samples.append((index, rgb_rel, depth_rel))
    return samples


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32) / 20.0


def inverse_project_windows_to_erp(
    pred_z: torch.Tensor,
    camera_meta: dict[str, torch.Tensor],
    pano_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull each ERP ray from every covering pinhole view and blend radial depth."""
    pano_h, pano_w = pano_hw
    device = pred_z.device
    dtype = torch.float32
    theta = (torch.arange(pano_w, device=device, dtype=dtype) + 0.5) / pano_w
    theta = (theta - 0.5) * (2.0 * math.pi)
    phi = 0.5 - (torch.arange(pano_h, device=device, dtype=dtype) + 0.5) / pano_h
    phi = phi * math.pi
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")
    cos_phi = torch.cos(phi_grid)
    rays = torch.stack(
        [cos_phi * torch.sin(theta_grid), torch.sin(phi_grid), cos_phi * torch.cos(theta_grid)],
        dim=-1,
    )

    yaw = camera_meta["yaw"].reshape(-1).to(device=device, dtype=dtype)
    pitch = camera_meta["pitch"].reshape(-1).to(device=device, dtype=dtype)
    fov_x = camera_meta["fov_x"].reshape(-1).to(device=device, dtype=dtype)
    fov_y = camera_meta["fov_y"].reshape(-1).to(device=device, dtype=dtype)
    forward, right, up = yaw_pitch_to_axes(yaw, pitch)
    weighted_sum = torch.zeros((pano_h, pano_w), device=device, dtype=dtype)
    weight_sum = torch.zeros_like(weighted_sum)
    for view_index in range(pred_z.shape[0]):
        z_factor = torch.einsum("hwc,c->hw", rays, forward[view_index])
        x_factor = torch.einsum("hwc,c->hw", rays, right[view_index])
        y_factor = torch.einsum("hwc,c->hw", rays, up[view_index])
        safe_z = z_factor.clamp_min(1e-8)
        grid_x = x_factor / (safe_z * torch.tan(fov_x[view_index] * 0.5))
        grid_y = -y_factor / (safe_z * torch.tan(fov_y[view_index] * 0.5))
        grid = torch.stack([grid_x, grid_y], dim=-1)[None]
        sampled_z = F.grid_sample(
            pred_z[view_index][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
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
    valid_erp = weight_sum > 0
    erp = weighted_sum / weight_sum.clamp_min(1e-8)
    return erp, valid_erp


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    pred = pred[valid]
    gt = gt[valid]
    diff = pred - gt
    log_diff = torch.log(pred) - torch.log(gt)
    ratio = torch.maximum(pred / gt, gt / pred)
    return {
        "abs_relative_difference": float((diff.abs() / gt).mean().item()),
        "squared_relative_difference": float((diff.square() / gt).mean().item()),
        "rmse_linear": float(diff.square().mean().sqrt().item()),
        "rmse_log": float(log_diff.square().mean().sqrt().item()),
        "log10": float((torch.log10(pred) - torch.log10(gt)).abs().mean().item()),
        "delta1_acc": float((ratio < 1.25).float().mean().item()),
        "delta2_acc": float((ratio < 1.25**2).float().mean().item()),
        "delta3_acc": float((ratio < 1.25**3).float().mean().item()),
        "i_rmse": float(((pred.reciprocal() - gt.reciprocal()).square().mean().sqrt()).item()),
        "silog_rmse": float(
            (log_diff.square().mean() - log_diff.mean().square()).clamp_min(0).sqrt().mul(100).item()
        ),
    }


def initialize_csv(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()


def append_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})
        handle.flush()


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def load_completed(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.is_file():
        return set(), []
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows_by_rgb: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for key in ("sample_index", "valid_pixels"):
            row[key] = int(float(raw[key]))
        for key in ("coverage", "median_scale", *METRIC_NAMES, "seconds"):
            row[key] = float(raw[key])
        rows_by_rgb[str(row["rgb_relative_path"])] = row
    rows = list(rows_by_rgb.values())
    return set(rows_by_rgb), rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mean": {}, "median": {}}
    keys = (*METRIC_NAMES, "coverage", "median_scale", "seconds")
    return {
        "count": len(rows),
        "mean": {key: float(np.mean([float(row[key]) for row in rows])) for key in keys},
        "median": {key: float(np.median([float(row[key]) for row in rows])) for key in keys},
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
