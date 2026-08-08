#!/usr/bin/env python3
"""Evaluate PanoSUNCG in the 0716 checkpoint-native four-window domain.

One complete ERP is passed to the model. The checkpoint restores its native
four-window sampler (checkpoint-native square resolution, four yaw angles, pitch -15 degrees,
75-degree FOV). Ground-truth radial ERP depth is sampled onto exactly the same
rays and converted to pinhole Z-depth. Metrics are computed directly in window
space with the repository's solid-angle weighting, overlap de-duplication and
10-step scale-only IRLS. No ERP fusion or latitude mask is used.

The evaluator is resumable and supports deterministic independent sharding.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_ROOT = PROJECT_ROOT.parent
for root in (PROJECT_ROOT, GIT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.evaluate_depth_checkpoint import (  # noqa: E402
    DEPTH_ACCUMULATOR_KEYS,
    DEPTH_METRIC_KEYS,
    DEPTH_METRIC_PROTOCOL,
    apply_checkpoint_eval_defaults,
    build_covered_sphere_weights,
    build_eval_model,
    compute_depth_metrics,
    load_checkpoint_payload,
    normalize_args_for_eval,
)
from scripts.evaluate_panosuncg_zeroshot import (  # noqa: E402
    atomic_write_json,
    read_depth,
    read_rgb,
    resolve_dataset,
)
from training.train_pano_omega import (  # noqa: E402
    parse_args as parse_training_args,
    resolve_device,
    sample_depth_targets,
    unwrap_model,
)
from vggt_omega.data.pano_sampler import resolve_fov_degrees  # noqa: E402


BASE_FIELDS = (
    "sample_index",
    "rgb_relative_path",
    "depth_relative_path",
    "views",
    "inference_seconds",
    "sample_seconds",
)
CSV_FIELDS = BASE_FIELDS + tuple(DEPTH_METRIC_KEYS) + tuple(DEPTH_ACCUMULATOR_KEYS)


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
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete official split")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def read_split(split_file: Path, dataset_dir: Path) -> list[tuple[int, str, str]]:
    samples: list[tuple[int, str, str]] = []
    for line_number, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{split_file}:{line_number}: expected RGB and depth paths")
        rgb_rel, depth_rel = fields
        if not (dataset_dir / rgb_rel).is_file() or not (dataset_dir / depth_rel).is_file():
            raise FileNotFoundError(f"Missing split files at line {line_number}: {raw}")
        samples.append((len(samples), rgb_rel, depth_rel))
    return samples


def parse_pitch_degrees(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    return (float(value),)


def initialize_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(row)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and not set(CSV_FIELDS).issubset(rows[0]):
        raise RuntimeError(f"Incompatible resumable CSV schema: {path}")
    return rows


def summarize_rows(
    rows: list[Mapping[str, Any]],
    expected_samples: int,
    *,
    total_split_samples: int,
) -> dict[str, Any]:
    valid = sum(float(row.get("depth_valid_pixels", 0.0)) for row in rows)
    micro: dict[str, Any] = {"depth_valid_pixels": int(valid)}
    if valid > 0:
        for prefix in ("depth", "depth_irls"):
            abs_error = sum(float(row[f"{prefix}_abs_error_sum"]) for row in rows)
            sq_error = sum(float(row[f"{prefix}_sq_error_sum"]) for row in rows)
            abs_rel = sum(float(row[f"{prefix}_abs_rel_sum"]) for row in rows)
            micro[f"{prefix}_mae"] = abs_error / valid
            micro[f"{prefix}_rmse"] = math.sqrt(max(sq_error / valid, 0.0))
            micro[f"{prefix}_abs_rel"] = abs_rel / valid
            for label in ("1p25", "1p25_2", "1p25_3"):
                count = sum(float(row[f"{prefix}_delta_{label}_count"]) for row in rows)
                micro[f"{prefix}_delta_{label}"] = count / valid
    macro = {
        "depth_irls_abs_rel": (
            float(np.mean([float(row["depth_irls_abs_rel"]) for row in rows])) if rows else None
        ),
        "depth_irls_scale": (
            float(np.mean([float(row["depth_irls_scale"]) for row in rows])) if rows else None
        ),
    }
    return {
        "status": "completed" if len(rows) == expected_samples else "running",
        "dataset": "PanoSUNCG",
        "method": "0716 full pipeline",
        "protocol": "checkpoint-native four-window direct evaluation",
        "completed_samples": len(rows),
        "expected_samples": expected_samples,
        "total_split_samples": total_split_samples,
        "primary_metric": {
            "name": "pixel_micro.depth_irls_abs_rel",
            "value": micro.get("depth_irls_abs_rel"),
        },
        "pixel_micro": micro,
        "image_macro": macro,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError("--shard-rank must be in [0, num-shards)")
    required = (args.config, args.checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    dataset_dir, split_file = resolve_dataset(args.dataset_root)
    all_samples = read_split(split_file, dataset_dir)
    if args.limit > 0:
        all_samples = all_samples[: args.limit]
    samples = all_samples[args.shard_rank :: args.num_shards]
    if not samples:
        raise RuntimeError(
            f"Shard {args.shard_rank}/{args.num_shards} has no samples from {len(all_samples)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_dir / ".eval.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another evaluation is writing {args.output_dir}") from exc

    csv_path = args.output_dir / "per_sample_metrics.csv"
    progress_path = args.output_dir / "progress.json"
    summary_path = args.output_dir / "metrics_summary.json"
    config_path = args.output_dir / "run_config.json"
    rows = load_rows(csv_path) if args.resume else []
    completed = {row["rgb_relative_path"] for row in rows}
    if not args.resume or not csv_path.is_file():
        initialize_csv(csv_path)

    device = resolve_device(args.device, {"distributed": False, "local_rank": 0})
    train_args = parse_training_args(["--config", str(args.config)])
    train_args.device = args.device
    train_args.distributed = "none"
    train_args.batch_size = 1
    train_args.num_workers = 0
    train_args.checkpoint = args.checkpoint
    train_args.amp_dtype = "bfloat16" if args.amp_dtype == "bfloat16" else "none"
    normalize_args_for_eval(train_args)
    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    apply_checkpoint_eval_defaults(train_args, checkpoint_payload)
    model = build_eval_model(train_args, args.checkpoint, checkpoint_payload, device)
    model.eval()

    pitches = parse_pitch_degrees(train_args.pitch_degrees)
    fov_x_degrees, fov_y_degrees = resolve_fov_degrees(
        train_args.fov_degrees,
        getattr(train_args, "fov_x_degrees", None),
        getattr(train_args, "fov_y_degrees", None),
    )
    native_sampler = {
        "window_size": int(train_args.window_size),
        "num_yaw": int(train_args.num_yaw),
        "pitch_degrees": list(pitches),
        "fov_degrees": float(train_args.fov_degrees),
        "fov_x_degrees": fov_x_degrees,
        "fov_y_degrees": fov_y_degrees,
        "views_per_panorama": int(train_args.num_yaw) * len(pitches),
    }
    expected_geometry = {
        "num_yaw": 4,
        "pitch_degrees": [-15.0],
        "fov_degrees": 75.0,
        "fov_x_degrees": 75.0,
        "fov_y_degrees": 75.0,
        "views_per_panorama": 4,
    }
    observed_geometry = {
        key: native_sampler[key] for key in expected_geometry
    }
    if observed_geometry != expected_geometry:
        raise RuntimeError(
            "This entry point is geometry-locked but uses the checkpoint-native "
            "square window resolution. "
            f"Expected {expected_geometry}, got {native_sampler}"
        )
    if native_sampler["window_size"] <= 0:
        raise RuntimeError(
            f"Invalid checkpoint-native window_size: {native_sampler['window_size']}"
        )

    run_config = {
        "dataset": "PanoSUNCG",
        "method": "0716 full pipeline",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_format": checkpoint_payload.get("checkpoint_format"),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": str(args.config.resolve()),
        "dataset_root": str(dataset_dir),
        "split_file": str(split_file),
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "total_split_samples": len(all_samples),
        "shard_samples": len(samples),
        "num_shards": args.num_shards,
        "shard_rank": args.shard_rank,
        "input": "one complete ERP; checkpoint-native sampler produces four pinhole windows",
        "native_sampler": native_sampler,
        "evaluation_domain": "four native pinhole windows, directly; no ERP fusion",
        "gt_sampling": "radial ERP depth sampled to identical rays and converted to Z-depth",
        "metric_protocol": DEPTH_METRIC_PROTOCOL,
        "alignment": "repository-default weighted scale-only IRLS, 10 iterations",
        "pixel_weighting": "pinhole solid angle divided by same-pano window coverage count",
        "latitude_mask": None,
        "depth_range": [float(args.min_depth), float(args.max_depth)],
        "input_height": args.input_height,
        "input_width": args.input_width,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
    }
    atomic_write_json(config_path, run_config)
    print(
        f"[native4] shard={args.shard_rank}/{args.num_shards} "
        f"samples={len(samples)} resumed={len(completed)}",
        flush=True,
    )
    print(f"[native4] sampler={json.dumps(native_sampler)}", flush=True)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True

    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    started = time.time()
    session_count = 0
    for sample_index, rgb_rel, depth_rel in samples:
        if rgb_rel in completed:
            continue
        sample_started = time.perf_counter()
        try:
            rgb = read_rgb(dataset_dir / rgb_rel)
            target_np = read_depth(dataset_dir / depth_rel)
            valid_np = (
                np.isfinite(target_np)
                & (target_np >= float(args.min_depth))
                & (target_np <= float(args.max_depth))
            )
            rgb_np = rgb.permute(1, 2, 0).numpy()
            if float(rgb_np.min()) == 0.0:
                valid_np &= rgb_np.mean(axis=-1) != 0.0

            interpolation = (
                cv2.INTER_AREA
                if rgb_np.shape[0] >= args.input_height
                else cv2.INTER_LINEAR
            )
            resized_np = cv2.resize(
                rgb_np,
                (args.input_width, args.input_height),
                interpolation=interpolation,
            )
            resized = (
                torch.from_numpy(np.ascontiguousarray(resized_np))
                .permute(2, 0, 1)[None, None]
                .to(device, non_blocking=True)
            )
            target_erp = torch.from_numpy(np.ascontiguousarray(target_np))[None, None, None].to(
                device=device, dtype=torch.float32
            )
            source_valid = torch.from_numpy(np.ascontiguousarray(valid_np))[None, None, None].to(
                device=device, dtype=torch.bool
            )
            target_depth, target_valid, camera_meta = sample_depth_targets(
                unwrap_model(model),
                target_erp,
                source_depth_semantics="range",
                max_range_depth=args.max_depth,
                return_camera_meta=True,
                source_valid_mask=source_valid,
            )
            if target_depth.shape[1] != 4:
                raise RuntimeError(f"Expected four target windows, got {target_depth.shape[1]}")
            spherical_weights = build_covered_sphere_weights(
                camera_meta,
                height=target_depth.shape[-3],
                width=target_depth.shape[-2],
                num_panos=1,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predictions = model(
                    pano_images=resized,
                    return_sampler_output=True,
                    return_window_pose=False,
                )
                pred_scale = predictions.get(
                    "_pred_depth_scale", predictions["depth"].new_tensor(1.0)
                )
                pred_depth = predictions["depth"].float() * pred_scale.float()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
            if pred_depth.shape != target_depth.shape:
                raise RuntimeError(
                    f"Prediction/target mismatch: {tuple(pred_depth.shape)} vs "
                    f"{tuple(target_depth.shape)}"
                )

            metrics = compute_depth_metrics(
                pred_depth,
                target_depth,
                target_valid,
                spherical_weights=spherical_weights,
            )
            row = {
                "sample_index": sample_index,
                "rgb_relative_path": rgb_rel,
                "depth_relative_path": depth_rel,
                "views": 4,
                "inference_seconds": inference_seconds,
                "sample_seconds": time.perf_counter() - sample_started,
                **metrics,
            }
            append_csv(csv_path, row)
            rows.append({key: str(value) for key, value in row.items()})
            completed.add(rgb_rel)
            session_count += 1
            if (
                session_count == 1
                or session_count % max(args.progress_every, 1) == 0
                or len(completed) == len(samples)
            ):
                current = summarize_rows(
                    rows,
                    len(samples),
                    total_split_samples=len(all_samples),
                )
                elapsed = time.time() - started
                rate = session_count / max(elapsed, 1e-9)
                current.update(
                    {
                        "last_sample": rgb_rel,
                        "num_shards": args.num_shards,
                        "shard_rank": args.shard_rank,
                        "eta_seconds": (
                            (len(samples) - len(completed)) / rate if rate > 0 else None
                        ),
                    }
                )
                atomic_write_json(progress_path, current)
                print(
                    f"[native4] shard={args.shard_rank} "
                    f"{len(completed)}/{len(samples)} "
                    f"IRLS_AbsRel={current['pixel_micro']['depth_irls_abs_rel']:.6f}",
                    flush=True,
                )
        except Exception as exc:
            error = {
                "sample_index": sample_index,
                "rgb_relative_path": rgb_rel,
                "depth_relative_path": depth_rel,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            with (args.output_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")
            if not args.continue_on_error:
                raise
            print(f"[native4][error] {rgb_rel}: {type(exc).__name__}: {exc}", flush=True)

    final = summarize_rows(rows, len(samples), total_split_samples=len(all_samples))
    final["run_config"] = run_config
    final["completed_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(summary_path, final)
    atomic_write_json(progress_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
