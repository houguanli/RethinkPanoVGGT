#!/usr/bin/env python3
"""Combine PanoSUNCG native-four depth and camera summaries into one JSON."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-camera", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    depth_path = output_dir / "depth_native4" / "metrics_summary.json"
    camera_path = output_dir / "camera_pose" / "camera_center_summary.json"
    depth = read_json(depth_path)
    camera = read_json(camera_path) if camera_path.is_file() else None
    if camera is None and not args.allow_missing_camera:
        raise FileNotFoundError(camera_path)

    run_config = depth.get("run_config", {})
    native_sampler = run_config.get("native_sampler", {})
    pixel_micro = depth.get("pixel_micro", {})
    camera_micro = camera.get("micro", {}) if camera else {}
    combined = {
        "status": "completed" if camera is not None else "depth_completed",
        "dataset": depth.get("dataset", "PanoSUNCG"),
        "method": depth.get("method", "0716 full pipeline"),
        "protocol": {
            "depth": depth.get("protocol"),
            "camera": camera.get("protocol") if camera else None,
        },
        "checkpoint": {
            "path": run_config.get("checkpoint"),
            "format": run_config.get("checkpoint_format"),
            "step": run_config.get("checkpoint_step"),
        },
        "input_resolution": {
            "erp_height": run_config.get("input_height"),
            "erp_width": run_config.get("input_width"),
            "checkpoint_native_window_size": native_sampler.get("window_size"),
            "views_per_panorama": native_sampler.get("views_per_panorama", 4),
        },
        "counts": {
            "depth_samples": depth.get("completed_samples"),
            "depth_valid_pixels": pixel_micro.get("depth_valid_pixels"),
            "camera_trajectories": camera.get("trajectory_count") if camera else None,
            "camera_frames": camera.get("frame_count") if camera else None,
            "camera_pairs": camera.get("pair_count") if camera else None,
        },
        "headline_metrics": {
            "depth_pixel_micro_irls_abs_rel": pixel_micro.get("depth_irls_abs_rel"),
            "depth_pixel_micro_irls_rmse": pixel_micro.get("depth_irls_rmse"),
            "depth_pixel_micro_irls_delta_1p25": pixel_micro.get(
                "depth_irls_delta_1p25"
            ),
            "camera_micro_ate_rmse": camera_micro.get("ate_rmse"),
            "camera_micro_direction_deg_mean": camera_micro.get(
                "direction_deg_mean"
            ),
            "camera_micro_relative_length_error_mean": camera_micro.get(
                "relative_length_error_mean"
            ),
        },
        "depth": depth,
        "camera": camera,
        "combined_at_local": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    destination = output_dir / "evaluation_summary.json"
    atomic_write_json(destination, combined)
    print(destination)


if __name__ == "__main__":
    main()
