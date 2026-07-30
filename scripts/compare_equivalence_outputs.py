#!/usr/bin/env python3
"""Compare official-window and pano-wrapper prediction tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reconstruct_pano_omega import (  # noqa: E402
    apply_range_depth_modifier,
    save_depth_sheet,
    write_known_window_point_cloud,
    write_official_point_cloud,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    parser.add_argument("--pred-depth-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.comparison_dir
    shared = torch.load(root / "shared_windows.pt", map_location="cpu", weights_only=False)
    official = torch.load(root / "official_windows.pt", map_location="cpu", weights_only=False)
    current = torch.load(root / "pano_current_setting.pt", map_location="cpu", weights_only=False)
    pure = torch.load(root / "pano_pure_sampler.pt", map_location="cpu", weights_only=False)

    summary = {
        "official_vs_pano_pure_sampler": compare_prediction_dicts(official, pure),
        "official_vs_pano_current_setting": compare_prediction_dicts(official, current),
        "pano_pure_sampler_vs_pano_current_setting": compare_prediction_dicts(pure, current),
    }
    (root / "equivalence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    windows = shared["windows"][0].numpy()
    camera_meta = {key: value[0] for key, value in shared["camera_meta"].items()}
    target_depth = shared["target_depth"][0, ..., 0].float().numpy()
    target_valid = shared["target_valid"][0, ..., 0].bool().numpy()
    official_dir = root / "official_windows_reconstruction"
    export_reconstruction(
        official_dir,
        official,
        windows,
        camera_meta,
        args,
        target_valid=target_valid,
        gt_depth_semantics=shared.get("gt_depth_semantics"),
        export_official_pose=True,
    )
    save_depth_sheet(target_depth, official_dir / "target_depth_windows.jpg", max_depth=args.depth_max_m)
    write_known_window_point_cloud(
        official_dir / "target_known_window_camera_points.ply",
        depth_z=target_depth,
        windows=windows,
        camera_meta=camera_meta,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
        extra_valid=target_valid,
    )
    export_reconstruction(root / "pano_pure_sampler_reconstruction", pure, windows, camera_meta, args)
    export_reconstruction(root / "pano_current_setting_reconstruction", current, windows, camera_meta, args)


def compare_prediction_dicts(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict:
    out = {}
    for key in sorted(set(a.keys()) & set(b.keys())):
        if key == "camera_and_register_tokens" and a[key].shape != b[key].shape:
            out[key] = {
                "shape_a": list(a[key].shape),
                "shape_b": list(b[key].shape),
                "note": "shape mismatch; current pano-global setting adds one special token",
            }
            continue
        out[key] = tensor_diff(a[key], b[key])
    return out


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float | list[int]]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "median_abs": float(diff.median().item()),
        "mean_rel": float((diff / denom).mean().item()),
    }


def export_reconstruction(
    path: Path,
    pred: Dict[str, torch.Tensor],
    windows: np.ndarray,
    camera_meta: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    target_valid: np.ndarray | None = None,
    gt_depth_semantics: str | None = None,
    export_official_pose: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    raw_depth = pred["depth"][0, ..., 0].float().numpy() * args.pred_depth_scale
    depth, pred_valid = apply_range_depth_modifier(raw_depth, camera_meta, args.depth_max_m)
    valid = pred_valid if target_valid is None else pred_valid & target_valid
    save_depth_sheet(raw_depth, path / "pred_depth_windows_unfiltered.jpg", max_depth=args.depth_max_m)
    save_depth_sheet(depth, path / "pred_depth_windows.jpg", max_depth=args.depth_max_m)
    write_known_window_point_cloud(
        path / "known_window_camera_points.ply",
        depth_z=depth,
        windows=windows,
        camera_meta=camera_meta,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
        extra_valid=target_valid,
    )
    if export_official_pose:
        write_official_point_cloud(
            path / "pred_official_camera_points_native.ply",
            pred_depth_z=depth,
            windows=windows,
            pose_enc=pred["pose_enc"][0].float(),
            max_depth=args.depth_max_m,
            max_points=args.max_points,
            display_y_up=False,
            rotate_y_180=False,
            extra_valid=target_valid,
        )
        write_official_point_cloud(
            path / "pred_official_camera_points.ply",
            pred_depth_z=depth,
            windows=windows,
            pose_enc=pred["pose_enc"][0].float(),
            max_depth=args.depth_max_m,
            max_points=args.max_points,
            display_y_up=True,
            rotate_y_180=True,
            extra_valid=target_valid,
        )
    stats = {
        "pred_depth_scale": args.pred_depth_scale,
        "depth_max_m": args.depth_max_m,
        "gt_source_depth_semantics": gt_depth_semantics,
        "gt_valid_mask_applied": target_valid is not None,
        "pred_valid_ratio_before_gt_mask": float(pred_valid.mean()),
        "output_valid_ratio": float(valid.mean()),
        "removed_by_gt_mask_ratio": float((pred_valid & ~valid).mean()),
        "official_display_transform": "flip Y then rotate 180 degrees around Y" if export_official_pose else None,
    }
    (path / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
