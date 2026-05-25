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

from scripts.reconstruct_pano_omega import save_depth_sheet, write_known_window_point_cloud  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
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
    export_reconstruction(root / "official_windows_reconstruction", official, windows, camera_meta, args)
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


def export_reconstruction(path: Path, pred: Dict[str, torch.Tensor], windows: np.ndarray, camera_meta: Dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    path.mkdir(parents=True, exist_ok=True)
    depth = pred["depth"][0, ..., 0].float().numpy()
    save_depth_sheet(depth, path / "pred_depth_windows.jpg", max_depth=args.depth_max_m)
    write_known_window_point_cloud(
        path / "known_window_camera_points.ply",
        depth_z=depth,
        windows=windows,
        camera_meta=camera_meta,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
