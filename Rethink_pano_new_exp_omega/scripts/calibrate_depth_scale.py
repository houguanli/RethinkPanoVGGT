#!/usr/bin/env python3
"""Estimate a fixed initial metric scale for VGGT-Omega panorama Z-depth."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import PanoVKittiOmegaDataset  # noqa: E402
from training.train_pano_omega import build_model, load_checkpoint, move_batch_to_device, sample_depth_targets, set_seed  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate VGGT predicted window Z-depth to metric GT Z-depth.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT.parent / "ckpt" / "vggt_omega_1b_512.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-mode", choices=["current", "pure"], default="current")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = PanoVKittiOmegaDataset(root=args.dataset_root, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_calibration_model(args.model_mode).to(device).eval()
    load_checkpoint(model, args.checkpoint, strict=(args.model_mode == "pure"))

    per_sample = []
    pooled_log_ratios = []
    with torch.no_grad():
        for batch in loader:
            batch_device = move_batch_to_device(batch, device)
            pred = model(pano_images=batch_device["pano_image"])["depth"]
            target, valid = sample_depth_targets(model, batch_device["pano_depth"])
            pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
            valid = valid & torch.isfinite(pred) & (pred > 1e-6)
            log_ratio = (torch.log(target.clamp_min(1e-6)) - torch.log(pred.clamp_min(1e-6)))[valid]
            if log_ratio.numel() == 0:
                continue
            log_ratio_np = log_ratio.cpu().numpy()
            pooled_log_ratios.append(log_ratio_np)
            sample_scale = float(math.exp(float(np.median(log_ratio_np))))
            per_sample.append(
                {
                    "scene_name": batch["scene_name"][0],
                    "valid_pixels": int(log_ratio_np.size),
                    "scale": sample_scale,
                }
            )

    if not pooled_log_ratios:
        raise RuntimeError("No valid depth ratios available for scale calibration.")
    pooled = np.concatenate(pooled_log_ratios)
    sample_scales = np.asarray([item["scale"] for item in per_sample], dtype=np.float64)
    global_scale = float(math.exp(float(np.median(pooled))))
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset.root),
        "model_mode": args.model_mode,
        "seed": args.seed,
        "depth_definition": "GT ERP radial range converted to virtual-pinhole Z-depth before calibration",
        "depth_sampling": "masked bilinear range sampling normalized by bilinear valid weight",
        "valid_mask": "nearest-sampled original valid depth mask",
        "samples": len(per_sample),
        "global_scale": global_scale,
        "per_sample_scale_median": float(np.median(sample_scales)),
        "per_sample_scale_p10": float(np.percentile(sample_scales, 10)),
        "per_sample_scale_p90": float(np.percentile(sample_scales, 90)),
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))
    print(f"[INFO] wrote calibration = {args.output}")


def build_calibration_model(mode: str) -> VGGTOmega_LUNA:
    if mode == "current":
        model_args = SimpleNamespace(
            patch_size=16,
            window_size=512,
            num_yaw=8,
            pitch_degrees="0",
            fov_degrees=75.0,
            enable_camera_head=True,
            smoke=False,
        )
        return build_model(model_args)
    return VGGTOmega_LUNA(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        enable_pano_global_token=False,
        enable_luna=False,
        sampler={
            "window_size": 512,
            "patch_size": 16,
            "fov_degrees": 75.0,
            "num_yaw": 8,
            "pitch_degrees": (0.0,),
        },
        checkpoint_path=None,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


if __name__ == "__main__":
    main()
