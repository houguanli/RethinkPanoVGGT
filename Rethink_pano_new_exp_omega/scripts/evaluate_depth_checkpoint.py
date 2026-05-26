#!/usr/bin/env python3
"""Evaluate corrected metric Z-depth loss for a pano checkpoint."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pred-depth-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-depth-semantics", choices=["range", "cubemap_z", "double_cubemap_z"], default="range")
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    scale = args.pred_depth_scale if args.pred_depth_scale is not None else float(ckpt_args.get("pred_depth_scale", 1.0))
    model = build_model(
        SimpleNamespace(
            patch_size=int(ckpt_args.get("patch_size", 16)),
            window_size=int(ckpt_args.get("window_size", 512)),
            num_yaw=int(ckpt_args.get("num_yaw", 8)),
            pitch_degrees=ckpt_args.get("pitch_degrees", "0"),
            fov_degrees=float(ckpt_args.get("fov_degrees", 75.0)),
            enable_camera_head=True,
            smoke=False,
        )
    ).to(device).eval()
    load_checkpoint(model, args.checkpoint, strict=False)
    dataset = PanoVKittiOmegaDataset(root=args.dataset_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    losses = []
    sample_rows = []
    with torch.no_grad():
        for batch in loader:
            moved = move_batch_to_device(batch, device)
            pred = model(pano_images=moved["pano_image"])["depth"].float() * scale
            target, valid = sample_depth_targets(
                model,
                moved["pano_depth"],
                source_depth_semantics=args.gt_depth_semantics,
                max_range_depth=args.depth_max_m,
            )
            valid = valid & torch.isfinite(pred) & torch.isfinite(target)
            values = (torch.log(pred.clamp_min(1e-6)) - torch.log(target.clamp_min(1e-6))).abs()[valid]
            if values.numel() == 0:
                continue
            loss = float(values.mean().cpu())
            losses.append(loss)
            sample_rows.append({"scene_name": batch["scene_name"][0], "log_l1_z_depth": loss})

    result = {
        "checkpoint": str(args.checkpoint),
        "pred_depth_scale": scale,
        "gt_source_depth_semantics": args.gt_depth_semantics,
        "depth_max_m": args.depth_max_m,
        "samples": len(losses),
        "mean_log_l1_z_depth": float(np.mean(losses)),
        "median_log_l1_z_depth": float(np.median(losses)),
        "p90_log_l1_z_depth": float(np.percentile(losses, 90)),
        "per_sample": sample_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


if __name__ == "__main__":
    main()
