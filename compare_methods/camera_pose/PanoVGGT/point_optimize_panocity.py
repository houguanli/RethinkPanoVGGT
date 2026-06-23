from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

COMPARE_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex  # noqa: E402
from evaluate_panocity import _build_model, _default_checkpoint, _load_checkpoint, _load_rgb  # noqa: E402


def _read_depth(path: Path, height: int, width: int, depth_scale: float) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32) / float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    return torch.from_numpy(depth)


def _records_for_sample(index: PanoCityPairedIndex, start: int, views: int):
    return [index[(start + offset) % len(index)] for offset in range(views)]


def _stack_inputs(records: Iterable, height: int, width: int, depth_scale: float):
    images = []
    depths = []
    rgb_paths = []
    depth_paths = []
    for record in records:
        images.append(_load_rgb(record.rgb_path, height, width))
        depths.append(_read_depth(record.depth_path, height, width, depth_scale))
        rgb_paths.append(str(record.rgb_path))
        depth_paths.append(str(record.depth_path))
    return (
        torch.stack(images, dim=0).unsqueeze(0),
        torch.stack(depths, dim=0).unsqueeze(0),
        rgb_paths,
        depth_paths,
    )


def _metric_dict(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = (
        mask.bool()
        & torch.isfinite(pred)
        & torch.isfinite(gt)
        & (pred > 1e-6)
        & (gt > 1e-4)
    )
    if not valid.any():
        return {
            "valid_pixels": 0,
            "abs_rel": None,
            "rmse": None,
            "median_ratio": None,
            "lstsq_scale": None,
            "scaled_abs_rel": None,
            "scaled_rmse": None,
        }

    pred_valid = pred[valid].float()
    gt_valid = gt[valid].float()
    denom = gt_valid.clamp_min(1e-3)
    scale = (
        (pred_valid * gt_valid).sum() / pred_valid.square().sum().clamp_min(1e-8)
    ).abs().clamp(1e-6, 1e6)
    scaled_pred = scale * pred_valid
    ratio = (gt_valid / pred_valid.clamp_min(1e-6)).median()

    return {
        "valid_pixels": int(valid.sum().item()),
        "abs_rel": float(((pred_valid - gt_valid).abs() / denom).mean().item()),
        "rmse": float(torch.sqrt((pred_valid - gt_valid).square().mean()).item()),
        "median_ratio": float(ratio.item()),
        "lstsq_scale": float(scale.item()),
        "scaled_abs_rel": float(((scaled_pred - gt_valid).abs() / denom).mean().item()),
        "scaled_rmse": float(torch.sqrt((scaled_pred - gt_valid).square().mean()).item()),
    }


def _mean_metric(samples: list[dict], key: str):
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_model(config_name: str, checkpoint: Path, device: torch.device):
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    model = _build_model(cfg).to(device).eval()
    _load_checkpoint(model, checkpoint, device)
    return model, cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PanoVGGT-only raw depth/point scale validation on PanoCity paired subsets."
    )
    parser.add_argument("--config", default="panocity_4rtx5000")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--depth-scale", type=float, default=100.0)
    parser.add_argument("--depth-max", type=float, default=100.0)
    parser.add_argument("--output-dir", default="outputs/panocity_point_opt")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    args = parser.parse_args()

    if args.views < 1:
        raise SystemExit("--views must be >= 1")

    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint()
    device = torch.device(args.device)
    model, cfg = _load_model(args.config, checkpoint, device)

    height = int(cfg.img_size)
    width = height * 2
    index = PanoCityPairedIndex(split=args.split, max_samples=max(args.max_samples + args.views, args.views))

    per_view_metrics = []
    per_sample = []
    global_pred_sum = 0.0
    global_cross_sum = 0.0
    global_valid = 0

    with torch.no_grad():
        for sample_idx in range(args.max_samples):
            records = _records_for_sample(index, sample_idx, args.views)
            images, gt_depth, rgb_paths, depth_paths = _stack_inputs(records, height, width, args.depth_scale)
            outputs = model(images.to(device))
            pred_depth = outputs["depth"].detach().cpu()
            if pred_depth.dim() == 5 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth[..., 0]

            mask = (gt_depth > 0.1) & (gt_depth <= args.depth_max) & torch.isfinite(gt_depth)
            sample_metrics = _metric_dict(pred_depth, gt_depth, mask)
            per_sample.append(
                {
                    "sample_index": sample_idx,
                    "rgb_paths": rgb_paths,
                    "depth_paths": depth_paths,
                    **sample_metrics,
                }
            )

            valid = (
                mask.bool()
                & torch.isfinite(pred_depth)
                & torch.isfinite(gt_depth)
                & (pred_depth > 1e-6)
                & (gt_depth > 1e-4)
            )
            if valid.any():
                pred_valid = pred_depth[valid].float()
                gt_valid = gt_depth[valid].float()
                global_pred_sum += float(pred_valid.square().sum().item())
                global_cross_sum += float((pred_valid * gt_valid).sum().item())
                global_valid += int(valid.sum().item())

            for view_idx in range(args.views):
                per_view_metrics.append(_metric_dict(pred_depth[:, view_idx], gt_depth[:, view_idx], mask[:, view_idx]))

    global_scale = abs(global_cross_sum / max(global_pred_sum, 1e-8)) if global_valid else None
    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "max_samples": args.max_samples,
        "views": args.views,
        "depth_scale": args.depth_scale,
        "depth_max": args.depth_max,
        "global_valid_pixels": global_valid,
        "suggested_panovggt_geometry_output_scale": global_scale,
        "mean_per_view": {
            "abs_rel": _mean_metric(per_view_metrics, "abs_rel"),
            "rmse": _mean_metric(per_view_metrics, "rmse"),
            "lstsq_scale": _mean_metric(per_view_metrics, "lstsq_scale"),
            "scaled_abs_rel": _mean_metric(per_view_metrics, "scaled_abs_rel"),
            "scaled_rmse": _mean_metric(per_view_metrics, "scaled_rmse"),
        },
        "samples": per_sample,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "point_depth_scale_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if global_scale is not None:
        (output_dir / "recommended_scale.env").write_text(
            f"export PANOVGGT_GEOMETRY_OUTPUT_SCALE={global_scale:.8g}\n",
            encoding="utf-8",
        )

    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
