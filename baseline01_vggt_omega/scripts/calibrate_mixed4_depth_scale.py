#!/usr/bin/env python3
"""Estimate one global depth scale for the mixed official PanoVGGT datasets.

The readers convert every dataset's raw depth PNG to meters first. This script
then estimates a single pred_depth_scale from raw VGGT-Omega depth predictions:

    pred_depth_meters = raw_pred_depth * pred_depth_scale

The scale is computed as the median of per-sample median(gt_m / raw_pred),
sampled evenly across datasets so a large dataset cannot dominate calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
for path in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.datasets.pano_minimal import PanoMinimalPinholeDataset  # noqa: E402
from vggt_omega.models.vggt_omega import VGGTOmega  # noqa: E402


DATASETS = ("panocity", "matterport3d", "stanford2d3ds", "structured3d")
DEFAULT_OLD_SCALE = 5.491308212280273


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets")
    parser.add_argument("--checkpoint", default="../ckpt/vggt_omega_1b_512.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", default="all", help="Comma list or all.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-dataset", type=int, default=32)
    parser.add_argument("--max-pixels-per-sample", type=int, default=50000)
    parser.add_argument("--num-yaw", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--pitch-degrees", type=float, default=-15.0)
    parser.add_argument("--fov-degrees", type=float, default=75.0)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    parser.add_argument("--old-scale", type=float, default=DEFAULT_OLD_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-scale", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset_names = parse_dataset_names(args.datasets)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[calibrate] root={root}")
    print(f"[calibrate] datasets={','.join(dataset_names)}")
    print(f"[calibrate] split={args.split} samples_per_dataset={args.samples_per_dataset}")

    model = VGGTOmega(
        patch_size=args.patch_size,
        embed_dim=1024,
        enable_camera=False,
        enable_depth=True,
        enable_alignment=False,
        checkpoint_path=args.checkpoint,
        checkpoint_strict=False,
        activation_checkpointing=False,
    ).to(args.device)
    model.eval()

    dataset_summaries: dict[str, dict] = {}
    global_ratios: list[np.ndarray] = []
    global_sample_scales: list[float] = []

    for dataset_name in dataset_names:
        summary, ratios, sample_scales = calibrate_dataset(model, dataset_name, root, args)
        dataset_summaries[dataset_name] = summary
        global_ratios.extend(ratios)
        global_sample_scales.extend(sample_scales)
        torch.cuda.empty_cache()

    if not global_sample_scales:
        raise RuntimeError("No valid samples were collected; cannot estimate pred_depth_scale.")

    recommended_scale = float(np.median(np.asarray(global_sample_scales, dtype=np.float64)))
    global_ratio_array = concat_or_empty(global_ratios)
    result = {
        "recommended_pred_depth_scale": recommended_scale,
        "method": "median(per_sample_median(gt_meters/raw_pred_depth)); equal samples per dataset",
        "depth_unit": "meters_after_reader_conversion",
        "old_pred_depth_scale": float(args.old_scale),
        "root": str(root),
        "split": args.split,
        "num_yaw": int(args.num_yaw),
        "image_size": int(args.image_size),
        "pitch_degrees": float(args.pitch_degrees),
        "fov_degrees": float(args.fov_degrees),
        "samples_per_dataset": int(args.samples_per_dataset),
        "datasets": dataset_summaries,
        "global": {
            "valid_sample_count": int(len(global_sample_scales)),
            "sample_scale_median": recommended_scale,
            "sample_scale_p10": percentile(global_sample_scales, 10),
            "sample_scale_p90": percentile(global_sample_scales, 90),
            "ratio_pixel_count": int(global_ratio_array.size),
            "old_scale_metrics": log_metrics(global_ratio_array, args.old_scale),
            "recommended_scale_metrics": log_metrics(global_ratio_array, recommended_scale),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[calibrate] recommended_pred_depth_scale={recommended_scale:.9g}")
    print(f"[calibrate] wrote {output}")
    if args.print_scale:
        print(f"RECOMMENDED_SCALE={recommended_scale:.12g}")


def calibrate_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict, list[np.ndarray], list[float]]:
    common_conf = make_common_conf(args)
    dataset = PanoMinimalPinholeDataset(
        common_conf=common_conf,
        split=args.split,
        root=str(root),
        datasets=dataset_name,
        len_train=1000000,
        len_test=10000,
        output_depth_scale=1000.0,
        invalid_depth_value=65535.0,
        depth_max_m=args.depth_max_m,
        num_yaw=args.num_yaw,
        pitch_degrees=args.pitch_degrees,
        fov_degrees=args.fov_degrees,
        train_split_fraction=0.95,
        split_seed=args.seed,
        metadata_path=None,
        bad_sample_list=None,
        curriculum_bins=None,
        use_metadata_weights=False,
    )
    item_count = len(getattr(dataset, "items", []))
    if item_count <= 0:
        raise RuntimeError(f"No indexed samples for {dataset_name} under {root}")

    sample_count = min(int(args.samples_per_dataset), item_count)
    indices = random.Random(args.seed + stable_dataset_offset(dataset_name)).sample(range(item_count), sample_count)
    print(f"[calibrate] {dataset_name}: indexed={item_count} sampling={sample_count}")

    ratio_arrays: list[np.ndarray] = []
    sample_scales: list[float] = []
    valid_pixels = 0
    skipped = 0

    for ordinal, index in enumerate(indices, start=1):
        try:
            ratios = collect_sample_ratios(model, dataset, index, args)
        except Exception as exc:  # keep calibration robust against a few bad files
            skipped += 1
            print(f"[calibrate][WARN] {dataset_name} sample {index} failed: {exc}")
            continue
        if ratios.size == 0:
            skipped += 1
            continue
        sample_scale = float(np.median(ratios.astype(np.float64)))
        ratio_arrays.append(ratios.astype(np.float32, copy=False))
        sample_scales.append(sample_scale)
        valid_pixels += int(ratios.size)
        if ordinal == 1 or ordinal == sample_count or ordinal % 8 == 0:
            print(
                f"[calibrate] {dataset_name}: {ordinal}/{sample_count} "
                f"sample_scale={sample_scale:.6g} pixels={ratios.size}"
            )

    ratio_array = concat_or_empty(ratio_arrays)
    dataset_scale = float(np.median(np.asarray(sample_scales, dtype=np.float64))) if sample_scales else math.nan
    summary = {
        "indexed_samples": int(item_count),
        "requested_samples": int(sample_count),
        "valid_samples": int(len(sample_scales)),
        "skipped_samples": int(skipped),
        "ratio_pixel_count": int(valid_pixels),
        "sample_scale_median": dataset_scale,
        "sample_scale_p10": percentile(sample_scales, 10),
        "sample_scale_p90": percentile(sample_scales, 90),
        "old_scale_metrics": log_metrics(ratio_array, args.old_scale),
        "dataset_scale_metrics": log_metrics(ratio_array, dataset_scale) if sample_scales else {},
    }
    print(
        f"[calibrate] {dataset_name}: scale_median={dataset_scale:.6g} "
        f"valid_samples={len(sample_scales)} skipped={skipped}"
    )
    return summary, ratio_arrays, sample_scales


@torch.no_grad()
def collect_sample_ratios(
    model: torch.nn.Module,
    dataset: PanoMinimalPinholeDataset,
    index: int,
    args: argparse.Namespace,
) -> np.ndarray:
    sample = dataset.get_data(seq_index=index, img_per_seq=args.num_yaw, aspect_ratio=1.0)
    images_np = np.stack(sample["images"]).astype(np.float32)
    depths_np = np.stack(sample["depths"]).astype(np.float32)
    masks_np = np.stack(sample["point_masks"]).astype(bool)

    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).contiguous().div(255.0)
    images = images.unsqueeze(0).to(args.device, non_blocking=True)
    gt_depth = torch.from_numpy(depths_np).unsqueeze(0).to(args.device, non_blocking=True)
    gt_mask = torch.from_numpy(masks_np).unsqueeze(0).to(args.device, non_blocking=True)

    outputs = model(images=images)
    pred_depth = outputs["depth"].detach().float()
    if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
        pred_depth = pred_depth[..., 0]
    if pred_depth.ndim != gt_depth.ndim:
        raise RuntimeError(f"Unexpected prediction shape {tuple(pred_depth.shape)} for gt {tuple(gt_depth.shape)}")

    valid = (
        gt_mask.bool()
        & torch.isfinite(gt_depth)
        & torch.isfinite(pred_depth)
        & (gt_depth > 0)
        & (pred_depth > 1e-4)
    )
    if int(valid.sum().item()) < 100:
        return np.empty((0,), dtype=np.float32)

    ratios = (gt_depth[valid] / pred_depth[valid]).float()
    if ratios.numel() > args.max_pixels_per_sample:
        perm = torch.randperm(ratios.numel(), device=ratios.device)[: args.max_pixels_per_sample]
        ratios = ratios[perm]
    return ratios.cpu().numpy().astype(np.float32, copy=False)


def make_common_conf(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        fix_img_num=args.num_yaw,
        fix_aspect_ratio=1.0,
        load_track=False,
        track_num=1024,
        training=False,
        inside_random=False,
        img_size=args.image_size,
        patch_size=args.patch_size,
        rescale=False,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        get_nearby=False,
        load_depth=True,
        img_nums=[args.num_yaw, args.num_yaw],
        max_img_per_gpu=args.num_yaw,
        allow_duplicate_img=False,
        repeat_batch=False,
        augs=SimpleNamespace(
            cojitter=False,
            cojitter_ratio=0.0,
            scales=None,
            aspects=[1.0, 1.0],
            color_jitter=None,
            gray_scale=False,
            gau_blur=False,
        ),
    )


def parse_dataset_names(raw: str) -> list[str]:
    if raw in ("", "all", None):
        return list(DATASETS)
    aliases = {
        "pano_city": "panocity",
        "panocityofficial": "panocity",
        "mp3d": "matterport3d",
        "matterport": "matterport3d",
        "stanford": "stanford2d3ds",
        "2d3ds": "stanford2d3ds",
        "s3d": "structured3d",
    }
    names = []
    for value in str(raw).split(","):
        name = aliases.get(value.strip().lower(), value.strip().lower())
        if name:
            names.append(name)
    invalid = sorted(set(names) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}. Supported: {DATASETS}")
    return names


def concat_or_empty(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [array for array in arrays if array.size > 0]
    if not arrays:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def log_metrics(ratios: np.ndarray, scale: float) -> dict:
    if ratios.size == 0 or not math.isfinite(float(scale)) or scale <= 0:
        return {}
    ratios = ratios.astype(np.float64, copy=False)
    err = np.abs(np.log(float(scale) / np.clip(ratios, 1e-12, None)))
    clipped = np.minimum(err, 0.5)
    return {
        "log_abs_mean": float(np.mean(err)),
        "log_abs_median": float(np.median(err)),
        "clipped_log_l1_mean": float(np.mean(clipped)),
        "clipped_saturation_fraction": float(np.mean(err >= 0.5)),
        "approx_abs_rel_median": float(np.median(np.abs(np.exp(err) - 1.0))),
        "approx_abs_rel_mean": float(np.mean(np.abs(np.exp(err) - 1.0))),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stable_dataset_offset(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name))


if __name__ == "__main__":
    main()
