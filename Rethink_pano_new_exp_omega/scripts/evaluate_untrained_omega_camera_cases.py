#!/usr/bin/env python3
"""Evaluate base Omega same-scene pano camera estimates on selected datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.pano_minimal import PanoMinimalDataset  # noqa: E402
from training.train_pano_omega import (  # noqa: E402
    build_relative_pano_pose_targets,
    estimate_sample_depth_alignment_scale,
    load_checkpoint,
    sample_depth_targets,
)
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="matterport3d,structured3d")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cases-per-dataset", type=int, default=1)
    parser.add_argument("--panos-per-case", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-yaw", type=int, default=4)
    parser.add_argument("--pitch-degrees", default="-15")
    parser.add_argument("--fov-degrees", type=float, default=75.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-range-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale-alignment", default="sample_lstsq")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_base_omega(args).to(device).eval()
    load_checkpoint(model, args.checkpoint, strict=False)

    rows: list[dict[str, Any]] = []
    for dataset_name in [name.strip() for name in args.datasets.split(",") if name.strip()]:
        dataset = PanoMinimalDataset(
            root=args.dataset_root,
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=args.panos_per_case,
            pano_max_count=args.panos_per_case,
            split=args.split,
            datasets=dataset_name,
            grouping="nearest",
            bad_sample_list=PROJECT_ROOT / "configs" / "structured3d_bad_scenes.txt",
        )
        selected = select_case_indices(dataset, args.panos_per_case, args.cases_per_dataset)
        for case_id, index in enumerate(selected):
            row = evaluate_case(args, model, dataset, dataset_name, index, case_id, device)
            rows.append(row)
            print(
                f"[case] {dataset_name}#{case_id} panos={row['pano_count']} "
                f"t_l2_mean_m={row['translation_l2_mean_m']:.4f} "
                f"r_deg_mean={row['rotation_deg_mean']}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize_rows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_dir / "per_case_metrics.csv", rows)
    print(f"[done] wrote {args.output_dir / 'summary.json'}")


def build_base_omega(args: argparse.Namespace) -> VGGTOmega_LUNA:
    pitch = [float(value.strip()) for value in str(args.pitch_degrees).split(",") if value.strip()]
    sampler = {
        "window_size": int(args.window_size),
        "patch_size": int(args.patch_size),
        "fov_degrees": float(args.fov_degrees),
        "num_yaw": int(args.num_yaw),
        "pitch_degrees": pitch,
    }
    return VGGTOmega_LUNA(
        patch_size=int(args.patch_size),
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        enable_pano_global_token=False,
        enable_pano_geometry_residual=False,
        enable_luna=False,
        sampler=sampler,
        dense_head_frames_chunk_size=1,
        dense_head_return_confidence=False,
        enable_pano_camera_head=True,
        checkpoint_path=None,
        checkpoint_strict=False,
    )


def select_case_indices(dataset: PanoMinimalDataset, panos_per_case: int, limit: int) -> list[int]:
    indices: list[int] = []
    for index, group in enumerate(dataset.groups):
        if len(group) < panos_per_case:
            continue
        scene_keys = {dataset.items[item_index].get("scene_group_key") for item_index in group[:panos_per_case]}
        if len(scene_keys) != 1:
            continue
        indices.append(index)
        if len(indices) >= limit:
            break
    if not indices:
        raise RuntimeError(f"No same-scene groups with at least {panos_per_case} panos.")
    return indices


def evaluate_case(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataset: PanoMinimalDataset,
    dataset_name: str,
    index: int,
    case_id: int,
    device: torch.device,
) -> dict[str, Any]:
    sample = dataset[index]
    batch = move_sample_to_device(sample, device)
    pano_images = batch["pano_image"].unsqueeze(0)
    pano_depth = batch["pano_depth"].unsqueeze(0)

    target_depth, target_valid = sample_depth_targets(
        model,
        pano_depth,
        source_depth_semantics="range",
        max_range_depth=float(args.max_range_depth),
    )
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float32
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        predictions = model(
            pano_images=pano_images,
            return_sampler_output=True,
            return_window_pose=True,
        )

    pred_center = predictions["pano_camera_center_init"].float()
    pred_quat = F.normalize(predictions["pano_rotation_quat_w2c_init"].float(), dim=-1, eps=1e-6)
    target_center, target_quat, position_valid, rotation_valid = build_relative_pano_pose_targets(
        batch={key: value.unsqueeze(0) if torch.is_tensor(value) else value for key, value in batch.items()},
        position_mode="relative_anchor",
        device=device,
        dtype=pred_center.dtype,
    )

    translation = translation_metrics(pred_center, target_center, position_valid)
    rotation = rotation_metrics(pred_quat, target_quat, rotation_valid)

    raw_pred_depth = squeeze_depth(predictions["depth"].float())
    target_depth_s = squeeze_depth(target_depth.float())
    target_valid_s = squeeze_depth(target_valid.float()).bool()
    scale = estimate_sample_depth_alignment_scale(
        raw_pred_depth,
        target_depth_s,
        target_valid_s,
        mode=args.depth_scale_alignment,
        min_scale=0.05,
        max_scale=50.0,
        eps=1e-6,
    )
    pred_depth = raw_pred_depth * scale.reshape(-1, *([1] * (raw_pred_depth.ndim - 1)))
    depth = depth_metrics(pred_depth, target_depth_s, target_valid_s)

    scene_name = str(sample["scene_name"])
    safe_scene = safe_name(scene_name)
    case_dir = args.output_dir / f"{dataset_name}_case{case_id:02d}_{safe_scene}"
    case_dir.mkdir(parents=True, exist_ok=True)
    make_case_visualization(case_dir / "case_panel.png", sample, pred_depth, target_depth_s, target_valid_s)
    details = {
        "dataset": dataset_name,
        "case_id": case_id,
        "dataset_index": int(index),
        "scene_name": scene_name,
        "sequence_name": sample["sequence_name"],
        "rgb_path": sample["rgb_path"],
        "depth_path": sample["depth_path"],
        "pano_position_m": tensor_to_list(batch["pano_position_m"]),
        "pano_position_valid": tensor_to_list(batch["pano_position_valid"]),
        "pano_rotation_valid": tensor_to_list(batch["pano_rotation_valid"]),
        "pred_center_relative": tensor_to_list(pred_center[0].detach().cpu()),
        "target_center_relative": tensor_to_list(target_center[0].detach().cpu()),
        "pred_quat_w2c_relative_xyzw": tensor_to_list(pred_quat[0].detach().cpu()),
        "target_quat_w2c_relative_xyzw": tensor_to_list(target_quat[0].detach().cpu()),
        "translation": translation,
        "rotation": rotation,
        "depth": depth,
        "visualization": str((case_dir / "case_panel.png").resolve()),
    }
    (case_dir / "case_metrics.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset": dataset_name,
        "case_id": case_id,
        "scene_name": scene_name,
        "pano_count": int(pano_images.shape[1]),
        "translation_l2_mean_m": translation["l2_mean_m"],
        "translation_l2_median_m": translation["l2_median_m"],
        "translation_norm_l2_mean": translation["norm_l2_mean"],
        "rotation_deg_mean": rotation["deg_mean"],
        "rotation_deg_median": rotation["deg_median"],
        "depth_abs_rel": depth["abs_rel"],
        "depth_delta_1p25": depth["delta_1p25"],
        "depth_scale": float(scale.detach().cpu().reshape(-1)[0]),
        "visualization": str((case_dir / "case_panel.png").resolve()),
        "metrics_json": str((case_dir / "case_metrics.json").resolve()),
    }


def move_sample_to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()}


def translation_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    valid = valid.to(device=pred.device).bool().clone()
    valid[:, 0] = False
    delta = pred - target
    l2 = torch.linalg.vector_norm(delta, dim=-1)
    norms = torch.linalg.vector_norm(target.float(), dim=-1)
    valid_f = valid.to(norms.dtype)
    scale = ((norms.square() * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)).sqrt().clamp_min(1.0)
    norm_l2 = l2 / scale[:, None]
    values = l2[valid].detach().cpu().numpy()
    norm_values = norm_l2[valid].detach().cpu().numpy()
    return {
        "valid_count": int(values.size),
        "l2_mean_m": finite_mean(values),
        "l2_median_m": finite_median(values),
        "norm_l2_mean": finite_mean(norm_values),
        "per_pano_l2_m": tensor_to_list(l2[0].detach().cpu()),
        "per_pano_norm_l2": tensor_to_list(norm_l2[0].detach().cpu()),
    }


def rotation_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    valid = valid.to(device=pred.device).bool().clone()
    valid = valid & valid[:, :1]
    valid[:, 0] = False
    cosine = (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(cosine.clamp_max(1.0 - 1e-7)) * (180.0 / math.pi)
    values = angle[valid].detach().cpu().numpy()
    return {
        "valid_count": int(values.size),
        "deg_mean": finite_mean(values) if values.size else None,
        "deg_median": finite_median(values) if values.size else None,
        "per_pano_deg": tensor_to_list(angle[0].detach().cpu()),
    }


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    mask = valid & torch.isfinite(pred) & torch.isfinite(target) & (target > 0)
    if not bool(mask.any()):
        return {"valid_ratio": 0.0, "abs_rel": float("nan"), "delta_1p25": float("nan")}
    p = pred[mask].detach().float()
    t = target[mask].detach().float()
    ratio = torch.maximum(p / t.clamp_min(1e-6), t / p.clamp_min(1e-6))
    return {
        "valid_ratio": float(mask.float().mean().detach().cpu()),
        "abs_rel": float(((p - t).abs() / t.clamp_min(1e-6)).mean().detach().cpu()),
        "delta_1p25": float((ratio < 1.25).float().mean().detach().cpu()),
    }


def squeeze_depth(value: torch.Tensor) -> torch.Tensor:
    while value.ndim >= 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim >= 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim >= 4 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def make_case_visualization(
    path: Path,
    sample: dict[str, Any],
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_valid: torch.Tensor,
) -> None:
    rgb_tiles = [rgb_tile(image) for image in sample["pano_image"]]
    gt_pano_tiles = [depth_tile(depth[0].detach().cpu().numpy()) for depth in sample["pano_depth"]]
    pred = pred_depth.detach().cpu().numpy()
    target = target_depth.detach().cpu().numpy()
    valid = target_valid.detach().cpu().numpy().astype(bool)
    pred_tiles = [depth_tile(pred[idx]) for idx in range(min(pred.shape[0], 8))]
    target_tiles = [depth_tile(target[idx]) for idx in range(min(target.shape[0], 8))]
    error_tiles = [error_tile(pred[idx], target[idx], valid[idx]) for idx in range(min(pred.shape[0], 8))]
    rows = [
        add_title(tile_row(rgb_tiles), "Input ERP RGB panos"),
        add_title(tile_row(gt_pano_tiles), "GT ERP depth panos"),
        add_title(tile_row(target_tiles), "Sampled window GT depth"),
        add_title(tile_row(pred_tiles), "Base Omega predicted window depth (sample-scale aligned)"),
        add_title(tile_row(error_tiles), "Window abs-relative error"),
    ]
    width = max(row.shape[1] for row in rows)
    padded = [pad_width(row, width) for row in rows]
    cv2.imwrite(str(path), np.concatenate(padded, axis=0))


def rgb_tile(tensor: torch.Tensor, size: tuple[int, int] = (160, 80)) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return cv2.resize(array, size, interpolation=cv2.INTER_AREA)


def depth_tile(depth: np.ndarray, size: tuple[int, int] = (160, 80)) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        maximum = float(np.nanpercentile(depth[valid], 98))
        output[valid] = np.clip(np.log1p(depth[valid]) / np.log1p(max(maximum, 1e-3)) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(output, cv2.COLORMAP_TURBO)
    colored[~valid] = (18, 18, 18)
    return cv2.resize(colored, size, interpolation=cv2.INTER_AREA)


def error_tile(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, size: tuple[int, int] = (160, 80)) -> np.ndarray:
    mask = valid & np.isfinite(pred) & np.isfinite(target) & (target > 0)
    error = np.zeros(pred.shape, dtype=np.uint8)
    if mask.any():
        rel = np.abs(pred[mask] - target[mask]) / np.maximum(target[mask], 1e-6)
        error[mask] = np.clip(rel / 1.0 * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(error, cv2.COLORMAP_INFERNO)
    colored[~mask] = (18, 18, 18)
    return cv2.resize(colored, size, interpolation=cv2.INTER_AREA)


def tile_row(tiles: list[np.ndarray]) -> np.ndarray:
    if not tiles:
        return np.zeros((80, 160, 3), dtype=np.uint8)
    return np.concatenate(tiles, axis=1)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((28, image.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def pad_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    pad = np.full((image.shape[0], width - image.shape[1], 3), 16, dtype=np.uint8)
    return np.concatenate([image, pad], axis=1)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "cases": len(selected),
            "translation_l2_mean_m": finite_mean([row["translation_l2_mean_m"] for row in selected]),
            "translation_l2_median_m": finite_median([row["translation_l2_median_m"] for row in selected]),
            "rotation_deg_mean": finite_mean([row["rotation_deg_mean"] for row in selected if row["rotation_deg_mean"] is not None]),
            "depth_abs_rel": finite_mean([row["depth_abs_rel"] for row in selected]),
            "depth_delta_1p25": finite_mean([row["depth_delta_1p25"] for row in selected]),
        }
    return {"datasets": by_dataset, "cases": rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "case_id",
        "scene_name",
        "pano_count",
        "translation_l2_mean_m",
        "translation_l2_median_m",
        "translation_norm_l2_mean",
        "rotation_deg_mean",
        "rotation_deg_median",
        "depth_abs_rel",
        "depth_delta_1p25",
        "depth_scale",
        "visualization",
        "metrics_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:90]


def tensor_to_list(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def finite_mean(values: Any) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_median(values: Any) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


if __name__ == "__main__":
    main()
