#!/usr/bin/env python3
"""Run a single pano through VGGT-Omega LUNA and export reconstruction previews."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import PanoCityPairedOmegaDataset, PanoMinimalDataset, PanoVKittiOmegaDataset  # noqa: E402
from training.train_pano_omega import build_model, load_checkpoint, sample_depth_targets, set_seed  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays, rays_to_equirectangular  # noqa: E402


DEFAULT_PANOCITY_PRED_DEPTH_SCALE = 5.491308212280273


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pano reconstruction previews from a trained LUNA checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--dataset-format", choices=["vkitti", "panocity_paired", "pano_minimal"], default="vkitti")
    parser.add_argument("--pano-path", type=Path, default=None, help="Direct RGB panorama path. Bypasses dataset indexing.")
    parser.add_argument("--depth-path", type=Path, default=None, help="Optional direct depth path used for GT preview/scale fitting.")
    parser.add_argument("--direct-scene-name", type=str, default=None, help="Optional scene name for direct --pano-path export summary.")
    parser.add_argument(
        "--minimal-datasets",
        default="all",
        help=(
            "For --dataset-format pano_minimal: comma-separated subset, e.g. "
            "panocity,matterport3d,stanford2d3ds,structured3d."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="For --dataset-format pano_minimal: split index to read.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--sample-scene-name",
        type=str,
        default=None,
        help="Optional exact scene_name selector. If no exact match exists, a unique substring match is accepted.",
    )
    parser.add_argument(
        "--sample-rgb-path",
        type=str,
        default=None,
        help="Optional RGB path/stem selector. Useful for pulling a specific mixed4 pano case.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    parser.add_argument("--pred-depth-scale", type=float, default=None)
    parser.add_argument(
        "--fit-pred-depth-scale-from-target",
        action="store_true",
        help="Estimate pred depth scale from finite valid sampled GT window depth for this sample.",
    )
    parser.add_argument(
        "--pred-depth-scale-stat",
        choices=["median", "mean"],
        default="median",
        help="Statistic used with --fit-pred-depth-scale-from-target.",
    )
    parser.add_argument("--gt-depth-semantics", choices=["range", "cubemap_z", "double_cubemap_z"], default="range")
    parser.add_argument(
        "--mask-pred-by-target-valid",
        action="store_true",
        help="Only export predicted depth/points where the sampled GT depth target is valid.",
    )
    parser.add_argument("--pano-height", type=int, default=None)
    parser.add_argument("--pano-width", type=int, default=None)
    parser.add_argument("--output-depth-scale", type=float, default=100.0)
    parser.add_argument("--invalid-depth-value", type=float, default=None)
    parser.add_argument("--num-yaw", type=int, default=None, help="Override checkpoint window yaw count for eval/export.")
    parser.add_argument("--pitch-degrees", type=str, default=None, help="Override checkpoint pitch list for eval/export.")
    parser.add_argument("--fov-degrees", type=float, default=None, help="Override checkpoint window FOV for eval/export.")
    parser.add_argument("--fov-x-degrees", type=float, default=None, help="Override checkpoint horizontal window FoV.")
    parser.add_argument("--fov-y-degrees", type=float, default=None, help="Override checkpoint vertical window FoV.")
    parser.add_argument("--window-size", type=int, default=None, help="Override checkpoint window size for eval/export.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_args = model_args_from_checkpoint(ckpt_args)
    pred_depth_scale = resolve_pred_depth_scale(args, ckpt_args)
    if args.pano_height is not None and args.pano_width is not None:
        model_args.pano_height = args.pano_height
        model_args.pano_width = args.pano_width
    if args.num_yaw is not None:
        model_args.num_yaw = args.num_yaw
    if args.pitch_degrees is not None:
        model_args.pitch_degrees = args.pitch_degrees
    if args.fov_degrees is not None:
        model_args.fov_degrees = args.fov_degrees
        if args.fov_x_degrees is None:
            model_args.fov_x_degrees = None
        if args.fov_y_degrees is None:
            model_args.fov_y_degrees = None
    if args.fov_x_degrees is not None:
        model_args.fov_x_degrees = args.fov_x_degrees
    if args.fov_y_degrees is not None:
        model_args.fov_y_degrees = args.fov_y_degrees
    if args.window_size is not None:
        model_args.window_size = args.window_size

    pano_size = pano_size_from_args(model_args)
    if args.pano_path is not None:
        sample = read_direct_sample(args, pano_size)
        args.resolved_sample_index = None
    else:
        dataset = build_dataset(args, pano_size)
        sample_index = resolve_sample_index(dataset, args)
        args.resolved_sample_index = sample_index
        sample = dataset[sample_index]
    model = build_model(model_args).to(device)
    load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()

    output_dir = resolve_output_dir(args, sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    pano_image = sample["pano_image"][None].to(device)
    target_available = sample.get("pano_depth") is not None

    with torch.no_grad():
        predictions = model(pano_images=pano_image, return_sampler_output=True)
        if target_available:
            pano_depth = sample["pano_depth"][None].to(device)
            target_depth, target_valid = sample_depth_targets(
                model,
                pano_depth,
                source_depth_semantics=args.gt_depth_semantics,
                max_range_depth=args.depth_max_m,
            )

    model_pred_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    pred_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    if target_available:
        target_depth_np = target_depth[0, ..., 0].detach().float().cpu().numpy()
        target_valid_np = target_valid[0, ..., 0].detach().cpu().numpy()
    else:
        target_depth_np = np.zeros_like(model_pred_depth, dtype=np.float32)
        target_valid_np = np.zeros_like(model_pred_depth, dtype=bool)
    fitted_pred_depth_scale = None
    fitted_pred_depth_scale_valid_count = 0
    if args.fit_pred_depth_scale_from_target:
        if not target_available:
            raise ValueError("--fit-pred-depth-scale-from-target requires --depth-path or a dataset sample with depth.")
        pred_depth_scale, fitted_pred_depth_scale_valid_count = fit_pred_depth_scale_from_target(
            model_pred_depth,
            target_depth_np,
            target_valid_np,
            stat=args.pred_depth_scale_stat,
        )
        fitted_pred_depth_scale = pred_depth_scale
    raw_pred_depth = model_pred_depth * pred_depth_scale
    windows = predictions["pano_windows"][0].detach().float().cpu().numpy()
    camera_meta = {key: value[0].detach().float().cpu() for key, value in predictions["pano_camera_meta"].items()}
    pano_np = tensor_image_to_uint8(sample["pano_image"])
    pred_depth, pred_valid = apply_range_depth_modifier(raw_pred_depth, camera_meta, args.depth_max_m)
    pred_valid_after_range = pred_valid.copy()
    if args.mask_pred_by_target_valid:
        if not target_available:
            raise ValueError("--mask-pred-by-target-valid requires --depth-path or a dataset sample with depth.")
        pred_valid = pred_valid & target_valid_np.astype(bool)
        pred_depth = pred_depth.copy()
        pred_depth[~pred_valid] = np.inf

    Image.fromarray(pano_np).save(output_dir / "input_pano.jpg")
    save_contact_sheet(windows, output_dir / "sampled_windows.jpg")
    save_depth_sheet(raw_pred_depth, output_dir / "pred_z_depth_windows_unfiltered.jpg", max_depth=args.depth_max_m)
    save_depth_sheet(pred_depth, output_dir / "pred_z_depth_windows.jpg", max_depth=args.depth_max_m)
    if target_available:
        save_depth_sheet(target_depth_np, output_dir / "target_z_depth_windows.jpg", max_depth=args.depth_max_m)
    save_conf_sheet(pred_conf, output_dir / "pred_conf_windows.jpg")

    pred_erp, valid_erp = splat_windows_to_erp(
        pred_depth=pred_depth,
        camera_meta=camera_meta,
        pano_hw=pano_np.shape[:2],
    )
    save_depth_image(pred_erp, valid_erp, output_dir / "pred_z_depth_erp_splat.png", max_depth=args.depth_max_m)
    if target_available:
        target_range_erp = erp_range_depth_image(sample["pano_depth"][0].numpy(), args.gt_depth_semantics)
        target_range_valid = np.isfinite(target_range_erp) & (target_range_erp > 0) & (target_range_erp <= args.depth_max_m)
        save_depth_image(target_range_erp, target_range_valid, output_dir / "target_range_depth_erp.png", max_depth=args.depth_max_m)
    else:
        target_range_valid = None
    write_known_window_point_cloud(
        output_dir / "pred_points.ply",
        depth_z=pred_depth,
        windows=windows,
        camera_meta=camera_meta,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
    )
    if target_available:
        write_erp_point_cloud(
            output_dir / "gt_points.ply",
            depth=sample["pano_depth"][0].numpy(),
            rgb=pano_np,
            max_depth=args.depth_max_m,
            max_points=args.max_points,
            depth_semantics=args.gt_depth_semantics,
            extra_valid=target_range_valid,
        )
    write_summary(
        output_dir / "summary.json",
        args,
        sample,
        raw_pred_depth,
        pred_depth,
        pred_valid,
        target_depth_np,
        target_valid_np,
        pred_valid_after_range,
        pred_depth_scale,
        fitted_pred_depth_scale,
        fitted_pred_depth_scale_valid_count,
    )
    print(f"[INFO] exported reconstruction = {output_dir}")


def model_args_from_checkpoint(ckpt_args: Dict) -> SimpleNamespace:
    defaults = {
        "patch_size": 16,
        "window_size": 512,
        "num_yaw": 8,
        "pitch_degrees": "0",
        "fov_degrees": 75.0,
        "fov_x_degrees": None,
        "fov_y_degrees": None,
        "pano_height": 0,
        "pano_width": 0,
        "enable_camera_head": True,
        "enable_pano_global_token": False,
        "luna_patch_layers": 2,
        "luna_camera_layers": 2,
        "pred_depth_scale": 1.0,
        "smoke": False,
    }
    defaults.update({key: ckpt_args[key] for key in defaults.keys() & ckpt_args.keys()})
    return SimpleNamespace(**defaults)


def pano_size_from_args(args: SimpleNamespace) -> Tuple[int, int] | None:
    if args.pano_height and args.pano_width:
        return int(args.pano_height), int(args.pano_width)
    return None


def resolve_output_dir(args: argparse.Namespace, sample: Dict) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    scene_name = str(sample.get("scene_name") or Path(str(sample.get("rgb_path", "pano"))).stem)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in scene_name)[:120]
    return PROJECT_ROOT / "logs" / f"reconstruct_{safe_name}"


def read_direct_sample(args: argparse.Namespace, pano_size: Tuple[int, int] | None) -> Dict:
    if args.pano_path is None:
        raise ValueError("--pano-path is required for direct sample loading.")
    image = read_rgb_tensor(args.pano_path)
    depth = read_depth_tensor(args.depth_path, args.output_depth_scale, args.invalid_depth_value) if args.depth_path else None
    if pano_size is not None:
        height, width = pano_size
        image = F.interpolate(image[None], size=(height, width), mode="bilinear", align_corners=False)[0]
        if depth is not None:
            depth = F.interpolate(depth[None], size=(height, width), mode="nearest")[0]
    scene_name = args.direct_scene_name or args.pano_path.stem
    return {
        "pano_image": image,
        "pano_depth": depth,
        "sequence_name": "direct",
        "scene_name": scene_name,
        "rgb_path": str(args.pano_path),
        "depth_path": str(args.depth_path) if args.depth_path else None,
        "pano_position_m": torch.zeros(3, dtype=torch.float32),
    }


def read_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_depth_tensor(path: Path, output_depth_scale: float, invalid_depth_value: float | None) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32) / float(output_depth_scale)
    if invalid_depth_value is not None:
        depth_m[depth.astype(np.float32) >= float(invalid_depth_value)] = np.inf
    depth_m[depth_m <= 0] = np.inf
    return torch.from_numpy(depth_m)[None].contiguous()


def build_dataset(args: argparse.Namespace, pano_size: Tuple[int, int] | None):
    if args.dataset_format == "vkitti":
        return PanoVKittiOmegaDataset(
            root=args.dataset_root,
            pano_size=pano_size,
        )
    if args.dataset_format == "panocity_paired":
        return PanoCityPairedOmegaDataset(
            root=args.dataset_root,
            pano_size=pano_size,
            output_depth_scale=args.output_depth_scale,
            invalid_depth_value=args.invalid_depth_value,
        )
    if args.dataset_format == "pano_minimal":
        return PanoMinimalDataset(
            root=args.dataset_root,
            pano_size=pano_size,
            pano_sample_mode="single",
            pano_min_count=1,
            pano_max_count=1,
            split=args.dataset_split,
            datasets=args.minimal_datasets,
            output_depth_scale=args.output_depth_scale,
            invalid_depth_value=args.invalid_depth_value,
        )
    raise ValueError(f"Unknown dataset format: {args.dataset_format}")


def resolve_sample_index(dataset, args: argparse.Namespace) -> int:
    if args.sample_scene_name:
        return resolve_sample_index_from_items(dataset, "scene_name", args.sample_scene_name)
    if args.sample_rgb_path:
        return resolve_sample_index_from_items(dataset, "rgb_path", args.sample_rgb_path)
    if len(dataset) <= 0:
        raise IndexError("Dataset is empty.")
    index = int(args.sample_index)
    if index < 0:
        index += len(dataset)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} is outside dataset length {len(dataset)}.")
    return index


def resolve_sample_index_from_items(dataset, key: str, query: str) -> int:
    items = getattr(dataset, "items", None)
    groups = getattr(dataset, "groups", None)
    if items is None or groups is None:
        option = f"--sample-{key.replace('_', '-')}"
        raise ValueError(f"{option} is only supported for datasets exposing items/groups.")
    query_text = str(query)
    exact_matches = []
    substring_matches = []
    for group_index, group in enumerate(groups):
        if not group:
            continue
        item = items[int(group[0])]
        value = str(item.get(key, ""))
        if key.endswith("path"):
            path = Path(value)
            exact_candidates = {value, path.name, path.stem}
            substring_candidates = {value, path.name, path.stem}
        else:
            exact_candidates = {value}
            substring_candidates = {value}
        if query_text in exact_candidates:
            exact_matches.append(group_index)
        elif any(query_text in candidate for candidate in substring_candidates):
            substring_matches.append(group_index)
    matches = exact_matches or substring_matches
    if not matches:
        raise ValueError(f"No sample matched {key}={query!r}.")
    if len(matches) > 1 and not exact_matches:
        preview = []
        for group_index in matches[:10]:
            item = items[int(groups[group_index][0])]
            preview.append(str(item.get("scene_name", item.get(key, ""))))
        raise ValueError(f"{key}={query!r} matched {len(matches)} samples; use a more specific value. Examples: {preview}")
    return int(matches[0])


def resolve_pred_depth_scale(args: argparse.Namespace, ckpt_args: Dict) -> float:
    if args.pred_depth_scale is not None:
        return float(args.pred_depth_scale)
    checkpoint_scale = ckpt_args.get("pred_depth_scale", None)
    if checkpoint_scale is not None and float(checkpoint_scale) != 1.0:
        return float(checkpoint_scale)
    if args.dataset_format == "panocity_paired":
        return DEFAULT_PANOCITY_PRED_DEPTH_SCALE
    return float(checkpoint_scale if checkpoint_scale is not None else 1.0)


def fit_pred_depth_scale_from_target(
    pred_depth: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    stat: str,
) -> Tuple[float, int]:
    valid = (
        target_valid.astype(bool)
        & np.isfinite(target_depth)
        & np.isfinite(pred_depth)
        & (target_depth > 0)
        & (pred_depth > 0)
    )
    ratios = target_depth[valid].astype(np.float64) / pred_depth[valid].astype(np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        raise ValueError("Cannot fit pred depth scale: no finite positive pred/target depth pairs.")
    if stat == "median":
        return float(np.median(ratios)), int(ratios.size)
    if stat == "mean":
        return float(np.mean(ratios)), int(ratios.size)
    raise ValueError(f"Unknown pred-depth-scale-stat: {stat}")


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def save_contact_sheet(windows: np.ndarray, path: Path) -> None:
    images = [np.clip(window.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8) for window in windows]
    Image.fromarray(make_grid(images)).save(path)


def save_depth_sheet(depths: np.ndarray, path: Path, max_depth: float) -> None:
    images = [colorize_depth(depth, depth > 0, max_depth) for depth in depths]
    Image.fromarray(make_grid(images)).save(path)


def save_conf_sheet(conf: np.ndarray, path: Path) -> None:
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.percentile(conf[conf > 0], 98) if np.any(conf > 0) else 1.0
    images = []
    for item in conf:
        norm = np.clip(item / max(vmax, 1e-6), 0, 1)
        images.append(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1])
    Image.fromarray(make_grid(images)).save(path)


def make_grid(images: list[np.ndarray]) -> np.ndarray:
    count = len(images)
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        y = (idx // cols) * h
        x = (idx % cols) * w
        canvas[y : y + h, x : x + w] = image
    return canvas


def splat_windows_to_erp(pred_depth: np.ndarray, camera_meta: Dict[str, torch.Tensor], pano_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    pano_h, pano_w = pano_hw
    erp = np.zeros((pano_h, pano_w), dtype=np.float32)
    counts = np.zeros((pano_h, pano_w), dtype=np.float32)
    u, v = window_uv(camera_meta, pred_depth.shape[-2], pred_depth.shape[-1])
    x = np.clip(np.rint(u.numpy() * (pano_w - 1)).astype(np.int64), 0, pano_w - 1)
    y = np.clip(np.rint(v.numpy() * (pano_h - 1)).astype(np.int64), 0, pano_h - 1)
    values = np.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = values > 0
    np.add.at(erp, (y[valid], x[valid]), values[valid])
    np.add.at(counts, (y[valid], x[valid]), 1.0)
    valid_erp = counts > 0
    erp[valid_erp] /= counts[valid_erp]
    return erp, valid_erp


def save_depth_image(depth: np.ndarray, valid: np.ndarray, path: Path, max_depth: float) -> None:
    Image.fromarray(colorize_depth(depth, valid, max_depth)).save(path)


def colorize_depth(depth: np.ndarray, valid: np.ndarray, max_depth: float) -> np.ndarray:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = valid & (depth > 0)
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip(depth[valid] / max_depth, 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    vis[~valid] = 0
    return vis


def window_uv(camera_meta: Dict[str, torch.Tensor], height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    yaw = camera_meta["yaw"].flatten()
    pitch = camera_meta["pitch"].flatten()
    fov_x = camera_meta["fov_x"].flatten()
    fov_y = camera_meta["fov_y"].flatten()
    rays = pinhole_rays(yaw, pitch, fov_x, fov_y, height, width, device=yaw.device, dtype=yaw.dtype)
    _, _, u, v = rays_to_equirectangular(rays)
    return u, v


def window_z_factor(camera_meta: Dict[str, torch.Tensor], height: int, width: int) -> np.ndarray:
    yaw = camera_meta["yaw"].flatten()
    pitch = camera_meta["pitch"].flatten()
    fov_x = camera_meta["fov_x"].flatten()
    fov_y = camera_meta["fov_y"].flatten()
    rays = pinhole_rays(yaw, pitch, fov_x, fov_y, height, width, device=yaw.device, dtype=yaw.dtype)
    forward = camera_meta["rotations"][..., :, 2].reshape(-1, 3)
    return (rays * forward[:, None, None, :]).sum(dim=-1).numpy()


def apply_range_depth_modifier(
    pred_depth_z: np.ndarray,
    camera_meta: Dict[str, torch.Tensor],
    max_range_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mark predictions beyond the ERP radial-depth support as non-output values."""
    depth = pred_depth_z.astype(np.float32, copy=True)
    z_factor = window_z_factor(camera_meta, depth.shape[-2], depth.shape[-1])
    radial_depth = depth / np.maximum(z_factor, 1e-6)
    valid = np.isfinite(depth) & (depth > 0) & (z_factor > 1e-6)
    if max_range_depth > 0:
        valid &= np.isfinite(radial_depth) & (radial_depth <= max_range_depth)
    depth[~valid] = np.inf
    return depth, valid


def omega_y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    """Convert Omega internal [right, up, forward] points to ERP/GT [forward, right, up]."""
    return np.stack([points[..., 2], points[..., 0], points[..., 1]], axis=-1)


def write_known_window_point_cloud(
    path: Path,
    depth_z: np.ndarray,
    windows: np.ndarray,
    camera_meta: Dict[str, torch.Tensor],
    max_depth: float,
    max_points: int,
    extra_valid: np.ndarray | None = None,
) -> None:
    yaw = camera_meta["yaw"].flatten()
    pitch = camera_meta["pitch"].flatten()
    fov_x = camera_meta["fov_x"].flatten()
    fov_y = camera_meta["fov_y"].flatten()
    rays = pinhole_rays(yaw, pitch, fov_x, fov_y, depth_z.shape[-2], depth_z.shape[-1], device=yaw.device, dtype=yaw.dtype)
    rays_np = rays.numpy()
    z_factor = window_z_factor(camera_meta, depth_z.shape[-2], depth_z.shape[-1])
    depth = np.nan_to_num(depth_z, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (depth > 0) & (z_factor > 1e-6)
    if extra_valid is not None:
        valid &= extra_valid
    radial_depth = depth / np.maximum(z_factor, 1e-6)
    valid &= radial_depth <= max_depth
    points = omega_y_up_to_z_up(rays_np[valid] * radial_depth[valid, None])
    colors = np.clip(windows.transpose(0, 2, 3, 1)[valid] * 255.0, 0, 255).astype(np.uint8)
    write_ply(path, points, colors, max_points)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int) -> None:
    if points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def write_erp_point_cloud(
    path: Path,
    depth: np.ndarray,
    rgb: np.ndarray,
    max_depth: float,
    max_points: int,
    depth_semantics: str = "range",
    extra_valid: np.ndarray | None = None,
) -> None:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rays = erp_rays_np(*depth.shape)
    radial_depth = erp_range_depth_np(depth, rays, depth_semantics)
    valid = np.isfinite(radial_depth) & (radial_depth > 0) & (radial_depth <= max_depth)
    if extra_valid is not None:
        valid &= extra_valid
    points = rays[valid] * radial_depth[valid, None]
    colors = rgb[valid]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def erp_range_depth_np(depth: np.ndarray, rays: np.ndarray, semantics: str) -> np.ndarray:
    if semantics == "range":
        return depth
    if semantics not in {"cubemap_z", "double_cubemap_z"}:
        raise ValueError(f"Unknown ERP depth semantics: {semantics}")
    source_rays = merged_cubemap_source_rays_np(*depth.shape)
    face_z_factor = cube_projection_factor_np(source_rays)
    if semantics == "double_cubemap_z":
        back_rays = rotate_source_rays_about_vertical_np(source_rays, -math.pi / 4.0)
        back_factor = cube_projection_factor_np(back_rays)
        front_weight = cube_edge_weight_np(source_rays)
        back_weight = cube_edge_weight_np(back_rays)
        face_z_factor = (front_weight * face_z_factor + back_weight * back_factor) / np.maximum(
            front_weight + back_weight, 1e-6
        )
    return depth / face_z_factor


def erp_range_depth_image(depth: np.ndarray, semantics: str) -> np.ndarray:
    return erp_range_depth_np(depth.astype(np.float32), erp_rays_np(*depth.shape), semantics)


def erp_rays_np(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(height, dtype=np.float32) + 0.5, np.arange(width, dtype=np.float32) + 0.5, indexing="ij")
    theta = (x / max(width, 1) - 0.5) * (2.0 * math.pi)
    phi = (0.5 - y / max(height, 1)) * math.pi
    cos_phi = np.cos(phi)
    return np.stack(
        [cos_phi * np.cos(theta), cos_phi * np.sin(theta), np.sin(phi)],
        axis=-1,
    ).astype(np.float32)


def merged_cubemap_source_rays_np(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(height, dtype=np.float32) + 0.5, np.arange(width, dtype=np.float32) + 0.5, indexing="ij")
    longitude = (x / max(width, 1) * 2.0 - 1.0) * math.pi
    latitude = (0.5 - y / max(height, 1)) * math.pi
    cos_lat = np.cos(latitude)
    return np.stack(
        [cos_lat * np.cos(longitude), cos_lat * np.sin(longitude), np.sin(latitude)],
        axis=-1,
    ).astype(np.float32)


def rotate_source_rays_about_vertical_np(rays: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = np.float32(math.cos(angle_rad))
    sine = np.float32(math.sin(angle_rad))
    return np.stack(
        [
            cosine * rays[..., 0] - sine * rays[..., 1],
            sine * rays[..., 0] + cosine * rays[..., 1],
            rays[..., 2],
        ],
        axis=-1,
    ).astype(np.float32)


def cube_projection_factor_np(rays: np.ndarray) -> np.ndarray:
    return np.maximum(np.max(np.abs(rays), axis=-1), 1e-6)


def cube_edge_weight_np(rays: np.ndarray) -> np.ndarray:
    sorted_abs = np.sort(np.abs(rays), axis=-1)
    factor = np.maximum(sorted_abs[..., 2], 1e-6)
    return np.maximum(1.0 - sorted_abs[..., 1] / factor + 1e-6, 0.0).astype(np.float32)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    sample: Dict,
    raw_pred_depth: np.ndarray,
    pred_depth: np.ndarray,
    pred_valid: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    pred_valid_after_range: np.ndarray,
    pred_depth_scale: float,
    fitted_pred_depth_scale: float | None,
    fitted_pred_depth_scale_valid_count: int,
) -> None:
    valid_pred_raw = np.isfinite(raw_pred_depth) & (raw_pred_depth > 0)
    valid_pred = np.isfinite(pred_depth) & (pred_depth > 0) & pred_valid
    valid_target = np.isfinite(target_depth) & (target_depth > 0) & target_valid
    target_available = bool(sample.get("pano_depth") is not None)
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "dataset_format": args.dataset_format,
        "pano_path": str(args.pano_path) if args.pano_path else None,
        "direct_depth_path": str(args.depth_path) if args.depth_path else None,
        "target_available": target_available,
        "sample_index": args.sample_index,
        "resolved_sample_index": getattr(args, "resolved_sample_index", args.sample_index),
        "minimal_datasets": getattr(args, "minimal_datasets", None),
        "dataset_split": getattr(args, "dataset_split", None),
        "sample_scene_name_selector": getattr(args, "sample_scene_name", None),
        "sample_rgb_path_selector": getattr(args, "sample_rgb_path", None),
        "scene_name": sample["scene_name"],
        "rgb_path": sample["rgb_path"],
        "depth_path": sample.get("depth_path"),
        "depth_definition": "window Z-depth meters; target_range_depth_erp.png is decoded ERP radial range in meters",
        "gt_source_depth_semantics": args.gt_depth_semantics,
        "prediction_modifier": f"predictions with reconstructed radial range > {args.depth_max_m:g}m are marked non-output (inf)",
        "mask_pred_by_target_valid": bool(args.mask_pred_by_target_valid),
        "point_cloud_outputs": {
            "prediction": "pred_points.ply",
            "ground_truth": "gt_points.ply" if target_available else None,
            "coordinates": "single-pano Z-up export [forward, right, up] from fixed window geometry",
        },
        "pred_depth_scale": pred_depth_scale,
        "fitted_pred_depth_scale": fitted_pred_depth_scale,
        "fitted_pred_depth_scale_valid_count": fitted_pred_depth_scale_valid_count,
        "pred_valid_ratio_before_modifier": float(valid_pred_raw.mean()),
        "pred_valid_ratio_after_range_modifier": float(pred_valid_after_range.mean()),
        "pred_valid_ratio": float(valid_pred.mean()),
        "pred_removed_by_gt_valid_mask_ratio": float((pred_valid_after_range & ~valid_pred).mean()),
        "pred_removed_by_range_modifier_ratio": float((valid_pred_raw & ~pred_valid_after_range).mean()),
        "pred_removed_by_all_masks_ratio": float((valid_pred_raw & ~valid_pred).mean()),
        "target_valid_ratio": float(valid_target.mean()),
        "pred_depth_median_m": float(np.median(pred_depth[valid_pred])) if np.any(valid_pred) else None,
        "target_depth_median_m": float(np.median(target_depth[valid_target])) if np.any(valid_target) else None,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


if __name__ == "__main__":
    main()
