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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import PanoVKittiOmegaDataset  # noqa: E402
from training.train_pano_omega import build_model, load_checkpoint, sample_depth_targets, set_seed  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays, rays_to_equirectangular  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pano reconstruction previews from a trained LUNA checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    parser.add_argument("--pred-depth-scale", type=float, default=None)
    parser.add_argument("--pano-height", type=int, default=None)
    parser.add_argument("--pano-width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_args = model_args_from_checkpoint(ckpt_args)
    pred_depth_scale = args.pred_depth_scale
    if pred_depth_scale is None:
        pred_depth_scale = float(ckpt_args.get("pred_depth_scale", 1.0))
    if args.pano_height is not None and args.pano_width is not None:
        model_args.pano_height = args.pano_height
        model_args.pano_width = args.pano_width

    dataset = PanoVKittiOmegaDataset(
        root=args.dataset_root,
        pano_size=pano_size_from_args(model_args),
    )
    sample = dataset[args.sample_index]
    model = build_model(model_args).to(device)
    load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pano_image = sample["pano_image"][None].to(device)
    pano_depth = sample["pano_depth"][None].to(device)

    with torch.no_grad():
        predictions = model(pano_images=pano_image, return_sampler_output=True)
        target_depth, target_valid = sample_depth_targets(model, pano_depth)

    pred_depth = (predictions["depth"] * pred_depth_scale)[0, ..., 0].detach().float().cpu().numpy()
    pred_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    target_depth_np = target_depth[0, ..., 0].detach().float().cpu().numpy()
    target_valid_np = target_valid[0, ..., 0].detach().cpu().numpy()
    windows = predictions["pano_windows"][0].detach().float().cpu().numpy()
    camera_meta = {key: value[0].detach().float().cpu() for key, value in predictions["pano_camera_meta"].items()}
    pano_np = tensor_image_to_uint8(sample["pano_image"])

    Image.fromarray(pano_np).save(output_dir / "input_pano.jpg")
    save_contact_sheet(windows, output_dir / "sampled_windows.jpg")
    save_depth_sheet(pred_depth, output_dir / "pred_z_depth_windows.jpg", max_depth=args.depth_max_m)
    save_depth_sheet(target_depth_np, output_dir / "target_z_depth_windows.jpg", max_depth=args.depth_max_m)
    save_conf_sheet(pred_conf, output_dir / "pred_conf_windows.jpg")

    pred_erp, valid_erp = splat_windows_to_erp(
        pred_depth=pred_depth,
        camera_meta=camera_meta,
        pano_hw=pano_np.shape[:2],
    )
    save_depth_image(pred_erp, valid_erp, output_dir / "pred_z_depth_erp_splat.png", max_depth=args.depth_max_m)
    save_depth_image(sample["pano_depth"][0].numpy(), sample["pano_depth"][0].numpy() > 0, output_dir / "target_range_depth_erp.png", max_depth=args.depth_max_m)
    write_official_point_cloud(
        output_dir / "pred_official_camera_points.ply",
        pred_depth_z=pred_depth,
        windows=windows,
        pose_enc=predictions["pose_enc"][0].detach().float().cpu(),
        max_depth=args.depth_max_m,
        max_points=args.max_points,
    )
    write_known_window_point_cloud(
        output_dir / "pred_known_window_camera_points.ply",
        depth_z=pred_depth,
        windows=windows,
        camera_meta=camera_meta,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
    )
    write_known_window_point_cloud(
        output_dir / "target_known_window_camera_points.ply",
        depth_z=target_depth_np,
        windows=windows,
        camera_meta=camera_meta,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
        extra_valid=target_valid_np,
    )
    write_erp_point_cloud(
        output_dir / "target_erp_points.ply",
        depth=sample["pano_depth"][0].numpy(),
        rgb=pano_np,
        max_depth=args.depth_max_m,
        max_points=args.max_points,
    )
    write_summary(output_dir / "summary.json", args, sample, pred_depth, target_depth_np, target_valid_np, pred_depth_scale)
    print(f"[INFO] exported reconstruction = {output_dir}")


def model_args_from_checkpoint(ckpt_args: Dict) -> SimpleNamespace:
    defaults = {
        "patch_size": 16,
        "window_size": 512,
        "num_yaw": 8,
        "pitch_degrees": "0",
        "fov_degrees": 75.0,
        "pano_height": 0,
        "pano_width": 0,
        "enable_camera_head": True,
        "pred_depth_scale": 1.0,
        "smoke": False,
    }
    defaults.update({key: ckpt_args[key] for key in defaults.keys() & ckpt_args.keys()})
    return SimpleNamespace(**defaults)


def pano_size_from_args(args: SimpleNamespace) -> Tuple[int, int] | None:
    if args.pano_height and args.pano_width:
        return int(args.pano_height), int(args.pano_width)
    return None


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
    forward = camera_meta["rotations"][..., :, 2].reshape(-1, 3).numpy()
    z_factor = np.sum(rays_np * forward[:, None, None, :], axis=-1)
    depth = np.nan_to_num(depth_z, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (depth > 0) & (z_factor > 1e-6)
    if extra_valid is not None:
        valid &= extra_valid
    radial_depth = depth / np.maximum(z_factor, 1e-6)
    valid &= radial_depth <= max_depth
    points = rays_np[valid] * radial_depth[valid, None]
    colors = np.clip(windows.transpose(0, 2, 3, 1)[valid] * 255.0, 0, 255).astype(np.uint8)
    write_ply(path, points, colors, max_points)


def write_official_point_cloud(
    path: Path,
    pred_depth_z: np.ndarray,
    windows: np.ndarray,
    pose_enc: torch.Tensor,
    max_depth: float,
    max_points: int,
) -> None:
    extrinsics, intrinsics = encoding_to_camera(pose_enc[None], pred_depth_z.shape[-2:])
    extrinsics = extrinsics[0].numpy()
    intrinsics = intrinsics[0].numpy()
    num_frames, height, width = pred_depth_z.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    depth = np.nan_to_num(pred_depth_z, nan=0.0, posinf=0.0, neginf=0.0)
    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth],
        axis=-1,
    )
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    points = np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )
    radial_depth = np.linalg.norm(camera_points, axis=-1)
    valid = (depth > 0) & (radial_depth <= max_depth) & np.isfinite(points).all(axis=-1)
    colors = np.clip(windows.transpose(0, 2, 3, 1)[valid] * 255.0, 0, 255).astype(np.uint8)
    write_ply(path, points[valid], colors, max_points)


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
) -> None:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    h, w = depth.shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    theta = (x / max(w - 1, 1) - 0.5) * (2.0 * math.pi)
    phi = (0.5 - y / max(h - 1, 1)) * math.pi
    cos_phi = np.cos(phi)
    rays = np.stack(
        [
            cos_phi * np.sin(theta),
            np.sin(phi),
            cos_phi * np.cos(theta),
        ],
        axis=-1,
    ).astype(np.float32)
    valid = (depth > 0) & (depth <= max_depth)
    points = rays[valid] * depth[valid, None]
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


def write_summary(
    path: Path,
    args: argparse.Namespace,
    sample: Dict,
    pred_depth: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    pred_depth_scale: float,
) -> None:
    valid_pred = np.isfinite(pred_depth) & (pred_depth > 0)
    valid_target = np.isfinite(target_depth) & (target_depth > 0) & target_valid
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "sample_index": args.sample_index,
        "scene_name": sample["scene_name"],
        "rgb_path": sample["rgb_path"],
        "depth_definition": "window Z-depth meters; ERP source/target_range_depth_erp.png stores radial range meters",
        "pred_depth_scale": pred_depth_scale,
        "pred_valid_ratio": float(valid_pred.mean()),
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
