#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert pano_rl4 equirectangular panoramas into a full-panorama package for
VGGT-Omega style inspection.

This script intentionally keeps the original 360x180 ERP panorama. It does not
create pinhole crops. For every pano_rl4 triplet:

  x_-2370_y_-23000_rgb.png
  x_-2370_y_-23000_normal.exr / .png
  x_-2370_y_-23000_depth.exr / .png

it writes:

  output_root/
    sequence_list.txt
    conversion_meta.json
    omega_images/
      000000_pano_x_-2370_y_-23000.jpg
    preview/
      contact_sheet_first3.jpg
    pano_x_-2370_y_-23000/
      clone/
        pano_meta.json
        camera_meta.json
        frames/
          rgb/Camera_0/rgb_00000.jpg
          normal/Camera_0/normal_00000.png
          depth/Camera_0/depth_00000.png
          depth_valid/Camera_0/valid_00000.png
          depth_vis/Camera_0/depth_00000_vis.png
          preview/Camera_0/pano_preview_00000.jpg

Depth policy:
  - raw_depth * input_depth_scale = meters in source_depth_semantics coordinates
  - non-finite, <= invalid_depth_raw_min, >= invalid_depth_raw_max are invalid
  - metric depth <= 0 or > max_depth_m is invalid and saved as 0
  - saved depth PNG is uint16, depth_m * output_depth_scale
  - depth visualization is cold-to-warm (OpenCV TURBO), invalid pixels black
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


RGB_RE = re.compile(r"^x_(-?\d+)_y_(-?\d+)_rgb\.png$")
NORMAL_SUFFIXES = ["_normal.exr", "_normal.png"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_xy_from_rgb_name(path: Path) -> Optional[Tuple[int, int]]:
    match = RGB_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_degrees_list(value: str) -> Tuple[float, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(float(item) for item in items) or (0.0,)


def find_normal_path(rgb_path: Path) -> Optional[Path]:
    base_name = rgb_path.name.replace("_rgb.png", "")
    for suffix in NORMAL_SUFFIXES:
        candidate = rgb_path.with_name(f"{base_name}{suffix}")
        if candidate.exists():
            return candidate
    return None


def find_rgb_depth_normal_sets(
    input_dir: Path,
    depth_suffix: str,
    max_pairs: int = -1,
) -> List[Dict]:
    rgb_files = sorted(input_dir.glob("x_*_y_*_rgb.png"))
    items: List[Dict] = []

    for rgb_path in rgb_files:
        xy = parse_xy_from_rgb_name(rgb_path)
        if xy is None:
            continue

        depth_path = rgb_path.with_name(rgb_path.name.replace("_rgb.png", depth_suffix))
        if not depth_path.exists():
            print(f"[WARN] missing depth, skip: {rgb_path.name} -> {depth_path.name}")
            continue

        normal_path = find_normal_path(rgb_path)
        if normal_path is None:
            print(f"[WARN] missing normal, will only output RGB/depth: {rgb_path.name}")

        x_cm, y_cm = xy
        items.append(
            {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "normal_path": normal_path,
                "x_cm": x_cm,
                "y_cm": y_cm,
                "scene_name": f"pano_x_{x_cm}_y_{y_cm}",
            }
        )

        if max_pairs is not None and max_pairs >= 0 and len(items) >= max_pairs:
            break

    return items


# ---------------------------------------------------------------------------
# EXR / image readers
# ---------------------------------------------------------------------------

def read_exr_channels(path: Path) -> Dict[str, np.ndarray]:
    try:
        import OpenEXR
        import Imath
    except ImportError as exc:
        raise ImportError("OpenEXR/Imath is required for reading .exr files.") from exc

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)

    channels: Dict[str, np.ndarray] = {}
    for name in sorted(header["channels"].keys()):
        raw = exr.channel(name, float_type)
        channels[name] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    return channels


def choose_depth_channel(
    channels: Dict[str, np.ndarray],
    preferred: Optional[str] = None,
) -> Tuple[str, np.ndarray]:
    if preferred is not None:
        if preferred not in channels:
            raise KeyError(f"Depth channel {preferred!r} not found. Available: {list(channels.keys())}")
        return preferred, channels[preferred]

    for name in ["Z", "Depth", "depth", "R", "Y"]:
        if name in channels:
            return name, channels[name]

    for name in channels:
        lower = name.lower()
        if "depth" in lower or lower.endswith(".z") or lower == "z":
            return name, channels[name]

    first = next(iter(channels.keys()))
    return first, channels[first]


def read_depth_any(
    path: Path,
    exr_channel: Optional[str] = None,
    print_info: bool = False,
) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        channels = read_exr_channels(path)
        channel_name, depth = choose_depth_channel(channels, exr_channel)
        if print_info:
            print(f"[EXR] {path.name}: channels={list(channels.keys())}, selected={channel_name}")
        return depth.astype(np.float32)

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    if depth.ndim == 3:
        raise ValueError(
            f"Depth must be single-channel raw data, got {path} with shape {depth.shape}. "
            "Use the .exr or a raw uint16 depth PNG, not a visualized depth PNG."
        )
    return depth.astype(np.float32)


def read_normal_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        channels = read_exr_channels(path)
        candidates = [
            ("R", "G", "B"),
            ("r", "g", "b"),
            ("red", "green", "blue"),
            ("X", "Y", "Z"),
            ("x", "y", "z"),
        ]
        for names in candidates:
            if all(name in channels for name in names):
                return np.stack([channels[name] for name in names], axis=-1).astype(np.float32)

        first_three = sorted(channels.keys())[:3]
        if len(first_three) < 3:
            raise ValueError(f"Normal EXR has fewer than 3 channels: {path}")
        return np.stack([channels[name] for name in first_three], axis=-1).astype(np.float32)

    normal_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if normal_bgr is None:
        raise FileNotFoundError(f"Cannot read normal: {path}")
    if normal_bgr.ndim != 3 or normal_bgr.shape[2] < 3:
        raise ValueError(f"Normal image must have 3 channels: {path}, shape={normal_bgr.shape}")
    normal_bgr = normal_bgr[..., :3]
    return cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Depth / normal conversion
# ---------------------------------------------------------------------------

def clean_depth_to_meters(
    depth_raw: np.ndarray,
    input_depth_scale: float,
    invalid_depth_raw_min: float,
    invalid_depth_raw_max: float,
    max_depth_m: float,
) -> Tuple[np.ndarray, Dict]:
    raw = depth_raw.astype(np.float32)
    finite_mask = np.isfinite(raw)
    raw_valid_mask = (
        finite_mask
        & (raw > invalid_depth_raw_min)
        & (raw < invalid_depth_raw_max)
    )

    depth_m = np.zeros_like(raw, dtype=np.float32)
    depth_m[raw_valid_mask] = raw[raw_valid_mask] * float(input_depth_scale)

    metric_valid_mask = np.isfinite(depth_m) & (depth_m > 0)
    if max_depth_m > 0:
        metric_valid_mask &= depth_m <= float(max_depth_m)

    clean = np.zeros_like(depth_m, dtype=np.float32)
    clean[metric_valid_mask] = depth_m[metric_valid_mask]

    stats = depth_stats(raw, "raw_depth", valid_mask=raw_valid_mask)
    stats.update(depth_stats(clean, "clean_depth_m", valid_mask=metric_valid_mask))
    stats["raw_valid_pixels"] = int(raw_valid_mask.sum())
    stats["metric_valid_pixels"] = int(metric_valid_mask.sum())
    stats["invalid_pixels"] = int(raw.size - metric_valid_mask.sum())
    stats["far_or_clipped_pixels"] = int(raw_valid_mask.sum() - metric_valid_mask.sum())
    stats["total_pixels"] = int(raw.size)
    stats["valid_ratio"] = float(metric_valid_mask.sum() / max(raw.size, 1))
    return clean, stats


def depth_stats(arr: np.ndarray, prefix: str, valid_mask: Optional[np.ndarray] = None) -> Dict:
    data = arr.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(data) & (data > 0)
    values = data[valid_mask]

    out: Dict[str, object] = {
        f"{prefix}_shape": list(data.shape),
        f"{prefix}_valid_count": int(values.size),
    }
    if values.size:
        percentiles = np.percentile(values, [1, 5, 50, 95, 98, 99])
        out.update(
            {
                f"{prefix}_min": float(values.min()),
                f"{prefix}_max": float(values.max()),
                f"{prefix}_mean": float(values.mean()),
                f"{prefix}_p1_p5_p50_p95_p98_p99": [float(v) for v in percentiles],
            }
        )
    return out


def save_metric_depth_png(depth_m: np.ndarray, out_path: Path, output_depth_scale: float) -> None:
    ensure_dir(out_path.parent)
    depth_saved = np.round(depth_m.astype(np.float32) * float(output_depth_scale))
    depth_saved = np.clip(depth_saved, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_saved)


def normal_to_vis_rgb(normal: np.ndarray) -> np.ndarray:
    n = normal.astype(np.float32)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError(f"Normal map must be HxWx3, got {n.shape}")

    finite = np.isfinite(n).all(axis=2)
    n_min = float(np.nanmin(n)) if finite.any() else 0.0
    n_max = float(np.nanmax(n)) if finite.any() else 1.0

    if n_min >= -1.01 and n_max <= 1.01:
        vis = (np.clip(n, -1.0, 1.0) * 0.5 + 0.5) * 255.0
    elif n_min >= -0.01 and n_max <= 1.01:
        vis = np.clip(n, 0.0, 1.0) * 255.0
    else:
        vis = np.clip(n, 0.0, 255.0)

    vis[~finite] = 0.0
    return np.round(vis).clip(0, 255).astype(np.uint8)


def depth_to_colormap_bgr(
    depth_m: np.ndarray,
    vis_min_m: Optional[float],
    vis_max_m: Optional[float],
    percentile_low: float,
    percentile_high: float,
    max_depth_m: float,
) -> Tuple[np.ndarray, Dict]:
    d = depth_m.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if max_depth_m > 0:
        valid &= d <= float(max_depth_m)

    gray = np.zeros(d.shape, dtype=np.uint8)
    meta: Dict[str, object] = {
        "colormap": "OpenCV COLORMAP_TURBO",
        "invalid_depth_color": "black",
        "small_distance": "cold",
        "large_distance": "warm",
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
    }

    if valid.any():
        values = d[valid]
        lo = float(vis_min_m) if vis_min_m is not None else float(np.percentile(values, percentile_low))
        hi = float(vis_max_m) if vis_max_m is not None else float(np.percentile(values, percentile_high))
        if max_depth_m > 0:
            hi = min(hi, float(max_depth_m))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        norm = np.zeros_like(d, dtype=np.float32)
        norm[valid] = np.clip((d[valid] - lo) / (hi - lo), 0.0, 1.0)
        gray = np.round(norm * 255.0).astype(np.uint8)
        meta["vis_min_m"] = lo
        meta["vis_max_m"] = hi
    else:
        meta["vis_min_m"] = None
        meta["vis_max_m"] = None

    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, meta


def save_depth_colormap_vis(depth_m: np.ndarray, out_path: Path, **kwargs) -> Dict:
    color_bgr, meta = depth_to_colormap_bgr(depth_m, **kwargs)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), color_bgr)
    return meta


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------

def add_label(image_bgr: np.ndarray, label: str, label_height: int = 28) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    canvas = np.full((h + label_height, w, 3), 255, dtype=np.uint8)
    canvas[label_height:, :, :] = image_bgr
    cv2.putText(
        canvas,
        label,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


def resize_to_width(image_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w == width:
        return image_bgr
    scale = width / float(w)
    height = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def make_triptych_preview(
    rgb_bgr: np.ndarray,
    normal_vis_rgb: Optional[np.ndarray],
    depth_vis_bgr: np.ndarray,
    panel_width: int,
) -> np.ndarray:
    rgb_panel = add_label(resize_to_width(rgb_bgr, panel_width), "RGB panorama")

    if normal_vis_rgb is None:
        normal_bgr = np.zeros_like(rgb_bgr)
    else:
        normal_bgr = cv2.cvtColor(normal_vis_rgb, cv2.COLOR_RGB2BGR)
    normal_panel = add_label(resize_to_width(normal_bgr, panel_width), "Normal visualization")

    depth_panel = add_label(resize_to_width(depth_vis_bgr, panel_width), "Depth visualization (m, TURBO)")

    min_height = min(rgb_panel.shape[0], normal_panel.shape[0], depth_panel.shape[0])
    panels = [
        cv2.resize(panel, (panel.shape[1], min_height), interpolation=cv2.INTER_AREA)
        for panel in [rgb_panel, normal_panel, depth_panel]
    ]
    return np.concatenate(panels, axis=1)


def write_contact_sheet(previews: List[np.ndarray], out_path: Path) -> None:
    if not previews:
        return
    width = max(p.shape[1] for p in previews)
    normalized = []
    for preview in previews:
        if preview.shape[1] != width:
            scale = width / float(preview.shape[1])
            height = max(1, int(round(preview.shape[0] * scale)))
            preview = cv2.resize(preview, (width, height), interpolation=cv2.INTER_AREA)
        normalized.append(preview)
    sheet = np.concatenate(normalized, axis=0)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), sheet)


def build_camera_alignment_meta(
    x_cm: int,
    y_cm: int,
    camera_id: int,
    window_size: int,
    num_yaw: int,
    pitch_degrees: Sequence[float],
    fov_degrees: float,
) -> Dict:
    """Build local/world camera metadata for pano-centered virtual cameras."""
    yaw, pitch = make_default_camera_grid(num_yaw=num_yaw, pitch_degrees=pitch_degrees)
    fov = math.radians(float(fov_degrees))
    intrinsics = make_intrinsics(window_size, window_size, fov, fov)
    pano_position_m = np.array([x_cm * 0.01, 0.0, y_cm * 0.01], dtype=np.float32)

    cameras = []
    for view_idx, (yaw_i, pitch_i) in enumerate(zip(yaw, pitch)):
        r_c2w = camera_c2w_from_yaw_pitch(float(yaw_i), float(pitch_i))
        r_w2c = r_c2w.T
        t_local = np.zeros(3, dtype=np.float32)
        t_world = -r_w2c @ pano_position_m
        cameras.append(
            {
                "view_index": int(view_idx),
                "camera_id": int(camera_id),
                "yaw_rad": float(yaw_i),
                "pitch_rad": float(pitch_i),
                "fov_x_rad": float(fov),
                "fov_y_rad": float(fov),
                "intrinsics": intrinsics.tolist(),
                "rotation_c2w": r_c2w.tolist(),
                "rotation_w2c": r_w2c.tolist(),
                "translation_local": t_local.tolist(),
                "translation_world": t_world.astype(np.float32).tolist(),
                "extrinsic_local_w2c": np.concatenate([r_w2c, t_local[:, None]], axis=1).tolist(),
                "extrinsic_world_w2c": np.concatenate([r_w2c, t_world[:, None]], axis=1).tolist(),
            }
        )

    return {
        "camera_model": "virtual pinhole windows sampled from one equirectangular panorama",
        "alignment_policy": (
            "All virtual cameras are co-located with the source panorama. "
            "Use translation_local=[0,0,0] for pano-local training; translation_world "
            "places the shared camera center at the filename x/y position."
        ),
        "position_source": "pano filename x_<cm>_y_<cm>",
        "panorama_position_cm_xy": [int(x_cm), int(y_cm)],
        "panorama_position_m_xyz": pano_position_m.tolist(),
        "window_size": int(window_size),
        "num_yaw": int(num_yaw),
        "pitch_degrees": [float(v) for v in pitch_degrees],
        "fov_degrees": float(fov_degrees),
        "virtual_cameras": cameras,
    }


def make_default_camera_grid(num_yaw: int, pitch_degrees: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    yaw = np.arange(num_yaw, dtype=np.float32) * (2.0 * math.pi / float(num_yaw)) - math.pi
    pitch = np.array([math.radians(float(v)) for v in pitch_degrees], dtype=np.float32)
    yaw_grid, pitch_grid = np.meshgrid(yaw, pitch, indexing="ij")
    return yaw_grid.reshape(-1).astype(np.float32), pitch_grid.reshape(-1).astype(np.float32)


def make_intrinsics(width: int, height: int, fov_x: float, fov_y: float) -> np.ndarray:
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = 0.5 * width / math.tan(0.5 * float(fov_x))
    intrinsics[1, 1] = 0.5 * height / math.tan(0.5 * float(fov_y))
    intrinsics[0, 2] = (width - 1) * 0.5
    intrinsics[1, 2] = (height - 1) * 0.5
    return intrinsics


def camera_c2w_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cos_pitch = math.cos(pitch)
    forward = normalize_np(np.array([cos_pitch * math.sin(yaw), math.sin(pitch), cos_pitch * math.cos(yaw)]))
    right = normalize_np(np.array([math.cos(yaw), 0.0, -math.sin(yaw)]))
    up = normalize_np(np.cross(forward, right))
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def normalize_np(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), eps)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_one(
    item: Dict,
    output_root: Path,
    item_index: int,
    camera_id: int,
    camera_window_size: int,
    camera_num_yaw: int,
    camera_pitch_degrees: Sequence[float],
    camera_fov_degrees: float,
    image_ext: str,
    exr_depth_channel: Optional[str],
    source_depth_semantics: str,
    input_depth_scale: float,
    invalid_depth_raw_min: float,
    invalid_depth_raw_max: float,
    max_depth_m: float,
    output_depth_scale: float,
    depth_vis_min_m: Optional[float],
    depth_vis_max_m: Optional[float],
    depth_vis_percentile_low: float,
    depth_vis_percentile_high: float,
    preview_panel_width: int,
    debug_depth: bool,
) -> Tuple[Dict, np.ndarray]:
    scene_name = item["scene_name"]
    scene_root = output_root / scene_name / "clone"
    frames_root = scene_root / "frames"

    rgb_out_dir = frames_root / "rgb" / f"Camera_{camera_id}"
    normal_out_dir = frames_root / "normal" / f"Camera_{camera_id}"
    depth_out_dir = frames_root / "depth" / f"Camera_{camera_id}"
    depth_valid_out_dir = frames_root / "depth_valid" / f"Camera_{camera_id}"
    depth_vis_out_dir = frames_root / "depth_vis" / f"Camera_{camera_id}"
    preview_out_dir = frames_root / "preview" / f"Camera_{camera_id}"

    for directory in [rgb_out_dir, normal_out_dir, depth_out_dir, depth_valid_out_dir, depth_vis_out_dir, preview_out_dir]:
        ensure_dir(directory)

    rgb_path: Path = item["rgb_path"]
    depth_path: Path = item["depth_path"]
    normal_path: Optional[Path] = item.get("normal_path")

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
    pano_h, pano_w = rgb_bgr.shape[:2]

    depth_raw = read_depth_any(depth_path, exr_channel=exr_depth_channel, print_info=debug_depth)
    if depth_raw.shape[:2] != (pano_h, pano_w):
        raise ValueError(
            f"RGB/depth shape mismatch for {rgb_path.name}: "
            f"rgb={(pano_h, pano_w)}, depth={depth_raw.shape[:2]}"
        )

    depth_m, depth_meta = clean_depth_to_meters(
        depth_raw,
        input_depth_scale=input_depth_scale,
        invalid_depth_raw_min=invalid_depth_raw_min,
        invalid_depth_raw_max=invalid_depth_raw_max,
        max_depth_m=max_depth_m,
    )

    normal_vis_rgb: Optional[np.ndarray] = None
    if normal_path is not None:
        normal = read_normal_any(normal_path)
        if normal.shape[:2] != (pano_h, pano_w):
            raise ValueError(
                f"RGB/normal shape mismatch for {rgb_path.name}: "
                f"rgb={(pano_h, pano_w)}, normal={normal.shape[:2]}"
            )
        normal_vis_rgb = normal_to_vis_rgb(normal)

    rgb_name = f"rgb_00000{image_ext}"
    normal_name = "normal_00000.png"
    depth_name = "depth_00000.png"
    depth_valid_name = "valid_00000.png"
    depth_vis_name = "depth_00000_vis.png"
    preview_name = "pano_preview_00000.jpg"

    cv2.imwrite(str(rgb_out_dir / rgb_name), rgb_bgr)
    ensure_dir(output_root / "omega_images")
    cv2.imwrite(str(output_root / "omega_images" / f"{item_index:06d}_{scene_name}{image_ext}"), rgb_bgr)

    if normal_vis_rgb is not None:
        cv2.imwrite(str(normal_out_dir / normal_name), cv2.cvtColor(normal_vis_rgb, cv2.COLOR_RGB2BGR))

    save_metric_depth_png(depth_m, depth_out_dir / depth_name, output_depth_scale)
    cv2.imwrite(str(depth_valid_out_dir / depth_valid_name), ((depth_m > 0).astype(np.uint8) * 255))
    depth_vis_meta = save_depth_colormap_vis(
        depth_m,
        depth_vis_out_dir / depth_vis_name,
        vis_min_m=depth_vis_min_m,
        vis_max_m=depth_vis_max_m,
        percentile_low=depth_vis_percentile_low,
        percentile_high=depth_vis_percentile_high,
        max_depth_m=max_depth_m,
    )

    depth_vis_bgr = cv2.imread(str(depth_vis_out_dir / depth_vis_name), cv2.IMREAD_COLOR)
    if depth_vis_bgr is None:
        raise RuntimeError(f"Failed to reload depth visualization: {depth_vis_out_dir / depth_vis_name}")

    preview = make_triptych_preview(
        rgb_bgr=rgb_bgr,
        normal_vis_rgb=normal_vis_rgb,
        depth_vis_bgr=depth_vis_bgr,
        panel_width=preview_panel_width,
    )
    cv2.imwrite(str(preview_out_dir / preview_name), preview)

    if debug_depth:
        print(
            f"[DEPTH] {scene_name}: valid={depth_meta['metric_valid_pixels']}/"
            f"{depth_meta['total_pixels']} ({depth_meta['valid_ratio']:.3f}), "
            f"far_or_clipped={depth_meta['far_or_clipped_pixels']}"
        )

    camera_alignment = build_camera_alignment_meta(
        x_cm=int(item["x_cm"]),
        y_cm=int(item["y_cm"]),
        camera_id=camera_id,
        window_size=camera_window_size,
        num_yaw=camera_num_yaw,
        pitch_degrees=camera_pitch_degrees,
        fov_degrees=camera_fov_degrees,
    )
    save_json(camera_alignment, scene_root / "camera_meta.json")

    pano_meta = {
        "scene_name": scene_name,
        "rgb_path": str(rgb_path),
        "normal_path": str(normal_path) if normal_path is not None else None,
        "depth_path": str(depth_path),
        "pano_shape_hw": [int(pano_h), int(pano_w)],
        "projection": "equirectangular panorama, full 360x180 degrees",
        "depth_semantics": (
            f"metric ERP depth in meters using source semantics '{source_depth_semantics}' after input_depth_scale; "
            "decode source semantics to radial range before geometry export or loss"
        ),
        "depth_policy": {
            "unit": "meters",
            "source_depth_semantics": source_depth_semantics,
            "invalid_depth_value": 0,
            "input_depth_scale": float(input_depth_scale),
            "invalid_depth_raw_min": float(invalid_depth_raw_min),
            "invalid_depth_raw_max": float(invalid_depth_raw_max),
            "max_depth_m": float(max_depth_m),
            "max_depth_m_application": (
                "converter applies cap in stored source coordinates; training/export must reapply cap after radial decode"
            ),
            "output_depth_scale": float(output_depth_scale),
            "saved_depth_decode": f"depth_m = uint16_png / {float(output_depth_scale)}",
            "vggt_training_alignment": (
                "VGGT dataset samples use depth maps in meters with a positive-depth "
                "valid mask; VKitti-style samples are commonly capped at 80m."
            ),
        },
        "depth_stats": depth_meta,
        "depth_visualization": depth_vis_meta,
        "camera_alignment": camera_alignment,
        "outputs": {
            "rgb_relpath": str((rgb_out_dir / rgb_name).relative_to(scene_root)),
            "normal_vis_relpath": str((normal_out_dir / normal_name).relative_to(scene_root)) if normal_vis_rgb is not None else None,
            "depth_relpath": str((depth_out_dir / depth_name).relative_to(scene_root)),
            "depth_valid_relpath": str((depth_valid_out_dir / depth_valid_name).relative_to(scene_root)),
            "depth_vis_relpath": str((depth_vis_out_dir / depth_vis_name).relative_to(scene_root)),
            "preview_relpath": str((preview_out_dir / preview_name).relative_to(scene_root)),
            "camera_meta_relpath": "camera_meta.json",
        },
    }
    save_json(pano_meta, scene_root / "pano_meta.json")

    sequence_name = f"{scene_name}/clone/frames/rgb/Camera_{camera_id}"
    conversion_item = {
        "sequence_name": sequence_name,
        "scene_name": scene_name,
        "rgb_path": str(rgb_path),
        "normal_path": str(normal_path) if normal_path is not None else None,
        "depth_path": str(depth_path),
        "preview_path": str(preview_out_dir / preview_name),
        "depth_vis_path": str(depth_vis_out_dir / depth_vis_name),
        "camera_meta_path": str(scene_root / "camera_meta.json"),
        "depth_valid_ratio": depth_meta["valid_ratio"],
        "pano_shape_hw": [int(pano_h), int(pano_w)],
    }
    return conversion_item, preview


def build_parser() -> argparse.ArgumentParser:
    default_input_dir = Path("whitehole/AOKI/datasets/PANO_LUNA_omega")

    parser = argparse.ArgumentParser(
        description="Convert pano_rl4 RGB/normal/depth files to full-panorama VGGT-Omega visual assets."
    )
    parser.add_argument("--input_dir", type=Path, default=default_input_dir)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Default: <input_dir>/converted_pano4vggt_omega",
    )
    parser.add_argument("--max_pairs", type=int, default=-1)

    parser.add_argument("--depth_suffix", type=str, default="_depth.exr")
    parser.add_argument("--exr_depth_channel", type=str, default=None)
    parser.add_argument(
        "--source_depth_semantics",
        choices=["range", "cubemap_z", "double_cubemap_z"],
        default="double_cubemap_z",
        help="Meaning of input ERP depth. The current UE 12-shot merge stores blended perspective Z-depth.",
    )

    parser.add_argument(
        "--input_depth_scale",
        type=float,
        default=0.01,
        help="raw depth * this scale = meters. pano_rl4 / MatrixCity EXR is centimeter-like by default.",
    )
    parser.add_argument("--invalid_depth_raw_min", type=float, default=0.0)
    parser.add_argument(
        "--invalid_depth_raw_max",
        type=float,
        default=65000.0,
        help="Raw depth >= this value is treated as invalid/far sentinel.",
    )
    parser.add_argument(
        "--max_depth_m",
        type=float,
        default=80.0,
        help="Metric depth above this is saved as 0. This follows the VKitti-style VGGT training cap.",
    )
    parser.add_argument(
        "--output_depth_scale",
        type=float,
        default=100.0,
        help="Save uint16 depth as meters * this value.",
    )

    parser.add_argument("--depth_vis_min_m", type=float, default=None)
    parser.add_argument("--depth_vis_max_m", type=float, default=None)
    parser.add_argument("--depth_vis_percentile_low", type=float, default=2.0)
    parser.add_argument("--depth_vis_percentile_high", type=float, default=98.0)

    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument(
        "--camera_window_size",
        type=int,
        default=512,
        help="Virtual pinhole window size used when writing camera_meta.json.",
    )
    parser.add_argument(
        "--camera_num_yaw",
        type=int,
        default=8,
        help="Number of yaw views used when writing camera_meta.json.",
    )
    parser.add_argument(
        "--camera_pitch_degrees",
        type=str,
        default="0",
        help="Comma-separated pitch angles used when writing camera_meta.json.",
    )
    parser.add_argument(
        "--camera_fov_degrees",
        type=float,
        default=75.0,
        help="Virtual pinhole FoV used when writing camera_meta.json.",
    )
    parser.add_argument("--image_ext", type=str, default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument("--preview_panel_width", type=int, default=512)
    parser.add_argument("--contact_sheet_name", type=str, default="contact_sheet_first3.jpg")
    parser.add_argument("--debug_depth", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_root = args.output_root if args.output_root is not None else input_dir / "converted_pano4vggt_omega"

    ensure_dir(output_root)

    items = find_rgb_depth_normal_sets(
        input_dir=input_dir,
        depth_suffix=args.depth_suffix,
        max_pairs=args.max_pairs,
    )
    if not items:
        raise RuntimeError(f"No valid RGB/depth pairs found in {input_dir}")

    print(f"[INFO] input_dir = {input_dir}")
    print(f"[INFO] output_root = {output_root}")
    print(f"[INFO] actual_pairs = {len(items)}")
    print(f"[INFO] depth_suffix = {args.depth_suffix}")
    print(f"[INFO] source_depth_semantics = {args.source_depth_semantics}")
    print(f"[INFO] input_depth_scale = {args.input_depth_scale}")
    print(f"[INFO] invalid_depth_raw_max = {args.invalid_depth_raw_max}")
    print(f"[INFO] max_depth_m = {args.max_depth_m}")
    camera_pitch_degrees = parse_degrees_list(args.camera_pitch_degrees)
    print(
        f"[INFO] camera_meta = window_size:{args.camera_window_size}, "
        f"num_yaw:{args.camera_num_yaw}, pitch:{camera_pitch_degrees}, fov:{args.camera_fov_degrees}"
    )

    sequence_list: List[str] = []
    conversion_meta: List[Dict] = []
    previews: List[np.ndarray] = []

    for index, item in enumerate(items):
        print(f"\n[{index + 1}/{len(items)}] {item['rgb_path'].name}")
        converted, preview = convert_one(
            item=item,
            output_root=output_root,
            item_index=index,
            camera_id=args.camera_id,
            camera_window_size=args.camera_window_size,
            camera_num_yaw=args.camera_num_yaw,
            camera_pitch_degrees=camera_pitch_degrees,
            camera_fov_degrees=args.camera_fov_degrees,
            image_ext=args.image_ext,
            exr_depth_channel=args.exr_depth_channel,
            source_depth_semantics=args.source_depth_semantics,
            input_depth_scale=args.input_depth_scale,
            invalid_depth_raw_min=args.invalid_depth_raw_min,
            invalid_depth_raw_max=args.invalid_depth_raw_max,
            max_depth_m=args.max_depth_m,
            output_depth_scale=args.output_depth_scale,
            depth_vis_min_m=args.depth_vis_min_m,
            depth_vis_max_m=args.depth_vis_max_m,
            depth_vis_percentile_low=args.depth_vis_percentile_low,
            depth_vis_percentile_high=args.depth_vis_percentile_high,
            preview_panel_width=args.preview_panel_width,
            debug_depth=args.debug_depth,
        )
        sequence_list.append(converted["sequence_name"])
        conversion_meta.append(converted)
        previews.append(preview)

    with (output_root / "sequence_list.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(sequence_list))

    save_json(conversion_meta, output_root / "conversion_meta.json")

    contact_sheet_path = output_root / "preview" / args.contact_sheet_name
    write_contact_sheet(previews[: min(3, len(previews))], contact_sheet_path)

    print("\n[DONE]")
    print(f"sequence_list: {output_root / 'sequence_list.txt'}")
    print(f"meta:          {output_root / 'conversion_meta.json'}")
    print(f"contact_sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
