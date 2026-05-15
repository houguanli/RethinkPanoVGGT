#!/usr/bin/env python3
"""Remake MatrixCity-style pano RGB/depth/normal files into LUNA pano dataset.

The output is intentionally simpler than VKitti:

output_root/
  sequence_list.txt
  dataset_meta.json
  00000/
    rgb.png
    depth.png
    normal.png
    camera_6dof.txt
    pose_c2w.txt
    pose_w2c.txt
    meta.json

`camera_6dof.txt` stores one whole-pano pose as
`tx ty tz roll pitch yaw` in meters/radians. By default the filename
coordinates `x_*_y_*` are interpreted as world X/Z centimeters, matching the
existing conversion script in E:/pano_rl4/python_scripts/convert_pano.py.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


RGB_RE = re.compile(r"^x_(-?\d+)_y_(-?\d+)_rgb\.(png|jpg|jpeg)$", re.IGNORECASE)
NORMAL_SUFFIXES = ("_normal.exr", "_normal.png", "_normal.jpg")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_xy_from_rgb_name(path: Path) -> Optional[Tuple[int, int]]:
    match = RGB_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def find_normal_path(rgb_path: Path) -> Optional[Path]:
    stem = re.sub(r"_rgb\.(png|jpg|jpeg)$", "", rgb_path.name, flags=re.IGNORECASE)
    for suffix in NORMAL_SUFFIXES:
        candidate = rgb_path.with_name(f"{stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def find_rgb_depth_normal_items(
    input_dir: Path,
    depth_suffix: str,
    max_items: int = -1,
    require_normal: bool = False,
) -> List[Dict]:
    rgb_files = sorted(input_dir.glob("x_*_y_*_rgb.*"))
    items = []
    for rgb_path in rgb_files:
        xy = parse_xy_from_rgb_name(rgb_path)
        if xy is None:
            continue
        depth_path = rgb_path.with_name(re.sub(r"_rgb\.(png|jpg|jpeg)$", depth_suffix, rgb_path.name, flags=re.IGNORECASE))
        if not depth_path.exists():
            print(f"[WARN] missing depth, skip: {rgb_path.name} -> {depth_path.name}")
            continue
        normal_path = find_normal_path(rgb_path)
        if normal_path is None:
            message = f"missing normal for {rgb_path.name}"
            if require_normal:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}; output item will not contain normal.png")
        items.append(
            {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "normal_path": normal_path,
                "x_cm": xy[0],
                "y_cm": xy[1],
            }
        )
    if max_items is not None and max_items >= 0:
        items = items[:max_items]
    return items


def read_exr_channels(path: Path) -> Dict[str, np.ndarray]:
    try:
        import OpenEXR
        import Imath
    except ImportError as exc:
        raise ImportError("OpenEXR/Imath is required for EXR input.") from exc

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = {}
    for name in sorted(header["channels"].keys()):
        channels[name] = np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width)
    return channels


def choose_depth_channel(channels: Dict[str, np.ndarray], preferred: Optional[str]) -> Tuple[str, np.ndarray]:
    if preferred is not None:
        if preferred not in channels:
            raise KeyError(f"Depth channel {preferred!r} not found. Available: {list(channels.keys())}")
        return preferred, channels[preferred]
    for name in ("Z", "Depth", "depth", "R", "Y"):
        if name in channels:
            return name, channels[name]
    for name in channels:
        lower = name.lower()
        if "depth" in lower or lower.endswith(".z") or lower == "z":
            return name, channels[name]
    first = next(iter(channels))
    return first, channels[first]


def read_depth_any(path: Path, exr_channel: Optional[str] = None) -> Tuple[np.ndarray, Optional[str]]:
    suffix = path.suffix.lower()
    if suffix == ".exr":
        channel_name, depth = choose_depth_channel(read_exr_channels(path), exr_channel)
        return depth.astype(np.float32), channel_name

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    if depth.ndim == 3:
        raise ValueError(f"Depth must be single-channel, got {path} shape={depth.shape}")
    return depth.astype(np.float32), None


def read_normal_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        channels = read_exr_channels(path)
        for names in (("R", "G", "B"), ("r", "g", "b"), ("X", "Y", "Z"), ("x", "y", "z")):
            if all(name in channels for name in names):
                return np.stack([channels[name] for name in names], axis=-1).astype(np.float32)
        names = sorted(channels.keys())[:3]
        if len(names) < 3:
            raise ValueError(f"Normal EXR has fewer than 3 channels: {path}")
        return np.stack([channels[name] for name in names], axis=-1).astype(np.float32)

    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if normal is None:
        raise FileNotFoundError(f"Cannot read normal: {path}")
    if normal.ndim != 3 or normal.shape[2] != 3:
        raise ValueError(f"Normal must be HxWx3, got {path} shape={normal.shape}")
    return normal


def clean_metric_depth(
    raw_depth: np.ndarray,
    input_depth_scale: float,
    invalid_depth_raw_min: float,
    invalid_depth_raw_max: float,
    max_depth_m: float,
) -> np.ndarray:
    raw = raw_depth.astype(np.float32)
    valid = np.isfinite(raw) & (raw > invalid_depth_raw_min) & (raw < invalid_depth_raw_max)
    metric = np.zeros_like(raw, dtype=np.float32)
    metric[valid] = raw[valid] * input_depth_scale
    valid_metric = np.isfinite(metric) & (metric > 0.0) & (metric <= max_depth_m)
    clean = np.zeros_like(metric, dtype=np.float32)
    clean[valid_metric] = metric[valid_metric]
    return clean


def save_metric_depth_png(depth_m: np.ndarray, out_path: Path, output_depth_scale: float) -> None:
    saved = np.round(depth_m.astype(np.float32) * output_depth_scale)
    saved = np.clip(saved, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), saved)


def normalize_normal_map(normal: np.ndarray) -> np.ndarray:
    if normal.dtype == np.uint8:
        return normal
    normal_f = normal.astype(np.float32)
    if np.all((normal_f >= -1.0) & (normal_f <= 1.0)):
        normal_u8 = (normal_f * 0.5 + 0.5) * 255.0
    else:
        normal_u8 = np.clip(normal_f, 0.0, 1.0) * 255.0
    return np.clip(np.round(normal_u8), 0, 255).astype(np.uint8)


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def make_pose_c2w(translation_xyz: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = euler_xyz_to_matrix(roll, pitch, yaw)
    pose[:3, 3] = translation_xyz.astype(np.float32)
    return pose


def remake_one_item(
    item: Dict,
    out_dir: Path,
    item_id: int,
    input_depth_scale: float,
    invalid_depth_raw_min: float,
    invalid_depth_raw_max: float,
    max_depth_m: float,
    output_depth_scale: float,
    xy_scale: float,
    z_m: float,
    roll: float,
    pitch: float,
    yaw: float,
    depth_suffix_channel: Optional[str],
    image_ext: str,
) -> Dict:
    sample_id = f"{item_id:05d}"
    sample_dir = out_dir / sample_id
    ensure_dir(sample_dir)

    rgb_bgr = cv2.imread(str(item["rgb_path"]), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB: {item['rgb_path']}")
    h, w = rgb_bgr.shape[:2]

    raw_depth, depth_channel = read_depth_any(item["depth_path"], exr_channel=depth_suffix_channel)
    if raw_depth.shape[:2] != (h, w):
        raise ValueError(f"RGB/depth mismatch for {item['rgb_path'].name}: rgb={(h, w)} depth={raw_depth.shape[:2]}")
    depth_m = clean_metric_depth(
        raw_depth,
        input_depth_scale=input_depth_scale,
        invalid_depth_raw_min=invalid_depth_raw_min,
        invalid_depth_raw_max=invalid_depth_raw_max,
        max_depth_m=max_depth_m,
    )

    rgb_name = f"rgb{image_ext}"
    cv2.imwrite(str(sample_dir / rgb_name), rgb_bgr)
    save_metric_depth_png(depth_m, sample_dir / "depth.png", output_depth_scale)

    normal_relpath = None
    if item.get("normal_path") is not None:
        normal = read_normal_any(item["normal_path"])
        if normal.shape[:2] != (h, w):
            raise ValueError(f"RGB/normal mismatch for {item['rgb_path'].name}: rgb={(h, w)} normal={normal.shape[:2]}")
        cv2.imwrite(str(sample_dir / "normal.png"), normalize_normal_map(normal))
        normal_relpath = "normal.png"

    translation = np.array([item["x_cm"] * xy_scale, z_m, item["y_cm"] * xy_scale], dtype=np.float32)
    pose_c2w = make_pose_c2w(translation, roll=roll, pitch=pitch, yaw=yaw)
    pose_w2c = np.linalg.inv(pose_c2w).astype(np.float32)

    np.savetxt(sample_dir / "pose_c2w.txt", pose_c2w, fmt="%.9g")
    np.savetxt(sample_dir / "pose_w2c.txt", pose_w2c, fmt="%.9g")
    with (sample_dir / "camera_6dof.txt").open("w", encoding="utf-8") as f:
        f.write("# tx ty tz roll pitch yaw\n")
        f.write("# units: meters radians; euler order: Rz(yaw) @ Ry(pitch) @ Rx(roll)\n")
        f.write("%.9g %.9g %.9g %.9g %.9g %.9g\n" % (translation[0], translation[1], translation[2], roll, pitch, yaw))

    meta = {
        "id": sample_id,
        "rgb": rgb_name,
        "depth": "depth.png",
        "normal": normal_relpath,
        "camera_6dof": "camera_6dof.txt",
        "pose_c2w": "pose_c2w.txt",
        "pose_w2c": "pose_w2c.txt",
        "source_rgb": str(item["rgb_path"]),
        "source_depth": str(item["depth_path"]),
        "source_normal": str(item["normal_path"]) if item.get("normal_path") is not None else None,
        "source_xy_cm": [int(item["x_cm"]), int(item["y_cm"])],
        "image_hw": [int(h), int(w)],
        "translation_xyz_m": translation.tolist(),
        "roll_pitch_yaw_rad": [float(roll), float(pitch), float(yaw)],
        "depth_info": {
            "raw_channel": depth_channel,
            "input_depth_scale": float(input_depth_scale),
            "invalid_depth_raw_min": float(invalid_depth_raw_min),
            "invalid_depth_raw_max": float(invalid_depth_raw_max),
            "max_depth_m": float(max_depth_m),
            "output_depth_scale": float(output_depth_scale),
            "semantics": "metric ERP ray/range depth in meters before output scaling",
        },
        "axis_convention": {
            "world_up": "+Y",
            "filename_x_maps_to": "world X position",
            "filename_y_maps_to": "world Z position",
            "erp_center_lon_0": "local +Z in the LUNA sampler convention",
        },
    }
    save_json(meta, sample_dir / "meta.json")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remake pano RGB/depth/normal dataset for LUNA training.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--depth_suffix", type=str, default="_depth.exr")
    parser.add_argument("--exr_depth_channel", type=str, default=None)
    parser.add_argument("--input_depth_scale", type=float, default=0.01)
    parser.add_argument("--invalid_depth_raw_min", type=float, default=0.0)
    parser.add_argument("--invalid_depth_raw_max", type=float, default=65000.0)
    parser.add_argument("--max_depth_m", type=float, default=80.0)
    parser.add_argument("--output_depth_scale", type=float, default=100.0)
    parser.add_argument("--xy_scale", type=float, default=0.01)
    parser.add_argument("--z_m", type=float, default=0.5)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--image_ext", type=str, default=".png", choices=[".png", ".jpg"])
    parser.add_argument("--require_normal", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    items = find_rgb_depth_normal_items(
        input_dir=input_dir,
        depth_suffix=args.depth_suffix,
        max_items=args.max_items,
        require_normal=args.require_normal,
    )
    if not items:
        raise RuntimeError(f"No valid pano RGB/depth pairs found in {input_dir}")

    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] items={len(items)}")

    all_meta = []
    for idx, item in enumerate(items):
        sample_id = f"{idx:05d}"
        print(f"[{idx + 1}/{len(items)}] {sample_id} <- {item['rgb_path'].name}")
        all_meta.append(
            remake_one_item(
                item=item,
                out_dir=output_root,
                item_id=idx,
                input_depth_scale=args.input_depth_scale,
                invalid_depth_raw_min=args.invalid_depth_raw_min,
                invalid_depth_raw_max=args.invalid_depth_raw_max,
                max_depth_m=args.max_depth_m,
                output_depth_scale=args.output_depth_scale,
                xy_scale=args.xy_scale,
                z_m=args.z_m,
                roll=args.roll,
                pitch=args.pitch,
                yaw=args.yaw,
                depth_suffix_channel=args.exr_depth_channel,
                image_ext=args.image_ext,
            )
        )

    (output_root / "sequence_list.txt").write_text("\n".join(meta["id"] for meta in all_meta), encoding="utf-8")
    save_json(
        {
            "format": "pano_luna_numbered_v1",
            "num_items": len(all_meta),
            "items": all_meta,
        },
        output_root / "dataset_meta.json",
    )
    print("[DONE]")
    print(f"sequence_list: {output_root / 'sequence_list.txt'}")
    print(f"dataset_meta:  {output_root / 'dataset_meta.json'}")


if __name__ == "__main__":
    main()
