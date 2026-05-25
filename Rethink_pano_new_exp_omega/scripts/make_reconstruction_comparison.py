#!/usr/bin/env python3
"""Create a compact visual comparison for exported pano reconstructions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--raw-baseline-dir", type=Path, default=None)
    parser.add_argument("--luna-dir", type=Path, required=True)
    parser.add_argument("--loss-plot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    panels = [("Target range ERP", args.luna_dir / "target_range_depth_erp.png")]
    if args.raw_baseline_dir is not None:
        panels.append(("Init raw pred Z", args.raw_baseline_dir / "pred_z_depth_erp_splat.png"))
    panels.extend(
        [
            ("Init scaled pred Z", args.baseline_dir / "pred_z_depth_erp_splat.png"),
            ("LUNA 1h pred Z", args.luna_dir / "pred_z_depth_erp_splat.png"),
            ("1h loss curve", args.loss_plot),
        ]
    )
    images = [load_panel(path, title) for title, path in panels]
    height = max(image.shape[0] for image in images)
    padded = [pad_to_height(image, height) for image in images]
    canvas = np.concatenate(padded, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.output)
    print(f"[INFO] wrote comparison = {args.output}")


def load_panel(path: Path, title: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        array = np.asarray(image).copy()
    header_h = 34
    canvas = np.full((array.shape[0] + header_h, array.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h:] = array
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(20, 20, 20), font=font)
    return np.asarray(pil)


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


if __name__ == "__main__":
    main()
