#!/usr/bin/env python3
"""Create a compact visual panel for official-window equivalence outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = [
        ("official windows", args.root / "official_windows_reconstruction" / "pred_depth_windows.jpg"),
        ("pano pure sampler", args.root / "pano_pure_sampler_reconstruction" / "pred_depth_windows.jpg"),
        ("pano current setting", args.root / "pano_current_setting_reconstruction" / "pred_depth_windows.jpg"),
    ]
    images = [load_panel(title, path) for title, path in panels]
    height = max(image.shape[0] for image in images)
    padded = [pad_to_height(image, height) for image in images]
    canvas = np.concatenate(padded, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.output)
    print(f"[INFO] wrote {args.output}")


def load_panel(title: str, path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((560, 560), Image.Resampling.LANCZOS)
        body = np.asarray(image).copy()
    header_h = 34
    canvas = np.full((body.shape[0] + header_h, body.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h:] = body
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
