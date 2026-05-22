"""VKitti-style reader for panoramas produced by convert_pano4vggt_omega.py.

The converter writes a VKitti-like directory layout:

  <root>/sequence_list.txt
  <root>/<scene>/clone/pano_meta.json
  <root>/<scene>/clone/frames/rgb/Camera_0/rgb_00000.jpg
  <root>/<scene>/clone/frames/depth/Camera_0/depth_00000.png

This dataset keeps the full equirectangular panorama. The model's
PanoWindowSampler is responsible for turning it into virtual pinhole windows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = Path("whitehole/AOKI/datasets/PANO_LUNA_omega")
DEFAULT_CONVERTED_DIRNAME = "converted_pano4vggt_omega"
_RGB_RE = re.compile(r"^rgb_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def resolve_converted_dataset_root(root: str | Path) -> Path:
    """Resolve either the raw PANO_LUNA_omega folder or its converted output."""
    root = Path(root)
    candidates = [root, root / DEFAULT_CONVERTED_DIRNAME]
    for candidate in candidates:
        if (candidate / "sequence_list.txt").exists():
            return candidate
    return root


class PanoVKittiOmegaDataset(Dataset):
    """Read converted panorama samples using a VKitti-style sequence list."""

    def __init__(
        self,
        root: str | Path = DEFAULT_DATASET_ROOT,
        pano_size: Optional[Tuple[int, int]] = None,
        max_samples: Optional[int] = None,
        strict: bool = True,
    ) -> None:
        self.root = resolve_converted_dataset_root(root)
        self.pano_size = pano_size
        self.items = self._build_index(max_samples=max_samples)
        if strict and not self.items:
            raise FileNotFoundError(
                f"No pano VKitti samples found under {self.root}. "
                "Expected sequence_list.txt plus frames/rgb and frames/depth folders."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        image = _read_rgb_tensor(item["rgb_path"])
        depth = _read_depth_tensor(item["depth_path"], item["output_depth_scale"])

        if self.pano_size is not None:
            height, width = self.pano_size
            image = F.interpolate(
                image[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
            depth = F.interpolate(
                depth[None],
                size=(height, width),
                mode="nearest",
            )[0]

        return {
            "pano_image": image,
            "pano_depth": depth,
            "sequence_name": item["sequence_name"],
            "scene_name": item["scene_name"],
            "rgb_path": str(item["rgb_path"]),
            "depth_path": str(item["depth_path"]),
            "pano_position_m": torch.tensor(item["pano_position_m"], dtype=torch.float32),
        }

    def _build_index(self, max_samples: Optional[int]) -> List[Dict]:
        sequence_list_path = self.root / "sequence_list.txt"
        if not sequence_list_path.exists():
            return []

        sequence_names = [
            line.strip().replace("\\", "/")
            for line in sequence_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        items: List[Dict] = []
        for sequence_name in sequence_names:
            rgb_dir = self.root / sequence_name
            for rgb_path in _iter_rgb_files(rgb_dir):
                depth_path = _depth_path_for_rgb(rgb_path)
                if not depth_path.exists():
                    continue
                meta = _load_pano_meta(rgb_path)
                items.append(
                    {
                        "sequence_name": sequence_name,
                        "scene_name": meta.get("scene_name", _scene_name_from_rgb(rgb_path)),
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "output_depth_scale": _depth_scale_from_meta(meta),
                        "pano_position_m": _pano_position_from_meta(meta),
                    }
                )
                if max_samples is not None and len(items) >= max_samples:
                    return items
        return items


def _iter_rgb_files(rgb_dir: Path) -> Iterable[Path]:
    if not rgb_dir.exists():
        return []
    return sorted(path for path in rgb_dir.iterdir() if path.is_file() and _RGB_RE.match(path.name))


def _depth_path_for_rgb(rgb_path: Path) -> Path:
    parts = list(rgb_path.parts)
    try:
        rgb_idx = max(idx for idx, part in enumerate(parts) if part.lower() == "rgb")
    except ValueError as exc:
        raise ValueError(f"RGB path does not contain a frames/rgb component: {rgb_path}") from exc

    parts[rgb_idx] = "depth"
    depth_name = re.sub(r"^rgb_", "depth_", rgb_path.name, flags=re.IGNORECASE)
    depth_name = str(Path(depth_name).with_suffix(".png"))
    return Path(*parts[:-1]) / depth_name


def _pano_meta_path_for_rgb(rgb_path: Path) -> Path:
    parts = list(rgb_path.parts)
    try:
        frames_idx = max(idx for idx, part in enumerate(parts) if part.lower() == "frames")
    except ValueError as exc:
        raise ValueError(f"RGB path does not contain a frames component: {rgb_path}") from exc
    return Path(*parts[:frames_idx]) / "pano_meta.json"


def _load_pano_meta(rgb_path: Path) -> Dict:
    meta_path = _pano_meta_path_for_rgb(rgb_path)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _depth_scale_from_meta(meta: Dict) -> float:
    depth_policy = meta.get("depth_policy") or {}
    scale = float(depth_policy.get("output_depth_scale", 100.0))
    if scale <= 0:
        raise ValueError(f"Invalid output_depth_scale in pano_meta.json: {scale}")
    return scale


def _pano_position_from_meta(meta: Dict) -> List[float]:
    camera_alignment = meta.get("camera_alignment") or {}
    position = camera_alignment.get("panorama_position_m_xyz", None)
    if position is None:
        return [0.0, 0.0, 0.0]
    if len(position) != 3:
        raise ValueError(f"Expected panorama_position_m_xyz with 3 values, got {position}")
    return [float(value) for value in position]


def _scene_name_from_rgb(rgb_path: Path) -> str:
    parts = list(rgb_path.parts)
    try:
        clone_idx = max(idx for idx, part in enumerate(parts) if part.lower() == "clone")
    except ValueError:
        return rgb_path.parent.name
    if clone_idx == 0:
        return rgb_path.parent.name
    return parts[clone_idx - 1]


def _read_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _read_depth_tensor(path: Path, output_depth_scale: float) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32) / float(output_depth_scale)
    return torch.from_numpy(depth_m)[None].contiguous()
