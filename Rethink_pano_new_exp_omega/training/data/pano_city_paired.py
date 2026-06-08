"""Reader for PanoCity paired panorama data.

Expected layout:

  <root>/rgb/*.png
  <root>/depth/*.png

RGB and depth files are paired by identical stem when available. Some PanoCity
exports use ``000123_rgb_000123.png`` / ``000123_depth_000123.png`` naming, so
the reader also falls back to matching by the first numeric token.
"""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_PANOCITY_ROOT = Path("/mnt/f/panovggt/PanoCity_paired")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PanoCityPairedOmegaDataset(Dataset):
    """Read flat PanoCity rgb/depth panorama pairs."""

    def __init__(
        self,
        root: str | Path = DEFAULT_PANOCITY_ROOT,
        pano_size: Optional[Tuple[int, int]] = None,
        pano_sample_mode: str = "single",
        pano_min_count: int = 1,
        pano_max_count: int = 1,
        panos_per_sample: Optional[int] = None,
        grouping: str = "nearest",
        max_samples: Optional[int] = None,
        output_depth_scale: float = 1000.0,
        invalid_depth_value: Optional[float] = 65535.0,
        position_step_m: float = 1.0,
        strict: bool = True,
    ) -> None:
        if panos_per_sample is not None:
            pano_sample_mode = "single" if panos_per_sample == 1 else "fixed_neighborhood"
            pano_min_count = panos_per_sample
            pano_max_count = panos_per_sample
        if pano_sample_mode not in {"single", "fixed_neighborhood", "variable_neighborhood"}:
            raise ValueError(f"Unknown pano_sample_mode: {pano_sample_mode}")
        if pano_min_count < 1 or pano_max_count < pano_min_count:
            raise ValueError(f"Invalid pano count range: min={pano_min_count}, max={pano_max_count}")
        if pano_sample_mode == "single":
            pano_min_count = 1
            pano_max_count = 1
        if pano_sample_mode == "variable_neighborhood" and pano_min_count < 2:
            raise ValueError("variable_neighborhood requires pano_min_count >= 2")
        if grouping not in {"nearest", "sequential"}:
            raise ValueError(f"Unknown grouping mode: {grouping}")
        if output_depth_scale <= 0:
            raise ValueError(f"output_depth_scale must be positive, got {output_depth_scale}")

        self.root = Path(root)
        self.pano_size = pano_size
        self.pano_sample_mode = pano_sample_mode
        self.pano_min_count = pano_min_count
        self.pano_max_count = pano_max_count
        self.grouping = grouping
        self.output_depth_scale = float(output_depth_scale)
        self.invalid_depth_value = None if invalid_depth_value is None else float(invalid_depth_value)
        self.position_step_m = float(position_step_m)
        self.items = self._build_index(max_samples=max_samples)
        self.groups = self._build_groups()
        if strict and not self.items:
            raise FileNotFoundError(
                f"No PanoCity paired samples found under {self.root}. "
                "Expected rgb/*.png and depth/*.png folders."
            )
        if self.pano_sample_mode != "single" and strict and len(self.items) < self.pano_min_count:
            raise ValueError(
                f"{self.pano_sample_mode} needs at least {self.pano_min_count} panos, "
                f"but only found {len(self.items)} under {self.root}."
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> Dict:
        group = self.groups[index]
        if self.pano_sample_mode == "single":
            return self._read_item(self.items[group[0]])
        if self.pano_sample_mode == "variable_neighborhood":
            max_count = min(self.pano_max_count, len(group))
            min_count = min(self.pano_min_count, max_count)
            group = group[: random.randint(min_count, max_count)]

        samples = [self._read_item(self.items[item_index]) for item_index in group]
        return {
            "pano_image": torch.stack([sample["pano_image"] for sample in samples], dim=0),
            "pano_depth": torch.stack([sample["pano_depth"] for sample in samples], dim=0),
            "sequence_name": [sample["sequence_name"] for sample in samples],
            "scene_name": "|".join(sample["scene_name"] for sample in samples),
            "rgb_path": [sample["rgb_path"] for sample in samples],
            "depth_path": [sample["depth_path"] for sample in samples],
            "pano_position_m": torch.stack([sample["pano_position_m"] for sample in samples], dim=0),
        }

    def _read_item(self, item: Dict) -> Dict:
        image = _read_rgb_tensor(item["rgb_path"])
        depth = _read_depth_tensor(
            item["depth_path"],
            output_depth_scale=self.output_depth_scale,
            invalid_depth_value=self.invalid_depth_value,
        )

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

    def _build_groups(self) -> List[List[int]]:
        if self.pano_sample_mode == "single":
            return [[idx] for idx in range(len(self.items))]
        if not self.items:
            return []

        groups = []
        for idx in range(len(self.items)):
            group = list(range(idx, min(idx + self.pano_max_count, len(self.items))))
            if len(group) < self.pano_max_count:
                group.extend(range(0, self.pano_max_count - len(group)))
            groups.append(group)
        return groups

    def _build_index(self, max_samples: Optional[int]) -> List[Dict]:
        rgb_dir = self.root / "rgb"
        depth_dir = self.root / "depth"
        if not rgb_dir.exists() or not depth_dir.exists():
            return []

        items: List[Dict] = []
        seen_depth_paths = set()
        for rgb_path in _iter_image_files(rgb_dir):
            depth_path = _depth_path_for_rgb(rgb_path, depth_dir)
            if depth_path is None:
                continue
            depth_key = str(depth_path)
            if depth_key in seen_depth_paths:
                continue
            seen_depth_paths.add(depth_key)
            ordinal = len(items)
            items.append(
                {
                    "sequence_name": "PanoCity_paired",
                    "scene_name": rgb_path.stem,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "pano_position_m": [float(ordinal) * self.position_step_m, 0.0, 0.0],
                }
            )
            if max_samples is not None and len(items) >= max_samples:
                return items
        return items


def _iter_image_files(folder: Path) -> Iterable[Path]:
    for entry in os.scandir(folder):
        if entry.is_file():
            path = Path(entry.path)
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                yield path


def _first_token(path: Path) -> str:
    return path.name.split("_", 1)[0]


def _depth_path_for_rgb(rgb_path: Path, depth_dir: Path) -> Optional[Path]:
    token = _first_token(rgb_path)
    candidates = [
        depth_dir / rgb_path.name,
        depth_dir / rgb_path.name.replace("_rgb_", "_depth_"),
        depth_dir / f"{token}_depth_{token}.png",
        depth_dir / f"{token}_pano_{token}.png",
    ]
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _read_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _read_depth_tensor(
    path: Path,
    output_depth_scale: float,
    invalid_depth_value: Optional[float],
) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_raw = depth.astype(np.float32)
    depth_m = depth_raw / float(output_depth_scale)
    if invalid_depth_value is not None:
        depth_m = np.where(depth_raw >= invalid_depth_value, np.inf, depth_m)
    return torch.from_numpy(depth_m)[None].contiguous()
