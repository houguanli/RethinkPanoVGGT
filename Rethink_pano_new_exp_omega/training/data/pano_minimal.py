"""Readers for the minimal PanoVGGT dataset bundle.

Supported layouts under one parent directory:

  Matterport3D/<scan>/pano_skybox_color/*.jpg
  Matterport3D/<scan>/pano_depth/*.png
  Matterport3D/<scan>/pano_poses/*.txt

  Stanford2D3DS/<area>/pano/rgb/*.png
  Stanford2D3DS/<area>/pano/depth/*.png
  Stanford2D3DS/<area>/pano/pose/*.json

  Structured3D/<scene>/2D_rendering/<camera>/panorama/full/rgb_rawlight.png
  Structured3D/<scene>/2D_rendering/<camera>/panorama/full/depth.png
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class MixedPanoDataset(Dataset):
    """Concatenate multiple pano datasets while preserving the training batch schema."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        self.root = " + ".join(str(getattr(dataset, "root", "unknown")) for dataset in self.datasets)
        self.split = ",".join(str(getattr(dataset, "split", "train")) for dataset in self.datasets)
        self.train_split_fraction = "mixed"
        self.split_seed = "mixed"
        self.cumulative: List[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative.append(total)
        if not self.datasets:
            raise FileNotFoundError("No non-empty datasets were provided to MixedPanoDataset.")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> Dict:
        if index < 0:
            index += len(self)
        dataset_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = 0 if dataset_idx == 0 else self.cumulative[dataset_idx - 1]
        return self.datasets[dataset_idx][index - previous]


class PanoMinimalDataset(Dataset):
    """Read Matterport3D, Stanford2D3DS, and Structured3D from the minimal bundle."""

    def __init__(
        self,
        root: str | Path,
        pano_size: Optional[Tuple[int, int]] = None,
        pano_sample_mode: str = "single",
        pano_min_count: int = 1,
        pano_max_count: int = 1,
        panos_per_sample: Optional[int] = None,
        grouping: str = "nearest",
        max_samples: Optional[int] = None,
        split: str = "train",
        train_split_fraction: float = 0.95,
        split_seed: int = 42,
        datasets: Optional[str | Iterable[str]] = None,
        output_depth_scale: float = 1000.0,
        invalid_depth_value: Optional[float] = 65535.0,
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

        self.root = Path(root)
        self.pano_size = pano_size
        self.pano_sample_mode = pano_sample_mode
        self.pano_min_count = pano_min_count
        self.pano_max_count = pano_max_count
        self.grouping = grouping
        self.split = str(split)
        self.train_split_fraction = float(train_split_fraction)
        self.split_seed = int(split_seed)
        self.dataset_names = _parse_dataset_names(datasets)
        self.output_depth_scale = float(output_depth_scale)
        self.invalid_depth_value = None if invalid_depth_value is None else float(invalid_depth_value)
        self.items = self._build_index(max_samples=max_samples)
        self.groups = self._build_groups()
        if strict and not self.items:
            raise FileNotFoundError(f"No minimal PanoVGGT samples found under {self.root}.")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> Dict:
        group = self.groups[index]
        if self.pano_sample_mode == "single":
            return self._read_item_with_fallback(group[0])
        if self.pano_sample_mode == "variable_neighborhood":
            max_count = min(self.pano_max_count, len(group))
            min_count = min(self.pano_min_count, max_count)
            group = group[: random.randint(min_count, max_count)]
        samples = [self._read_item_with_fallback(item_index) for item_index in group]
        return {
            "pano_image": torch.stack([sample["pano_image"] for sample in samples], dim=0),
            "pano_depth": torch.stack([sample["pano_depth"] for sample in samples], dim=0),
            "sequence_name": [sample["sequence_name"] for sample in samples],
            "scene_name": "|".join(sample["scene_name"] for sample in samples),
            "rgb_path": [sample["rgb_path"] for sample in samples],
            "depth_path": [sample["depth_path"] for sample in samples],
            "pano_position_m": torch.stack([sample["pano_position_m"] for sample in samples], dim=0),
            "sample_weight": torch.stack([sample["sample_weight"] for sample in samples], dim=0),
            "metadata_valid_ratio": torch.stack([sample["metadata_valid_ratio"] for sample in samples], dim=0),
            "metadata_structure_score": torch.stack([sample["metadata_structure_score"] for sample in samples], dim=0),
            "metadata_quality_bin": [sample["metadata_quality_bin"] for sample in samples],
        }

    def _read_item_with_fallback(self, item_index: int) -> Dict:
        start = int(item_index) % len(self.items)
        last_error: Exception | None = None
        for offset in range(len(self.items)):
            candidate_index = (start + offset) % len(self.items)
            try:
                return self._read_item(self.items[candidate_index])
            except (FileNotFoundError, OSError, ValueError) as exc:
                last_error = exc
                print(f"[WARN] skipping unreadable minimal pano sample {self.items[candidate_index].get('scene_name')}: {exc}")
        raise RuntimeError("All minimal pano samples failed to load.") from last_error

    def _read_item(self, item: Dict) -> Dict:
        image = _read_rgb_tensor(item["rgb_path"])
        depth = _read_depth_tensor(
            item["depth_path"],
            output_depth_scale=float(item.get("output_depth_scale", self.output_depth_scale)),
            invalid_depth_value=self.invalid_depth_value,
        )
        if self.pano_size is not None:
            height, width = self.pano_size
            image = F.interpolate(image[None], size=(height, width), mode="bilinear", align_corners=False)[0]
            depth = F.interpolate(depth[None], size=(height, width), mode="nearest")[0]
        return {
            "pano_image": image,
            "pano_depth": depth,
            "sequence_name": item["sequence_name"],
            "scene_name": item["scene_name"],
            "rgb_path": str(item["rgb_path"]),
            "depth_path": str(item["depth_path"]),
            "pano_position_m": torch.tensor(item["pano_position_m"], dtype=torch.float32),
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
            "metadata_valid_ratio": torch.tensor(1.0, dtype=torch.float32),
            "metadata_structure_score": torch.tensor(0.0, dtype=torch.float32),
            "metadata_quality_bin": "unknown",
        }

    def _build_groups(self) -> List[List[int]]:
        if self.pano_sample_mode == "single":
            return [[idx] for idx in range(len(self.items))]
        groups = []
        for idx in range(len(self.items)):
            group = list(range(idx, min(idx + self.pano_max_count, len(self.items))))
            if len(group) < self.pano_max_count and self.items:
                group.extend(range(0, self.pano_max_count - len(group)))
            groups.append(group)
        return groups

    def _build_index(self, max_samples: Optional[int]) -> List[Dict]:
        items: List[Dict] = []
        if "matterport3d" in self.dataset_names:
            items.extend(_index_matterport3d(self.root / "Matterport3D", self.split, self.output_depth_scale))
        if "stanford2d3ds" in self.dataset_names:
            items.extend(_index_stanford2d3ds(self.root / "Stanford2D3DS", self.split, self.output_depth_scale))
        if "structured3d" in self.dataset_names:
            items.extend(_index_structured3d(self.root / "Structured3D", self.split, self.output_depth_scale))
        if max_samples is not None:
            items = items[: int(max_samples)]
        return items


def _parse_dataset_names(raw: Optional[str | Iterable[str]]) -> set[str]:
    if raw in (None, "", "all"):
        return {"matterport3d", "stanford2d3ds", "structured3d"}
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = [str(value) for value in raw]
    aliases = {"2d3ds": "stanford2d3ds", "stanford": "stanford2d3ds", "mp3d": "matterport3d", "s3d": "structured3d"}
    return {aliases.get(value.strip().lower(), value.strip().lower()) for value in values if value.strip()}


def _index_matterport3d(root: Path, split: str, scale: float) -> List[Dict]:
    index_path = root / "cache" / f"matterport3d_{split}_index.json"
    rows = _read_json_list(index_path)
    items: List[Dict] = []
    for row in rows:
        scan, room_id, _room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = root / str(scan) / "pano_skybox_color" / f"{pano_id}.jpg"
            depth_path = root / str(scan) / "pano_depth" / f"{pano_id}.png"
            pose_path = root / str(scan) / "pano_poses" / f"{pano_id}.txt"
            if not rgb_path.exists() or not depth_path.exists():
                continue
            items.append(_item("Matterport3D", f"{scan}_{room_id}_{pano_id}", rgb_path, depth_path, _read_pose_translation(pose_path), scale))
    return items


def _index_stanford2d3ds(root: Path, split: str, scale: float) -> List[Dict]:
    index_path = root / "cache" / f"2d3ds_{split}_index.json"
    rows = _read_json_list(index_path)
    items: List[Dict] = []
    for row in rows:
        area, _room_id, room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = _first_match(root / str(area) / "pano" / "rgb", f"camera_{pano_id}_*_rgb.png")
            depth_path = _first_match(root / str(area) / "pano" / "depth", f"camera_{pano_id}_*_depth.png")
            pose_path = _first_match(root / str(area) / "pano" / "pose", f"camera_{pano_id}_*_pose.json")
            if rgb_path is None or depth_path is None:
                continue
            items.append(_item("Stanford2D3DS", f"{area}_{room_name}_{pano_id}", rgb_path, depth_path, _read_stanford_position(pose_path), scale))
    return items


def _index_structured3d(root: Path, split: str, scale: float) -> List[Dict]:
    index_path = root / "cache" / f"structured3d_{split}_index.json"
    rows = _read_json_list(index_path)
    if not rows:
        for fallback_split in ("val", "test"):
            rows = _read_json_list(root / "cache" / f"structured3d_{fallback_split}_index.json")
            if rows:
                print(
                    f"[WARN] Structured3D {split} index not found under {root}; "
                    f"using structured3d_{fallback_split}_index.json instead."
                )
                break
    items: List[Dict] = []
    for row in rows:
        scene, pano_ids, _size = row
        for pano_id in pano_ids:
            pano_dir = root / str(scene) / "2D_rendering" / str(pano_id) / "panorama"
            rgb_path = pano_dir / "full" / "rgb_rawlight.png"
            depth_path = pano_dir / "full" / "depth.png"
            if not rgb_path.exists() or not depth_path.exists():
                continue
            items.append(_item("Structured3D", f"{scene}_{pano_id}", rgb_path, depth_path, _read_structured3d_position(pano_dir / "camera_xyz.txt"), scale))
    return items


def _item(sequence_name: str, scene_name: str, rgb_path: Path, depth_path: Path, position: List[float], scale: float) -> Dict:
    return {
        "sequence_name": sequence_name,
        "scene_name": scene_name,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "pano_position_m": position,
        "output_depth_scale": scale,
    }


def _read_json_list(path: Path) -> List:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _first_match(folder: Path, pattern: str) -> Optional[Path]:
    if not folder.exists():
        return None
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _read_pose_translation(path: Path) -> List[float]:
    if not path.exists():
        return [0.0, 0.0, 0.0]
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape == (4, 4):
        return [float(value) for value in matrix[:3, 3]]
    return [0.0, 0.0, 0.0]


def _read_stanford_position(path: Optional[Path]) -> List[float]:
    if path is None or not path.exists():
        return [0.0, 0.0, 0.0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [0.0, 0.0, 0.0]
    for key in ("camera_location", "position", "translation"):
        value = data.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
    return [0.0, 0.0, 0.0]


def _read_structured3d_position(path: Path) -> List[float]:
    if not path.exists():
        return [0.0, 0.0, 0.0]
    values = [float(value) for value in path.read_text(encoding="utf-8").split()[:3]]
    if len(values) != 3:
        return [0.0, 0.0, 0.0]
    return [value / 1000.0 for value in values]


def _read_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _read_depth_tensor(path: Path, output_depth_scale: float, invalid_depth_value: Optional[float]) -> torch.Tensor:
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
