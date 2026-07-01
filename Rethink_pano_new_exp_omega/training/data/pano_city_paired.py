"""Reader for PanoCity paired panorama data.

Expected layout:

  <root>/rgb/*.png
  <root>/depth/*.png

RGB and depth files are paired by identical stem when available. Some PanoCity
exports use ``000123_rgb_000123.png`` / ``000123_depth_000123.png`` naming, so
the reader also falls back to matching by the first numeric token.
"""

from __future__ import annotations

import json
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
        split: str = "train",
        train_split_fraction: float = 0.95,
        split_seed: int = 42,
        metadata_path: Optional[str | Path] = None,
        bad_sample_list: Optional[str | Path] = None,
        curriculum_bins: Optional[str | Iterable[str]] = None,
        use_metadata_weights: bool = True,
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
        self.split = str(split)
        self.train_split_fraction = float(train_split_fraction)
        self.split_seed = int(split_seed)
        self.metadata_path = None if metadata_path in (None, "") else Path(metadata_path)
        self.bad_sample_list = None if bad_sample_list in (None, "") else Path(bad_sample_list)
        self.curriculum_bins = _parse_bins(curriculum_bins)
        self.use_metadata_weights = bool(use_metadata_weights)
        self.metadata_by_name = _load_metadata(self.metadata_path, self.root)
        self.bad_samples = _load_bad_samples(self.bad_sample_list, self.root)
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
        if not self.items:
            raise IndexError("No PanoCity items are available.")
        start = int(item_index) % len(self.items)
        last_error: Exception | None = None
        for offset in range(len(self.items)):
            candidate_index = (start + offset) % len(self.items)
            try:
                return self._read_item(self.items[candidate_index])
            except (FileNotFoundError, OSError, ValueError) as exc:
                last_error = exc
                print(f"[WARN] skipping unreadable PanoCity sample {self.items[candidate_index].get('scene_name')}: {exc}")
        raise RuntimeError("All PanoCity samples failed to load.") from last_error

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
            "sample_weight": torch.tensor(float(item.get("sample_weight", 1.0)), dtype=torch.float32),
            "metadata_valid_ratio": torch.tensor(float(item.get("metadata_valid_ratio", 1.0)), dtype=torch.float32),
            "metadata_structure_score": torch.tensor(float(item.get("metadata_structure_score", 0.0)), dtype=torch.float32),
            "metadata_quality_bin": str(item.get("metadata_quality_bin", "unknown")),
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
            if _is_bad_sample(rgb_path, depth_path, self.bad_samples):
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
        items = _apply_metadata(
            items,
            metadata_by_name=self.metadata_by_name,
            curriculum_bins=self.curriculum_bins,
            use_metadata_weights=self.use_metadata_weights,
        )
        items = _split_items(
            items,
            split=self.split,
            train_fraction=self.train_split_fraction,
            seed=self.split_seed,
        )
        if max_samples is not None:
            items = items[: int(max_samples)]
        return items


def _split_items(items: List[Dict], split: str, train_fraction: float, seed: int) -> List[Dict]:
    if not items:
        return []
    train_fraction = min(max(float(train_fraction), 0.0), 1.0)
    order = list(range(len(items)))
    random.Random(int(seed)).shuffle(order)
    train_count = int(round(len(order) * train_fraction))
    if len(order) > 1:
        train_count = min(max(train_count, 1), len(order) - 1)
    split_name = str(split).lower()
    if split_name == "train":
        keep = set(order[:train_count])
    elif split_name in {"val", "valid", "validation", "test"}:
        keep = set(order[train_count:])
    else:
        raise ValueError(f"Unknown PanoCity split: {split}")
    return [item for idx, item in enumerate(items) if idx in keep]


def _parse_bins(raw: Optional[str | Iterable[str]]) -> Optional[set[str]]:
    if raw in (None, "", "all"):
        return None
    if isinstance(raw, str):
        values = [part.strip().lower() for part in raw.split(",")]
    else:
        values = [str(part).strip().lower() for part in raw]
    bins = {value for value in values if value}
    return bins or None


def _load_metadata(path: Optional[Path], root: Path) -> Dict[str, Dict]:
    if path is None:
        return {}
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        print(f"[WARN] PanoCity metadata not found: {resolved}; training without curriculum metadata.")
        return {}
    metadata: Dict[str, Dict] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            keys = {
                str(entry.get("name", "")),
                Path(str(entry.get("rgb_rel", ""))).stem,
                Path(str(entry.get("rgb_path", ""))).stem,
            }
            for key in keys:
                if key:
                    metadata[key] = entry
    print(f"[INFO] loaded PanoCity metadata entries = {len(metadata)} from {resolved}")
    return metadata


def _load_bad_samples(path: Optional[Path], root: Path) -> set[str]:
    if path is None:
        return set()
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        return set()
    values: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path_value = Path(value)
            values.add(value)
            values.add(path_value.name)
            values.add(path_value.stem)
    print(f"[INFO] loaded PanoCity bad sample entries = {len(values)} from {resolved}")
    return values


def _is_bad_sample(rgb_path: Path, depth_path: Path, bad_samples: set[str]) -> bool:
    if not bad_samples:
        return False
    keys = {
        str(rgb_path),
        str(depth_path),
        rgb_path.name,
        depth_path.name,
        rgb_path.stem,
        depth_path.stem,
    }
    return bool(keys & bad_samples)


def _apply_metadata(
    items: List[Dict],
    metadata_by_name: Dict[str, Dict],
    curriculum_bins: Optional[set[str]],
    use_metadata_weights: bool,
) -> List[Dict]:
    if not metadata_by_name:
        for item in items:
            item["sample_weight"] = 1.0
            item["metadata_valid_ratio"] = 1.0
            item["metadata_structure_score"] = 0.0
            item["metadata_quality_bin"] = "unknown"
        return items
    filtered: List[Dict] = []
    for item in items:
        metadata = metadata_by_name.get(str(item["scene_name"])) or metadata_by_name.get(Path(item["rgb_path"]).stem)
        if metadata is None:
            quality_bin = "unknown"
            sample_weight = 1.0
            valid_ratio = 1.0
            structure_score = 0.0
        else:
            quality_bin = str(metadata.get("quality_bin", "unknown")).lower()
            sample_weight = float(metadata.get("sample_weight", 1.0)) if use_metadata_weights else 1.0
            valid_ratio = float(metadata.get("valid_ratio", 1.0))
            structure_score = float(metadata.get("structure_score", 0.0))
        if curriculum_bins is not None and quality_bin not in curriculum_bins:
            continue
        updated = dict(item)
        updated["sample_weight"] = sample_weight
        updated["metadata_valid_ratio"] = valid_ratio
        updated["metadata_structure_score"] = structure_score
        updated["metadata_quality_bin"] = quality_bin
        filtered.append(updated)
    return filtered


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
