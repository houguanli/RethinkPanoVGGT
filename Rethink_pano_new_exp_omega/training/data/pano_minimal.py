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

  Panocity/<city>/<block>/pano_images/pano_*.png
  Panocity/<city>/<block>/panodepth_images/pano_depth_*.png
  Panocity/<city>/<block>/*_poses.json

Official PanoVGGT depth units are dataset-specific:
  Panocity: cm -> meters (/100)
  Matterport3D: /4000
  Stanford2D3DS: /512
  Structured3D: mm -> meters (/1000)
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

_PANOCITY_DEPTH_SCALE = 100.0
_MATTERPORT3D_DEPTH_SCALE = 4000.0
_STANFORD2D3DS_DEPTH_SCALE = 512.0
_STRUCTURED3D_DEPTH_SCALE = 1000.0

_MP3D_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_MP3D_WORLD_TO_OPENCV = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_S3D_WORLD_TO_OPENCV = _MP3D_WORLD_TO_OPENCV


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
    """Read Matterport3D, Stanford2D3DS, Structured3D, and Panocity from the bundle."""

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
        bad_sample_list: Optional[str | Path] = None,
        dataset_sampling_weights: Optional[str | Dict[str, float]] = None,
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
        self.bad_samples = _load_bad_samples(bad_sample_list, self.root)
        self.dataset_sampling_weights = _parse_dataset_sampling_weights(dataset_sampling_weights)
        self.output_depth_scale = float(output_depth_scale)
        self.invalid_depth_value = None if invalid_depth_value is None else float(invalid_depth_value)
        self.items = self._build_index(max_samples=max_samples)
        self.sample_indices, self.dataset_sampling_summary = _build_balanced_sample_indices(
            self.items,
            self.dataset_sampling_weights if self.split == "train" else None,
            seed=self.split_seed,
        )
        self.indices_by_scene = self._build_indices_by_scene()
        self.groups = self._build_groups()
        if strict and not self.items:
            raise FileNotFoundError(f"No minimal PanoVGGT samples found under {self.root}.")
        if strict and self.pano_sample_mode != "single" and not self.groups:
            raise FileNotFoundError(
                f"No minimal PanoVGGT multi-pano groups with at least {self.pano_min_count} panos found under {self.root}."
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
        samples = self._read_group_with_scene_fallback(group)
        return {
            "pano_image": torch.stack([sample["pano_image"] for sample in samples], dim=0),
            "pano_depth": torch.stack([sample["pano_depth"] for sample in samples], dim=0),
            "sequence_name": [sample["sequence_name"] for sample in samples],
            "scene_name": "|".join(sample["scene_name"] for sample in samples),
            "rgb_path": [sample["rgb_path"] for sample in samples],
            "depth_path": [sample["depth_path"] for sample in samples],
            "pano_position_m": torch.stack([sample["pano_position_m"] for sample in samples], dim=0),
            "pano_position_valid": torch.stack([sample["pano_position_valid"] for sample in samples], dim=0),
            "pano_rotation_c2w": torch.stack([sample["pano_rotation_c2w"] for sample in samples], dim=0),
            "pano_rotation_valid": torch.stack([sample["pano_rotation_valid"] for sample in samples], dim=0),
            "sample_weight": torch.stack([sample["sample_weight"] for sample in samples], dim=0),
            "metadata_valid_ratio": torch.stack([sample["metadata_valid_ratio"] for sample in samples], dim=0),
            "metadata_structure_score": torch.stack([sample["metadata_structure_score"] for sample in samples], dim=0),
            "metadata_quality_bin": [sample["metadata_quality_bin"] for sample in samples],
        }

    def _read_item_with_fallback(self, item_index: int) -> Dict:
        sample, _chosen_index = self._read_item_with_fallback_from_candidates(item_index)
        return sample

    def _read_group_with_scene_fallback(self, group: Sequence[int]) -> List[Dict]:
        if not group:
            raise IndexError("Cannot read an empty multi-pano group.")
        scene_key = _scene_group_key(self.items[group[0]])
        group_scene_keys = {_scene_group_key(self.items[item_index]) for item_index in group}
        if group_scene_keys != {scene_key}:
            raise RuntimeError(f"Multi-pano group crosses scenes: {sorted(group_scene_keys)}")
        scene_indices = self.indices_by_scene.get(scene_key, list(group))
        samples: List[Dict] = []
        used_indices: set[int] = set()
        for item_index in group:
            try:
                sample, chosen_index = self._read_item_with_fallback_from_candidates(
                    item_index,
                    fallback_indices=scene_indices,
                    used_indices=used_indices,
                )
            except RuntimeError:
                # Keep the replacement in-scene even if the scene has too few
                # readable alternatives to avoid a duplicate panorama.
                sample, chosen_index = self._read_item_with_fallback_from_candidates(
                    item_index,
                    fallback_indices=scene_indices,
                    used_indices=None,
                )
            samples.append(sample)
            used_indices.add(chosen_index)
        return samples

    def _read_item_with_fallback_from_candidates(
        self,
        item_index: int,
        fallback_indices: Optional[Sequence[int]] = None,
        used_indices: Optional[set[int]] = None,
    ) -> Tuple[Dict, int]:
        start = int(item_index) % len(self.items)
        candidates = self._fallback_candidates(start, fallback_indices)
        last_error: Exception | None = None
        for candidate_index in candidates:
            if used_indices is not None and candidate_index in used_indices:
                continue
            try:
                return self._read_item(self.items[candidate_index]), candidate_index
            except (FileNotFoundError, OSError, ValueError, SyntaxError) as exc:
                last_error = exc
                print(f"[WARN] skipping unreadable minimal pano sample {self.items[candidate_index].get('scene_name')}: {exc}")
        raise RuntimeError("All minimal pano samples failed to load.") from last_error

    def _fallback_candidates(
        self,
        item_index: int,
        fallback_indices: Optional[Sequence[int]],
    ) -> List[int]:
        if fallback_indices is None:
            return [(item_index + offset) % len(self.items) for offset in range(len(self.items))]
        anchor = self.items[item_index]
        unique_indices = sorted(set(int(index) for index in fallback_indices))
        return sorted(
            unique_indices,
            key=lambda candidate: (
                candidate != item_index,
                _position_distance_sq(anchor, self.items[candidate]),
                abs(candidate - item_index),
                candidate,
            ),
        )

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
            "pano_position_valid": torch.tensor(bool(item.get("pano_position_valid", True)), dtype=torch.bool),
            "pano_rotation_c2w": torch.tensor(item.get("pano_rotation_c2w", _identity_rotation()), dtype=torch.float32),
            "pano_rotation_valid": torch.tensor(bool(item.get("pano_rotation_valid", False)), dtype=torch.bool),
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
            "metadata_valid_ratio": torch.tensor(1.0, dtype=torch.float32),
            "metadata_structure_score": torch.tensor(0.0, dtype=torch.float32),
            "metadata_quality_bin": "unknown",
        }

    def _build_groups(self) -> List[List[int]]:
        sample_indices = self.sample_indices if self.sample_indices else list(range(len(self.items)))
        if self.pano_sample_mode == "single":
            return [[idx] for idx in sample_indices]

        offset_by_index: Dict[int, int] = {}
        for scene_indices in self.indices_by_scene.values():
            for offset, item_index in enumerate(scene_indices):
                offset_by_index[item_index] = offset

        groups = []
        for idx in sample_indices:
            item = self.items[idx]
            scene_indices = self.indices_by_scene.get(_scene_group_key(item), [idx])
            if len(scene_indices) < self.pano_min_count:
                continue
            anchor_offset = offset_by_index.get(idx, 0)
            radius = max(self.pano_max_count * 4, self.pano_min_count)
            start = max(0, anchor_offset - radius)
            end = min(len(scene_indices), anchor_offset + radius + 1)
            candidates = scene_indices[start:end]
            if self.grouping == "nearest":
                candidates.sort(
                    key=lambda candidate: (
                        candidate != idx,
                        _position_distance_sq(item, self.items[candidate]),
                        abs(candidate - idx),
                        candidate,
                    )
                )
            else:
                candidates.sort(key=lambda candidate: (abs(candidate - idx), candidate != idx, candidate))
            group = candidates[: self.pano_max_count]
            groups.append(group)
        return groups

    def _build_indices_by_scene(self) -> Dict[str, List[int]]:
        indices_by_scene: Dict[str, List[int]] = {}
        for item_index, item in enumerate(self.items):
            indices_by_scene.setdefault(_scene_group_key(item), []).append(item_index)
        return indices_by_scene

    def _build_index(self, max_samples: Optional[int]) -> List[Dict]:
        items: List[Dict] = []
        if "matterport3d" in self.dataset_names:
            items.extend(_index_matterport3d(self.root / "Matterport3D", self.split, _MATTERPORT3D_DEPTH_SCALE))
        if "stanford2d3ds" in self.dataset_names:
            items.extend(_index_stanford2d3ds(self.root / "Stanford2D3DS", self.split, _STANFORD2D3DS_DEPTH_SCALE))
        if "structured3d" in self.dataset_names:
            items.extend(_index_structured3d(self.root / "Structured3D", self.split, _STRUCTURED3D_DEPTH_SCALE))
        if "panocity" in self.dataset_names:
            items.extend(_index_panocity_official(self.root / "Panocity", self.split, _PANOCITY_DEPTH_SCALE))
        if self.bad_samples:
            before = len(items)
            items = [
                item
                for item in items
                if not _is_bad_sample(
                    str(item.get("rgb_path", "")),
                    str(item.get("depth_path", "")),
                    str(item.get("scene_name", "")),
                    self.bad_samples,
                )
            ]
            skipped = before - len(items)
            if skipped > 0:
                print(f"[INFO] skipped bad minimal pano samples from bad_sample_list = {skipped}")
        if max_samples is not None:
            items = items[: int(max_samples)]
        return items


def _load_bad_samples(path: Optional[str | Path], root: Path) -> set[str]:
    if path in (None, ""):
        return set()
    requested = Path(path)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(Path(__file__).resolve().parents[2] / requested)
        candidates.append(root / requested)
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        print(f"[WARN] minimal pano bad_sample_list not found: {path}")
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
    print(f"[INFO] loaded minimal pano bad sample entries = {len(values)} from {resolved}")
    return values


def _is_bad_sample(rgb_path: str, depth_path: str, scene_name: str, bad_samples: set[str]) -> bool:
    if not bad_samples:
        return False
    keys: set[str] = {scene_name}
    for raw in (rgb_path, depth_path):
        path = Path(raw)
        keys.add(str(path))
        keys.add(path.name)
        keys.add(path.stem)
        keys.update(part for part in path.parts if part)
    return bool(keys & bad_samples)


def _scene_group_key(item: Dict) -> str:
    explicit = item.get("scene_group_key")
    if explicit not in (None, ""):
        return str(explicit)
    dataset = str(item.get("dataset") or item.get("sequence_name") or "unknown")
    scene_name = str(item.get("scene_name") or "")
    scene_prefix = scene_name.rsplit("_", 1)[0] if "_" in scene_name else scene_name
    return f"{dataset}:{scene_prefix}"


def _join_scene_key(dataset: object, *parts: object) -> str:
    values = [str(dataset)]
    values.extend(str(part) for part in parts if part not in (None, ""))
    return ":".join(values)


def _position_distance_sq(anchor: Dict, candidate: Dict) -> float:
    try:
        a = anchor.get("pano_position_m", [0.0, 0.0, 0.0])
        b = candidate.get("pano_position_m", [0.0, 0.0, 0.0])
        return float(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _parse_dataset_names(raw: Optional[str | Iterable[str]]) -> set[str]:
    if raw in (None, "", "all"):
        return {"matterport3d", "stanford2d3ds", "structured3d", "panocity"}
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = [str(value) for value in raw]
    aliases = {
        "2d3ds": "stanford2d3ds",
        "stanford": "stanford2d3ds",
        "mp3d": "matterport3d",
        "s3d": "structured3d",
        "pano_city": "panocity",
        "panocityofficial": "panocity",
    }
    return {aliases.get(value.strip().lower(), value.strip().lower()) for value in values if value.strip()}


def _dataset_key(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "pano_city": "panocity",
        "panocityofficial": "panocity",
        "panocity_paired": "panocity",
        "matterport": "matterport3d",
        "mp3d": "matterport3d",
        "stanford": "stanford2d3ds",
        "2d3ds": "stanford2d3ds",
        "s3d": "structured3d",
    }
    return aliases.get(raw, raw)


def _parse_dataset_sampling_weights(raw: Optional[str | Dict[str, float]]) -> Optional[Dict[str, float]]:
    if raw in (None, "", "none"):
        return None
    if isinstance(raw, dict):
        pairs = raw.items()
    else:
        pairs = []
        for part in str(raw).split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"Dataset sampling weight must be name:weight, got {part!r}")
            name, value = part.split(":", 1)
            pairs.append((name, value))
    weights: Dict[str, float] = {}
    for name, value in pairs:
        key = _dataset_key(name)
        weight = float(value)
        if weight < 0:
            raise ValueError(f"Dataset sampling weight must be non-negative, got {name}:{value}")
        weights[key] = weights.get(key, 0.0) + weight
    return weights or None


def _build_balanced_sample_indices(
    items: List[Dict],
    requested_weights: Optional[Dict[str, float]],
    seed: int,
) -> Tuple[List[int], Dict]:
    natural_indices: Dict[str, List[int]] = {}
    for index, item in enumerate(items):
        key = _dataset_key(item.get("dataset") or item.get("sequence_name"))
        natural_indices.setdefault(key, []).append(index)
    natural_counts = {key: len(indices) for key, indices in sorted(natural_indices.items())}
    if not requested_weights:
        return list(range(len(items))), {
            "enabled": False,
            "natural_counts": natural_counts,
            "target_counts": natural_counts,
            "weights": {},
        }

    available = {key for key, indices in natural_indices.items() if indices}
    provided = {key: float(value) for key, value in requested_weights.items() if key in available and value > 0}
    if not provided:
        return list(range(len(items))), {
            "enabled": False,
            "natural_counts": natural_counts,
            "target_counts": natural_counts,
            "weights": {},
            "warning": "no requested dataset weights matched available datasets",
        }

    provided_sum = sum(provided.values())
    weights = dict(provided)
    missing = sorted(available - set(weights))
    if provided_sum < 1.0 and missing:
        missing_total = sum(natural_counts[key] for key in missing)
        if missing_total > 0:
            remaining = 1.0 - provided_sum
            for key in missing:
                weights[key] = remaining * natural_counts[key] / missing_total

    weight_sum = sum(weights.values())
    weights = {key: value / weight_sum for key, value in weights.items() if value > 0}
    total = len(items)
    exact = {key: total * weight for key, weight in weights.items()}
    target_counts = {key: int(np.floor(value)) for key, value in exact.items()}
    remainder = total - sum(target_counts.values())
    for key in sorted(weights, key=lambda name: (exact[name] - target_counts[name]), reverse=True)[:remainder]:
        target_counts[key] += 1

    rng = random.Random(int(seed))
    sampled: List[int] = []
    for key, target in sorted(target_counts.items()):
        indices = natural_indices.get(key, [])
        if not indices or target <= 0:
            continue
        if target <= len(indices):
            sampled.extend(rng.sample(indices, target))
        else:
            sampled.extend(rng.choice(indices) for _ in range(target))
    rng.shuffle(sampled)
    return sampled, {
        "enabled": True,
        "natural_counts": natural_counts,
        "target_counts": {key: int(value) for key, value in sorted(target_counts.items())},
        "weights": {key: float(value) for key, value in sorted(weights.items())},
        "virtual_total": len(sampled),
    }


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
            position, position_valid, rotation, rotation_valid = _read_pose_position_rotation(pose_path)
            items.append(
                _item(
                    "Matterport3D",
                    f"{scan}_{room_id}_{pano_id}",
                    rgb_path,
                    depth_path,
                    position,
                    scale,
                    position_valid=position_valid,
                    rotation_c2w=rotation,
                    rotation_valid=rotation_valid,
                    scene_group_key=_join_scene_key("Matterport3D", scan, room_id),
                )
            )
    return items


def _index_stanford2d3ds(root: Path, split: str, scale: float) -> List[Dict]:
    index_path = root / "cache" / f"2d3ds_{split}_index.json"
    rows = _read_json_list(index_path)
    items: List[Dict] = []
    for row in rows:
        area, room_id, room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = _first_match(root / str(area) / "pano" / "rgb", f"camera_{pano_id}_*_rgb.png")
            depth_path = _first_match(root / str(area) / "pano" / "depth", f"camera_{pano_id}_*_depth.png")
            pose_path = _first_match(root / str(area) / "pano" / "pose", f"camera_{pano_id}_*_pose.json")
            if rgb_path is None or depth_path is None:
                continue
            position, position_valid, rotation, rotation_valid = _read_stanford_pose(pose_path)
            items.append(
                _item(
                    "Stanford2D3DS",
                    f"{area}_{room_name}_{pano_id}",
                    rgb_path,
                    depth_path,
                    position,
                    scale,
                    position_valid=position_valid,
                    rotation_c2w=rotation,
                    rotation_valid=rotation_valid,
                    scene_group_key=_join_scene_key("Stanford2D3DS", area, room_id),
                )
            )
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
            position_path = pano_dir / "camera_xyz.txt"
            position, position_valid = _read_structured3d_position(position_path)
            items.append(
                _item(
                    "Structured3D",
                    f"{scene}_{pano_id}",
                    rgb_path,
                    depth_path,
                    position,
                    scale,
                    position_valid=position_valid,
                    rotation_c2w=_identity_rotation(),
                    rotation_valid=False,
                    scene_group_key=_join_scene_key("Structured3D", scene),
                )
            )
    return items


def _index_panocity_official(root: Path, split: str, scale: float) -> List[Dict]:
    index_path = root / "cache" / f"panocity_{split}_index.json"
    rows = _read_json_list(index_path)
    if not rows and split != "all":
        raise FileNotFoundError(
            f"Panocity {split} index not found: {index_path}. "
            "Run scripts/build_panocity_official_index.py or scripts/build_mixed4_official_indexes.py "
            "so train/val/test follow the official split instead of falling back to all data."
        )
    if not rows and split == "all":
        rows = build_panocity_official_rows(root)
    items: List[Dict] = []
    pose_cache: Dict[Path, Dict[str, List[float]]] = {}
    pose_path_cache: Dict[Tuple[str, str, str], Path] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rgb_path = _resolve_cached_path(root, row.get("rgb_path"))
        depth_path = _resolve_cached_path(root, row.get("depth_path"))
        if rgb_path is None or depth_path is None:
            continue
        position, position_valid, rotation, rotation_valid = _panocity_pose_from_row(
            root,
            row,
            rgb_path,
            depth_path,
            pose_cache,
            pose_path_cache,
        )
        city = str(row.get("city") or row.get("scene") or "")
        block = str(row.get("block") or "")
        items.append(
            _item(
                "Panocity",
                str(row.get("scene_name") or rgb_path.stem),
                rgb_path,
                depth_path,
                position,
                scale,
                position_valid=position_valid,
                rotation_c2w=rotation,
                rotation_valid=rotation_valid,
                scene_group_key=str(row.get("scene_group_key") or _join_scene_key("Panocity", city, block)),
            )
        )
    return items


def build_panocity_official_rows(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for pose_path in sorted(root.glob("*/*/*_poses.json")):
        block_dir = pose_path.parent
        city = block_dir.parent.name
        block = block_dir.name
        try:
            payload = json.loads(pose_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARN] skipping malformed Panocity pose file {pose_path}: {exc}")
            continue
        frames = payload.get("frames", []) if isinstance(payload, dict) else []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            rgb_name = frame.get("name")
            depth_name = frame.get("depth")
            if not rgb_name or not depth_name:
                continue
            rgb_path = block_dir / "pano_images" / str(rgb_name)
            depth_path = block_dir / "panodepth_images" / str(depth_name)
            if not rgb_path.exists() or not depth_path.exists():
                continue
            matrix = frame.get("transformation_matrix") or []
            rotation = _rotation_from_matrix_c2w(matrix)
            position = _translation_from_matrix(matrix)
            rows.append(
                {
                    "dataset": "Panocity",
                    "city": city,
                    "block": block,
                    "scene_name": f"{city}_{block}_{Path(str(rgb_name)).stem}",
                    "scene_group_key": _join_scene_key("Panocity", city, block),
                    "rgb_path": str(rgb_path.relative_to(root)),
                    "depth_path": str(depth_path.relative_to(root)),
                    "pano_position_m": position,
                    "pano_position_valid": _matrix_has_translation(matrix),
                    "pano_rotation_c2w": rotation,
                    "pano_rotation_valid": rotation is not None,
                }
            )
    return rows


def _item(
    sequence_name: str,
    scene_name: str,
    rgb_path: Path,
    depth_path: Path,
    position: List[float],
    scale: float,
    position_valid: bool = True,
    rotation_c2w: Optional[List[List[float]]] = None,
    rotation_valid: bool = False,
    scene_group_key: Optional[str] = None,
) -> Dict:
    return {
        "dataset": sequence_name,
        "sequence_name": sequence_name,
        "scene_name": scene_name,
        "scene_group_key": scene_group_key,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "pano_position_m": position,
        "pano_position_valid": bool(position_valid),
        "pano_rotation_c2w": rotation_c2w if rotation_c2w is not None else _identity_rotation(),
        "pano_rotation_valid": bool(rotation_valid and rotation_c2w is not None),
        "output_depth_scale": scale,
    }


def _resolve_cached_path(root: Path, value: object) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _translation_from_matrix(matrix: object) -> List[float]:
    try:
        if len(matrix) >= 3:
            return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
    except (TypeError, ValueError, IndexError):
        pass
    return [0.0, 0.0, 0.0]


def _matrix_has_translation(matrix: object) -> bool:
    try:
        return len(matrix) >= 3 and all(len(matrix[index]) >= 4 for index in range(3))
    except (TypeError, IndexError):
        return False


def _rotation_from_matrix_c2w(matrix: object) -> Optional[List[List[float]]]:
    try:
        if len(matrix) >= 3:
            rotation = [[float(matrix[row][col]) for col in range(3)] for row in range(3)]
            return _orthonormalize_rotation(rotation)
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _orthonormalize_rotation(rotation: List[List[float]]) -> List[List[float]]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return _identity_rotation()
    u, _s, vh = np.linalg.svd(matrix)
    ortho = u @ vh
    if np.linalg.det(ortho) < 0:
        u[:, -1] *= -1.0
        ortho = u @ vh
    return [[float(value) for value in row] for row in ortho.astype(np.float32)]


def _identity_rotation() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _read_pose_position_rotation(path: Path) -> Tuple[List[float], bool, List[List[float]], bool]:
    if not path.exists():
        return [0.0, 0.0, 0.0], False, _identity_rotation(), False
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape == (4, 4):
        matrix = _MP3D_WORLD_TO_OPENCV @ matrix @ np.linalg.inv(_MP3D_CAMERA_TO_OPENCV)
        rotation = _rotation_from_matrix_c2w(matrix)
        return (
            [float(value) for value in matrix[:3, 3]],
            True,
            rotation or _identity_rotation(),
            rotation is not None,
        )
    return [0.0, 0.0, 0.0], False, _identity_rotation(), False


def _panocity_pose_from_row(
    root: Path,
    row: Dict,
    rgb_path: Path,
    depth_path: Path,
    pose_cache: Dict[Path, Dict[str, Dict]],
    pose_path_cache: Dict[Tuple[str, str, str], Path],
) -> Tuple[List[float], bool, List[List[float]], bool]:
    position = _coerce_position(row.get("pano_position_m"))
    position_valid = bool(row.get("pano_position_valid", row.get("pano_position_m") is not None))
    rotation = _coerce_rotation(row.get("pano_rotation_c2w"))
    rotation_valid = bool(row.get("pano_rotation_valid", rotation is not None))
    if position_valid and rotation is not None and rotation_valid:
        return position, True, rotation, True

    pose_path = _panocity_pose_path(root, row, rgb_path, pose_path_cache)
    pose_records = _load_panocity_pose_records(pose_path, pose_cache)
    for key in (rgb_path.name, depth_path.name, rgb_path.stem, depth_path.stem):
        if key in pose_records:
            record = pose_records[key]
            return (
                record.get("position", position),
                bool(record.get("position_valid", position_valid)),
                record.get("rotation_c2w", rotation or _identity_rotation()),
                bool(record.get("rotation_valid", False)),
            )
    return (
        position,
        position_valid,
        rotation or _identity_rotation(),
        bool(rotation_valid and rotation is not None),
    )


def _coerce_position(raw: object) -> List[float]:
    try:
        values = list(raw) if raw is not None else []
        if len(values) >= 3:
            return [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        pass
    return [0.0, 0.0, 0.0]


def _coerce_rotation(raw: object) -> Optional[List[List[float]]]:
    try:
        rows = list(raw) if raw is not None else []
        if len(rows) >= 3:
            rotation = [[float(rows[row][col]) for col in range(3)] for row in range(3)]
            return _orthonormalize_rotation(rotation)
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _is_zero_position(position: List[float]) -> bool:
    return all(abs(float(value)) <= 1e-8 for value in position[:3])


def _panocity_pose_path(
    root: Path,
    row: Dict,
    rgb_path: Path,
    pose_path_cache: Dict[Tuple[str, str, str], Path],
) -> Path:
    pose_value = row.get("poses_file")
    if pose_value not in (None, ""):
        pose_path = Path(str(pose_value))
        return pose_path if pose_path.is_absolute() else root / pose_path
    block_dir = rgb_path.parents[1] if rgb_path.parent.name == "pano_images" else rgb_path.parent
    city = str(row.get("city") or block_dir.parent.name)
    block = str(row.get("block") or block_dir.name)
    cache_key = (str(block_dir), city, block)
    if cache_key in pose_path_cache:
        return pose_path_cache[cache_key]
    block_suffix = block[len(city) + 1 :] if block.startswith(f"{city}_") else block
    direct = block_dir / f"{city}_Pano_{block_suffix}_poses.json"
    if direct.exists():
        pose_path_cache[cache_key] = direct
        return direct
    candidates = sorted(block_dir.glob("*_poses.json"))
    resolved = candidates[0] if candidates else block_dir / "_missing_poses.json"
    pose_path_cache[cache_key] = resolved
    return resolved


def _load_panocity_pose_records(
    pose_path: Path,
    cache: Dict[Path, Dict[str, Dict]],
) -> Dict[str, Dict]:
    if not pose_path.is_file():
        return {}
    pose_path = pose_path.resolve()
    if pose_path in cache:
        return cache[pose_path]
    try:
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] skipping malformed Panocity pose file {pose_path}: {exc}")
        cache[pose_path] = {}
        return cache[pose_path]

    records: Dict[str, Dict] = {}
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        matrix = frame.get("transformation_matrix") or []
        rotation = _rotation_from_matrix_c2w(matrix)
        record = {
            "position": _translation_from_matrix(matrix),
            "position_valid": _matrix_has_translation(matrix),
            "rotation_c2w": rotation or _identity_rotation(),
            "rotation_valid": rotation is not None,
        }
        for key in (frame.get("name"), frame.get("depth")):
            if not key:
                continue
            path_key = Path(str(key))
            records[str(key)] = record
            records[path_key.name] = record
            records[path_key.stem] = record
    cache[pose_path] = records
    return records


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


def _read_stanford_pose(path: Optional[Path]) -> Tuple[List[float], bool, List[List[float]], bool]:
    if path is None or not path.exists():
        return [0.0, 0.0, 0.0], False, _identity_rotation(), False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [0.0, 0.0, 0.0], False, _identity_rotation(), False
    position = [0.0, 0.0, 0.0]
    position_valid = False
    for key in ("camera_location", "position", "translation"):
        value = data.get(key)
        if isinstance(value, list) and len(value) >= 3:
            position = [float(value[0]), float(value[1]), float(value[2])]
            position_valid = True
            break
    rt_matrix = data.get("camera_rt_matrix")
    rotation = None
    try:
        rt = np.asarray(rt_matrix, dtype=np.float32)
        if rt.shape[0] >= 3 and rt.shape[1] >= 3:
            # Stanford stores camera_rt_matrix as world-to-camera. Convert it to
            # camera-to-world so all pano rotations share the same convention.
            rotation = _orthonormalize_rotation(rt[:3, :3].T.tolist())
    except (TypeError, ValueError, IndexError):
        rotation = None
    return position, position_valid, rotation or _identity_rotation(), rotation is not None


def _read_structured3d_position(path: Path) -> Tuple[List[float], bool]:
    if not path.exists():
        return [0.0, 0.0, 0.0], False
    try:
        values = [float(value) for value in path.read_text(encoding="utf-8").split()[:3]]
    except (OSError, ValueError):
        return [0.0, 0.0, 0.0], False
    if len(values) != 3:
        return [0.0, 0.0, 0.0], False
    position = np.asarray([value / 1000.0 for value in values], dtype=np.float32)
    position = (_S3D_WORLD_TO_OPENCV[:3, :3] @ position[:, None])[:, 0]
    return [float(value) for value in position], True


def _read_rgb_tensor(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (FileNotFoundError, OSError, ValueError, SyntaxError) as exc:
        raise OSError(f"Cannot read RGB image: {path}: {exc}") from exc
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _read_depth_tensor(path: Path, output_depth_scale: float, invalid_depth_value: Optional[float]) -> torch.Tensor:
    try:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except (cv2.error, OSError, ValueError, SyntaxError) as exc:
        raise OSError(f"Cannot read depth map: {path}: {exc}") from exc
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32) / float(output_depth_scale)
    if invalid_depth_value is not None:
        depth_m[depth.astype(np.float32) >= float(invalid_depth_value)] = np.inf
    depth_m[depth_m <= 0] = np.inf
    return torch.from_numpy(depth_m)[None].contiguous()
