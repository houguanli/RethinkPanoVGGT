import glob
import json
import math
import os
import os.path as osp
import random
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points


class PanoCityPairedPinholeDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        root: str = "../../panovggt/PanoCity_paired",
        len_train: int = 100000,
        len_test: int = 10000,
        output_depth_scale: float = 1000.0,
        invalid_depth_value: Optional[float] = 65535.0,
        depth_max_m: float = 80.0,
        num_yaw: int = 8,
        pitch_degrees: float = 0.0,
        fov_degrees: float = 75.0,
        max_samples: Optional[int] = None,
        train_split_fraction: float = 0.95,
        split_seed: int = 42,
        metadata_path: Optional[str] = None,
        bad_sample_list: Optional[str] = None,
        curriculum_bins: Optional[str | Iterable[str]] = None,
        use_metadata_weights: bool = True,
    ):
        super().__init__(common_conf=common_conf)
        self.root = root
        self.split = str(split)
        self.training = common_conf.training
        self.len_train = len_train if split == "train" else len_test
        self.train_split_fraction = float(train_split_fraction)
        self.split_seed = int(split_seed)
        self.metadata_path = metadata_path
        self.bad_sample_list = bad_sample_list
        self.curriculum_bins = _parse_bins(curriculum_bins)
        self.use_metadata_weights = bool(use_metadata_weights)
        self.metadata_by_name = _load_metadata(metadata_path, self.root)
        self.bad_samples = _load_bad_samples(bad_sample_list, self.root)
        self.output_depth_scale = float(output_depth_scale)
        self.invalid_depth_value = None if invalid_depth_value is None else float(invalid_depth_value)
        self.depth_max_m = float(depth_max_m)
        self.num_yaw = int(num_yaw)
        self.pitch_degrees = float(pitch_degrees)
        self.fov_degrees = float(fov_degrees)
        self.items = self._build_index(max_samples=max_samples)
        if not self.items:
            raise FileNotFoundError(f"No PanoCity paired samples found under {self.root}/rgb and {self.root}/depth")

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        if seq_index is None:
            seq_index = 0
        item_index = self._resolve_item_index(int(seq_index))
        item, image, range_depth = self._read_item_with_fallback(item_index)

        target_shape = self.get_target_shape(aspect_ratio)
        height, width = int(target_shape[0]), int(target_shape[1])
        view_count = int(img_per_seq) if img_per_seq is not None else self.num_yaw
        view_count = max(1, view_count)
        yaw_values = np.linspace(0.0, 2.0 * math.pi, view_count, endpoint=False, dtype=np.float32)
        pitch = math.radians(self.pitch_degrees)
        fov_x = math.radians(self.fov_degrees)
        fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) * float(height) / float(width))

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for yaw in yaw_values:
            window_image, window_depth, intrinsic, extrinsic = _sample_pinhole_window(
                image=image,
                range_depth=range_depth,
                yaw=float(yaw),
                pitch=pitch,
                fov_x=fov_x,
                fov_y=fov_y,
                height=height,
                width=width,
            )
            world_coords, cam_coords, point_mask = depth_to_world_coords_points(window_depth, extrinsic, intrinsic)
            images.append(window_image)
            depths.append(window_depth.astype(np.float32))
            extrinsics.append(extrinsic.astype(np.float32))
            intrinsics.append(intrinsic.astype(np.float32))
            cam_points.append(cam_coords.astype(np.float32))
            world_points.append(world_coords.astype(np.float32))
            point_masks.append(point_mask)
            original_sizes.append(np.array([height, width], dtype=np.int32))

        sample_weight = float(item.get("sample_weight", 1.0))
        metadata_valid_ratio = float(item.get("metadata_valid_ratio", 1.0))
        metadata_structure_score = float(item.get("metadata_structure_score", 0.0))
        return {
            "seq_name": "panocity_paired_" + item["name"],
            "ids": np.arange(view_count, dtype=np.int64),
            "frame_num": view_count,
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "sample_weight": np.full((view_count,), sample_weight, dtype=np.float32),
            "metadata_valid_ratio": np.full((view_count,), metadata_valid_ratio, dtype=np.float32),
            "metadata_structure_score": np.full((view_count,), metadata_structure_score, dtype=np.float32),
            "metadata_quality_bin": str(item.get("metadata_quality_bin", "unknown")),
        }

    def _resolve_item_index(self, seq_index: int) -> int:
        return int(seq_index) % len(self.items)

    def _build_index(self, max_samples: Optional[int] = None):
        rgb_dir = osp.join(self.root, "rgb")
        depth_dir = osp.join(self.root, "depth")
        items = []
        for rgb_path in sorted(glob.glob(osp.join(rgb_dir, "*"))):
            if osp.splitext(rgb_path)[1].lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            depth_path = _depth_path_for_rgb(rgb_path, depth_dir)
            if depth_path is None:
                continue
            if _is_bad_sample(rgb_path, depth_path, self.bad_samples):
                continue
            items.append({"rgb_path": rgb_path, "depth_path": depth_path, "name": osp.splitext(osp.basename(rgb_path))[0]})
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

    def _read_item_with_fallback(self, item_index: int):
        last_error: Exception | None = None
        for offset in range(len(self.items)):
            candidate = self.items[(item_index + offset) % len(self.items)]
            try:
                image = _read_rgb(candidate["rgb_path"])
                range_depth = _read_depth(
                    candidate["depth_path"],
                    output_depth_scale=float(candidate.get("output_depth_scale", self.output_depth_scale)),
                    invalid_depth_value=self.invalid_depth_value,
                    depth_max_m=self.depth_max_m,
                )
                return candidate, image, range_depth
            except (FileNotFoundError, OSError, ValueError) as exc:
                last_error = exc
                print(f"[WARN] skipping unreadable PanoCity sample {candidate.get('name')}: {exc}")
        raise RuntimeError("All PanoCity samples failed to load.") from last_error


def _split_items(items: list[dict], split: str, train_fraction: float, seed: int) -> list[dict]:
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


def _load_metadata(path: Optional[str], root: str) -> dict[str, dict]:
    if path in (None, ""):
        return {}
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(root) / resolved
    if not resolved.exists():
        print(f"[WARN] PanoCity metadata not found: {resolved}; training without curriculum metadata.")
        return {}
    metadata: dict[str, dict] = {}
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


def _load_bad_samples(path: Optional[str], root: str) -> set[str]:
    if path in (None, ""):
        return set()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(root) / resolved
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


def _is_bad_sample(rgb_path: str, depth_path: str, bad_samples: set[str]) -> bool:
    if not bad_samples:
        return False
    rgb = Path(rgb_path)
    depth = Path(depth_path)
    keys = {
        str(rgb),
        str(depth),
        rgb.name,
        depth.name,
        rgb.stem,
        depth.stem,
    }
    return bool(keys & bad_samples)


def _apply_metadata(
    items: list[dict],
    metadata_by_name: dict[str, dict],
    curriculum_bins: Optional[set[str]],
    use_metadata_weights: bool,
) -> list[dict]:
    if not metadata_by_name:
        for item in items:
            item["sample_weight"] = 1.0
            item["metadata_valid_ratio"] = 1.0
            item["metadata_structure_score"] = 0.0
            item["metadata_quality_bin"] = "unknown"
        return items
    filtered = []
    for item in items:
        name = str(item["name"])
        metadata = metadata_by_name.get(name) or metadata_by_name.get(Path(item["rgb_path"]).stem)
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


def _depth_path_for_rgb(rgb_path: str, depth_dir: str) -> Optional[str]:
    name = osp.basename(rgb_path)
    stem = osp.splitext(name)[0]
    token = name.split("_", 1)[0]
    candidates = [
        osp.join(depth_dir, name),
        osp.join(depth_dir, name.replace("_rgb_", "_depth_")),
        osp.join(depth_dir, f"{token}_depth_{token}.png"),
        osp.join(depth_dir, f"{token}_pano_{token}.png"),
        osp.join(depth_dir, stem + ".png"),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return None


def _read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_depth(
    path: str,
    output_depth_scale: float,
    invalid_depth_value: Optional[float],
    depth_max_m: float,
) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    raw = depth.astype(np.float32)
    depth_m = raw / float(output_depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if invalid_depth_value is not None:
        valid &= raw < float(invalid_depth_value)
    if depth_max_m > 0:
        valid &= depth_m <= float(depth_max_m)
    return np.where(valid, depth_m, 0.0).astype(np.float32)


def _sample_pinhole_window(
    image: np.ndarray,
    range_depth: np.ndarray,
    yaw: float,
    pitch: float,
    fov_x: float,
    fov_y: float,
    height: int,
    width: int,
):
    x = (np.arange(width, dtype=np.float32) + 0.5) / float(width) * 2.0 - 1.0
    y = (np.arange(height, dtype=np.float32) + 0.5) / float(height) * 2.0 - 1.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    x_cam = xx * math.tan(fov_x * 0.5)
    y_cam = yy * math.tan(fov_y * 0.5)

    forward, right, up = _yaw_pitch_axes(yaw, pitch)
    rays = forward[None, None, :] + x_cam[..., None] * right[None, None, :] - y_cam[..., None] * up[None, None, :]
    rays = rays / np.clip(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8, None)

    theta = np.arctan2(rays[..., 0], rays[..., 2])
    phi = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    map_x = ((theta / (2.0 * math.pi) + 0.5) % 1.0) * (image.shape[1] - 1)
    map_y = np.clip(0.5 - phi / math.pi, 0.0, 1.0) * (image.shape[0] - 1)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    window_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    sampled_range = cv2.remap(range_depth, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    z_factor = np.clip(np.sum(rays * forward[None, None, :], axis=-1), 0.0, None)
    window_depth = (sampled_range * z_factor).astype(np.float32)

    fx = 0.5 * float(width) / math.tan(fov_x * 0.5)
    fy = 0.5 * float(height) / math.tan(fov_y * 0.5)
    intrinsic = np.array(
        [[fx, 0.0, (width - 1.0) * 0.5], [0.0, fy, (height - 1.0) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    down = -up
    camera_to_world = np.stack([right, down, forward], axis=1)
    world_to_camera = camera_to_world.T
    extrinsic = np.concatenate([world_to_camera, np.zeros((3, 1), dtype=np.float32)], axis=1)
    return window_image, window_depth, intrinsic, extrinsic


def _yaw_pitch_axes(yaw: float, pitch: float):
    cos_pitch = math.cos(pitch)
    forward = np.array([cos_pitch * math.sin(yaw), math.sin(pitch), cos_pitch * math.cos(yaw)], dtype=np.float32)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)
    up = np.cross(forward, right)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return forward, right, up
