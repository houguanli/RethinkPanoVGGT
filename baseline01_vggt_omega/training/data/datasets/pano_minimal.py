import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points
from data.datasets.panocity_paired import (
    PanoCityPairedPinholeDataset,
    _apply_metadata,
    _is_bad_sample,
    _sample_pinhole_window,
)

_PANOCITY_DEPTH_SCALE = 100.0
_MATTERPORT3D_DEPTH_SCALE = 4000.0
_STANFORD2D3DS_DEPTH_SCALE = 512.0
_STRUCTURED3D_DEPTH_SCALE = 1000.0


class PanoMinimalPinholeDataset(PanoCityPairedPinholeDataset):
    """Pinhole-window reader for the official PanoVGGT minimal dataset bundle.

    Depth scales follow the official PanoVGGT readers:
    Panocity /100, Matterport3D /4000, Stanford2D3DS /512, Structured3D /1000.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        root: str = "../../panovggt",
        len_train: int = 100000,
        len_test: int = 10000,
        datasets: Optional[str | Iterable[str]] = "all",
        dataset_sampling_weights: Optional[str | dict[str, float]] = None,
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
        self.dataset_names = _parse_dataset_names(datasets)
        self.dataset_sampling_weights = _parse_dataset_sampling_weights(dataset_sampling_weights)
        super().__init__(
            common_conf=common_conf,
            split=split,
            root=root,
            len_train=len_train,
            len_test=len_test,
            output_depth_scale=output_depth_scale,
            invalid_depth_value=invalid_depth_value,
            depth_max_m=depth_max_m,
            num_yaw=num_yaw,
            pitch_degrees=pitch_degrees,
            fov_degrees=fov_degrees,
            max_samples=max_samples,
            train_split_fraction=train_split_fraction,
            split_seed=split_seed,
            metadata_path=metadata_path,
            bad_sample_list=bad_sample_list,
            curriculum_bins=curriculum_bins,
            use_metadata_weights=use_metadata_weights,
        )
        self.balanced_sample_indices, self.dataset_sampling_summary = _build_balanced_sample_indices(
            self.items,
            self.dataset_sampling_weights if self.split == "train" else None,
            seed=self.split_seed,
        )
        if self.dataset_sampling_summary.get("enabled"):
            print(f"[INFO] PanoMinimal dataset_sampling = {self.dataset_sampling_summary}")

    def _resolve_item_index(self, seq_index: int) -> int:
        if getattr(self, "balanced_sample_indices", None):
            return self.balanced_sample_indices[int(seq_index) % len(self.balanced_sample_indices)]
        return int(seq_index) % len(self.items)

    def _build_index(self, max_samples: Optional[int] = None):
        root = Path(self.root)
        items = []
        if "matterport3d" in self.dataset_names:
            items.extend(_index_matterport3d(root / "Matterport3D", self.split))
        if "stanford2d3ds" in self.dataset_names:
            items.extend(_index_stanford2d3ds(root / "Stanford2D3DS", self.split))
        if "structured3d" in self.dataset_names:
            items.extend(_index_structured3d(root / "Structured3D", self.split))
        if "panocity" in self.dataset_names:
            items.extend(_index_panocity_official(root / "Panocity", self.split))

        if self.bad_samples:
            items = [
                item
                for item in items
                if not _is_bad_sample(str(item["rgb_path"]), str(item["depth_path"]), self.bad_samples)
            ]
        items = _apply_metadata(
            items,
            metadata_by_name=self.metadata_by_name,
            curriculum_bins=self.curriculum_bins,
            use_metadata_weights=self.use_metadata_weights,
        )
        if max_samples is not None:
            items = items[: int(max_samples)]
        return items


class PanoMinimalMultiPanoPinholeDataset(BaseDataset):
    """Flatten validated multi-pano samples into Omega-style pinhole multiviews.

    The main multi-pano reader owns scene grouping, split handling, and pose
    conventions. This adapter keeps that input logic and only converts each
    pano group into a regular VGGT-Omega sequence of pinhole windows.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        root: str = "../../panovggt",
        len_train: int = 100000,
        len_test: int = 10000,
        datasets: Optional[str | Iterable[str]] = "all",
        dataset_sampling_weights: Optional[str | dict[str, float]] = None,
        output_depth_scale: float = 1000.0,
        invalid_depth_value: Optional[float] = 65535.0,
        depth_max_m: float = 80.0,
        windows_per_pano: int = 4,
        pitch_degrees: float = -15.0,
        fov_degrees: float = 75.0,
        max_samples: Optional[int] = None,
        train_split_fraction: float = 0.95,
        split_seed: int = 42,
        bad_sample_list: Optional[str] = None,
        pano_sample_mode: str = "fixed_neighborhood",
        pano_min_count: int = 2,
        pano_max_count: int = 2,
        dataset_pano_counts: Optional[str | dict[str, int]] = None,
        grouping: str = "nearest",
        camera_supervised_datasets: Optional[str | Iterable[str]] = "panocity,stanford2d3ds",
        main_dataset_path: Optional[str] = None,
    ):
        super().__init__(common_conf=common_conf)
        self.root = root
        self.split = str(split)
        self.training = common_conf.training
        self.len_train = int(len_train if split == "train" else len_test)
        self.len_test = int(len_test)
        self.split_seed = int(split_seed)
        self.windows_per_pano = max(1, int(windows_per_pano))
        self.pitch_degrees = float(pitch_degrees)
        self.fov_degrees = float(fov_degrees)
        self.pano_min_count = max(1, int(pano_min_count))
        self.pano_max_count = max(self.pano_min_count, int(pano_max_count))
        self.dataset_pano_counts = _parse_dataset_pano_counts(dataset_pano_counts)
        self.depth_max_m = float(depth_max_m)
        self.camera_supervised_datasets = _parse_dataset_names(camera_supervised_datasets)
        self._runtime_dataset_kwargs = {
            "root": root,
            "datasets": datasets,
            "dataset_sampling_weights": dataset_sampling_weights,
            "output_depth_scale": output_depth_scale,
            "invalid_depth_value": invalid_depth_value,
            "max_samples": max_samples,
            "train_split_fraction": train_split_fraction,
            "bad_sample_list": bad_sample_list,
            "pano_sample_mode": pano_sample_mode,
            "grouping": grouping,
            "main_dataset_path": main_dataset_path,
        }
        self.pano_dataset = self._build_runtime_dataset()
        if len(self.pano_dataset) <= 0:
            raise FileNotFoundError(f"No multi-pano samples found under {root}")
        summary = getattr(self.pano_dataset, "dataset_sampling_summary", None)
        if summary and summary.get("enabled"):
            print(f"[INFO] Omega multi-pano source sampling = {summary}")

    def _build_runtime_dataset(self):
        kwargs = self._runtime_dataset_kwargs
        module = _load_main_pano_minimal_module(kwargs["main_dataset_path"])
        if self.dataset_pano_counts:
            runtime_datasets = []
            for dataset_name in _ordered_dataset_names(_parse_dataset_names(kwargs["datasets"])):
                pano_count = int(self.dataset_pano_counts.get(dataset_name, self.pano_max_count))
                if pano_count <= 0:
                    continue
                runtime_dataset = module.PanoMinimalDataset(
                    root=kwargs["root"],
                    pano_sample_mode="fixed_neighborhood" if pano_count > 1 else "single",
                    pano_min_count=pano_count,
                    pano_max_count=pano_count,
                    grouping=kwargs["grouping"],
                    max_samples=kwargs["max_samples"],
                    split=self.split,
                    train_split_fraction=kwargs["train_split_fraction"],
                    split_seed=self.split_seed,
                    datasets=dataset_name,
                    bad_sample_list=kwargs["bad_sample_list"],
                    dataset_sampling_weights=None,
                    output_depth_scale=kwargs["output_depth_scale"],
                    invalid_depth_value=kwargs["invalid_depth_value"],
                    strict=False,
                )
                if len(runtime_dataset) > 0:
                    runtime_datasets.append((dataset_name, runtime_dataset))
            if not runtime_datasets:
                raise FileNotFoundError(
                    f"No dataset-specific multi-pano samples found under {kwargs['root']} "
                    f"for counts {self.dataset_pano_counts}."
                )
            pano_dataset = _WeightedRuntimeMixedPanoDataset(
                runtime_datasets,
                kwargs["dataset_sampling_weights"] if self.split == "train" else None,
                seed=self.split_seed,
            )
            self.pano_min_count = min(int(self.dataset_pano_counts.get(key, self.pano_max_count)) for key, _ in runtime_datasets)
            self.pano_max_count = max(int(self.dataset_pano_counts.get(key, self.pano_max_count)) for key, _ in runtime_datasets)
        else:
            pano_dataset = module.PanoMinimalDataset(
                root=kwargs["root"],
                pano_sample_mode=kwargs["pano_sample_mode"],
                pano_min_count=self.pano_min_count,
                pano_max_count=self.pano_max_count,
                grouping=kwargs["grouping"],
                max_samples=kwargs["max_samples"],
                split=self.split,
                train_split_fraction=kwargs["train_split_fraction"],
                split_seed=self.split_seed,
                datasets=kwargs["datasets"],
                bad_sample_list=kwargs["bad_sample_list"],
                dataset_sampling_weights=kwargs["dataset_sampling_weights"],
                output_depth_scale=kwargs["output_depth_scale"],
                invalid_depth_value=kwargs["invalid_depth_value"],
                strict=True,
            )
        return pano_dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        # The main dataset is loaded from a runtime path, so its class is not
        # importable while Python 3.14 forkserver workers unpickle this adapter.
        state["pano_dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pano_dataset = self._build_runtime_dataset()

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        if seq_index is None:
            seq_index = 0
        group = self.pano_dataset[int(seq_index) % len(self.pano_dataset)]
        pano_images = group["pano_image"]
        pano_depths = group["pano_depth"]
        if pano_images.ndim == 3:
            pano_images = pano_images[None]
            pano_depths = pano_depths[None]

        target_pano_count = self._target_pano_count(group)
        desired_views = int(img_per_seq) if img_per_seq is not None else self.pano_max_count * self.windows_per_pano
        max_panos_by_views = max(1, desired_views // self.windows_per_pano)
        if target_pano_count is not None:
            pano_count = min(int(pano_images.shape[0]), int(target_pano_count))
        else:
            pano_count = min(int(pano_images.shape[0]), self.pano_max_count, max_panos_by_views)
        pano_count = max(1, pano_count)
        view_count = pano_count * self.windows_per_pano

        target_shape = self.get_target_shape(aspect_ratio)
        height, width = int(target_shape[0]), int(target_shape[1])
        yaw_values = np.linspace(0.0, 2.0 * math.pi, self.windows_per_pano, endpoint=False, dtype=np.float32)
        pitch = math.radians(self.pitch_degrees)
        fov_x = math.radians(self.fov_degrees)
        fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) * float(height) / float(width))

        dataset_names = _as_string_list(group.get("sequence_name"), pano_count)
        camera_dataset_enabled = all(_dataset_key(name) in self.camera_supervised_datasets for name in dataset_names)
        translation_valid = _as_bool_array(group.get("pano_translation_valid"), pano_count, default=False)
        rotation_valid = _as_bool_array(group.get("pano_rotation_valid"), pano_count, default=False)
        sample_weight_by_pano = _as_float_array(group.get("sample_weight"), pano_count, default=1.0)
        valid_ratio_by_pano = _as_float_array(group.get("metadata_valid_ratio"), pano_count, default=1.0)
        structure_by_pano = _as_float_array(group.get("metadata_structure_score"), pano_count, default=0.0)
        quality_bins = _as_string_list(group.get("metadata_quality_bin"), pano_count, default="unknown")

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        camera_valid = []
        camera_weight = []
        view_pano_index = []
        view_window_index = []
        view_yaw = []
        view_pitch = []
        view_fov_x = []
        view_fov_y = []
        pano_range_depths = []

        for pano_idx in range(pano_count):
            image = _pano_image_to_uint8_numpy(pano_images[pano_idx])
            range_depth = _pano_depth_to_numpy(pano_depths[pano_idx], depth_max_m=self.depth_max_m)
            pano_range_depths.append(range_depth)
            pano_c2w = _pano_c2w_from_group(group, pano_idx)
            pano_camera_valid = bool(camera_dataset_enabled and translation_valid[pano_idx] and rotation_valid[pano_idx])

            for window_idx, yaw in enumerate(yaw_values):
                window_image, window_depth, intrinsic, local_extrinsic = _sample_pinhole_window(
                    image=image,
                    range_depth=range_depth,
                    yaw=float(yaw),
                    pitch=pitch,
                    fov_x=fov_x,
                    fov_y=fov_y,
                    height=height,
                    width=width,
                )
                local_c2w = _extrinsic_w2c_to_c2w(local_extrinsic)
                world_c2w = pano_c2w @ local_c2w
                extrinsic = _c2w_to_extrinsic_w2c(world_c2w)
                world_coords, cam_coords, point_mask = depth_to_world_coords_points(window_depth, extrinsic, intrinsic)

                images.append(window_image)
                depths.append(window_depth.astype(np.float32))
                extrinsics.append(extrinsic.astype(np.float32))
                intrinsics.append(intrinsic.astype(np.float32))
                cam_points.append(cam_coords.astype(np.float32))
                world_points.append(world_coords.astype(np.float32))
                point_masks.append(point_mask)
                original_sizes.append(np.array([height, width], dtype=np.int32))
                camera_valid.append(pano_camera_valid)
                camera_weight.append(float(sample_weight_by_pano[pano_idx]))
                view_pano_index.append(pano_idx)
                view_window_index.append(window_idx)
                view_yaw.append(float(yaw))
                view_pitch.append(float(pitch))
                view_fov_x.append(float(fov_x))
                view_fov_y.append(float(fov_y))

        scene_name = group.get("scene_name", "")
        if isinstance(scene_name, (list, tuple)):
            scene_name = "|".join(str(value) for value in scene_name)
        return {
            "seq_name": "omega_multipano_" + str(scene_name),
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
            "sample_weight": np.repeat(sample_weight_by_pano[:pano_count], self.windows_per_pano).astype(np.float32),
            "metadata_valid_ratio": np.repeat(valid_ratio_by_pano[:pano_count], self.windows_per_pano).astype(np.float32),
            "metadata_structure_score": np.repeat(structure_by_pano[:pano_count], self.windows_per_pano).astype(np.float32),
            "metadata_quality_bin": "|".join(quality_bins[:pano_count]),
            "camera_valid": np.asarray(camera_valid, dtype=np.bool_),
            "camera_weight": np.asarray(camera_weight, dtype=np.float32),
            "view_pano_index": np.asarray(view_pano_index, dtype=np.int64),
            "view_window_index": np.asarray(view_window_index, dtype=np.int64),
            "view_yaw": np.asarray(view_yaw, dtype=np.float32),
            "view_pitch": np.asarray(view_pitch, dtype=np.float32),
            "view_fov_x": np.asarray(view_fov_x, dtype=np.float32),
            "view_fov_y": np.asarray(view_fov_y, dtype=np.float32),
            "pano_range_depth_erp": np.stack(pano_range_depths).astype(np.float32),
            "pano_count": np.full((view_count,), pano_count, dtype=np.int64),
            "windows_per_pano": np.full((view_count,), self.windows_per_pano, dtype=np.int64),
        }

    def _target_pano_count(self, group: dict) -> Optional[int]:
        if not self.dataset_pano_counts:
            return None
        sequence_names = _as_string_list(group.get("sequence_name"), 1)
        dataset_key = _dataset_key(sequence_names[0] if sequence_names else "")
        count = self.dataset_pano_counts.get(dataset_key)
        return int(count) if count is not None else None


class _WeightedRuntimeMixedPanoDataset:
    """Mix exact-count per-dataset PanoMinimalDataset instances."""

    def __init__(
        self,
        datasets: list[tuple[str, object]],
        dataset_sampling_weights: Optional[str | dict[str, float]],
        seed: int,
    ) -> None:
        self.datasets = [(key, dataset) for key, dataset in datasets if len(dataset) > 0]
        if not self.datasets:
            raise FileNotFoundError("No non-empty runtime pano datasets.")
        self.dataset_sampling_weights = _parse_dataset_sampling_weights(dataset_sampling_weights)
        self.sample_indices, self.dataset_sampling_summary = self._build_indices(int(seed))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int):
        dataset_index, local_index = self.sample_indices[int(index) % len(self.sample_indices)]
        return self.datasets[dataset_index][1][local_index % len(self.datasets[dataset_index][1])]

    def _build_indices(self, seed: int) -> tuple[list[tuple[int, int]], dict]:
        natural_counts = {key: len(dataset) for key, dataset in self.datasets}
        natural = [
            (dataset_index, local_index)
            for dataset_index, (_key, dataset) in enumerate(self.datasets)
            for local_index in range(len(dataset))
        ]
        if not self.dataset_sampling_weights:
            return natural, {
                "enabled": False,
                "natural_counts": natural_counts,
                "target_counts": natural_counts,
                "weights": {},
            }

        available = {key for key, count in natural_counts.items() if count > 0}
        provided = {
            key: float(value)
            for key, value in self.dataset_sampling_weights.items()
            if key in available and float(value) > 0
        }
        if not provided:
            return natural, {
                "enabled": False,
                "natural_counts": natural_counts,
                "target_counts": natural_counts,
                "weights": {},
                "warning": "no requested dataset weights matched available datasets",
            }
        weight_sum = sum(provided.values())
        weights = {key: value / weight_sum for key, value in provided.items()}
        total = sum(natural_counts.values())
        exact = {key: total * weights[key] for key in weights}
        target_counts = {key: int(math.floor(value)) for key, value in exact.items()}
        remainder = total - sum(target_counts.values())
        for key in sorted(weights, key=lambda name: (exact[name] - target_counts[name]), reverse=True)[:remainder]:
            target_counts[key] += 1

        rng = random.Random(int(seed))
        mixed: list[tuple[int, int]] = []
        dataset_index_by_key = {key: idx for idx, (key, _dataset) in enumerate(self.datasets)}
        for key, target in sorted(target_counts.items()):
            dataset_index = dataset_index_by_key[key]
            dataset_len = len(self.datasets[dataset_index][1])
            if target <= dataset_len:
                local_indices = rng.sample(range(dataset_len), target)
            else:
                local_indices = [rng.randrange(dataset_len) for _ in range(target)]
            mixed.extend((dataset_index, local_index) for local_index in local_indices)
        rng.shuffle(mixed)
        return mixed, {
            "enabled": True,
            "natural_counts": natural_counts,
            "target_counts": {key: int(value) for key, value in sorted(target_counts.items())},
            "weights": {key: float(value) for key, value in sorted(weights.items())},
            "virtual_total": len(mixed),
        }


def _load_main_pano_minimal_module(main_dataset_path: Optional[str]):
    if main_dataset_path not in (None, ""):
        module_path = Path(str(main_dataset_path))
    else:
        module_path = (
            Path(__file__).resolve().parents[4]
            / "Rethink_pano_new_exp_omega"
            / "training"
            / "data"
            / "pano_minimal.py"
        )
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find main multi-pano dataset module: {module_path}")
    spec = importlib.util.spec_from_file_location("_runtime_multipano_pano_minimal", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pano_image_to_uint8_numpy(image) -> np.ndarray:
    array = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.rint(array * 255.0).clip(0, 255).astype(np.uint8)


def _pano_depth_to_numpy(depth, depth_max_m: float) -> np.ndarray:
    array = depth.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    valid = np.isfinite(array) & (array > 0)
    if depth_max_m > 0:
        valid &= array <= float(depth_max_m)
    return np.where(valid, array, 0.0).astype(np.float32)


def _pano_c2w_from_group(group: dict, pano_idx: int) -> np.ndarray:
    rotation = group.get("pano_rotation_c2w")
    position = group.get("pano_position_m")
    matrix = np.eye(4, dtype=np.float32)
    if rotation is not None:
        rot = rotation[pano_idx].detach().cpu().float().numpy()
        if rot.shape == (3, 3) and np.isfinite(rot).all():
            matrix[:3, :3] = rot.astype(np.float32)
    if position is not None:
        pos = position[pano_idx].detach().cpu().float().numpy()
        if pos.shape[0] >= 3 and np.isfinite(pos[:3]).all():
            matrix[:3, 3] = pos[:3].astype(np.float32)
    return matrix


def _extrinsic_w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    rotation = extrinsic[:3, :3].astype(np.float32)
    translation = extrinsic[:3, 3].astype(np.float32)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = -rotation.T @ translation
    return matrix


def _c2w_to_extrinsic_w2c(c2w: np.ndarray) -> np.ndarray:
    rotation = c2w[:3, :3].astype(np.float32)
    translation = c2w[:3, 3].astype(np.float32)
    extrinsic = np.zeros((3, 4), dtype=np.float32)
    extrinsic[:3, :3] = rotation.T
    extrinsic[:3, 3] = -rotation.T @ translation
    return extrinsic


def _as_float_array(value, length: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=np.float32)
    if hasattr(value, "detach"):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = array.reshape(-1)
    if array.size < length:
        array = np.pad(array, (0, length - array.size), constant_values=default)
    return array[:length].astype(np.float32)


def _as_bool_array(value, length: int, default: bool = False) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=np.bool_)
    if hasattr(value, "detach"):
        array = value.detach().cpu().bool().numpy()
    else:
        array = np.asarray(value, dtype=np.bool_)
    array = array.reshape(-1)
    if array.size < length:
        array = np.pad(array, (0, length - array.size), constant_values=default)
    return array[:length].astype(np.bool_)


def _as_string_list(value, length: int, default: str = "") -> list[str]:
    if value is None:
        values = [default]
    elif isinstance(value, str):
        values = [value]
    else:
        values = [str(item) for item in list(value)]
    if len(values) < length:
        values.extend([values[-1] if values else default] * (length - len(values)))
    return values[:length]


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


def _ordered_dataset_names(names: set[str]) -> list[str]:
    order = ["panocity", "matterport3d", "stanford2d3ds", "structured3d"]
    ordered = [name for name in order if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


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


def _parse_dataset_pano_counts(raw: Optional[str | dict[str, int]]) -> dict[str, int]:
    if raw in (None, "", "none"):
        return {}
    if isinstance(raw, dict):
        pairs = raw.items()
    else:
        pairs = []
        for part in str(raw).split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"Dataset pano count must be name:count, got {part!r}")
            name, value = part.split(":", 1)
            pairs.append((name, value))
    counts: dict[str, int] = {}
    for name, value in pairs:
        key = _dataset_key(name)
        count = int(value)
        if count < 1:
            raise ValueError(f"Dataset pano count must be positive, got {name}:{value}")
        counts[key] = count
    return counts


def _parse_dataset_sampling_weights(raw: Optional[str | dict[str, float]]) -> Optional[dict[str, float]]:
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
    weights: dict[str, float] = {}
    for name, value in pairs:
        key = _dataset_key(name)
        weight = float(value)
        if weight < 0:
            raise ValueError(f"Dataset sampling weight must be non-negative, got {name}:{value}")
        weights[key] = weights.get(key, 0.0) + weight
    return weights or None


def _build_balanced_sample_indices(
    items: list[dict],
    requested_weights: Optional[dict[str, float]],
    seed: int,
) -> tuple[list[int], dict]:
    natural_indices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        key = _dataset_key(item.get("dataset") or str(item.get("name", "")).split("_", 1)[0])
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
    target_counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = total - sum(target_counts.values())
    for key in sorted(weights, key=lambda name: (exact[name] - target_counts[name]), reverse=True)[:remainder]:
        target_counts[key] += 1

    rng = random.Random(int(seed))
    sampled: list[int] = []
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


def _index_matterport3d(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"matterport3d_{split}_index.json")
    items = []
    for row in rows:
        scan, room_id, _room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = root / str(scan) / "pano_skybox_color" / f"{pano_id}.jpg"
            depth_path = root / str(scan) / "pano_depth" / f"{pano_id}.png"
            items.append(
                _item(
                    "Matterport3D",
                    f"{scan}_{room_id}_{pano_id}",
                    rgb_path,
                    depth_path,
                    _MATTERPORT3D_DEPTH_SCALE,
                )
            )
    return items


def _index_stanford2d3ds(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"2d3ds_{split}_index.json")
    items = []
    for row in rows:
        area, _room_id, room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = _first_match(root / str(area) / "pano" / "rgb", f"camera_{pano_id}_*_rgb.png")
            depth_path = _first_match(root / str(area) / "pano" / "depth", f"camera_{pano_id}_*_depth.png")
            if rgb_path is not None and depth_path is not None:
                items.append(
                    _item(
                        "Stanford2D3DS",
                        f"{area}_{room_name}_{pano_id}",
                        rgb_path,
                        depth_path,
                        _STANFORD2D3DS_DEPTH_SCALE,
                    )
                )
    return items


def _index_structured3d(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"structured3d_{split}_index.json")
    if not rows:
        for fallback_split in ("val", "test"):
            rows = _read_json_list(root / "cache" / f"structured3d_{fallback_split}_index.json")
            if rows:
                print(
                    f"[WARN] Structured3D {split} index not found under {root}; "
                    f"using structured3d_{fallback_split}_index.json instead."
                )
                break
    items = []
    for row in rows:
        scene, pano_ids, _size = row
        for pano_id in pano_ids:
            pano_dir = root / str(scene) / "2D_rendering" / str(pano_id) / "panorama" / "full"
            rgb_path = pano_dir / "rgb_rawlight.png"
            depth_path = pano_dir / "depth.png"
            items.append(_item("Structured3D", f"{scene}_{pano_id}", rgb_path, depth_path, _STRUCTURED3D_DEPTH_SCALE))
    return items


def _index_panocity_official(root: Path, split: str) -> list[dict]:
    index_path = root / "cache" / f"panocity_{split}_index.json"
    rows = _read_json_list(index_path)
    if not rows and split != "all":
        raise FileNotFoundError(
            f"Panocity {split} index not found: {index_path}. "
            "Run Rethink_pano_new_exp_omega/scripts/build_panocity_official_index.py or "
            "scripts/build_mixed4_official_indexes.py so train/val/test follow the official split."
        )
    if not rows and split == "all":
        rows = _build_panocity_official_rows(root)
    items = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rgb_path = _resolve_cached_path(root, row.get("rgb_path"))
        depth_path = _resolve_cached_path(root, row.get("depth_path"))
        if rgb_path is not None and depth_path is not None:
            items.append(_item("Panocity", str(row.get("scene_name") or rgb_path.stem), rgb_path, depth_path, _PANOCITY_DEPTH_SCALE))
    return items


def _build_panocity_official_rows(root: Path) -> list[dict]:
    rows = []
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
            if rgb_path.exists() and depth_path.exists():
                rows.append(
                    {
                        "dataset": "Panocity",
                        "city": city,
                        "block": block,
                        "scene_name": f"{city}_{block}_{Path(str(rgb_name)).stem}",
                        "rgb_path": str(rgb_path.relative_to(root)),
                        "depth_path": str(depth_path.relative_to(root)),
                    }
                )
    return rows


def _item(dataset: str, name: str, rgb_path: Path, depth_path: Path, output_depth_scale: float) -> dict:
    return {
        "dataset": dataset,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "name": f"{dataset}_{name}",
        "output_depth_scale": float(output_depth_scale),
    }


def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _resolve_cached_path(root: Path, value: object) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _first_match(folder: Path, pattern: str) -> Optional[Path]:
    if not folder.exists():
        return None
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None
