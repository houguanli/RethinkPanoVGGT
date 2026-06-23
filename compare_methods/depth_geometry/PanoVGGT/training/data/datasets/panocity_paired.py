"""PanoCity paired CSV adapter for PanoVGGT depth training configs."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

COMPARE_ROOT = Path(__file__).resolve().parents[5]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex, smoke_summary  # noqa: E402,F401
from training.data.base_dataset import BaseDataset  # noqa: E402
from training.data.dataset_util import threshold_depth_map  # noqa: E402


class PanoCityPairedDataset(BaseDataset):
    """PanoVGGT-compatible reader for the paired RGB/depth PanoCity layout.

    The paired export currently has no calibrated pose labels. For finetune
    plumbing we provide a deterministic weak camera path with small translations
    between selected panoramas. Replace these placeholders when calibrated poses
    are available.
    """

    def __init__(
        self,
        common_conf,
        root: str | None = None,
        split: str = "train",
        pairs_file: str = "pairs.csv",
        len_train: int | None = None,
        len_test: int | None = None,
        max_samples: int | None = None,
        depth_scale: float = 100.0,
        depth_max: float = 100.0,
        **_: object,
    ) -> None:
        super().__init__(common_conf=common_conf)
        self.training = bool(common_conf.training)
        self.split = split
        self.depth_scale = float(depth_scale)
        self.depth_max = float(depth_max)
        self.index = PanoCityPairedIndex(root=root, pairs_file=pairs_file, split=split, max_samples=max_samples)
        if split == "train" and len_train is not None:
            self.dataset_length = min(int(len_train), len(self.index))
        elif split != "train" and len_test is not None:
            self.dataset_length = min(int(len_test), len(self.index))
        else:
            self.dataset_length = len(self.index)

    def __len__(self):
        return self.dataset_length

    def _target_resolution(self):
        return int(self.img_size), int(self.img_size) * 2

    def _read_rgb(self, path: Path, target_resolution: tuple[int, int]) -> np.ndarray:
        height, width = target_resolution
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read RGB image: {path}")
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def _read_depth(self, path: Path, target_resolution: tuple[int, int]) -> np.ndarray:
        height, width = target_resolution
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth image: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / self.depth_scale
        depth = threshold_depth_map(depth, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
        depth[~np.isfinite(depth)] = 0.0
        return depth[None, ...]

    @staticmethod
    def _placeholder_w2c(frame_offset: int) -> np.ndarray:
        pose = np.zeros((3, 4), dtype=np.float32)
        pose[:3, :3] = np.eye(3, dtype=np.float32)
        pose[0, 3] = -0.05 * float(frame_offset)
        return pose

    def get_data(
        self,
        seq_index: int | None = None,
        img_per_seq: int | None = None,
        seq_name: str | None = None,
        ids: list[int] | None = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if len(self.index) == 0:
            raise RuntimeError("No PanoCity paired records are available.")
        if seq_index is None:
            seq_index = np.random.randint(0, len(self.index))
        if img_per_seq is None:
            img_per_seq = 1
        img_per_seq = max(1, int(img_per_seq))
        ids = ids if ids is not None else list(range(img_per_seq))

        target_resolution = self._target_resolution()
        records = [self.index[(seq_index + int(offset)) % len(self.index)] for offset in ids]

        batch_data = {
            "images": [],
            "depths": [],
            "extrinsics": [],
            "cam_points": [],
            "world_points": [],
            "point_masks": [],
            "original_sizes": [],
        }
        successful_ids = []

        for frame_offset, record in enumerate(records):
            image = self._read_rgb(record.rgb_path, target_resolution)
            depth_map = self._read_depth(record.depth_path, target_resolution)
            frame = self.process_one_image(
                image=image,
                depth_map=depth_map,
                extrinsic_w2c=self._placeholder_w2c(frame_offset),
                shape=target_resolution,
                equi_rotate=None,
                R_delta=None,
                depth_max=self.depth_max,
            )
            batch_data["images"].append(frame["rgb"])
            batch_data["depths"].append(frame["depth_tensor"])
            batch_data["extrinsics"].append(frame["extrinsic"])
            batch_data["cam_points"].append(frame["cam_coords"])
            batch_data["world_points"].append(frame["world_coords"])
            batch_data["point_masks"].append(frame["valid_mask"])
            batch_data["original_sizes"].append(np.array(target_resolution, dtype=np.int32))
            successful_ids.append(record.pair_idx)

        return {
            "seq_name": seq_name or f"PanoCityPaired_{records[0].block}_{records[0].num_id}",
            "ids": successful_ids,
            "frame_num": len(successful_ids),
            **batch_data,
        }
