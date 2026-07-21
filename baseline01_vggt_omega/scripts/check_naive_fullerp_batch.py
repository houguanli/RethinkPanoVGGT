#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from train_utils.normalization import normalize_camera_extrinsics_and_points_batch  # noqa: E402


def main() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(TRAINING_ROOT / "config")):
        cfg = compose(
            config_name="mixed4_naive_fullerp_vggtomega_scalealigned_2pano",
            overrides=[
                "data.train.dataset.dataset_configs.0.datasets=stanford2d3ds",
                "data.train.dataset.dataset_configs.0.dataset_sampling_weights=null",
                "data.train.dataset.dataset_configs.0.max_samples=20",
            ],
        )
    dataset = instantiate(
        cfg.data.train.dataset,
        common_config=cfg.data.train.common_config,
        _recursive_=False,
    )
    sample = dataset[(0, 2, 0.5)]
    extrinsics, cam_points, world_points, depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=sample["extrinsics"].unsqueeze(0),
        cam_points=sample["cam_points"].unsqueeze(0),
        world_points=sample["world_points"].unsqueeze(0),
        depths=sample["depths"].unsqueeze(0),
        point_masks=sample["point_masks"].unsqueeze(0),
    )
    tensors = {
        "extrinsics": extrinsics,
        "cam_points": cam_points,
        "world_points": world_points,
        "depths": depths,
    }
    summary = {
        "seq_name": sample["seq_name"],
        "input_representation": sample.get("input_representation"),
        "images_shape": list(sample["images"].shape),
        "depths_shape": list(sample["depths"].shape),
        "view_pano_index": sample["view_pano_index"].tolist(),
        "view_window_index": sample["view_window_index"].tolist(),
        "windows_per_pano": sample["windows_per_pano"].tolist(),
        "camera_valid": sample["camera_valid"].tolist(),
        "depth_valid_fraction": float(sample["point_masks"].float().mean()),
        "normalized_finite": {name: bool(torch.isfinite(value).all()) for name, value in tensors.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["images_shape"] != [2, 3, 192, 384]:
        raise RuntimeError(f"Unexpected full ERP image shape: {summary['images_shape']}")
    if summary["view_window_index"] != [-1, -1] or summary["windows_per_pano"] != [0, 0]:
        raise RuntimeError("Naive full ERP sample unexpectedly contains perspective windows.")
    if not all(summary["normalized_finite"].values()):
        raise RuntimeError("Scene-normalized full ERP tensors contain NaN or Inf.")


if __name__ == "__main__":
    main()
