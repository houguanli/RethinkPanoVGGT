"""Tests for pano range-depth to VGGT-Omega Z-depth conventions."""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.train_pano_omega import build_window_z_factor  # noqa: E402
from vggt_omega.data.pano_sampler import PanoWindowSampler  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays  # noqa: E402


def test_range_depth_is_converted_to_pinhole_z_depth():
    sampler = PanoWindowSampler(window_size=3, patch_size=1, fov_degrees=75.0, num_yaw=1)
    output = sampler(torch.ones(1, 3, 8, 16))
    factor = build_window_z_factor(output.camera_meta, height=3, width=3)
    assert torch.isclose(factor[0, 0, 1, 1], torch.tensor(1.0)), factor
    assert factor[0, 0, 0, 0] < 1.0, factor


def test_camera_rotation_uses_opencv_down_axis():
    sampler = PanoWindowSampler(window_size=3, patch_size=1, fov_degrees=75.0, num_yaw=1)
    output = sampler(torch.ones(1, 3, 8, 16))
    meta = output.camera_meta
    rays = pinhole_rays(
        meta["yaw"].reshape(-1),
        meta["pitch"].reshape(-1),
        meta["fov_x"].reshape(-1),
        meta["fov_y"].reshape(-1),
        3,
        3,
    )
    down_axis = meta["rotations"][0, 0, :, 1]
    assert torch.dot(rays[0, 2, 1], down_axis) > 0


if __name__ == "__main__":
    test_range_depth_is_converted_to_pinhole_z_depth()
    test_camera_rotation_uses_opencv_down_axis()
    print("pano depth supervision (omega) ok")
