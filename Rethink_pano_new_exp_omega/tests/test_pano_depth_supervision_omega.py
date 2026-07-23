"""Tests for pano range-depth to VGGT-Omega Z-depth conventions."""

import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.train_pano_omega import build_target_translation, build_window_z_factor, erp_depth_to_range_depth, masked_log_l1_depth, sample_depth_targets  # noqa: E402
from scripts.reconstruct_pano_omega import apply_range_depth_modifier  # noqa: E402
from vggt_omega.data.pano_sampler import PanoWindowSampler  # noqa: E402


def test_range_depth_is_converted_to_pinhole_z_depth():
    sampler = PanoWindowSampler(window_size=3, patch_size=1, fov_degrees=75.0, num_yaw=1)
    output = sampler(torch.ones(1, 3, 8, 16))
    factor = build_window_z_factor(output.camera_meta, height=3, width=3)
    assert torch.isclose(factor[0, 0, 1, 1], torch.tensor(1.0)), factor
    assert factor[0, 0, 0, 0] < 1.0, factor


def test_camera_rotation_stays_in_proper_rotation_space():
    sampler = PanoWindowSampler(window_size=3, patch_size=1, fov_degrees=75.0, num_yaw=1)
    output = sampler(torch.ones(1, 3, 8, 16))
    rotation = output.camera_meta["rotations"][0, 0]
    assert torch.isclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-6), rotation


def test_depth_loss_ignores_infinite_values():
    pred = torch.tensor([[[[[1.0], [float("inf")], [4.0]]]]])
    target = torch.tensor([[[[[1.0], [2.0], [float("inf")]]]]])
    loss = masked_log_l1_depth(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0)), loss


def test_infinite_source_depth_does_not_pollute_sampled_targets():
    sampler = PanoWindowSampler(window_size=3, patch_size=1, fov_degrees=75.0, num_yaw=1)
    pano_depth = torch.ones(1, 1, 8, 16)
    pano_depth[0, 0, 4, 8] = float("inf")
    target, _ = sample_depth_targets(type("Model", (), {"pano_sampler": sampler})(), pano_depth)
    assert torch.isfinite(target).all(), target


def test_cubemap_z_depth_is_expanded_off_face_center():
    stored_depth = torch.ones(1, 1, 5, 9)
    range_depth = erp_depth_to_range_depth(stored_depth, "cubemap_z")
    assert torch.isclose(range_depth[0, 0, 2, 4], torch.tensor(1.0))
    assert range_depth[0, 0, 2, 5] > 1.0


def test_double_cubemap_z_uses_offset_cube_to_reduce_corner_expansion():
    stored_depth = torch.ones(1, 1, 5, 16)
    single_range = erp_depth_to_range_depth(stored_depth, "cubemap_z")
    double_range = erp_depth_to_range_depth(stored_depth, "double_cubemap_z")
    assert double_range[0, 0, 2, 5] < single_range[0, 0, 2, 5]
    assert torch.isfinite(double_range).all()


def test_predicted_range_modifier_removes_far_window_corners():
    sampler = PanoWindowSampler(window_size=3, patch_size=1, fov_degrees=75.0, num_yaw=1)
    output = sampler(torch.ones(1, 3, 8, 16))
    camera_meta = {key: value[0] for key, value in output.camera_meta.items()}
    depth = np.full((1, 3, 3), 50.0, dtype=np.float32)
    filtered, valid = apply_range_depth_modifier(depth, camera_meta, max_range_depth=60.0)
    assert valid[0, 1, 1]
    assert not valid[0, 0, 0]
    assert np.isinf(filtered[0, 0, 0])


def test_single_pano_camera_translation_target_is_zero():
    rotations_w2c = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1)
    batch = {"pano_position_m": torch.tensor([[10.0, 0.0, 20.0]])}
    target = build_target_translation(rotations_w2c, batch, "local_zero")
    assert torch.equal(target, torch.zeros(1, 4, 3)), target


def test_multi_pano_camera_translation_target_is_relative_to_anchor():
    rotations_w2c = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1)
    batch = {"pano_position_m": torch.tensor([[[10.0, 0.0, 20.0], [13.0, 0.0, 24.0]]])}
    target = build_target_translation(rotations_w2c, batch, "relative_anchor")
    expected = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.0, 0.0, -4.0], [-3.0, 0.0, -4.0]]])
    assert torch.allclose(target, expected), target


if __name__ == "__main__":
    test_range_depth_is_converted_to_pinhole_z_depth()
    test_camera_rotation_stays_in_proper_rotation_space()
    test_depth_loss_ignores_infinite_values()
    test_infinite_source_depth_does_not_pollute_sampled_targets()
    test_cubemap_z_depth_is_expanded_off_face_center()
    test_double_cubemap_z_uses_offset_cube_to_reduce_corner_expansion()
    test_predicted_range_modifier_removes_far_window_corners()
    test_single_pano_camera_translation_target_is_zero()
    test_multi_pano_camera_translation_target_is_relative_to_anchor()
    print("pano depth supervision (omega) ok")
