"""Tests for pano-level camera supervision."""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.train_pano_omega import camera_alignment_loss  # noqa: E402


def test_camera_supervision_none_returns_zero_loss():
    predictions = {
        "depth": torch.ones(1, 2, 4, 4, 1),
        "pose_enc": torch.zeros(1, 2, 9),
        "pano_camera_meta": {
            "rotations": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
            "fov_x": torch.ones(1, 2),
            "fov_y": torch.ones(1, 2),
        },
    }
    losses = camera_alignment_loss(
        predictions=predictions,
        batch={},
        translation_weight=1.0,
        rotation_weight=1.0,
        fov_weight=1.0,
        position_mode="local_zero",
        supervision_mode="none",
    )
    assert losses["loss_camera"].item() == 0.0
    assert losses["loss_camera_t"].item() == 0.0


def test_pano_relative_translation_supervises_pano_centers_not_window_pose():
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()
    pose_enc = torch.zeros(1, 4, 9)
    # Identity w2c convention: t = -C. Two windows belong to each pano.
    pose_enc[:, 2:, :3] = torch.tensor([-2.0, 0.0, 0.0])
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pose_enc": pose_enc,
        "pano_camera_meta": {
            "rotations": rotations,
            "fov_x": torch.ones(1, 4),
            "fov_y": torch.ones(1, 4),
        },
    }
    batch = {"pano_position_m": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])}
    losses = camera_alignment_loss(
        predictions=predictions,
        batch=batch,
        translation_weight=1.0,
        rotation_weight=1.0,
        fov_weight=1.0,
        position_mode="relative_anchor",
        supervision_mode="pano_relative",
    )
    assert torch.isclose(losses["loss_camera"], torch.tensor(0.0))
    assert torch.isclose(losses["loss_camera_r"], torch.tensor(0.0))
    assert torch.isclose(losses["loss_camera_fov"], torch.tensor(0.0))


if __name__ == "__main__":
    test_camera_supervision_none_returns_zero_loss()
    test_pano_relative_translation_supervises_pano_centers_not_window_pose()
    print("camera supervision modes (omega) ok")
