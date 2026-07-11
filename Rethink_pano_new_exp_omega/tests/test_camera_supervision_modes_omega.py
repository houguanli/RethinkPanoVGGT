"""Tests for pano-level camera supervision."""

import os
import sys
from types import SimpleNamespace

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.train_pano_omega import (  # noqa: E402
    build_optimizer,
    camera_alignment_loss,
    transition_optimizer_for_stage,
)
from vggt_omega.models.heads import PanoCameraHead  # noqa: E402


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
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pano_camera_center": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        "pano_rotation_quat_w2c": identity_quat.reshape(1, 1, 4).expand(1, 2, 4).clone(),
    }
    batch = {
        "pano_position_m": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3).clone(),
        "pano_rotation_valid": torch.tensor([[True, True]]),
    }
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
    assert losses["camera_translation_valid_count"].item() == 1.0
    assert losses["camera_rotation_valid_count"].item() == 1.0


def test_pano_relative_masks_missing_rotation_without_masking_translation():
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pano_camera_center": torch.zeros(1, 2, 3),
        "pano_rotation_quat_w2c": torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).expand(1, 2, 4),
    }
    batch = {
        "pano_position_m": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        "pano_rotation_valid": torch.tensor([[False, False]]),
    }
    losses = camera_alignment_loss(
        predictions=predictions,
        batch=batch,
        translation_weight=1.0,
        rotation_weight=1.0,
        fov_weight=0.0,
        position_mode="relative_anchor",
        supervision_mode="pano_relative",
    )
    assert losses["loss_camera_t"].item() > 0.0
    assert losses["loss_camera_r"].item() == 0.0
    assert losses["camera_rotation_valid_count"].item() == 0.0


def test_pano_camera_head_outputs_one_zero_initialized_pose_per_pano():
    head = PanoCameraHead(dim_in=32, hidden_dim=32, camera_meta_dim=16, num_heads=4, num_cross_pano_layers=1)
    outputs = head(
        torch.randn(2, 12, 32),
        torch.randn(2, 12, 16),
        num_panos=3,
    )
    assert outputs["pano_camera_center"].shape == (2, 3, 3)
    assert outputs["pano_rotation_quat_w2c"].shape == (2, 3, 4)
    assert torch.allclose(outputs["pano_camera_center"], torch.zeros(2, 3, 3))
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).expand(2, 3, 4)
    assert torch.allclose(outputs["pano_rotation_quat_w2c"], expected)


def test_pano_camera_rotation_geodesic_loss_reaches_new_head():
    head = PanoCameraHead(dim_in=32, hidden_dim=32, camera_meta_dim=16, num_heads=4, num_cross_pano_layers=1)
    outputs = head(torch.randn(1, 8, 32), torch.randn(1, 8, 16), num_panos=2)
    rotation_y_90_c2w = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    rotations = torch.stack([torch.eye(3), rotation_y_90_c2w], dim=0).unsqueeze(0)
    predictions = {
        "depth": torch.ones(1, 8, 4, 4, 1),
        **outputs,
    }
    batch = {
        "pano_position_m": torch.zeros(1, 2, 3),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_rotation_c2w": rotations,
        "pano_rotation_valid": torch.tensor([[True, True]]),
    }
    losses = camera_alignment_loss(
        predictions=predictions,
        batch=batch,
        translation_weight=0.0,
        rotation_weight=1.0,
        fov_weight=0.0,
        position_mode="relative_anchor",
        supervision_mode="pano_relative",
    )
    losses["loss_camera"].backward()
    gradient = head.rotation_branch[-1].weight.grad
    assert losses["camera_rotation_deg"].item() > 80.0
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_stage_transition_preserves_optimizer_state_when_parameter_set_is_unchanged():
    model = torch.nn.Linear(4, 2)
    args = SimpleNamespace(
        lr=5e-5,
        weight_decay=0.05,
        optimizer_type="adafactor",
        pred_depth_scale_lr=None,
    )
    optimizer = build_optimizer(model, args, optimizer_type="adafactor")
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    state_size = len(optimizer.state)
    transitioned, preserved = transition_optimizer_for_stage(
        optimizer,
        model,
        args,
        {
            "lr": 1e-5,
            "weight_decay": 0.05,
            "optimizer_type": "adafactor",
        },
    )
    assert preserved
    assert transitioned is optimizer
    assert len(transitioned.state) == state_size
    assert transitioned.param_groups[0]["lr"] == 1e-5


if __name__ == "__main__":
    test_camera_supervision_none_returns_zero_loss()
    test_pano_relative_translation_supervises_pano_centers_not_window_pose()
    test_pano_relative_masks_missing_rotation_without_masking_translation()
    test_pano_camera_head_outputs_one_zero_initialized_pose_per_pano()
    test_pano_camera_rotation_geodesic_loss_reaches_new_head()
    test_stage_transition_preserves_optimizer_state_when_parameter_set_is_unchanged()
    print("camera supervision modes (omega) ok")
