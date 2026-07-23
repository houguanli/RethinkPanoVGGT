"""Tests for pano-level camera supervision."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.train_pano_omega import (  # noqa: E402
    build_optimizer,
    camera_alignment_loss,
    load_checkpoint,
    shared_frame_point_loss,
    transition_optimizer_for_stage,
)
from vggt_omega.models.heads import PanoCameraHead  # noqa: E402
from vggt_omega.models.vggt_omega_luna import _omega_window_pose_to_relative_pano_pose  # noqa: E402
from vggt_omega.utils.pano_pose import omega_y_up_pose_to_official_y_down  # noqa: E402
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat  # noqa: E402


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
    assert losses["camera_translation_valid_count"].item() == 2.0
    assert losses["camera_rotation_valid_count"].item() == 2.0
    assert torch.isclose(losses["camera_translation_deg"], torch.tensor(0.0), atol=1e-4)


def test_pano_relative_supervision_converts_omega_y_up_to_official_y_down():
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pano_camera_center": torch.tensor([[[0.0, 0.0, 0.0], [0.0, -2.0, 0.0]]]),
        "pano_rotation_quat_w2c": identity_quat.reshape(1, 1, 4).expand(1, 2, 4).clone(),
    }
    batch = {
        "pano_position_m": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3).clone(),
        "pano_rotation_valid": torch.tensor([[True, True]]),
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
    assert torch.isclose(losses["loss_camera_t"], torch.tensor(0.0))


def test_omega_to_official_rotation_is_proper_and_conjugated():
    angle = torch.tensor(torch.pi / 3.0)
    rotation_omega = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(angle), -torch.sin(angle)],
            [0.0, torch.sin(angle), torch.cos(angle)],
        ]
    )
    center = torch.tensor([[1.0, 2.0, 3.0]])
    quat = mat_to_quat(rotation_omega).reshape(1, 4)
    center_official, quat_official = omega_y_up_pose_to_official_y_down(center, quat)
    basis = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    expected_rotation = basis @ rotation_omega @ basis
    assert torch.allclose(center_official, torch.tensor([[1.0, -2.0, 3.0]]))
    assert torch.allclose(quat_to_mat(quat_official)[0], expected_rotation, atol=1e-6)
    assert torch.isclose(torch.linalg.det(quat_to_mat(quat_official)[0]), torch.tensor(1.0), atol=1e-6)


def test_invalid_omega_window_pose_falls_back_to_finite_identity_pano_pose():
    pose_enc = torch.zeros(1, 4, 9)
    crop_c2w = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()
    centers, quaternions = _omega_window_pose_to_relative_pano_pose(
        pose_enc,
        crop_c2w,
        num_panos=2,
    )
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).expand_as(quaternions)
    assert torch.isfinite(centers).all()
    assert torch.isfinite(quaternions).all()
    assert torch.allclose(centers, torch.zeros_like(centers))
    assert torch.allclose(quaternions, identity, atol=1e-6)


def test_delta_checkpoint_restores_foundation_then_parent_then_delta():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        foundation = root / "foundation.pt"
        parent = root / "parent.pt"
        delta = root / "delta.pt"
        torch.save({"weight": torch.tensor([[1.0]]), "bias": torch.tensor([1.0])}, foundation)
        torch.save({"bias": torch.tensor([2.0])}, parent)
        torch.save(
            {
                "model_delta": {"weight": torch.tensor([[3.0]])},
                "foundation_checkpoint": str(foundation),
                "base_checkpoint": str(parent),
            },
            delta,
        )
        model = torch.nn.Linear(1, 1)
        load_checkpoint(model, delta, strict=False)
        assert torch.allclose(model.weight, torch.tensor([[3.0]]))
        assert torch.allclose(model.bias, torch.tensor([2.0]))


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


def test_pano_relative_translation_metric_is_direction_angle_in_degrees():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pano_camera_center": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]),
        "pano_rotation_quat_w2c": identity.reshape(1, 1, 4).expand(1, 2, 4),
    }
    batch = {
        "pano_position_m": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_translation_valid": torch.tensor([[True, True]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        "pano_rotation_valid": torch.tensor([[False, False]]),
    }
    losses = camera_alignment_loss(
        predictions=predictions,
        batch=batch,
        translation_weight=1.0,
        rotation_weight=0.0,
        fov_weight=0.0,
        position_mode="relative_anchor",
        supervision_mode="pano_relative",
    )
    assert torch.isclose(losses["camera_translation_deg"], torch.tensor(90.0), atol=1e-4)
    assert losses["camera_translation_valid_count"].item() == 2.0


def test_pano_relative_uses_all_ordered_camera_pairs():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    centers = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    predictions = {
        "depth": torch.ones(1, 6, 4, 4, 1),
        "pano_camera_center": centers.clone(),
        "pano_rotation_quat_w2c": identity.reshape(1, 1, 4).expand(1, 3, 4),
    }
    batch = {
        "pano_position_m": centers.clone(),
        "pano_position_valid": torch.tensor([[True, True, True]]),
        "pano_translation_valid": torch.tensor([[True, True, True]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, 3, 3),
        "pano_rotation_valid": torch.tensor([[True, True, True]]),
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
    assert losses["camera_translation_valid_count"].item() == 6.0
    assert losses["camera_rotation_valid_count"].item() == 6.0
    assert torch.isclose(losses["loss_camera_t"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(losses["loss_camera_r"], torch.tensor(0.0), atol=1e-4)


def test_structured_style_translation_mask_disables_loss_and_metric():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pano_camera_center": torch.zeros(1, 2, 3),
        "pano_rotation_quat_w2c": identity.reshape(1, 1, 4).expand(1, 2, 4),
    }
    batch = {
        "pano_position_m": torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_translation_valid": torch.tensor([[False, False]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        "pano_rotation_valid": torch.tensor([[False, False]]),
    }
    losses = camera_alignment_loss(
        predictions=predictions,
        batch=batch,
        translation_weight=1.0,
        rotation_weight=0.0,
        fov_weight=0.0,
        position_mode="relative_anchor",
        supervision_mode="pano_relative",
    )
    assert losses["loss_camera_t"].item() == 0.0
    assert losses["camera_translation_deg"].item() == 0.0
    assert losses["camera_translation_valid_count"].item() == 0.0


def test_invalid_nan_pose_placeholders_do_not_poison_masked_camera_loss():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    predictions = {
        "depth": torch.ones(1, 4, 4, 4, 1),
        "pano_camera_center": torch.zeros(1, 2, 3),
        "pano_rotation_quat_w2c": identity.reshape(1, 1, 4).expand(1, 2, 4),
    }
    batch = {
        "pano_position_m": torch.full((1, 2, 3), float("nan")),
        "pano_translation_valid": torch.zeros(1, 2, dtype=torch.bool),
        "pano_rotation_c2w": torch.full((1, 2, 3, 3), float("nan")),
        "pano_rotation_valid": torch.zeros(1, 2, dtype=torch.bool),
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
    assert torch.isfinite(losses["loss_camera"])
    assert losses["loss_camera_t"].item() == 0.0
    assert losses["loss_camera_r"].item() == 0.0
    assert losses["camera_translation_deg"].item() == 0.0


def test_shared_frame_point_loss_is_zero_for_matching_depth_and_pose():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    depth = torch.ones(1, 2, 8, 8, 1)
    centers = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    predictions = {
        "pano_camera_center": centers.clone(),
        "pano_rotation_quat_w2c": identity.reshape(1, 1, 4).expand(1, 2, 4),
        "pano_camera_meta": {
            "yaw": torch.zeros(1, 2),
            "pitch": torch.zeros(1, 2),
            "fov_x": torch.full((1, 2), torch.pi / 2),
            "fov_y": torch.full((1, 2), torch.pi / 2),
            "rotations": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        },
    }
    batch = {
        "pano_position_m": centers.clone(),
        "pano_position_valid": torch.tensor([[True, True]]),
        "pano_translation_valid": torch.tensor([[True, True]]),
        "pano_rotation_c2w": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        "pano_rotation_valid": torch.tensor([[True, True]]),
    }
    result = shared_frame_point_loss(
        pred_depth=depth,
        target_depth=depth,
        valid_mask=torch.ones_like(depth, dtype=torch.bool),
        predictions=predictions,
        batch=batch,
        pred_translation_scale=None,
        stride=2,
    )
    assert torch.isclose(result["loss_global_point"], torch.tensor(0.0), atol=1e-6)

    mismatched_target = depth * 2.0
    mismatched_target[:, :, 0, 0] = float("nan")
    mismatched_valid = torch.ones_like(depth, dtype=torch.bool)
    mismatched_valid[:, :, 0, 0] = False
    mismatched = shared_frame_point_loss(
        pred_depth=depth,
        target_depth=mismatched_target,
        valid_mask=mismatched_valid,
        predictions=predictions,
        batch=batch,
        pred_translation_scale=None,
        stride=2,
    )
    assert mismatched["loss_global_point"].item() > 0.0
    assert torch.isclose(result["global_point_valid_ratio"], torch.tensor(1.0))


def test_pano_camera_head_outputs_one_zero_initialized_pose_per_pano():
    head = PanoCameraHead(dim_in=32, hidden_dim=32, camera_meta_dim=16, num_heads=4, num_cross_pano_layers=1)
    outputs = head(
        torch.randn(2, 12, 32),
        torch.randn(2, 12, 16),
        num_panos=3,
    )
    assert outputs["pano_camera_center_residual"].shape == (2, 3, 3)
    assert outputs["pano_rotation_quat_w2c_residual"].shape == (2, 3, 4)
    assert torch.allclose(outputs["pano_camera_center_residual"], torch.zeros(2, 3, 3))
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).expand(2, 3, 4)
    assert torch.allclose(outputs["pano_rotation_quat_w2c_residual"], expected)


def test_pano_camera_rotation_geodesic_loss_reaches_new_head():
    head = PanoCameraHead(dim_in=32, hidden_dim=32, camera_meta_dim=16, num_heads=4, num_cross_pano_layers=1)
    outputs = head(torch.randn(1, 8, 32), torch.randn(1, 8, 16), num_panos=2)
    rotation_y_90_c2w = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    rotations = torch.stack([torch.eye(3), rotation_y_90_c2w], dim=0).unsqueeze(0)
    predictions = {
        "depth": torch.ones(1, 8, 4, 4, 1),
        "pano_camera_center": outputs["pano_camera_center_residual"],
        "pano_rotation_quat_w2c": outputs["pano_rotation_quat_w2c_residual"],
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
