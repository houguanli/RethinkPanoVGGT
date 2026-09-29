"""Coordinate conversions for panorama-level camera poses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rotation import mat_to_quat, quat_to_mat


def omega_y_up_vectors_to_official_y_down(vectors: torch.Tensor) -> torch.Tensor:
    """Reflect right/up/forward vectors into right/down/forward coordinates."""
    if vectors.shape[-1] != 3:
        raise ValueError(f"Expected vectors[..., 3], got {tuple(vectors.shape)}")
    basis = vectors.new_tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return torch.einsum("ij,...j->...i", basis, vectors)


def omega_y_up_pose_to_official_y_down(
    centers: torch.Tensor,
    quaternions_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert relative Omega pano poses to the official OpenCV pano basis.

    Omega's panorama sampler represents ERP directions as right/up/forward,
    while official dataset poses use right/down/forward. The basis change is a
    reflection. Centers therefore receive one basis matrix, while rotations
    require conjugation so the result remains a proper rotation.
    """
    if centers.shape[-1] != 3:
        raise ValueError(f"Expected centers[..., 3], got {tuple(centers.shape)}")
    if quaternions_w2c.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternions_w2c[..., 4], got {tuple(quaternions_w2c.shape)}"
        )
    if centers.shape[:-1] != quaternions_w2c.shape[:-1]:
        raise ValueError(
            "Center and quaternion batch shapes differ: "
            f"{tuple(centers.shape[:-1])} vs {tuple(quaternions_w2c.shape[:-1])}"
        )

    basis = centers.new_tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    centers_official = omega_y_up_vectors_to_official_y_down(centers)
    quaternion_finite = torch.isfinite(quaternions_w2c).all(dim=-1, keepdim=True)
    quaternion_nonzero = torch.linalg.vector_norm(quaternions_w2c, dim=-1, keepdim=True) > 1e-6
    identity_quaternion = torch.zeros_like(quaternions_w2c)
    identity_quaternion[..., 3] = 1.0
    safe_quaternion = torch.where(
        quaternion_finite & quaternion_nonzero,
        F.normalize(torch.nan_to_num(quaternions_w2c), dim=-1, eps=1e-6),
        identity_quaternion,
    )
    rotations_omega = quat_to_mat(safe_quaternion)
    rotations_official = basis @ rotations_omega @ basis
    quaternions_official = F.normalize(
        mat_to_quat(rotations_official),
        dim=-1,
        eps=1e-6,
    )
    return centers_official, quaternions_official
