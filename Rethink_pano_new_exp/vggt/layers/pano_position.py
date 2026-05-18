import math
from typing import Tuple

import torch
import torch.nn.functional as F


def yaw_pitch_to_axes(yaw: torch.Tensor, pitch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return forward/right/up unit axes for spherical yaw and pitch angles."""
    cos_pitch = torch.cos(pitch)
    forward = torch.stack(
        [
            cos_pitch * torch.sin(yaw),
            torch.sin(pitch),
            cos_pitch * torch.cos(yaw),
        ],
        dim=-1,
    )
    right = torch.stack([torch.cos(yaw), torch.zeros_like(yaw), -torch.sin(yaw)], dim=-1)
    up = torch.cross(forward, right, dim=-1)
    return F.normalize(forward, dim=-1), F.normalize(right, dim=-1), F.normalize(up, dim=-1)


def pinhole_rays(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Build world-space rays for virtual pinhole cameras.

    Args:
        yaw, pitch, fov_x, fov_y: Tensors with shape [V].
        height, width: Output ray grid size.

    Returns:
        Rays with shape [V, height, width, 3].
    """
    device = device or yaw.device
    dtype = dtype or yaw.dtype
    yaw = yaw.to(device=device, dtype=dtype)
    pitch = pitch.to(device=device, dtype=dtype)
    fov_x = fov_x.to(device=device, dtype=dtype)
    fov_y = fov_y.to(device=device, dtype=dtype)

    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    x_plane = xx[None, :, :] * torch.tan(fov_x[:, None, None] * 0.5)
    y_plane = -yy[None, :, :] * torch.tan(fov_y[:, None, None] * 0.5)

    forward, right, up = yaw_pitch_to_axes(yaw, pitch)
    rays = (
        forward[:, None, None, :]
        + x_plane[..., None] * right[:, None, None, :]
        + y_plane[..., None] * up[:, None, None, :]
    )
    return F.normalize(rays, dim=-1)


def rays_to_equirectangular(rays: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert unit rays to yaw/pitch and normalized equirectangular uv."""
    rays = F.normalize(rays, dim=-1)
    theta = torch.atan2(rays[..., 0], rays[..., 2])
    phi = torch.asin(rays[..., 1].clamp(-1.0, 1.0))
    u = torch.remainder(theta / (2.0 * math.pi) + 0.5, 1.0)
    v = (0.5 - phi / math.pi).clamp(0.0, 1.0)
    return theta, phi, u, v


def build_sphere_encoding(theta: torch.Tensor, phi: torch.Tensor, sphere_dir: torch.Tensor) -> torch.Tensor:
    """Build a compact spherical encoding used by LUNA-Patch."""
    return torch.cat(
        [
            torch.sin(theta)[..., None],
            torch.cos(theta)[..., None],
            torch.sin(phi)[..., None],
            torch.cos(phi)[..., None],
            sphere_dir,
        ],
        dim=-1,
    )


def build_camera_encoding(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
) -> torch.Tensor:
    """Build [sin/cos yaw, sin/cos pitch, fovs, aspect, center/right/up rays]."""
    center_ray, right_ray, up_ray = yaw_pitch_to_axes(yaw, pitch)
    aspect = fov_x / fov_y.clamp_min(torch.finfo(fov_y.dtype).eps)
    return torch.cat(
        [
            torch.sin(yaw)[..., None],
            torch.cos(yaw)[..., None],
            torch.sin(pitch)[..., None],
            torch.cos(pitch)[..., None],
            fov_x[..., None],
            fov_y[..., None],
            aspect[..., None],
            center_ray,
            right_ray,
            up_ray,
        ],
        dim=-1,
    )


def camera_rotation_from_yaw_pitch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Return camera-to-world rotation matrices with right/up/forward columns."""
    forward, right, up = yaw_pitch_to_axes(yaw, pitch)
    return torch.stack([right, up, forward], dim=-1)
