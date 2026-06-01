"""Pano window sampler for VGGT-Omega + LUNA.

Same sampling semantics as `Rethink_pano_new_exp/vggt/data/pano_sampler.py`,
but:
  - default `patch_size` is 16 instead of 14 to match the Omega backbone;
  - imports the pano-position helpers from inside the omega layers package.

A `PanoWindowSampler` accepts a panorama image `[B, 3, H_pano, W_pano]` and
returns virtual pinhole windows `[B, S, 3, H, W]` plus per-token spherical
metadata and per-view camera-encoding metadata ready for the LUNA adapters.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers.pano_position import (
    build_camera_encoding,
    build_sphere_encoding,
    camera_rotation_from_yaw_pitch,
    pinhole_rays,
    rays_to_equirectangular,
)


@dataclass
class PanoSamplerOutput:
    windows: torch.Tensor
    camera_meta: Dict[str, torch.Tensor]
    token_meta: Dict[str, torch.Tensor]


def make_default_view_grid(
    num_yaw: int = 8,
    pitch_degrees: Sequence[float] = (0.0,),
    yaw_offset_degrees: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a simple yaw/pitch grid in radians."""
    yaw = torch.arange(num_yaw, dtype=torch.float32) * (2.0 * math.pi / num_yaw)
    yaw = yaw + math.radians(yaw_offset_degrees) - math.pi
    pitch = torch.tensor([math.radians(v) for v in pitch_degrees], dtype=torch.float32)
    yaw_grid, pitch_grid = torch.meshgrid(yaw, pitch, indexing="ij")
    return yaw_grid.reshape(-1).clone(), pitch_grid.reshape(-1).clone()


class PanoWindowSampler(nn.Module):
    """Sample VGGT-compatible virtual pinhole windows from equirectangular panos.

    Defaults match the Omega backbone (``patch_size=16``, ``window_size=512``);
    these can still be overridden from the model config.
    """

    def __init__(
        self,
        window_size: int = 512,
        patch_size: int = 16,
        fov_degrees: float = 75.0,
        num_yaw: int = 8,
        pitch_degrees: Sequence[float] = (0.0,),
        seam_width: float = 0.02,
    ):
        super().__init__()
        yaw, pitch = make_default_view_grid(num_yaw=num_yaw, pitch_degrees=pitch_degrees)
        self.window_size = window_size
        self.patch_size = patch_size
        self.fov_radians = math.radians(fov_degrees)
        self.seam_width = seam_width
        self.register_buffer("default_yaw", yaw, persistent=False)
        self.register_buffer("default_pitch", pitch, persistent=False)

    def forward(
        self,
        pano_images: torch.Tensor,
        yaw: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        fov: Optional[torch.Tensor] = None,
        interpolation_mode: str = "bilinear",
    ) -> PanoSamplerOutput:
        return sample_pano_windows(
            pano_images=pano_images,
            yaw=self.default_yaw if yaw is None else yaw,
            pitch=self.default_pitch if pitch is None else pitch,
            fov=self.fov_radians if fov is None else fov,
            window_size=self.window_size,
            patch_size=self.patch_size,
            seam_width=self.seam_width,
            interpolation_mode=interpolation_mode,
        )


def sample_pano_windows(
    pano_images: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    fov,
    window_size: int = 512,
    patch_size: int = 16,
    seam_width: float = 0.02,
    interpolation_mode: str = "bilinear",
) -> PanoSamplerOutput:
    """Sample virtual pinhole windows and build LUNA metadata."""
    if pano_images.ndim == 3:
        pano_images = pano_images.unsqueeze(0)
    if pano_images.ndim != 4 or pano_images.shape[1] != 3:
        raise ValueError(f"Expected pano_images shape [B, 3, H, W], got {tuple(pano_images.shape)}")

    B, _, pano_h, pano_w = pano_images.shape
    device = pano_images.device
    dtype = pano_images.dtype

    yaw = yaw.to(device=device, dtype=dtype).flatten()
    pitch = pitch.to(device=device, dtype=dtype).flatten()
    if yaw.shape != pitch.shape:
        raise ValueError(f"yaw and pitch must share shape, got {tuple(yaw.shape)} and {tuple(pitch.shape)}")

    fov_x, fov_y = _normalize_fov(fov, yaw.shape[0], device=device, dtype=dtype)
    S = yaw.shape[0]
    yaw_b = yaw[None, :].expand(B, S)
    pitch_b = pitch[None, :].expand(B, S)
    fov_x_b = fov_x[None, :].expand(B, S)
    fov_y_b = fov_y[None, :].expand(B, S)

    rays = pinhole_rays(
        yaw_b.reshape(-1),
        pitch_b.reshape(-1),
        fov_x_b.reshape(-1),
        fov_y_b.reshape(-1),
        window_size,
        window_size,
        device=device,
        dtype=dtype,
    )
    _, _, u, v = rays_to_equirectangular(rays)
    windows = _sample_equirectangular(
        pano_images,
        u.reshape(B, S, window_size, window_size),
        v.reshape(B, S, window_size, window_size),
        mode=interpolation_mode,
    )

    token_meta = _build_token_meta(
        yaw_b=yaw_b,
        pitch_b=pitch_b,
        fov_x_b=fov_x_b,
        fov_y_b=fov_y_b,
        pano_hw=(pano_h, pano_w),
        window_size=window_size,
        patch_size=patch_size,
        seam_width=seam_width,
    )
    camera_meta = _build_camera_meta(yaw_b, pitch_b, fov_x_b, fov_y_b)

    return PanoSamplerOutput(windows=windows, camera_meta=camera_meta, token_meta=token_meta)


def _normalize_fov(fov, num_views: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(fov):
        fov_tensor = fov.to(device=device, dtype=dtype)
    else:
        fov_tensor = torch.tensor(fov, device=device, dtype=dtype)

    if fov_tensor.ndim == 0:
        fov_x = fov_tensor.expand(num_views)
        fov_y = fov_tensor.expand(num_views)
    elif fov_tensor.ndim == 1:
        if fov_tensor.numel() == 1:
            fov_x = fov_tensor.expand(num_views)
            fov_y = fov_tensor.expand(num_views)
        elif fov_tensor.numel() == num_views:
            fov_x = fov_tensor
            fov_y = fov_tensor
        elif fov_tensor.numel() == 2:
            fov_x = fov_tensor[0].expand(num_views)
            fov_y = fov_tensor[1].expand(num_views)
        else:
            raise ValueError(f"Cannot broadcast fov shape {tuple(fov_tensor.shape)} to {num_views} views")
    elif fov_tensor.ndim == 2 and fov_tensor.shape == (num_views, 2):
        fov_x = fov_tensor[:, 0]
        fov_y = fov_tensor[:, 1]
    else:
        raise ValueError(f"Unsupported fov shape {tuple(fov_tensor.shape)}")
    return fov_x, fov_y


def _sample_equirectangular(
    pano_images: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    B, C, pano_h, pano_w = pano_images.shape
    _, S, out_h, out_w = u.shape
    pano_pad = torch.cat([pano_images[..., -1:], pano_images, pano_images[..., :1]], dim=-1)
    pano_pad = pano_pad[:, None].expand(B, S, C, pano_h, pano_w + 2).reshape(B * S, C, pano_h, pano_w + 2)

    x_pixel = u.reshape(B * S, out_h, out_w) * (pano_w - 1) + 1.0
    y_pixel = v.reshape(B * S, out_h, out_w) * (pano_h - 1)
    grid_x = 2.0 * x_pixel / (pano_w + 1) - 1.0
    grid_y = 2.0 * y_pixel / (pano_h - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    windows = F.grid_sample(pano_pad, grid, mode=mode, padding_mode="border", align_corners=True)
    return windows.reshape(B, S, C, out_h, out_w)


def _build_token_meta(
    yaw_b: torch.Tensor,
    pitch_b: torch.Tensor,
    fov_x_b: torch.Tensor,
    fov_y_b: torch.Tensor,
    pano_hw: Tuple[int, int],
    window_size: int,
    patch_size: int,
    seam_width: float,
) -> Dict[str, torch.Tensor]:
    B, S = yaw_b.shape
    pano_h, pano_w = pano_hw
    patch_h = window_size // patch_size
    patch_w = window_size // patch_size

    rays = pinhole_rays(
        yaw_b.reshape(-1),
        pitch_b.reshape(-1),
        fov_x_b.reshape(-1),
        fov_y_b.reshape(-1),
        patch_h,
        patch_w,
        device=yaw_b.device,
        dtype=yaw_b.dtype,
    )
    theta, phi, u, v = rays_to_equirectangular(rays)
    sphere_dir = rays.reshape(B, S, patch_h * patch_w, 3)
    theta = theta.reshape(B, S, patch_h * patch_w)
    phi = phi.reshape(B, S, patch_h * patch_w)
    u = u.reshape(B, S, patch_h * patch_w)
    v = v.reshape(B, S, patch_h * patch_w)

    global_w = max(1, math.ceil(pano_w / patch_size))
    global_h = max(1, math.ceil(pano_h / patch_size))
    global_x = torch.floor(u * global_w).to(torch.long).clamp(0, global_w - 1)
    global_y = torch.floor(v * global_h).to(torch.long).clamp(0, global_h - 1)
    global_patch_id = global_y * global_w + global_x

    local_y, local_x = torch.meshgrid(
        torch.arange(patch_h, device=yaw_b.device),
        torch.arange(patch_w, device=yaw_b.device),
        indexing="ij",
    )
    local_x = local_x.reshape(1, 1, -1).expand(B, S, -1)
    local_y = local_y.reshape(1, 1, -1).expand(B, S, -1)
    window_id = torch.arange(S, device=yaw_b.device).reshape(1, S, 1).expand(B, S, patch_h * patch_w)
    pano_id = torch.arange(B, device=yaw_b.device).reshape(B, 1, 1).expand(B, S, patch_h * patch_w)
    sphere_encoding = build_sphere_encoding(theta, phi, sphere_dir)

    return {
        "pano_id": pano_id,
        "window_id": window_id,
        "local_patch_x": local_x,
        "local_patch_y": local_y,
        "pano_u": u * (pano_w - 1),
        "pano_v": v * (pano_h - 1),
        "theta": theta,
        "phi": phi,
        "sphere_dir": sphere_dir,
        "sphere_encoding": sphere_encoding,
        "global_patch_id": global_patch_id,
        "is_seam_region": (u <= seam_width) | (u >= 1.0 - seam_width),
        "patch_hw": torch.tensor([patch_h, patch_w], device=yaw_b.device),
        "global_hw": torch.tensor([global_h, global_w], device=yaw_b.device),
    }


def _build_camera_meta(
    yaw_b: torch.Tensor,
    pitch_b: torch.Tensor,
    fov_x_b: torch.Tensor,
    fov_y_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    camera_encoding = build_camera_encoding(yaw_b, pitch_b, fov_x_b, fov_y_b)
    rotations = camera_rotation_from_yaw_pitch(yaw_b, pitch_b)
    translations = torch.zeros(*yaw_b.shape, 3, device=yaw_b.device, dtype=yaw_b.dtype)
    return {
        "yaw": yaw_b,
        "pitch": pitch_b,
        "fov_x": fov_x_b,
        "fov_y": fov_y_b,
        "view_params": torch.stack([yaw_b, pitch_b, fov_y_b, fov_x_b], dim=-1),
        "camera_encoding": camera_encoding,
        "rotations": rotations,
        "translations": translations,
    }
