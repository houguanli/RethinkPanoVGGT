"""Project pinhole-window Z depth back to panorama radial depth."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _as_bv(value: torch.Tensor, batch: int, views: int, *, device, dtype) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(batch, -1)
    if value.shape != (batch, views):
        raise ValueError(f"Expected [B,V]=[{batch},{views}], got {tuple(value.shape)}")
    return value


def _window_rays(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    height: int,
    width: int,
    *,
    align_corners: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = yaw.device, yaw.dtype
    if align_corners:
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    else:
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width) * 2.0 - 1.0
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height) * 2.0 - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    cos_pitch = torch.cos(pitch)
    forward = torch.stack(
        [cos_pitch * torch.sin(yaw), torch.sin(pitch), cos_pitch * torch.cos(yaw)], dim=-1
    )
    right = torch.stack([torch.cos(yaw), torch.zeros_like(yaw), -torch.sin(yaw)], dim=-1)
    up = torch.cross(forward, right, dim=-1)
    forward = F.normalize(forward, dim=-1)
    right = F.normalize(right, dim=-1)
    up = F.normalize(up, dim=-1)

    x_plane = xx[None, None] * torch.tan(fov_x[..., None, None] * 0.5)
    y_plane = -yy[None, None] * torch.tan(fov_y[..., None, None] * 0.5)
    rays = (
        forward[..., None, None, :]
        + x_plane[..., None] * right[..., None, None, :]
        + y_plane[..., None] * up[..., None, None, :]
    )
    rays = F.normalize(rays, dim=-1)
    z_factor = (rays * forward[..., None, None, :]).sum(dim=-1).clamp_min(1e-6)
    return rays, z_factor


def splat_window_z_depth_to_erp(
    window_z_depth: torch.Tensor,
    *,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    view_pano_index: torch.Tensor,
    num_panos: int,
    erp_height: int,
    erp_width: int,
    align_corners: bool,
) -> dict[str, torch.Tensor]:
    """Splat window Z depth into per-pano ERP radial-depth maps.

    Coverage is geometric and independent of prediction validity. ``valid_mask``
    additionally requires a finite positive prediction and is the mask that
    should be intersected with the GT-valid mask for metric computation.
    """
    if window_z_depth.ndim == 5 and window_z_depth.shape[-1] == 1:
        window_z_depth = window_z_depth[..., 0]
    if window_z_depth.ndim != 4:
        raise ValueError(f"Expected depth [B,V,H,W], got {tuple(window_z_depth.shape)}")
    batch, views, height, width = window_z_depth.shape
    device = window_z_depth.device
    work_dtype = torch.float32
    depth = window_z_depth.to(dtype=work_dtype)
    yaw = _as_bv(yaw, batch, views, device=device, dtype=work_dtype)
    pitch = _as_bv(pitch, batch, views, device=device, dtype=work_dtype)
    fov_x = _as_bv(fov_x, batch, views, device=device, dtype=work_dtype)
    fov_y = _as_bv(fov_y, batch, views, device=device, dtype=work_dtype)
    pano_ids = torch.as_tensor(view_pano_index, device=device, dtype=torch.long)
    if pano_ids.ndim == 1:
        pano_ids = pano_ids.unsqueeze(0).expand(batch, -1)
    if pano_ids.shape != (batch, views):
        raise ValueError(f"Expected view_pano_index [B,V], got {tuple(pano_ids.shape)}")
    if bool(((pano_ids < 0) | (pano_ids >= int(num_panos))).any()):
        raise ValueError("view_pano_index is outside [0, num_panos)")

    rays, z_factor = _window_rays(
        yaw, pitch, fov_x, fov_y, height, width, align_corners=align_corners
    )
    radial_depth = depth / z_factor
    theta = torch.atan2(rays[..., 0], rays[..., 2])
    phi = torch.asin(rays[..., 1].clamp(-1.0, 1.0))
    u = torch.remainder(theta / (2.0 * math.pi) + 0.5, 1.0)
    v = (0.5 - phi / math.pi).clamp(0.0, 1.0)
    x = u * float(erp_width - 1)
    y = v * float(erp_height - 1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long().clamp(0, erp_height - 1)
    x1 = (x0 + 1) % erp_width
    y1 = (y0 + 1).clamp(0, erp_height - 1)
    wx = x - x0.to(x.dtype)
    wy = y - y0.to(y.dtype)

    total_cells = batch * int(num_panos) * erp_height * erp_width
    depth_sum = depth.new_zeros(total_cells)
    valid_weight = depth.new_zeros(total_cells)
    coverage_weight = depth.new_zeros(total_cells)
    batch_offsets = torch.arange(batch, device=device)[:, None, None, None] * int(num_panos)
    pano_bank = batch_offsets + pano_ids[:, :, None, None]
    pred_valid = torch.isfinite(radial_depth) & (radial_depth > 0.0)
    # Downweight grazing rays that stretch a small window area over many ERP pixels.
    center_weight = z_factor.square()

    for xx_idx, yy_idx, bilinear_weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        weights = bilinear_weight * center_weight
        flat_index = (((pano_bank * erp_height + yy_idx) * erp_width) + xx_idx).reshape(-1)
        coverage_weight.scatter_add_(0, flat_index, weights.reshape(-1))
        valid_weights = torch.where(pred_valid, weights, torch.zeros_like(weights))
        valid_weight.scatter_add_(0, flat_index, valid_weights.reshape(-1))
        depth_sum.scatter_add_(
            0,
            flat_index,
            torch.where(pred_valid, radial_depth * weights, torch.zeros_like(weights)).reshape(-1),
        )

    shape = (batch, int(num_panos), erp_height, erp_width)
    coverage_weight = coverage_weight.reshape(shape)
    valid_weight = valid_weight.reshape(shape)
    erp_depth = (depth_sum.reshape(shape) / valid_weight.clamp_min(1e-8)).to(window_z_depth.dtype)
    return {
        "depth": erp_depth,
        "coverage_mask": coverage_weight > 1e-8,
        "valid_mask": valid_weight > 1e-8,
        "coverage_weight": coverage_weight,
        "valid_weight": valid_weight,
    }
