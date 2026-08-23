"""Lightweight spherical completion head conditioned on VGGT-Omega depth.

The head never replaces the canonical Omega reconstruction.  It predicts a
bounded log-depth residual for pixels not covered by the perspective windows;
the observed core is copied from the Omega ERP splat and only a narrow boundary
band is blended.  Horizontal circular padding respects the ERP seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers.pano_position import pinhole_rays, rays_to_equirectangular


def _pad_erp(x: torch.Tensor, padding: int) -> torch.Tensor:
    if padding <= 0:
        return x
    x = F.pad(x, (padding, padding, 0, 0), mode="circular")
    return F.pad(x, (0, 0, padding, padding), mode="replicate")


class ERPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_pad_erp(x, self.padding))


class ERPResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = ERPConv(channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = ERPConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(x)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return x + residual


class ERPRemainingBandHead(nn.Module):
    """Small circular U-Net that completes relative log depth on the ERP."""

    def __init__(self, width: int = 32, max_log_residual: float = 2.5) -> None:
        super().__init__()
        self.width = int(width)
        self.max_log_residual = float(max_log_residual)
        # RGB(3), Omega log depth, coverage, boundary distance, sin/cos latitude.
        self.stem = ERPConv(8, width)
        self.enc1 = nn.Sequential(ERPResidualBlock(width), ERPResidualBlock(width))
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(ERPResidualBlock(width * 2), ERPResidualBlock(width * 2))
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(ERPResidualBlock(width * 4), ERPResidualBlock(width * 4))
        self.up2 = nn.Conv2d(width * 4 + width * 2, width * 2, 1)
        self.dec2 = nn.Sequential(ERPResidualBlock(width * 2), ERPResidualBlock(width * 2))
        self.up1 = nn.Conv2d(width * 2 + width, width, 1)
        self.dec1 = nn.Sequential(ERPResidualBlock(width), ERPResidualBlock(width))
        self.out = ERPConv(width, 2)
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        omega_depth: torch.Tensor,
        coverage: torch.Tensor,
        *,
        blend_width_pixels: int = 12,
    ) -> Dict[str, torch.Tensor]:
        if rgb.ndim != 4 or omega_depth.ndim != 4 or coverage.ndim != 4:
            raise ValueError("Expected RGB/depth/coverage tensors in BCHW layout")
        coverage = coverage.float().clamp(0.0, 1.0)
        base_depth = fill_missing_depth(omega_depth, coverage)
        boundary_distance = soft_coverage_distance(coverage, blend_width_pixels)
        lat = latitude_channels(rgb.shape[-2], rgb.shape[-1], rgb.device, rgb.dtype).expand(
            rgb.shape[0], -1, -1, -1
        )
        x = torch.cat(
            [rgb, torch.log(base_depth.clamp_min(1e-6)), coverage, boundary_distance, lat],
            dim=1,
        )
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        u2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat([u2, e2], dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(self.up1(torch.cat([u1, e1], dim=1)))
        raw = self.out(features)
        delta_log = torch.tanh(raw[:, :1]) * self.max_log_residual
        confidence = torch.sigmoid(raw[:, 1:2])
        completion = base_depth * torch.exp(delta_log)
        # Copy the trusted core exactly; blend only immediately outside it.
        outside_blend = (1.0 - coverage) * boundary_distance
        output = base_depth * outside_blend + completion * (1.0 - outside_blend)
        output = torch.where(coverage > 0.5, omega_depth.clamp_min(1e-6), output)
        return {
            "depth": output,
            "completion_depth": completion,
            "confidence": confidence,
            "delta_log_depth": delta_log,
            "base_depth": base_depth,
            "coverage": coverage,
            "boundary_distance": boundary_distance,
        }


def load_erp_completion_head(path: str | Path, device: torch.device | str = "cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "completion_head" not in payload:
        raise ValueError(f"Not an ERP completion checkpoint: {path}")
    head_args = payload.get("head_args", {})
    head = ERPRemainingBandHead(
        width=int(head_args.get("width", 32)),
        max_log_residual=float(head_args.get("max_log_residual", 2.5)),
    )
    head.load_state_dict(payload["completion_head"])
    head.to(device).eval()
    return head, payload


def latitude_channels(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latitude = math.pi * (0.5 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height)
    sin_lat = latitude.sin().view(1, 1, height, 1).expand(1, 1, height, width)
    cos_lat = latitude.cos().view(1, 1, height, 1).expand(1, 1, height, width)
    return torch.cat([sin_lat, cos_lat], dim=1)


def spherical_pixel_weights(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latitude = math.pi * (0.5 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height)
    # A small floor keeps the exact pole trainable while retaining correct area emphasis.
    weights = latitude.cos().clamp_min(0.05)
    return weights.view(1, 1, height, 1).expand(1, 1, height, width)


def fill_missing_depth(depth: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    valid = (coverage > 0.5) & torch.isfinite(depth) & (depth > 0)
    flat_depth = depth.flatten(1)
    flat_valid = valid.flatten(1)
    fills = []
    for index in range(depth.shape[0]):
        values = flat_depth[index][flat_valid[index]]
        fills.append(values.median() if values.numel() else depth.new_tensor(1.0))
    fill = torch.stack(fills).view(-1, 1, 1, 1).clamp_min(1e-4)
    return torch.where(valid, depth, fill)


def soft_coverage_distance(coverage: torch.Tensor, width: int) -> torch.Tensor:
    """Return an outside-boundary blend weight in [0,1]."""
    width = max(int(width), 1)
    current = coverage.float().clamp(0.0, 1.0)
    reached = current.clone()
    distance = torch.zeros_like(current)
    for step in range(1, width + 1):
        expanded = F.max_pool2d(_pad_erp(reached, 1), kernel_size=3, stride=1, padding=0)
        ring = (expanded - reached).clamp(0.0, 1.0)
        distance = distance + ring * (1.0 - float(step - 1) / float(width))
        reached = expanded
    return distance * (1.0 - current)


@dataclass
class ERPSplat:
    depth: torch.Tensor
    valid_mask: torch.Tensor
    counts: torch.Tensor


def splat_omega_window_depth_to_erp(
    pred_window_z: torch.Tensor,
    camera_meta: Dict[str, torch.Tensor],
    *,
    num_panos: int,
    erp_height: int,
    erp_width: int,
) -> ERPSplat:
    """Differentiable-value nearest splat from Omega z-depth windows to ERP range depth."""
    if pred_window_z.ndim == 5:
        pred_window_z = pred_window_z[..., 0]
    batch, views, win_h, win_w = pred_window_z.shape
    if views % int(num_panos):
        raise ValueError(f"{views} views cannot be divided over {num_panos} panoramas")
    yaw = camera_meta["yaw"].reshape(-1)
    pitch = camera_meta["pitch"].reshape(-1)
    fov_x = camera_meta["fov_x"].reshape(-1)
    fov_y = camera_meta["fov_y"].reshape(-1)
    rays = pinhole_rays(yaw, pitch, fov_x, fov_y, win_h, win_w, device=yaw.device, dtype=yaw.dtype)
    _, _, u, v = rays_to_equirectangular(rays)
    rotations = camera_meta["rotations"].reshape(-1, 3, 3)
    forward = rotations[..., :, 2]
    z_factor = (rays * forward[:, None, None, :]).sum(dim=-1).clamp_min(1e-6)
    range_depth = pred_window_z.reshape(-1, win_h, win_w) / z_factor
    x = torch.remainder(torch.round(u * erp_width - 0.5).long(), erp_width)
    y = torch.round(v * erp_height - 0.5).long().clamp(0, erp_height - 1)
    pixels = y * erp_width + x
    out_depth = pred_window_z.new_zeros((batch * num_panos, erp_height * erp_width), dtype=torch.float32)
    out_count = pred_window_z.new_zeros((batch * num_panos, erp_height * erp_width), dtype=torch.float32)
    views_per_pano = views // num_panos
    for b in range(batch):
        for p in range(num_panos):
            start = b * views + p * views_per_pano
            end = start + views_per_pano
            value = range_depth[start:end].reshape(-1).float()
            index = pixels[start:end].reshape(-1)
            valid = torch.isfinite(value) & (value > 0)
            target = b * num_panos + p
            out_depth[target].scatter_add_(0, index[valid], value[valid])
            out_count[target].scatter_add_(0, index[valid], torch.ones_like(value[valid]))
    valid = out_count > 0
    out_depth = out_depth / out_count.clamp_min(1.0)
    shape = (batch * num_panos, 1, erp_height, erp_width)
    return ERPSplat(out_depth.view(shape), valid.view(shape), out_count.view(shape))
