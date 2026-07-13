"""Pano-level camera head for grouped ERP windows."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanoCameraHead(nn.Module):
    """Pool virtual-window camera tokens into one relative pose residual per panorama.

    The released Omega camera head predicts one pinhole pose per input frame.
    This head has different semantics: known virtual-view metadata is injected
    before within-pano pooling, and cross-pano attention then predicts a
    residual on top of the pretrained Omega window-pose initialization.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 512,
        camera_meta_dim: int = 16,
        num_heads: int = 8,
        num_cross_pano_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.camera_meta_dim = int(camera_meta_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(dim_in, eps=1e-5),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
        )
        self.meta_proj = nn.Sequential(
            nn.LayerNorm(self.camera_meta_dim, eps=1e-5),
            nn.Linear(self.camera_meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.window_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.pano_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.pano_query, std=0.02)
        self.intra_pano_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.pano_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_pano_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_cross_pano_layers,
            enable_nested_tensor=False,
        )
        self.relative_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.center_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.rotation_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self._initialize_relative_output()

    def _initialize_relative_output(self) -> None:
        center_out = self.center_branch[-1]
        rotation_out = self.rotation_branch[-1]
        nn.init.zeros_(center_out.weight)
        nn.init.zeros_(center_out.bias)
        nn.init.zeros_(rotation_out.weight)
        nn.init.zeros_(rotation_out.bias)
        # Omega quaternions store the real component last: [x, y, z, w].
        rotation_out.bias.data[-1] = 1.0

    def forward(
        self,
        window_camera_tokens: torch.Tensor,
        camera_encoding: torch.Tensor,
        num_panos: int,
    ) -> dict[str, torch.Tensor]:
        if window_camera_tokens.ndim != 3:
            raise ValueError(
                "window_camera_tokens must have shape [B, N*V, D], "
                f"got {tuple(window_camera_tokens.shape)}"
            )
        if camera_encoding.ndim != 3:
            raise ValueError(
                f"camera_encoding must have shape [B, N*V, M], got {tuple(camera_encoding.shape)}"
            )
        if window_camera_tokens.shape[:2] != camera_encoding.shape[:2]:
            raise ValueError(
                "window token and camera metadata shapes disagree: "
                f"{tuple(window_camera_tokens.shape[:2])} vs {tuple(camera_encoding.shape[:2])}"
            )
        if num_panos < 1 or window_camera_tokens.shape[1] % int(num_panos) != 0:
            raise ValueError(
                f"Cannot group {window_camera_tokens.shape[1]} windows into {num_panos} panos"
            )

        batch_size, total_windows, _ = window_camera_tokens.shape
        views_per_pano = total_windows // int(num_panos)
        meta = _fit_last_dim(camera_encoding, self.camera_meta_dim)
        windows = self.token_proj(window_camera_tokens) + self.meta_proj(meta)
        windows = self.window_norm(windows)
        windows = windows.reshape(batch_size * num_panos, views_per_pano, -1)

        query = self.pano_query.expand(batch_size * num_panos, -1, -1)
        pooled, _ = self.intra_pano_attention(query, windows, windows, need_weights=False)
        pooled = self.pano_norm(query + pooled).reshape(batch_size, num_panos, -1)
        contextual = self.cross_pano_attention(pooled)

        # The head predicts directly in pano-0 coordinates. Subtracting the
        # anchor feature makes the first center exactly zero and makes all
        # outputs invariant to a shared feature offset.
        relative = self.relative_norm(contextual - contextual[:, :1])
        raw_centers = self.center_branch(relative)
        centers = raw_centers - raw_centers[:, :1]
        raw_quaternions = F.normalize(self.rotation_branch(relative), dim=-1, eps=1e-6)
        anchor_inverse = _quaternion_conjugate(raw_quaternions[:, :1])
        quaternions = F.normalize(
            _quaternion_multiply(raw_quaternions, anchor_inverse),
            dim=-1,
            eps=1e-6,
        )
        return {
            "pano_camera_center_residual": centers,
            "pano_rotation_quat_w2c_residual": quaternions,
        }


def _fit_last_dim(value: torch.Tensor, target_dim: int) -> torch.Tensor:
    if value.shape[-1] == target_dim:
        return value
    if value.shape[-1] > target_dim:
        return value[..., :target_dim]
    return F.pad(value, (0, target_dim - value.shape[-1]))


def _quaternion_conjugate(value: torch.Tensor) -> torch.Tensor:
    return torch.cat([-value[..., :3], value[..., 3:4]], dim=-1)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dim=-1,
    )
