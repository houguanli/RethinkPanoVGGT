"""LUNA-Camera residual adapter for known-virtual-camera injection (Omega).

Mirrors `Rethink_pano_new_exp/vggt/layers/luna_camera.py`. The module body is
identical — only the import path differs.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class LunaCameraAdapter(nn.Module):
    """Inject known virtual-camera metadata into VGGT camera tokens."""

    def __init__(self, dim: int, camera_meta_dim: int = 16, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.camera_meta_dim = camera_meta_dim
        self.camera_mlp = nn.Sequential(
            nn.LayerNorm(camera_meta_dim),
            nn.Linear(camera_meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        camera_tokens: torch.Tensor,
        camera_meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        """Apply residual metadata injection.

        Args:
            camera_tokens: Camera tokens with shape [B, S, C].
            camera_meta: Tensor [B, S, D] or dict with ``camera_encoding``.
        """
        if camera_meta is None:
            return camera_tokens

        meta = camera_meta["camera_encoding"] if isinstance(camera_meta, dict) else camera_meta
        meta = meta.to(device=camera_tokens.device, dtype=camera_tokens.dtype)
        if meta.shape[:2] != camera_tokens.shape[:2]:
            raise ValueError(f"Expected camera_meta shape {camera_tokens.shape[:2]}, got {tuple(meta.shape[:2])}")

        if meta.shape[-1] < self.camera_meta_dim:
            pad = camera_tokens.new_zeros(*meta.shape[:-1], self.camera_meta_dim - meta.shape[-1])
            meta = torch.cat([meta, pad], dim=-1)
        elif meta.shape[-1] > self.camera_meta_dim:
            meta = meta[..., : self.camera_meta_dim]

        corr = self.camera_mlp(meta)
        return camera_tokens + self.alpha.to(dtype=camera_tokens.dtype) * corr
