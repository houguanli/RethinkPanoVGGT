from typing import Dict, Optional

import torch
import torch.nn as nn


class LunaPatchAdapter(nn.Module):
    """Lightweight residual adapter for cross-window pano patch sharing."""

    def __init__(self, dim: int, sphere_dim: int = 7, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.sphere_dim = sphere_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2 + sphere_dim),
            nn.Linear(dim * 2 + sphere_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, tokens: torch.Tensor, token_meta: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Apply global_patch_id mean pooling and residual correction.

        Args:
            tokens: Patch tokens with shape [B, S, N, C].
            token_meta: Dict containing `global_patch_id` [B, S, N] and
                optional `sphere_encoding` or `sphere_dir`.
        """
        if token_meta is None or "global_patch_id" not in token_meta:
            return tokens

        global_ids = token_meta["global_patch_id"].to(device=tokens.device, dtype=torch.long)
        if global_ids.ndim == 4:
            global_ids = global_ids.flatten(2)
        if global_ids.shape != tokens.shape[:3]:
            raise ValueError(
                f"Expected global_patch_id shape {tokens.shape[:3]}, got {tuple(global_ids.shape)}"
            )

        global_feat = scatter_mean_by_global_id(tokens, global_ids)
        sphere_enc = self._get_sphere_encoding(tokens, token_meta)
        corr = self.mlp(torch.cat([tokens, global_feat, sphere_enc], dim=-1))
        return tokens + self.alpha.to(dtype=tokens.dtype) * corr

    def _get_sphere_encoding(self, tokens: torch.Tensor, token_meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "sphere_encoding" in token_meta:
            enc = token_meta["sphere_encoding"].to(device=tokens.device, dtype=tokens.dtype)
        elif "sphere_dir" in token_meta:
            enc = token_meta["sphere_dir"].to(device=tokens.device, dtype=tokens.dtype)
        else:
            return tokens.new_zeros(*tokens.shape[:3], self.sphere_dim)

        if enc.ndim == 5:
            enc = enc.flatten(2, 3)
        elif enc.ndim == 4 and enc.shape[:3] != tokens.shape[:3]:
            enc = enc.flatten(2)

        if enc.shape[:3] != tokens.shape[:3]:
            raise ValueError(f"Expected sphere encoding shape {tokens.shape[:3]}, got {tuple(enc.shape[:3])}")

        if enc.shape[-1] < self.sphere_dim:
            pad = tokens.new_zeros(*enc.shape[:-1], self.sphere_dim - enc.shape[-1])
            enc = torch.cat([enc, pad], dim=-1)
        elif enc.shape[-1] > self.sphere_dim:
            enc = enc[..., : self.sphere_dim]
        return enc


def scatter_mean_by_global_id(tokens: torch.Tensor, global_ids: torch.Tensor) -> torch.Tensor:
    """Batch-local scatter mean without requiring torch_scatter."""
    B, S, N, C = tokens.shape
    flat_ids = global_ids.reshape(B, S * N)
    valid = flat_ids >= 0

    if not bool(valid.any()):
        return torch.zeros_like(tokens)

    num_ids = int(flat_ids[valid].max().item()) + 1
    offsets = torch.arange(B, device=tokens.device, dtype=torch.long)[:, None] * num_ids
    safe_ids = flat_ids.clamp_min(0) + offsets
    flat_tokens = tokens.reshape(B * S * N, C)
    flat_safe_ids = safe_ids.reshape(-1)
    flat_valid = valid.reshape(-1)

    bank_size = B * num_ids
    sums = tokens.new_zeros(bank_size, C)
    counts = tokens.new_zeros(bank_size, 1)
    sums.index_add_(0, flat_safe_ids[flat_valid], flat_tokens[flat_valid])
    counts.index_add_(0, flat_safe_ids[flat_valid], tokens.new_ones(int(flat_valid.sum().item()), 1))
    bank = sums / counts.clamp_min(1.0)

    gathered = bank[flat_safe_ids.reshape(-1)].reshape(B, S, N, C)
    return torch.where(valid.reshape(B, S, N, 1), gathered, torch.zeros_like(gathered))
