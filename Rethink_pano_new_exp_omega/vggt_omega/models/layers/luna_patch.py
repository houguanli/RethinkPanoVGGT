"""LUNA-Patch residual adapter for cross-window pano patch sharing (Omega).

Functionally identical to `Rethink_pano_new_exp/vggt/layers/luna_patch.py`.
Kept inside the VGGT-Omega package tree so the LUNA experiment can sit next to
the rest of the omega layers (SelfAttention/RoPE/etc.) without sys.path tricks.

The MLP/LayerNorm path is plain torch.nn — it does NOT depend on the omega
SelfAttentionBlock, so the b1 implementation transfers verbatim. The only thing
that changes between b1 and omega is *where* this adapter is called from
(see `vggt_omega.models.aggregator` in this experiment).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class LunaPatchAdapter(nn.Module):
    """Lightweight residual adapter for cross-window pano patch sharing."""

    def __init__(
        self,
        dim: int,
        sphere_dim: int = 7,
        hidden_dim: Optional[int] = None,
        patch_bank_mode: str = "aligned",
        patch_bank_shuffle_seed: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.sphere_dim = sphere_dim
        self.patch_bank_mode = normalize_patch_bank_mode(patch_bank_mode)
        self.patch_bank_shuffle_seed = patch_bank_shuffle_seed
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2 + sphere_dim),
            nn.Linear(dim * 2 + sphere_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # alpha kept fp32 — initial 0 means residual is no-op at load time.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, tokens: torch.Tensor, token_meta: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Apply global_patch_id mean pooling and residual correction.

        Args:
            tokens: Patch tokens with shape [B, S, N, C] (patch tokens only,
                special tokens already stripped by the aggregator).
            token_meta: Dict containing ``global_patch_id`` [B, S, N] and
                optional ``sphere_encoding`` or ``sphere_dir``.
        """
        if token_meta is None:
            return tokens

        global_feat = self._get_patch_bank_context(tokens, token_meta)
        if global_feat is None:
            return tokens
        sphere_enc = self._get_sphere_encoding(tokens, token_meta)
        corr = self.mlp(torch.cat([tokens, global_feat, sphere_enc], dim=-1))
        return tokens + self.alpha.to(dtype=tokens.dtype) * corr

    def _get_patch_bank_context(
        self,
        tokens: torch.Tensor,
        token_meta: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.patch_bank_mode == "none":
            # Parameter-matched control: retain the full concatenated input
            # width and MLP, but remove cross-window Patch Bank information.
            return torch.zeros_like(tokens)

        if "global_patch_id" not in token_meta:
            return None

        global_ids = token_meta["global_patch_id"].to(device=tokens.device, dtype=torch.long)
        if global_ids.ndim == 4:
            global_ids = global_ids.flatten(2)
        if global_ids.shape != tokens.shape[:3]:
            raise ValueError(
                f"Expected global_patch_id shape {tokens.shape[:3]}, got {tuple(global_ids.shape)}"
            )

        pano_ids = token_meta.get("pano_id")
        if pano_ids is not None:
            pano_ids = pano_ids.to(device=tokens.device, dtype=torch.long)
            if pano_ids.ndim == 4:
                pano_ids = pano_ids.flatten(2)
            if pano_ids.shape != tokens.shape[:3]:
                raise ValueError(f"Expected pano_id shape {tokens.shape[:3]}, got {tuple(pano_ids.shape)}")

        return scatter_mean_by_global_id(
            tokens,
            global_ids,
            pano_ids=pano_ids,
            shuffle=self.patch_bank_mode == "shuffled",
            shuffle_seed=self.patch_bank_shuffle_seed,
        )

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


def normalize_patch_bank_mode(mode: str) -> str:
    """Normalize user-facing aliases for Patch Bank ablation conditions."""
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "aligned": "aligned",
        "on": "aligned",
        "bank": "aligned",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "shuffled": "shuffled",
        "shuffle": "shuffled",
        "random_shuffle": "shuffled",
    }
    if normalized not in aliases:
        choices = ", ".join(sorted({"aligned", "none", "shuffled"}))
        raise ValueError(f"Unsupported patch_bank_mode={mode!r}; expected one of: {choices}")
    return aliases[normalized]


def scatter_mean_by_global_id(
    tokens: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    pano_ids: Optional[torch.Tensor] = None,
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
) -> torch.Tensor:
    """Pool repeated ERP patches within each panorama.

    ``global_patch_id`` identifies a direction on one ERP sphere. The same ID
    in another panorama belongs to another camera center and must not share a
    mean pool, so multi-pano inputs use ``(pano_id, global_patch_id)`` as the
    grouping key.
    """
    B, S, N, C = tokens.shape
    flat_global_ids = global_ids.reshape(B, S * N)
    valid = flat_global_ids >= 0

    if pano_ids is None:
        flat_pano_ids = torch.zeros_like(flat_global_ids)
    else:
        if pano_ids.shape != global_ids.shape:
            raise ValueError(f"Expected pano_ids shape {tuple(global_ids.shape)}, got {tuple(pano_ids.shape)}")
        flat_pano_ids = pano_ids.reshape(B, S * N)
        valid = valid & (flat_pano_ids >= 0)

    if not bool(valid.any()):
        return torch.zeros_like(tokens)

    num_global_ids = int(flat_global_ids[valid].max().item()) + 1
    combined_ids = flat_pano_ids.clamp_min(0) * num_global_ids + flat_global_ids.clamp_min(0)
    num_combined_ids = int(combined_ids[valid].max().item()) + 1
    offsets = torch.arange(B, device=tokens.device, dtype=torch.long)[:, None] * num_combined_ids
    safe_ids = combined_ids + offsets
    flat_tokens = tokens.reshape(B * S * N, C)
    flat_safe_ids = safe_ids.reshape(-1)
    flat_valid = valid.reshape(-1)

    bank_size = B * num_combined_ids
    sums = tokens.new_zeros(bank_size, C)
    counts = tokens.new_zeros(bank_size, 1)
    sums.index_add_(0, flat_safe_ids[flat_valid], flat_tokens[flat_valid])
    counts.index_add_(0, flat_safe_ids[flat_valid], tokens.new_ones(int(flat_valid.sum().item()), 1))
    bank = (sums / counts.clamp_min(1.0)).reshape(B, num_combined_ids, C)
    counts = counts.reshape(B, num_combined_ids, 1)

    if shuffle:
        bank = _shuffle_occupied_bank_cells_per_pano(
            bank,
            counts,
            num_global_ids=num_global_ids,
            shuffle_seed=shuffle_seed,
        )

    batch_idx = torch.arange(B, device=tokens.device)[:, None].expand(B, S * N)
    gathered = bank[batch_idx, combined_ids].reshape(B, S, N, C)
    return torch.where(valid.reshape(B, S, N, 1), gathered, torch.zeros_like(gathered))


def _shuffle_occupied_bank_cells_per_pano(
    bank: torch.Tensor,
    counts: torch.Tensor,
    *,
    num_global_ids: int,
    shuffle_seed: Optional[int],
) -> torch.Tensor:
    """Permute occupied ERP cells without ever crossing panorama boundaries."""
    shuffled = bank.clone()
    bank_ids = torch.arange(bank.shape[1], device=bank.device)
    bank_pano_ids = torch.div(bank_ids, num_global_ids, rounding_mode="floor")
    for batch_idx in range(bank.shape[0]):
        occupied_mask = counts[batch_idx, :, 0] > 0
        active_panos = torch.unique(bank_pano_ids[occupied_mask])
        for pano_id in active_panos.tolist():
            occupied = torch.nonzero(
                occupied_mask & (bank_pano_ids == int(pano_id)),
                as_tuple=False,
            ).flatten()
            if occupied.numel() < 2:
                continue

            generator = None
            if shuffle_seed is not None:
                generator = torch.Generator(device=bank.device)
                generator.manual_seed(
                    int(shuffle_seed) + batch_idx * 1_000_003 + int(pano_id) * 10_007
                )
            permutation = torch.randperm(occupied.numel(), device=bank.device, generator=generator)
            identity = torch.arange(occupied.numel(), device=bank.device)
            if torch.equal(permutation, identity):
                permutation = torch.roll(permutation, shifts=1)
            shuffled[batch_idx, occupied] = bank[batch_idx, occupied[permutation]]
    return shuffled
