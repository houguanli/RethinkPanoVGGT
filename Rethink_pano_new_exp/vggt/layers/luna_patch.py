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
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, tokens: torch.Tensor, token_meta: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Apply global_patch_id mean pooling and residual correction.

        Args:
            tokens: Patch tokens with shape [B, S, N, C].
            token_meta: Dict containing `global_patch_id` [B, S, N] and
                optional `sphere_encoding` or `sphere_dir`.
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
            # Parameter-matched Patch Bank ablation: preserve the full adapter
            # input width and MLP, but remove all cross-window bank information.
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

        return scatter_mean_by_global_id(
            tokens,
            global_ids,
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
    """Normalize user-facing aliases for the three Patch Bank conditions."""
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
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
) -> torch.Tensor:
    """Build and gather a batch-local Patch Bank without ``torch_scatter``.

    When ``shuffle`` is true, only the occupied bank cells are permuted within
    each panorama. This preserves the bank size and feature distribution while
    breaking the ERP correspondence used for gathering.
    """
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
    bank = (sums / counts.clamp_min(1.0)).reshape(B, num_ids, C)
    counts = counts.reshape(B, num_ids, 1)

    if shuffle:
        bank = _shuffle_occupied_bank_cells(bank, counts, shuffle_seed)

    batch_idx = torch.arange(B, device=tokens.device)[:, None].expand(B, S * N)
    gathered = bank[batch_idx, flat_ids.clamp_min(0)].reshape(B, S, N, C)
    return torch.where(valid.reshape(B, S, N, 1), gathered, torch.zeros_like(gathered))


def _shuffle_occupied_bank_cells(
    bank: torch.Tensor,
    counts: torch.Tensor,
    shuffle_seed: Optional[int],
) -> torch.Tensor:
    """Permute valid Patch Bank entries per panorama, leaving empty cells empty."""
    shuffled = bank.clone()
    for batch_idx in range(bank.shape[0]):
        occupied = torch.nonzero(counts[batch_idx, :, 0] > 0, as_tuple=False).flatten()
        if occupied.numel() < 2:
            continue

        generator = None
        if shuffle_seed is not None:
            generator = torch.Generator(device=bank.device)
            generator.manual_seed(int(shuffle_seed) + batch_idx)
        permutation = torch.randperm(occupied.numel(), device=bank.device, generator=generator)
        identity = torch.arange(occupied.numel(), device=bank.device)
        if torch.equal(permutation, identity):
            permutation = torch.roll(permutation, shifts=1)
        shuffled[batch_idx, occupied] = bank[batch_idx, occupied[permutation]]
    return shuffled
