# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT-Omega aggregator extended with optional LUNA injection points.

This is the omega counterpart of `Rethink_pano_new_exp/vggt/models/aggregator.py`.
Compared with the released omega aggregator we add:

  * ``enable_pano_global_token`` / ``pano_geometry_embed`` — an extra learnable
    pano-level special token (sits between camera and register tokens), driven
    by per-view sin/cos(yaw,pitch) + FoV geometry features. Patch start index
    shifts by +1 when enabled.
  * ``enable_luna`` + per-layer ``LunaPatchAdapter`` / ``LunaCameraAdapter``
    injected **after** the (frame, inter_frame) pair of each backbone layer.

The injection happens right before we decide whether to cache the layer output.
This matches the b1 semantics ("LUNA runs after a full attention pair, then we
package the frame||global concat for downstream heads") while respecting the
omega-specific differences:

  - The omega aggregator caches only ``cached_layer_indices`` layers (default
    ``[4, 11, 17, 23]``); we keep LUNA-Patch defaults aligned with these so the
    DenseHead picks up the corrected features.
  - The omega ``"register"`` inter-frame block slices patch tokens out and only
    runs attention over camera+register across frames. LUNA still runs over the
    full ``[B, S, P, C]`` view after this branch re-concatenates patches, so the
    adapter signature is identical to the b1 path.
  - Patch tokens carry RoPE through ``frame_rope``; LUNA never touches RoPE — it
    is a residual MLP adapter on the tokens themselves.
"""

from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt_omega.models.layers import (
    LunaCameraAdapter,
    LunaPatchAdapter,
    Mlp,
    RopePositionEmbedding,
    SelfAttentionBlock,
)
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over pano windows with LUNA hooks.

    The base behaviour is identical to ``vggt_omega.models.aggregator.Aggregator``
    when ``enable_pano_global_token=False`` and ``enable_luna=False``.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: Sequence[int] = (2, 6, 9, 14, 20),
        cached_layer_indices: Sequence[int] = (4, 11, 17, 23),
        # ---- pano / LUNA additions (default off → bit-equivalent to base) ----
        enable_pano_global_token: bool = False,
        enable_pano_geometry_residual: bool = False,
        pano_geom_dim: int = 6,
        enable_luna: bool = False,
        luna_patch_layers: Optional[Iterable[int]] = None,
        luna_camera_layers: Optional[Iterable[int]] = None,
        luna_sphere_dim: int = 7,
        luna_camera_meta_dim: int = 16,
        luna_hidden_dim: Optional[int] = None,
        luna_patch_bank_mode: str = "aligned",
        luna_patch_bank_shuffle_seed: Optional[int] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        block_factory = lambda: SelfAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            ffn_layer=Mlp,
            init_values=1e-5,
            use_qk_norm=True,
            mask_k_bias=True,
        )
        self.frame_blocks = nn.ModuleList([block_factory() for _ in range(depth)])
        self.inter_frame_blocks = nn.ModuleList([block_factory() for _ in range(depth)])

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices: Set[int] = set(cached_layer_indices)
        self.enable_pano_global_token = enable_pano_global_token
        self.enable_pano_geometry_residual = bool(enable_pano_geometry_residual)
        self.enable_luna = enable_luna
        self.luna_patch_bank_mode = luna_patch_bank_mode
        self.luna_patch_bank_shuffle_seed = luna_patch_bank_shuffle_seed
        self.use_checkpoint = bool(use_checkpoint)

        # Camera + register tokens (b1/omega share this two-slot scheme).
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))

        # Optional pano geometry path. The residual mode does not add a token or
        # change patch_token_start; it gates geometry into the existing camera
        # token with alpha=0 so loading an old checkpoint starts as a no-op.
        needs_pano_geometry = self.enable_pano_global_token or self.enable_pano_geometry_residual
        if self.enable_pano_global_token:
            self.pano_global_token = nn.Parameter(torch.empty(1, 1, 1, embed_dim))
        else:
            self.pano_global_token = None
        if needs_pano_geometry:
            self.pano_geometry_embed = nn.Sequential(
                nn.LayerNorm(pano_geom_dim),
                nn.Linear(pano_geom_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.pano_geometry_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.pano_geometry_embed = None
            self.pano_geometry_alpha = None

        # patch_token_start = camera (1) + optional pano_global (0 or 1) + register (R)
        self.patch_token_start = 1 + int(self.enable_pano_global_token) + num_register_tokens

        # Inter-frame attention type per layer ("global" by default, "register"
        # for the selected indices — same as released omega).
        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        # LUNA residual adapters
        self.luna_patch_layers: Set[int] = (
            _resolve_luna_layers(luna_patch_layers, depth) if enable_luna else set()
        )
        self.luna_camera_layers: Set[int] = (
            _resolve_luna_layers(luna_camera_layers, depth) if enable_luna else set()
        )
        self.luna_patch_adapters = nn.ModuleDict(
            {
                str(layer_idx): LunaPatchAdapter(
                    dim=embed_dim,
                    sphere_dim=luna_sphere_dim,
                    hidden_dim=luna_hidden_dim,
                    patch_bank_mode=luna_patch_bank_mode,
                    patch_bank_shuffle_seed=(
                        None
                        if luna_patch_bank_shuffle_seed is None
                        else int(luna_patch_bank_shuffle_seed) + layer_idx
                    ),
                )
                for layer_idx in sorted(self.luna_patch_layers)
            }
        )
        self.luna_camera_adapters = nn.ModuleDict(
            {
                str(layer_idx): LunaCameraAdapter(
                    dim=embed_dim,
                    camera_meta_dim=luna_camera_meta_dim,
                    hidden_dim=luna_hidden_dim,
                )
                for layer_idx in sorted(self.luna_camera_layers)
            }
        )

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        if self.pano_global_token is not None:
            nn.init.normal_(self.pano_global_token, std=1e-3)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        images: torch.Tensor,
        pano_geometry: Optional[torch.Tensor] = None,
        pano_token_meta: Optional[Dict[str, torch.Tensor]] = None,
        pano_camera_meta: Optional[torch.Tensor] = None,
    ) -> Tuple[list, int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        geometry_delta = None
        if self.pano_geometry_embed is not None and pano_geometry is not None:
            if pano_geometry.shape[:2] != (batch_size, num_frames):
                raise ValueError(
                    f"Expected pano_geometry shape [B, S, D] with B={batch_size}, S={num_frames}, "
                    f"got {tuple(pano_geometry.shape)}"
                )
            geometry_delta = self.pano_geometry_embed(
                pano_geometry.to(device=images.device, dtype=patch_tokens.dtype)
            )
            geometry_delta = geometry_delta.reshape(batch_size * num_frames, 1, -1)
            geometry_delta = self.pano_geometry_alpha.to(dtype=geometry_delta.dtype) * geometry_delta
            if self.enable_pano_geometry_residual:
                camera_token = camera_token + geometry_delta

        special_tokens = [camera_token]
        if self.enable_pano_global_token:
            pano_global_token = expand_and_flatten(self.pano_global_token, batch_size, num_frames)
            if geometry_delta is not None:
                pano_global_token = pano_global_token + geometry_delta
            special_tokens.append(pano_global_token)
        special_tokens.append(register_token)

        tokens = torch.cat([*special_tokens, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                def layer_forward(layer_tokens, current_block_idx=block_idx):
                    return self._run_layer(
                        layer_tokens,
                        batch_size,
                        num_frames,
                        num_tokens,
                        embed_dim,
                        current_block_idx,
                        frame_rope,
                        pano_token_meta,
                        pano_camera_meta,
                    )

                tokens, frame_tokens = checkpoint(
                    layer_forward,
                    tokens,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                tokens, frame_tokens = self._run_layer(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    frame_rope,
                    pano_token_meta,
                    pano_camera_meta,
                )

            if block_idx in self.cached_layer_indices:
                # frame_tokens reflects post-frame-block state; tokens reflects
                # post-inter-frame (+ optional LUNA) state. We follow the omega
                # convention of caching the [frame || global] concat at 2*C dim.
                inter_tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
                outputs.append(torch.cat([frame_tokens, inter_tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    # ----------------------------------------------------- block-level helpers

    def _run_layer(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        frame_rope: Tuple[torch.Tensor, torch.Tensor],
        pano_token_meta: Optional[Dict[str, torch.Tensor]] = None,
        pano_camera_meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, frame_tokens = self._run_frame_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            frame_rope,
        )
        tokens = self._run_inter_frame_attention_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            self.inter_frame_attention_types[block_idx],
        )

        # ---- LUNA residual injection right before the cache decision ----
        if self.enable_luna:
            tokens = self._apply_luna_adapters(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                pano_token_meta=pano_token_meta,
                pano_camera_meta=pano_camera_meta,
            )
        return tokens, frame_tokens

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start:].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)

    # ----------------------------------------------------- LUNA injection

    def _apply_luna_adapters(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        pano_token_meta: Optional[Dict[str, torch.Tensor]] = None,
        pano_camera_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run optional LUNA-Patch / LUNA-Camera residuals at the end of a layer.

        Inputs:
            tokens: ``[B, S, P, C]`` after frame + inter_frame attention.
        Outputs:
            Same shape, with optional residual corrections applied.
        """
        layer_key = str(block_idx)
        has_patch = layer_key in self.luna_patch_adapters and pano_token_meta is not None
        has_camera = layer_key in self.luna_camera_adapters and pano_camera_meta is not None
        if not has_patch and not has_camera:
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        tokens_bsp = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        if has_patch:
            special_tokens = tokens_bsp[:, :, : self.patch_token_start, :]
            patch_tokens = tokens_bsp[:, :, self.patch_token_start:, :]
            patch_tokens = self.luna_patch_adapters[layer_key](patch_tokens, pano_token_meta)
            tokens_bsp = torch.cat([special_tokens, patch_tokens], dim=2)

        if has_camera:
            camera_tokens = tokens_bsp[:, :, 0, :]
            camera_tokens = self.luna_camera_adapters[layer_key](camera_tokens, pano_camera_meta)
            tokens_bsp = torch.cat(
                [camera_tokens[:, :, None, :], tokens_bsp[:, :, 1:, :]],
                dim=2,
            )
        return tokens_bsp


# ----------------------------------------------------------------------------- helpers


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    """Expand a (1, 2, X, C) token bank into (B*S, X, C) for multi-frame use."""
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])


def expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    """Expand a single shared token (1, 1, X, C) to every (B, S) view."""
    expanded = token_tensor.expand(batch_size, num_frames, *token_tensor.shape[2:])
    return expanded.reshape(batch_size * num_frames, *expanded.shape[2:])


def _resolve_luna_layers(layers, depth: int) -> Set[int]:
    """Parse LUNA layer selectors.

    The implicit default is intentionally conservative: only the final three
    blocks get LUNA patch residuals, so early attention stays on the pretrained
    Omega feature distribution. Explicit ``second_half`` remains available for
    older ablations.
    """
    if layers is None:
        return set(range(max(depth - 3, 0), depth))
    if isinstance(layers, str):
        normalized = layers.strip().lower()
        if normalized in {"", "none", "false", "off"}:
            return set()
        if normalized in {"last_half", "second_half"}:
            return set(range(depth // 2, depth))
        if normalized in {"last2", "final2", "tail2"}:
            return set(range(max(depth - 2, 0), depth))
        if normalized in {"last3", "final3", "tail3"}:
            return set(range(max(depth - 3, 0), depth))
        if normalized in {"last4", "final4", "tail4"}:
            return set(range(max(depth - 4, 0), depth))
        if normalized in {"last", "final"}:
            return {depth - 1}
        return {int(item.strip()) for item in normalized.split(",") if item.strip()}
    return {int(layer_idx) for layer_idx in layers}
