# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT-Omega top-level model with LUNA-aware aggregator wiring.

This module mirrors `Rethink_pano_new_exp/vggt/models/vggt.py` but on top of
the omega backbone. Key changes vs the released `vggt_omega.VGGTOmega`:

  * The aggregator now accepts pano-related kwargs (LUNA insertion layers,
    pano-global token, geometry MLP). Defaults stay off → identical numerics
    to the released checkpoint.
  * The forward signature is extended with optional ``pano_view_params /
    pano_angles / pano_fov / pano_token_meta / pano_camera_meta`` so the
    pano-sampler path can feed in metadata. When all of them are ``None`` the
    forward behaves exactly like the released omega.
"""

import contextlib
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn

from vggt_omega.checkpoint import DEFAULT_CHECKPOINT_PATH, load_checkpoint
from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction.

    This experimental variant accepts optional pano metadata so that the
    LUNA-augmented aggregator can use it. Pure-omega behaviour is preserved
    when ``enable_luna`` / ``enable_pano_global_token`` are both ``False``.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        # ---- pano / LUNA additions (default off) ----
        enable_pano_global_token: bool = False,
        pano_geom_dim: int = 6,
        enable_luna: bool = False,
        luna_patch_layers=None,
        luna_camera_layers=None,
        luna_sphere_dim: int = 7,
        luna_camera_meta_dim: int = 16,
        luna_hidden_dim: Optional[int] = None,
        aggregator_use_checkpoint: bool = False,
        aggregator_kwargs: Optional[dict] = None,
        dense_head_frames_chunk_size: Optional[int] = 8,
        dense_head_use_checkpoint: bool = False,
        dense_head_return_confidence: bool = True,
        checkpoint_path: Optional[str] = str(DEFAULT_CHECKPOINT_PATH),
        checkpoint_strict: bool = True,
    ) -> None:
        super().__init__()
        self.dense_head_frames_chunk_size = (
            None if dense_head_frames_chunk_size is None or int(dense_head_frames_chunk_size) <= 0
            else int(dense_head_frames_chunk_size)
        )
        self.dense_head_use_checkpoint = bool(dense_head_use_checkpoint)
        self.dense_head_return_confidence = bool(dense_head_return_confidence)

        aggregator_kwargs = dict(aggregator_kwargs or {})
        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_pano_global_token=enable_pano_global_token,
            pano_geom_dim=pano_geom_dim,
            enable_luna=enable_luna,
            luna_patch_layers=luna_patch_layers,
            luna_camera_layers=luna_camera_layers,
            luna_sphere_dim=luna_sphere_dim,
            luna_camera_meta_dim=luna_camera_meta_dim,
            luna_hidden_dim=luna_hidden_dim,
            use_checkpoint=aggregator_use_checkpoint,
            **aggregator_kwargs,
        )
        _warn_if_rope_not_max(self.aggregator)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        if checkpoint_path is not None:
            load_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def forward(
        self,
        images: torch.Tensor,
        pano_view_params: Optional[torch.Tensor] = None,
        pano_angles: Optional[torch.Tensor] = None,
        pano_fov: Optional[torch.Tensor] = None,
        pano_token_meta: Optional[Dict[str, torch.Tensor]] = None,
        pano_camera_meta: Optional[torch.Tensor] = None,
        dense_head_frames_chunk_size: Optional[int] = None,
        dense_head_use_checkpoint: Optional[bool] = None,
        dense_head_return_confidence: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        pano_geometry = build_pano_geometry(
            pano_view_params=pano_view_params,
            pano_angles=pano_angles,
            pano_fov=pano_fov,
            device=images.device,
        )

        if images.device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            amp_context = contextlib.nullcontext()

        with amp_context:
            aggregated_tokens_list, patch_token_start = self.aggregator(
                images,
                pano_geometry=pano_geometry,
                pano_token_meta=pano_token_meta,
                pano_camera_meta=pano_camera_meta,
            )

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions: Dict[str, torch.Tensor] = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                frames_chunk_size = (
                    self.dense_head_frames_chunk_size
                    if dense_head_frames_chunk_size is None
                    else dense_head_frames_chunk_size
                )
                use_checkpoint = (
                    self.dense_head_use_checkpoint
                    if dense_head_use_checkpoint is None
                    else bool(dense_head_use_checkpoint)
                )
                return_confidence = (
                    self.dense_head_return_confidence
                    if dense_head_return_confidence is None
                    else bool(dense_head_return_confidence)
                )
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                    frames_chunk_size=frames_chunk_size,
                    use_checkpoint=use_checkpoint,
                    return_confidence=return_confidence,
                )
                predictions["depth"] = depth
                if depth_conf is not None:
                    predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def build_pano_geometry(
    pano_view_params: Optional[torch.Tensor] = None,
    pano_angles: Optional[torch.Tensor] = None,
    pano_fov: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Build [sin(theta), cos(theta), sin(phi), cos(phi), fov_h, fov_w].

    Identical helper to the b1 version. Returns ``None`` when no pano info is
    available so the aggregator path stays fully equivalent to vanilla omega.
    """
    if pano_view_params is None and pano_angles is None:
        return None

    if pano_view_params is not None:
        pano_view_params = pano_view_params.to(device=device)
        theta = pano_view_params[..., 0]
        phi = pano_view_params[..., 1]
        fov = pano_view_params[..., 2:] if pano_view_params.shape[-1] > 2 else None
    else:
        pano_angles = pano_angles.to(device=device)
        theta = pano_angles[..., 0]
        phi = pano_angles[..., 1]
        fov = pano_fov.to(device=device) if pano_fov is not None else None

    if fov is None:
        fov_h = torch.zeros_like(theta)
        fov_w = torch.zeros_like(theta)
    elif fov.shape[-1] == 1:
        fov_h = fov[..., 0]
        fov_w = fov[..., 0]
    else:
        fov_h = fov[..., 0]
        fov_w = fov[..., 1]

    return torch.stack(
        [torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi), fov_h, fov_w],
        dim=-1,
    ).float()


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
