"""End-to-end pano-aware VGGT-Omega wrapper with LUNA adapters.

Counterpart of ``Rethink_pano_new_exp/vggt/models/vggt_luna.py`` for the omega
backbone. The class:

  1. Configures the base ``VGGTOmega`` with ``enable_luna=True`` and (by
     default) the pano-global token enabled.
  2. Owns a ``PanoWindowSampler`` so the user can pass a raw panorama
     ``[B, 3, H_pano, W_pano]`` and let the model handle window sampling +
     metadata generation internally.

Regular omega input still works: pass ``images`` ([B, S, 3, H, W]) and leave
``pano_images`` as ``None``.
"""

from typing import Optional

import torch

from vggt_omega.checkpoint import DEFAULT_CHECKPOINT_PATH
from vggt_omega.data.pano_sampler import PanoWindowSampler
from vggt_omega.models.vggt_omega import VGGTOmega


class VGGTOmega_LUNA(VGGTOmega):
    """VGGT-Omega with pano sampler plus LUNA residual adapters."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        enable_pano_global_token: bool = True,
        enable_pano_geometry_residual: bool = False,
        pano_geom_dim: int = 6,
        enable_luna: bool = True,
        luna_patch_layers=None,
        luna_camera_layers=None,
        luna_sphere_dim: int = 7,
        luna_camera_meta_dim: int = 16,
        luna_hidden_dim: Optional[int] = None,
        aggregator_use_checkpoint: bool = False,
        sampler: Optional[dict] = None,
        aggregator_kwargs: Optional[dict] = None,
        dense_head_frames_chunk_size: Optional[int] = 8,
        dense_head_use_checkpoint: bool = False,
        dense_head_return_confidence: bool = True,
        checkpoint_path: Optional[str] = str(DEFAULT_CHECKPOINT_PATH),
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_depth=enable_depth,
            enable_alignment=enable_alignment,
            enable_pano_global_token=enable_pano_global_token,
            enable_pano_geometry_residual=enable_pano_geometry_residual,
            pano_geom_dim=pano_geom_dim,
            enable_luna=enable_luna,
            luna_patch_layers=luna_patch_layers,
            luna_camera_layers=luna_camera_layers,
            luna_sphere_dim=luna_sphere_dim,
            luna_camera_meta_dim=luna_camera_meta_dim,
            luna_hidden_dim=luna_hidden_dim,
            aggregator_use_checkpoint=aggregator_use_checkpoint,
            aggregator_kwargs=aggregator_kwargs,
            dense_head_frames_chunk_size=dense_head_frames_chunk_size,
            dense_head_use_checkpoint=dense_head_use_checkpoint,
            dense_head_return_confidence=dense_head_return_confidence,
            checkpoint_path=checkpoint_path,
            checkpoint_strict=checkpoint_strict,
        )

        # Sampler defaults reflect the omega patch size; users can still override
        # window size / FoV / yaw grid via the ``sampler`` dict.
        sampler_config = {"window_size": 512, "patch_size": patch_size}
        sampler_config.update(sampler or {})
        self.pano_sampler = PanoWindowSampler(**sampler_config)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        pano_images: Optional[torch.Tensor] = None,
        yaw: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        fov: Optional[torch.Tensor] = None,
        return_sampler_output: bool = False,
        **kwargs,
    ):
        """Run either regular VGGT input or end-to-end pano input.

        Regular mode keeps the omega API: pass ``images`` as ``[B, S, 3, H, W]``.
        Pano mode: pass ``pano_images`` as ``[B, 3, H_pano, W_pano]``; the
        sampler builds windows, token metadata and camera metadata, which we
        then forward through the LUNA-aware aggregator.
        """
        if pano_images is None:
            return super().forward(images, **kwargs)

        sampler_output = self.sample_pano_windows(pano_images, yaw=yaw, pitch=pitch, fov=fov)
        predictions = super().forward(
            sampler_output.windows,
            pano_view_params=sampler_output.camera_meta["view_params"],
            pano_token_meta=sampler_output.token_meta,
            pano_camera_meta=sampler_output.camera_meta["camera_encoding"],
            **kwargs,
        )
        if return_sampler_output:
            predictions["pano_windows"] = sampler_output.windows
            predictions["pano_camera_meta"] = sampler_output.camera_meta
            predictions["pano_token_meta"] = sampler_output.token_meta
        return predictions

    def sample_pano_windows(
        self,
        pano_images: torch.Tensor,
        yaw: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        fov: Optional[torch.Tensor] = None,
        interpolation_mode: str = "bilinear",
    ):
        """Sample either single-pano batches or multi-pano batches.

        Input shapes:
        - [B, C, H, W] -> [B, S, C, window, window]
        - [B, N, C, H, W] -> [B, N*S, C, window, window]
        """
        if pano_images.ndim != 5:
            return self.pano_sampler(
                pano_images,
                yaw=yaw,
                pitch=pitch,
                fov=fov,
                interpolation_mode=interpolation_mode,
            )

        batch_size, num_panos, channels, height, width = pano_images.shape
        flat = pano_images.reshape(batch_size * num_panos, channels, height, width)
        sampled = self.pano_sampler(
            flat,
            yaw=yaw,
            pitch=pitch,
            fov=fov,
            interpolation_mode=interpolation_mode,
        )
        views_per_pano = sampled.windows.shape[1]
        sampled.windows = sampled.windows.reshape(
            batch_size,
            num_panos * views_per_pano,
            *sampled.windows.shape[2:],
        )
        sampled.camera_meta = _merge_multi_pano_meta(sampled.camera_meta, batch_size, num_panos, views_per_pano)
        sampled.token_meta = _merge_multi_pano_meta(sampled.token_meta, batch_size, num_panos, views_per_pano)
        patch_tokens = sampled.token_meta.get("pano_id")
        if patch_tokens is not None:
            patches_per_view = patch_tokens.shape[-1]
            pano_id = torch.arange(num_panos, device=patch_tokens.device).reshape(1, num_panos, 1, 1)
            pano_id = pano_id.expand(batch_size, num_panos, views_per_pano, patches_per_view)
            sampled.token_meta["pano_id"] = pano_id.reshape(batch_size, num_panos * views_per_pano, patches_per_view)
        return sampled


def _merge_multi_pano_meta(meta, batch_size: int, num_panos: int, views_per_pano: int):
    merged = {}
    for key, value in meta.items():
        if not torch.is_tensor(value):
            merged[key] = value
            continue
        if value.shape[:2] == (batch_size * num_panos, views_per_pano):
            merged[key] = value.reshape(batch_size, num_panos * views_per_pano, *value.shape[2:])
        else:
            merged[key] = value
    return merged
