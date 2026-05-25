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
        pano_geom_dim: int = 6,
        enable_luna: bool = True,
        luna_patch_layers=None,
        luna_camera_layers=None,
        luna_sphere_dim: int = 7,
        luna_camera_meta_dim: int = 16,
        luna_hidden_dim: Optional[int] = None,
        sampler: Optional[dict] = None,
        aggregator_kwargs: Optional[dict] = None,
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
            pano_geom_dim=pano_geom_dim,
            enable_luna=enable_luna,
            luna_patch_layers=luna_patch_layers,
            luna_camera_layers=luna_camera_layers,
            luna_sphere_dim=luna_sphere_dim,
            luna_camera_meta_dim=luna_camera_meta_dim,
            luna_hidden_dim=luna_hidden_dim,
            aggregator_kwargs=aggregator_kwargs,
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

        sampler_output = self.pano_sampler(pano_images, yaw=yaw, pitch=pitch, fov=fov)
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
