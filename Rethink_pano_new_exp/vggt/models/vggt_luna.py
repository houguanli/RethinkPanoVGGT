from typing import Optional

import torch

from vggt.data.pano_sampler import PanoWindowSampler
from vggt.models.vggt import VGGT


class VGGT_LUNA(VGGT):
    """VGGT with pano sampler plus LUNA residual adapters."""

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        enable_pano_global_token=True,
        pano_geom_dim=6,
        enable_luna=True,
        luna_patch_layers=None,
        luna_camera_layers=None,
        luna_sphere_dim=7,
        luna_camera_meta_dim=16,
        luna_hidden_dim=None,
        luna_patch_bank_mode="aligned",
        luna_patch_bank_shuffle_seed=None,
        sampler=None,
        aggregator_kwargs=None,
        lora=None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            enable_pano_global_token=enable_pano_global_token,
            pano_geom_dim=pano_geom_dim,
            enable_luna=enable_luna,
            luna_patch_layers=luna_patch_layers,
            luna_camera_layers=luna_camera_layers,
            luna_sphere_dim=luna_sphere_dim,
            luna_camera_meta_dim=luna_camera_meta_dim,
            luna_hidden_dim=luna_hidden_dim,
            luna_patch_bank_mode=luna_patch_bank_mode,
            luna_patch_bank_shuffle_seed=luna_patch_bank_shuffle_seed,
            aggregator_kwargs=aggregator_kwargs,
            lora=lora,
        )
        sampler_config = {"window_size": img_size, "patch_size": patch_size}
        sampler_config.update(sampler or {})
        self.pano_sampler = PanoWindowSampler(**sampler_config)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        query_points: torch.Tensor = None,
        pano_images: Optional[torch.Tensor] = None,
        yaw: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        fov: Optional[torch.Tensor] = None,
        return_sampler_output: bool = False,
        **kwargs,
    ):
        """Run either regular VGGT input or end-to-end pano input.

        Regular mode keeps the B2 API: pass `images` as [B, S, 3, H, W].
        Pano mode: pass `pano_images` as [B, 3, H_pano, W_pano].
        """
        if pano_images is None:
            return super().forward(images, query_points=query_points, **kwargs)

        sampler_output = self.pano_sampler(pano_images, yaw=yaw, pitch=pitch, fov=fov)
        predictions = super().forward(
            sampler_output.windows,
            query_points=query_points,
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
