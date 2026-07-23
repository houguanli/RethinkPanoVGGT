# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_pano_global_token=False, pano_geom_dim=6,
                 enable_luna=False, luna_patch_layers=None, luna_camera_layers=None,
                 luna_sphere_dim=7, luna_camera_meta_dim=16, luna_hidden_dim=None,
                 aggregator_kwargs=None, lora=None):
        super().__init__()

        aggregator_kwargs = aggregator_kwargs or {}
        self.aggregator = Aggregator(
            img_size=img_size,
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
            **aggregator_kwargs,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        pano_view_params: torch.Tensor = None,
        pano_angles: torch.Tensor = None,
        pano_fov: torch.Tensor = None,
        pano_token_meta=None,
        pano_camera_meta=None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        pano_geometry = build_pano_geometry(
            pano_view_params=pano_view_params,
            pano_angles=pano_angles,
            pano_fov=pano_fov,
            device=images.device,
        )
        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            pano_geometry=pano_geometry,
            pano_token_meta=pano_token_meta,
            pano_camera_meta=pano_camera_meta,
        )

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions


def build_pano_geometry(
    pano_view_params: torch.Tensor = None,
    pano_angles: torch.Tensor = None,
    pano_fov: torch.Tensor = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Build [sin(theta), cos(theta), sin(phi), cos(phi), fov_h, fov_w]."""
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
