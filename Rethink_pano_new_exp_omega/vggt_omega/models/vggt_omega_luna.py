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
import torch.nn.functional as F

from vggt_omega.checkpoint import DEFAULT_CHECKPOINT_PATH
from vggt_omega.data.pano_sampler import PanoWindowSampler
from vggt_omega.models.heads import PanoCameraHead
from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat


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
        enable_pano_camera_head: bool = True,
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
        pano_camera_dim = min(512, 2 * embed_dim)
        self.pano_camera_head = (
            PanoCameraHead(
                dim_in=2 * embed_dim,
                hidden_dim=pano_camera_dim,
                camera_meta_dim=luna_camera_meta_dim,
                num_heads=8,
                num_cross_pano_layers=2,
            )
            if enable_camera and enable_pano_camera_head
            else None
        )

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

        num_panos = int(pano_images.shape[1]) if pano_images.ndim == 5 else 1
        sampler_output = self.sample_pano_windows(pano_images, yaw=yaw, pitch=pitch, fov=fov)
        requested_return_window_pose = bool(kwargs.get("return_window_pose", True))
        if self.pano_camera_head is not None:
            # The pano-level head predicts only a residual. Use Omega's
            # pretrained window camera head as the initial pano pose.
            kwargs = dict(kwargs)
            kwargs["return_window_pose"] = True
        predictions = super().forward(
            sampler_output.windows,
            pano_view_params=sampler_output.camera_meta["view_params"],
            pano_token_meta=sampler_output.token_meta,
            pano_camera_meta=sampler_output.camera_meta["camera_encoding"],
            **kwargs,
        )
        if self.pano_camera_head is not None:
            camera_tokens = predictions["camera_and_register_tokens"][:, :, 0]
            pano_camera_residual = self.pano_camera_head(
                camera_tokens,
                sampler_output.camera_meta["camera_encoding"],
                num_panos=num_panos,
            )
            base_center, base_quat = _omega_window_pose_to_relative_pano_pose(
                predictions["pose_enc"],
                sampler_output.camera_meta["rotations"],
                num_panos=num_panos,
            )
            # Treat the pretrained Omega window-pose estimate as the fixed
            # initializer; camera supervision should train the pano residual,
            # not distort the base camera path through the shared tokens.
            base_center = base_center.detach()
            base_quat = base_quat.detach()
            predictions["pano_camera_center_init"] = base_center
            predictions["pano_rotation_quat_w2c_init"] = base_quat
            center_residual = pano_camera_residual["pano_camera_center_residual"]
            rotation_residual = pano_camera_residual["pano_rotation_quat_w2c_residual"]
            predictions["pano_camera_center_residual"] = center_residual
            predictions["pano_rotation_quat_w2c_residual"] = rotation_residual
            predictions["pano_camera_center"] = base_center + center_residual
            predictions["pano_rotation_quat_w2c"] = F.normalize(
                _quaternion_multiply(rotation_residual, base_quat),
                dim=-1,
                eps=1e-6,
            )
            predictions["pano_pose_enc"] = torch.cat(
                [
                    predictions["pano_camera_center"],
                    predictions["pano_rotation_quat_w2c"],
                ],
                dim=-1,
            )
            if not requested_return_window_pose:
                predictions.pop("pose_enc", None)
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


def _omega_window_pose_to_relative_pano_pose(
    pose_enc: torch.Tensor,
    window_rotations_c2w: torch.Tensor,
    num_panos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate Omega window poses into pano-relative pose initialization.

    ``pose_enc`` stores window camera-from-world extrinsics. Each sampled
    window also has a known ERP crop rotation. Removing that known crop rotation
    gives a pano-level pose. The result is expressed relative to pano 0, which
    is the gauge used by the pano-relative camera loss.
    """
    if pose_enc.ndim != 3 or pose_enc.shape[-1] < 7:
        raise ValueError(f"Expected pose_enc [B, N*V, >=7], got {tuple(pose_enc.shape)}")
    if window_rotations_c2w.ndim != 4 or window_rotations_c2w.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected window_rotations_c2w [B, N*V, 3, 3], "
            f"got {tuple(window_rotations_c2w.shape)}"
        )
    if int(num_panos) < 1 or pose_enc.shape[1] % int(num_panos) != 0:
        raise ValueError(f"Cannot group {pose_enc.shape[1]} window poses into {num_panos} panos.")

    batch_size, total_windows, _ = pose_enc.shape
    views_per_pano = total_windows // int(num_panos)
    raw_pose = pose_enc.float()
    raw_window_quat = raw_pose[..., 3:7]
    window_pose_valid = (
        torch.isfinite(raw_pose[..., :7]).all(dim=-1)
        & (torch.linalg.vector_norm(raw_window_quat, dim=-1) > 1e-6)
    )
    pose = torch.nan_to_num(raw_pose, nan=0.0, posinf=0.0, neginf=0.0)
    crop_c2w = window_rotations_c2w.to(device=pose.device, dtype=pose.dtype)

    fallback_window_quat_w2c = mat_to_quat(crop_c2w.transpose(-1, -2).contiguous())
    window_quat_w2c = torch.where(
        window_pose_valid[..., None],
        F.normalize(pose[..., 3:7], dim=-1, eps=1e-6),
        fallback_window_quat_w2c,
    )
    window_w2c = quat_to_mat(window_quat_w2c)
    window_c2w = window_w2c.transpose(-1, -2).contiguous()
    window_translation = pose[..., :3]
    window_centers = -(window_c2w @ window_translation[..., None])[..., 0]
    window_centers = torch.where(window_pose_valid[..., None], window_centers, torch.zeros_like(window_centers))

    pano_c2w_by_window = window_c2w @ crop_c2w.transpose(-1, -2)
    grouped_valid = window_pose_valid.reshape(batch_size, int(num_panos), views_per_pano)
    grouped_centers = window_centers.reshape(batch_size, int(num_panos), views_per_pano, 3)
    valid_count = grouped_valid.sum(dim=2, keepdim=True).clamp_min(1)
    pano_centers = (
        torch.where(grouped_valid[..., None], grouped_centers, torch.zeros_like(grouped_centers)).sum(dim=2)
        / valid_count.to(dtype=grouped_centers.dtype)
    )
    pano_c2w_by_window = pano_c2w_by_window.reshape(
        batch_size,
        int(num_panos),
        views_per_pano,
        3,
        3,
    )
    pano_quat_c2w_by_window = mat_to_quat(pano_c2w_by_window.reshape(-1, 3, 3)).reshape(
        batch_size,
        int(num_panos),
        views_per_pano,
        4,
    )
    pano_quat_c2w = _average_quaternions(pano_quat_c2w_by_window, dim=2, valid=grouped_valid)
    pano_c2w = quat_to_mat(pano_quat_c2w)

    anchor_w2c = pano_c2w[:, :1].transpose(-1, -2).contiguous()
    relative_centers = (anchor_w2c @ (pano_centers - pano_centers[:, :1])[..., None])[..., 0]
    relative_c2w = anchor_w2c @ pano_c2w
    relative_w2c = relative_c2w.transpose(-1, -2).contiguous()
    relative_quat_w2c = F.normalize(mat_to_quat(relative_w2c), dim=-1, eps=1e-6)
    return relative_centers, relative_quat_w2c


def _average_quaternions(
    quaternions: torch.Tensor,
    dim: int,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    quaternions = F.normalize(quaternions, dim=-1, eps=1e-6)
    reference = quaternions.select(dim, 0).unsqueeze(dim)
    sign = torch.where(
        (quaternions * reference).sum(dim=-1, keepdim=True) < 0,
        -torch.ones((), device=quaternions.device, dtype=quaternions.dtype),
        torch.ones((), device=quaternions.device, dtype=quaternions.dtype),
    )
    signed = quaternions * sign
    if valid is None:
        return F.normalize(signed.mean(dim=dim), dim=-1, eps=1e-6)
    weights = valid.to(device=quaternions.device, dtype=quaternions.dtype)[..., None]
    summed = (signed * weights).sum(dim=dim)
    count = weights.sum(dim=dim)
    averaged = F.normalize(summed / count.clamp_min(1.0), dim=-1, eps=1e-6)
    identity = torch.zeros_like(averaged)
    identity[..., 3] = 1.0
    return torch.where(count > 0, averaged, identity)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dim=-1,
    )
