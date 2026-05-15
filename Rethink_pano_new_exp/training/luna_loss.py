from typing import Dict, Optional

import torch
import torch.nn.functional as F

try:
    from vggt.layers.luna_patch import scatter_mean_by_global_id
except ImportError:
    from layers.luna_patch import scatter_mean_by_global_id


def has_luna_supervision(batch: Dict) -> bool:
    return "pano_token_meta" in batch or "global_patch_id" in batch


def compute_luna_consistency_loss(
    predictions: Dict,
    batch: Dict,
    token_meta: Optional[Dict[str, torch.Tensor]] = None,
    lambda_point: float = 0.1,
    lambda_seam: float = 0.1,
    lambda_cam: float = 0.1,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """MVP LUNA losses for cross-window consistency.

    The point/seam terms softly pull tokens with the same global pano patch id
    toward their batch-local mean. The camera term keeps pano-derived windows
    close to a shared camera center.
    """
    token_meta = token_meta or batch.get("pano_token_meta", None)
    if token_meta is None and "global_patch_id" in batch:
        token_meta = {
            "global_patch_id": batch["global_patch_id"],
            "is_seam_region": batch.get("is_seam_region"),
        }

    zero = _zero_like_prediction(predictions)
    loss_point = _point_consistency_loss(predictions, token_meta) if token_meta is not None else zero
    loss_seam = _seam_consistency_loss(predictions, token_meta) if token_meta is not None else zero
    loss_cam = _camera_center_residual_loss(predictions)

    loss_luna = lambda_point * loss_point + lambda_seam * loss_seam + lambda_cam * loss_cam
    return {
        "loss_luna": loss_luna,
        "loss_luna_point": loss_point,
        "loss_luna_seam": loss_seam,
        "loss_luna_camera": loss_cam,
    }


def _point_consistency_loss(predictions: Dict, token_meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "world_points" not in predictions or "global_patch_id" not in token_meta:
        return _zero_like_prediction(predictions)

    point_tokens = _pool_points_to_token_grid(predictions["world_points"], token_meta["global_patch_id"])
    global_ids = token_meta["global_patch_id"].to(device=point_tokens.device, dtype=torch.long)
    if global_ids.ndim == 4:
        global_ids = global_ids.flatten(2)

    global_mean = scatter_mean_by_global_id(point_tokens, global_ids)
    return (point_tokens - global_mean).abs().mean()


def _seam_consistency_loss(predictions: Dict, token_meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    seam_mask = token_meta.get("is_seam_region", None)
    if "world_points" not in predictions or seam_mask is None or "global_patch_id" not in token_meta:
        return _zero_like_prediction(predictions)

    point_tokens = _pool_points_to_token_grid(predictions["world_points"], token_meta["global_patch_id"])
    global_ids = token_meta["global_patch_id"].to(device=point_tokens.device, dtype=torch.long)
    if global_ids.ndim == 4:
        global_ids = global_ids.flatten(2)
    seam_mask = seam_mask.to(device=point_tokens.device, dtype=torch.bool)
    if seam_mask.ndim == 4:
        seam_mask = seam_mask.flatten(2)

    if not bool(seam_mask.any()):
        return point_tokens.sum() * 0.0

    global_mean = scatter_mean_by_global_id(point_tokens, global_ids)
    return (point_tokens - global_mean).abs()[seam_mask].mean()


def _camera_center_residual_loss(predictions: Dict) -> torch.Tensor:
    pose = predictions.get("pose_enc", None)
    if pose is None:
        pose_list = predictions.get("pose_enc_list", None)
        pose = pose_list[-1] if pose_list else None
    if pose is None:
        return _zero_like_prediction(predictions)
    translation = pose[..., :3]
    centered = translation - translation.mean(dim=1, keepdim=True)
    return centered.abs().mean()


def _pool_points_to_token_grid(points: torch.Tensor, global_ids: torch.Tensor) -> torch.Tensor:
    B, S, H, W, C = points.shape
    if global_ids.ndim == 4:
        patch_h, patch_w = global_ids.shape[-2:]
    else:
        num_tokens = global_ids.shape[-1]
        patch_h = int(num_tokens ** 0.5)
        patch_w = num_tokens // patch_h
        if patch_h * patch_w != num_tokens:
            raise ValueError(f"Cannot infer token grid from {num_tokens} tokens")

    pooled = F.adaptive_avg_pool2d(
        points.reshape(B * S, H, W, C).permute(0, 3, 1, 2),
        output_size=(patch_h, patch_w),
    )
    return pooled.permute(0, 2, 3, 1).reshape(B, S, patch_h * patch_w, C)


def _zero_like_prediction(predictions: Dict) -> torch.Tensor:
    for value in predictions.values():
        if torch.is_tensor(value):
            return value.sum() * 0.0
        if isinstance(value, list) and value and torch.is_tensor(value[-1]):
            return value[-1].sum() * 0.0
    return torch.tensor(0.0)
