import itertools
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding
from vggt_omega.utils.rotation import quat_to_mat


def has_pano_supervision(batch: Dict) -> bool:
    """Return True when the batch declares pano geometry supervision."""
    flag = batch.get("is_pano", batch.get("pano_is_pano", None))
    if flag is None:
        return "pano_rotations" in batch or "pano_view_params" in batch
    if torch.is_tensor(flag):
        return bool(flag.any().item())
    return bool(flag)


def compute_pano_geometry_loss(
    predictions: Dict,
    batch: Dict,
    lambda_zero_t: float = 1.0,
    lambda_abs_rot: float = 1.0,
    lambda_rel_rot: float = 1.0,
    gamma: float = 0.6,
    pose_encoding_type: str = "absT_quaR_FoV",
    rel_loss_type: str = "geodesic",
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Losses for pinhole crops cut from the same panorama.

    Expected prediction pose encoding is VGGT's absT_quaR_FoV layout:
    translation [:3], quaternion [3:7], FoV [7:].
    Ground-truth rotations are read from `batch["pano_rotations"]` when present,
    otherwise from `batch["extrinsics"]`.
    """
    pred_pose_list = predictions.get("pose_enc_list")
    if pred_pose_list is None and "pose_enc" in predictions:
        pred_pose_list = [predictions["pose_enc"]]
    if pred_pose_list is None:
        zero = _zero_like_prediction(predictions)
        return _empty_loss_dict(zero)

    gt_pose = _get_gt_pose_encoding(batch, pose_encoding_type)
    valid_mask = _get_pano_valid_mask(batch, gt_pose)

    total_zero_t = total_abs_rot = total_rel_rot = None
    n_stages = len(pred_pose_list)

    for stage_idx, pred_pose in enumerate(pred_pose_list):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        zero_t = zero_translation_loss(pred_pose, valid_mask=valid_mask)
        abs_rot = absolute_rotation_loss(pred_pose, gt_pose, valid_mask=valid_mask)
        rel_rot = relative_rotation_loss(
            pred_pose,
            gt_pose,
            pairs=batch.get("pano_pairs", None),
            valid_mask=valid_mask,
            loss_type=rel_loss_type,
        )
        total_zero_t = zero_t * stage_weight if total_zero_t is None else total_zero_t + zero_t * stage_weight
        total_abs_rot = abs_rot * stage_weight if total_abs_rot is None else total_abs_rot + abs_rot * stage_weight
        total_rel_rot = rel_rot * stage_weight if total_rel_rot is None else total_rel_rot + rel_rot * stage_weight

    loss_zero_t = total_zero_t / n_stages
    loss_abs_rot = total_abs_rot / n_stages
    loss_rel_rot = total_rel_rot / n_stages
    loss_pano = (
        lambda_zero_t * loss_zero_t
        + lambda_abs_rot * loss_abs_rot
        + lambda_rel_rot * loss_rel_rot
    )

    return {
        "loss_pano": loss_pano,
        "loss_pano_zero_t": loss_zero_t,
        "loss_pano_abs_rot": loss_abs_rot,
        "loss_pano_rel_rot": loss_rel_rot,
    }


def zero_translation_loss(pred_pose: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    translation = pred_pose[..., :3]
    if valid_mask is None:
        centered = translation - translation.mean(dim=1, keepdim=True)
        return centered.abs().mean()

    losses = []
    for batch_idx in range(translation.shape[0]):
        mask = valid_mask[batch_idx]
        if mask.sum() <= 1:
            continue
        trans = translation[batch_idx, mask]
        losses.append((trans - trans.mean(dim=0, keepdim=True)).abs().mean())
    return torch.stack(losses).mean() if losses else translation.sum() * 0.0


def absolute_rotation_loss(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_q = F.normalize(pred_pose[..., 3:7], dim=-1)
    gt_q = F.normalize(gt_pose[..., 3:7].to(pred_q), dim=-1)
    err = torch.minimum((pred_q - gt_q).abs().sum(dim=-1), (pred_q + gt_q).abs().sum(dim=-1))
    if valid_mask is not None:
        err = err[valid_mask]
    return err.mean() if err.numel() > 0 else pred_pose.sum() * 0.0


def relative_rotation_loss(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    pairs: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    loss_type: str = "geodesic",
) -> torch.Tensor:
    pred_r = quat_to_mat(F.normalize(pred_pose[..., 3:7], dim=-1))
    gt_r = quat_to_mat(F.normalize(gt_pose[..., 3:7].to(pred_pose), dim=-1))

    losses = []
    for batch_idx in range(pred_pose.shape[0]):
        pair_idx = _pairs_for_batch(pred_pose.shape[1], pairs, batch_idx, pred_pose.device)
        if valid_mask is not None:
            pair_idx = pair_idx[valid_mask[batch_idx, pair_idx[:, 0]] & valid_mask[batch_idx, pair_idx[:, 1]]]
        if pair_idx.numel() == 0:
            continue

        i, j = pair_idx[:, 0], pair_idx[:, 1]
        pred_rel = pred_r[batch_idx, i].transpose(-1, -2) @ pred_r[batch_idx, j]
        gt_rel = gt_r[batch_idx, i].transpose(-1, -2) @ gt_r[batch_idx, j]
        delta = pred_rel.transpose(-1, -2) @ gt_rel
        if loss_type == "l1":
            losses.append((pred_rel - gt_rel).abs().mean())
        elif loss_type == "geodesic":
            trace = delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            losses.append(torch.acos(cos_angle).mean())
        else:
            raise ValueError(f"Unknown rel_loss_type: {loss_type}")

    return torch.stack(losses).mean() if losses else pred_pose.sum() * 0.0


def _get_gt_pose_encoding(batch: Dict, pose_encoding_type: str) -> torch.Tensor:
    if "pano_rotations" in batch:
        rotations = batch["pano_rotations"]
        translation = torch.zeros(rotations.shape[:-2] + (3,), device=rotations.device, dtype=rotations.dtype)
        extrinsics = torch.cat([rotations, translation[..., None]], dim=-1)
        intrinsics = batch["intrinsics"]
    else:
        extrinsics = batch["extrinsics"]
        intrinsics = batch["intrinsics"]
    image_hw = batch["images"].shape[-2:]
    try:
        return extri_intri_to_pose_encoding(
            extrinsics,
            intrinsics,
            image_hw,
            pose_encoding_type=pose_encoding_type,
        )
    except TypeError:
        return extri_intri_to_pose_encoding(extrinsics, intrinsics, image_hw)


def _get_pano_valid_mask(batch: Dict, gt_pose: torch.Tensor) -> Optional[torch.Tensor]:
    mask = batch.get("pano_valid_mask", None)
    if mask is None:
        flag = batch.get("is_pano", batch.get("pano_is_pano", None))
        if flag is None:
            return torch.ones(gt_pose.shape[:2], device=gt_pose.device, dtype=torch.bool)
        if torch.is_tensor(flag):
            flag = flag.to(device=gt_pose.device, dtype=torch.bool)
            return flag[:, None].expand(gt_pose.shape[:2]) if flag.ndim == 1 else flag
        return torch.ones(gt_pose.shape[:2], device=gt_pose.device, dtype=torch.bool) if flag else None
    return mask.to(device=gt_pose.device, dtype=torch.bool)


def _pairs_for_batch(num_views: int, pairs: Optional[torch.Tensor], batch_idx: int, device: torch.device) -> torch.Tensor:
    if pairs is None:
        return torch.tensor(list(itertools.combinations(range(num_views), 2)), device=device, dtype=torch.long)
    if pairs.ndim == 2:
        return pairs.to(device=device, dtype=torch.long)
    return pairs[batch_idx].to(device=device, dtype=torch.long)


def _zero_like_prediction(predictions: Dict) -> torch.Tensor:
    for value in predictions.values():
        if torch.is_tensor(value):
            return value.sum() * 0.0
        if isinstance(value, list) and value and torch.is_tensor(value[-1]):
            return value[-1].sum() * 0.0
    return torch.tensor(0.0)


def _empty_loss_dict(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "loss_pano": zero,
        "loss_pano_zero_t": zero,
        "loss_pano_abs_rot": zero,
        "loss_pano_rel_rot": zero,
    }
