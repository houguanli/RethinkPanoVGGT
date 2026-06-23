"""Small depth sanity checks shared by PanoCity compare-method adapters."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def as_bchw(depth: torch.Tensor) -> torch.Tensor:
    """Convert common depth tensor layouts to Bx1xHxW."""
    if depth.ndim == 5:
        depth = depth[:, 0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.permute(0, 3, 1, 2)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4:
        raise ValueError(f"Expected depth tensor with 3-5 dims, got shape={tuple(depth.shape)}")
    return depth.float()


def resize_like(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = as_bchw(pred)
    target = as_bchw(target)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, target.shape[-2:], mode="bilinear", align_corners=True)
    return pred


def depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Return absolute and best-scale-aligned depth metrics."""
    target = as_bchw(target)
    pred = resize_like(pred, target)
    if mask is None:
        valid = target > 0
    else:
        valid = as_bchw(mask).bool()
    valid = valid & torch.isfinite(pred) & torch.isfinite(target) & (target > 0)
    if not valid.any():
        return {
            "valid_pixels": 0.0,
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "best_scale": float("nan"),
            "scaled_abs_rel": float("nan"),
            "scaled_rmse": float("nan"),
        }

    p = pred[valid].detach().float()
    t = target[valid].detach().float()
    abs_rel = torch.mean(torch.abs(p - t) / torch.clamp(t, min=1e-6))
    rmse = torch.sqrt(torch.mean((p - t) ** 2))
    denom = torch.sum(p * p)
    if denom > 1e-12:
        best_scale = torch.sum(p * t) / denom
        scaled = p * best_scale
        scaled_abs_rel = torch.mean(torch.abs(scaled - t) / torch.clamp(t, min=1e-6))
        scaled_rmse = torch.sqrt(torch.mean((scaled - t) ** 2))
    else:
        best_scale = torch.tensor(float("nan"), device=p.device)
        scaled_abs_rel = torch.tensor(float("nan"), device=p.device)
        scaled_rmse = torch.tensor(float("nan"), device=p.device)

    return {
        "valid_pixels": float(valid.sum().detach().cpu()),
        "abs_rel": float(abs_rel.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "best_scale": float(best_scale.detach().cpu()),
        "scaled_abs_rel": float(scaled_abs_rel.detach().cpu()),
        "scaled_rmse": float(scaled_rmse.detach().cpu()),
    }


def tensor_stats(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    tensor = as_bchw(tensor)
    if mask is not None:
        valid = as_bchw(mask).bool() & torch.isfinite(tensor)
        values = tensor[valid]
    else:
        values = tensor[torch.isfinite(tensor)]
    if values.numel() == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": float(values.min().detach().cpu()),
        "mean": float(values.mean().detach().cpu()),
        "max": float(values.max().detach().cpu()),
    }


def load_partial_state(model: torch.nn.Module, checkpoint: str, device: torch.device) -> int:
    """Load matching tensors from common checkpoint payloads."""
    del device
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint payload in {checkpoint}: {type(state)}")

    target_state = model.state_dict()
    candidates = state
    if not any(k in target_state for k in candidates):
        stripped = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        prefixed = {"module." + k if not k.startswith("module.") else k: v for k, v in state.items()}
        candidates = stripped if any(k in target_state for k in stripped) else prefixed

    matched = {
        k: v
        for k, v in candidates.items()
        if k in target_state and hasattr(v, "shape") and v.shape == target_state[k].shape
    }
    if not matched:
        raise RuntimeError(
            f"Checkpoint {checkpoint} did not match any model parameters. "
            f"checkpoint keys sample={list(state.keys())[:5]}; model keys sample={list(target_state.keys())[:5]}"
        )
    model.load_state_dict(matched, strict=False)
    return len(matched)
