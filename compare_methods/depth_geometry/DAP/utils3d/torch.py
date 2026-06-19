from __future__ import annotations

import torch


def image_uv(width: int, height: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return normalized pixel-center UV coordinates with shape [H, W, 2]."""
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu, vv], dim=-1)


def points_to_normals(points: torch.Tensor, mask: torch.Tensor):
    """Estimate normals from neighboring 3D points."""
    points = points.to(dtype=torch.float32)
    mask = mask.bool()
    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]

    normal = torch.linalg.cross(dx, dy, dim=-1)
    norm = torch.linalg.norm(normal, dim=-1, keepdim=True)
    normal_mask = mask & torch.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    normal = normal / torch.clamp(norm, min=1e-8)
    normal = torch.where(normal_mask[..., None], normal, torch.zeros_like(normal))
    return normal, normal_mask
