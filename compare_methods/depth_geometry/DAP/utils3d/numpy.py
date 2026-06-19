from __future__ import annotations

import numpy as np


def image_uv(width: int, height: int) -> np.ndarray:
    """Return normalized pixel-center UV coordinates with shape [H, W, 2]."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu, vv], axis=-1)


def points_to_normals(points: np.ndarray, mask: np.ndarray):
    """Estimate normals from neighboring 3D points.

    Args:
        points: float array [H, W, 3].
        mask: boolean valid mask [H, W].

    Returns:
        normal: float array [H, W, 3].
        normal_mask: boolean array [H, W].
    """
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]

    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_mask = mask & np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    normal = normal / np.maximum(norm, 1e-8)
    normal[~normal_mask] = 0.0
    return normal.astype(np.float32), normal_mask
