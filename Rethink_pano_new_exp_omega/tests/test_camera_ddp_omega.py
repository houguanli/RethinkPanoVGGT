"""DDP regression test for mixed camera-label availability."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from training.train_pano_omega import camera_alignment_loss  # noqa: E402


def _worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
        rotation_y_90_c2w = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        rotations = torch.stack([torch.eye(3), rotation_y_90_c2w], dim=0).unsqueeze(0)
        predictions = {
            "depth": torch.ones(1, 8, 2, 2, 1),
            "pano_camera_center": torch.zeros(1, 2, 3, requires_grad=True),
            "pano_rotation_quat_w2c": identity_quat.reshape(1, 1, 4).expand(1, 2, 4).clone().requires_grad_(True),
        }
        batch = {
            "pano_position_m": torch.zeros(1, 2, 3),
            "pano_position_valid": torch.tensor([[True, True]]),
            "pano_rotation_c2w": rotations,
            "pano_rotation_valid": torch.tensor([[True, True]]) if rank == 0 else torch.tensor([[False, False]]),
        }
        losses = camera_alignment_loss(
            predictions=predictions,
            batch=batch,
            translation_weight=1.0,
            rotation_weight=1.0,
            fov_weight=0.0,
            position_mode="relative_anchor",
            supervision_mode="pano_relative",
        )
        reported_degrees = losses["camera_rotation_deg"].detach().clone()
        dist.all_reduce(reported_degrees, op=dist.ReduceOp.SUM)
        reported_degrees /= world_size
        assert 89.0 < float(reported_degrees) < 91.0
        losses["loss_camera"].backward()
    finally:
        dist.destroy_process_group()


def test_mixed_rotation_validity_uses_identical_collective_order() -> None:
    with tempfile.TemporaryDirectory(prefix="pano_camera_ddp_") as tmpdir:
        init_file = str(Path(tmpdir) / "init")
        mp.spawn(_worker, args=(2, init_file), nprocs=2, join=True)


if __name__ == "__main__":
    test_mixed_rotation_validity_uses_identical_collective_order()
    print("camera DDP mixed-label normalization (omega) ok")
