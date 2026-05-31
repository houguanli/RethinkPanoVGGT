"""Tests for dynamic pano view sampling used by multipano training."""

import math
import os
import sys
from types import SimpleNamespace

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.train_pano_omega import sample_training_views  # noqa: E402


def test_fixed_view_sampling_uses_sampler_default_grid():
    args = SimpleNamespace(view_sampling_mode="fixed", views_per_pano=0, num_yaw=4, pitch_degrees="0")
    yaw, pitch = sample_training_views(args, step=1, device=torch.device("cpu"), dtype=torch.float32)
    assert yaw is None
    assert pitch is None


def test_cyclic_view_sampling_keeps_budget_and_covers_pitch_bands():
    args = SimpleNamespace(view_sampling_mode="cyclic", views_per_pano=2, num_yaw=8, pitch_degrees="-35,0,35")
    yaw1, pitch1 = sample_training_views(args, step=1, device=torch.device("cpu"), dtype=torch.float32)
    yaw2, pitch2 = sample_training_views(args, step=2, device=torch.device("cpu"), dtype=torch.float32)

    assert yaw1.shape == (2,)
    assert pitch1.shape == (2,)
    assert not torch.equal(yaw1, yaw2) or not torch.equal(pitch1, pitch2)
    assert torch.all(yaw1 >= -math.pi)
    assert torch.all(yaw1 < math.pi)
    expected_pitches = torch.tensor([math.radians(-35.0), 0.0, math.radians(35.0)], dtype=torch.float32)
    assert all(bool(torch.any(torch.isclose(value, expected_pitches))) for value in pitch1)


if __name__ == "__main__":
    test_fixed_view_sampling_uses_sampler_default_grid()
    test_cyclic_view_sampling_keeps_budget_and_covers_pitch_bands()
    print("dynamic view sampling (omega) ok")
