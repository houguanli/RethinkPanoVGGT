import math
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, GIT_ROOT)

from evaluation_common.erp_depth import splat_window_z_depth_to_erp  # noqa: E402


def test_yaw8_constant_range_reconstructs_covered_erp():
    yaw = (torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0) - math.pi).reshape(1, 8)
    pitch = torch.full((1, 8), math.radians(-15.0))
    fov = torch.full((1, 8), math.radians(75.0))

    # Build the Z-depth that corresponds to a constant radial depth of 5 m.
    from evaluation_common.erp_depth import _window_rays

    _, z_factor = _window_rays(yaw, pitch, fov, fov, 192, 192, align_corners=True)
    window_z = 5.0 * z_factor
    result = splat_window_z_depth_to_erp(
        window_z,
        yaw=yaw,
        pitch=pitch,
        fov_x=fov,
        fov_y=fov,
        view_pano_index=torch.zeros(1, 8, dtype=torch.long),
        num_panos=1,
        erp_height=64,
        erp_width=128,
        align_corners=True,
    )

    coverage = result["coverage_mask"]
    coverage_fraction = float(coverage.float().mean())
    assert 0.35 < coverage_fraction < 0.65, coverage_fraction
    assert torch.allclose(result["depth"][coverage], torch.full_like(result["depth"][coverage], 5.0), atol=1e-4)


if __name__ == "__main__":
    test_yaw8_constant_range_reconstructs_covered_erp()
    print("ERP yaw8 covered-mask splat test ok")
