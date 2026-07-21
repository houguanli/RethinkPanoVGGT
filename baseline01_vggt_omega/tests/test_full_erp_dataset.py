import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, REPO_ROOT)

from data.datasets.pano_minimal import (  # noqa: E402
    _full_erp_points,
    _full_erp_pseudo_intrinsic,
    _opencv_camera_to_pano_basis,
)


def test_full_erp_unprojection_preserves_radial_depth():
    depth = np.full((8, 16), 3.0, dtype=np.float32)
    erp_c2w = _opencv_camera_to_pano_basis()
    world, camera, valid = _full_erp_points(depth, erp_c2w)

    assert world.shape == camera.shape == (8, 16, 3)
    assert valid.all()
    np.testing.assert_allclose(np.linalg.norm(camera, axis=-1), depth, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(world, axis=-1), depth, atol=1e-5)
    assert np.isfinite(world).all()


def test_full_erp_pseudo_intrinsic_tracks_angular_density():
    intrinsic = _full_erp_pseudo_intrinsic(height=256, width=512)
    np.testing.assert_allclose(intrinsic[0, 0], 512.0 / (2.0 * np.pi))
    np.testing.assert_allclose(intrinsic[1, 1], 256.0 / np.pi)
    assert intrinsic[2, 2] == 1.0


if __name__ == "__main__":
    test_full_erp_unprojection_preserves_radial_depth()
    test_full_erp_pseudo_intrinsic_tracks_angular_density()
    print("full ERP dataset geometry tests passed")
