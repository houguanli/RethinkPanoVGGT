import unittest

import numpy as np

from scripts.calibrate_camera_conventions import (
    PairRecord,
    PanoRecord,
    DOCUMENTED_PANOVGGT_CAMERA_BASIS_CANONICAL_TO_NATIVE,
    WORLD_NATIVE_TO_CANONICAL,
    canonicalize_camera_pose,
    erp_rays,
    prepare_pair_samples,
    rotation_about_y,
    score_candidate,
    search_coarse,
)
class CameraConventionCalibrationTest(unittest.TestCase):
    def test_official_dataset_pose_conversions_are_proper_and_round_trip(self):
        expected_rotations = {
            "panocity": np.eye(3),
            "matterport3d": (
                WORLD_NATIVE_TO_CANONICAL["matterport3d"]
                @ DOCUMENTED_PANOVGGT_CAMERA_BASIS_CANONICAL_TO_NATIVE["matterport3d"]
            ),
            "stanford2d3ds": np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            ),
            "structured3d": np.eye(3),
        }
        for dataset, expected_rotation in expected_rotations.items():
            with self.subTest(dataset=dataset):
                world_basis = WORLD_NATIVE_TO_CANONICAL[dataset]
                camera_basis = DOCUMENTED_PANOVGGT_CAMERA_BASIS_CANONICAL_TO_NATIVE[dataset]
                center, rotation = canonicalize_camera_pose(
                    dataset,
                    np.asarray([1.0, 2.0, 3.0]),
                    np.eye(3),
                )

                np.testing.assert_allclose(center, world_basis @ np.asarray([1.0, 2.0, 3.0]))
                np.testing.assert_allclose(rotation, expected_rotation, atol=1e-6)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)

                center_native = world_basis.T @ center
                rotation_native = world_basis.T @ rotation @ camera_basis.T
                np.testing.assert_allclose(center_native, [1.0, 2.0, 3.0], atol=1e-6)
                np.testing.assert_allclose(rotation_native, np.eye(3), atol=1e-6)

    def test_recovers_axis_mapping_and_position_scale_from_depth(self):
        height, width = 40, 80
        rays = erp_rays(height, width)
        true_basis = rotation_about_y(np.pi / 2.0)
        true_scale = 0.5
        first_center_native = np.asarray([0.0, 0.0, 0.0])
        second_center_native = np.asarray([4.0, 0.5, -1.0])
        rotation = np.eye(3)
        bounds_min = np.asarray([-4.0, -2.5, -6.0])
        bounds_max = np.asarray([8.0, 3.5, 10.0])

        first_depth = render_box_depth(
            true_scale * first_center_native,
            (rotation @ true_basis @ rays.reshape(-1, 3).T).T,
            bounds_min,
            bounds_max,
        ).reshape(height, width)
        second_depth = render_box_depth(
            true_scale * second_center_native,
            (rotation @ true_basis @ rays.reshape(-1, 3).T).T,
            bounds_min,
            bounds_max,
        ).reshape(height, width)
        pair = PairRecord(
            scene="synthetic",
            first=PanoRecord(
                scene="synthetic",
                name="first",
                depth=first_depth,
                center=first_center_native,
                rotation_c2w=rotation,
                rotation_valid=True,
            ),
            second=PanoRecord(
                scene="synthetic",
                name="second",
                depth=second_depth,
                center=second_center_native,
                rotation_c2w=rotation,
                rotation_valid=True,
            ),
        )
        samples = prepare_pair_samples(
            [pair],
            rays,
            points_per_direction=2500,
            max_depth_m=100.0,
            seed=7,
        )

        true_metrics = score_candidate(samples, true_basis, true_scale)
        identity_metrics = score_candidate(samples, np.eye(3), 1.0)
        self.assertLess(true_metrics["score"], identity_metrics["score"] * 0.5)
        self.assertGreater(true_metrics["inlier_10pct"], 0.7)

        candidates = search_coarse(samples, [0.5, 1.0])
        best = candidates[0]
        self.assertAlmostEqual(best["position_scale_to_m"], true_scale)
        self.assertLess(best["metrics"]["score"], identity_metrics["score"] * 0.5)


def render_box_depth(center, directions, bounds_min, bounds_max):
    safe = np.where(np.abs(directions) > 1e-8, directions, np.nan)
    first = (bounds_min[None, :] - center[None, :]) / safe
    second = (bounds_max[None, :] - center[None, :]) / safe
    candidates = np.concatenate([first, second], axis=1)
    candidates[candidates <= 1e-6] = np.nan
    return np.nanmin(candidates, axis=1).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
