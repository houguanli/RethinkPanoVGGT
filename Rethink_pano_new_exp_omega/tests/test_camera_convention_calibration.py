import unittest
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from training.data.pano_minimal import _index_matterport3d, _item, _read_stanford_pose


class CameraConventionCalibrationTest(unittest.TestCase):
    def test_training_rotation_target_preserves_converted_erp_heading(self):
        rotation_y_90 = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        item = _item(
            "PanoCity",
            "scene_pano",
            Path("rgb.jpg"),
            Path("depth.png"),
            [0.0, 0.0, 0.0],
            1.0,
            rotation_c2w=rotation_y_90,
            rotation_valid=True,
        )
        np.testing.assert_allclose(item["pano_rotation_c2w"], rotation_y_90, atol=1e-6)
        np.testing.assert_allclose(item["pano_rotation_raw_c2w"], rotation_y_90, atol=1e-6)
        self.assertTrue(item["pano_rotation_valid"])
        self.assertTrue(item["pano_rotation_raw_valid"])

    def test_stanford_pose_position_and_rotation_share_opencv_c2w_path(self):
        with tempfile.TemporaryDirectory(prefix="stanford_pose_") as tmpdir:
            pose_path = Path(tmpdir) / "pose.json"
            pose_path.write_text(
                json.dumps(
                    {
                        "camera_rt_matrix": [
                            [1.0, 0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0, -2.0],
                            [0.0, 0.0, 1.0, -3.0],
                        ]
                    }
                ),
                encoding="utf-8",
            )
            position, position_valid, rotation, rotation_valid = _read_stanford_pose(pose_path)

        self.assertTrue(position_valid)
        self.assertTrue(rotation_valid)
        np.testing.assert_allclose(position, [1.0, -3.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(np.asarray(rotation).T @ np.asarray(rotation), np.eye(3), atol=1e-6)

    def test_matterport_cache_all_is_regrouped_by_parsed_room(self):
        with tempfile.TemporaryDirectory(prefix="matterport_rooms_") as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            parsed_dir = root / "parsed_json"
            cache_dir.mkdir(parents=True)
            parsed_dir.mkdir(parents=True)
            (cache_dir / "matterport3d_train_index.json").write_text(
                json.dumps(
                    [
                        [
                            "scan_a",
                            "all",
                            "all",
                            ["pano_room_1", "pano_room_2"],
                            [1024, 2048],
                        ]
                    ]
                ),
                encoding="utf-8",
            )
            (parsed_dir / "scan_a.json").write_text(
                json.dumps(
                    {
                        "1": {"room_name": "room one", "panoramas": ["pano_room_1"]},
                        "2": {"room_name": "room two", "panoramas": ["pano_room_2"]},
                    }
                ),
                encoding="utf-8",
            )

            items = _index_matterport3d(root, "train", 4000.0)

        self.assertEqual([item["scene_group_key"] for item in items], ["Matterport3D:scan_a:1", "Matterport3D:scan_a:2"])
        self.assertEqual([item["scene_name"] for item in items], ["scan_a_1_pano_room_1", "scan_a_2_pano_room_2"])

    def test_official_dataset_pose_conversions_are_proper_and_round_trip(self):
        expected_rotations = {
            "panocity": np.eye(3),
            "matterport3d": np.eye(3),
            "stanford2d3ds": np.eye(3),
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
