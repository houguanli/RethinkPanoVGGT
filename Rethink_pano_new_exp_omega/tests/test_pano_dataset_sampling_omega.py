"""Tests for single and variable-neighborhood pano sampling."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.data import PanoMinimalDataset, PanoVKittiOmegaDataset  # noqa: E402
from training.data.pano_minimal import (  # noqa: E402
    _item,
    _parse_dataset_pano_max_counts,
    _rgb_depth_common_valid_mask,
    _read_pose_position_rotation,
    _read_structured3d_position,
)
from training.train_pano_omega import write_smoke_dataset  # noqa: E402
from scripts.evaluate_depth_checkpoint import apply_eval_max_panos, select_eval_indices  # noqa: E402


def test_single_pano_sampling_returns_one_erp():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "converted"
        write_smoke_dataset(root)
        dataset = PanoVKittiOmegaDataset(root=root, pano_sample_mode="single")
        sample = dataset[0]
        assert sample["pano_image"].shape == (3, 32, 64)
        assert sample["pano_position_m"].shape == (3,)


def test_variable_neighborhood_sampling_returns_anchor_first_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "converted"
        write_smoke_dataset(root)
        dataset = PanoVKittiOmegaDataset(
            root=root,
            pano_sample_mode="variable_neighborhood",
            pano_min_count=2,
            pano_max_count=3,
            grouping="nearest",
        )
        lengths = set()
        for _ in range(20):
            sample = dataset[1]
            lengths.add(int(sample["pano_image"].shape[0]))
            assert sample["pano_image"].shape[1:] == (3, 32, 64)
            assert sample["pano_position_m"].shape[0] == sample["pano_image"].shape[0]
            assert sample["scene_name"].split("|")[0] == "pano_smoke_01"
        assert lengths <= {2, 3}
        assert lengths


def test_minimal_multipano_groups_stay_inside_scene():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "minimal"
        _write_matterport_minimal_bundle(root)
        dataset = PanoMinimalDataset(
            root=root,
            datasets="matterport3d",
            pano_size=(16, 32),
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
        )
        assert dataset.groups
        for group in dataset.groups:
            scene_keys = {_scene_key(dataset, item_index) for item_index in group}
            assert len(scene_keys) == 1
        sample = dataset[0]
        assert all(name.startswith("scan_a_room0_") for name in sample["scene_name"].split("|"))


def test_minimal_dataset_specific_pano_caps_limit_groups():
    expected = {
        "panocity": 8,
        "matterport3d": 3,
        "stanford2d3ds": 3,
        "structured3d": 3,
    }
    for dataset_name, expected_max in expected.items():
        dataset = object.__new__(PanoMinimalDataset)
        dataset.pano_sample_mode = "variable_neighborhood"
        dataset.pano_min_count = 2
        dataset.pano_max_count = 8
        dataset.grouping = "nearest"
        dataset.dataset_pano_max_counts = _parse_dataset_pano_max_counts(
            "panocity:8,matterport3d:3,stanford2d3ds:3,structured3d:3"
        )
        dataset.items = [
            {
                "dataset": dataset_name,
                "sequence_name": dataset_name,
                "scene_group_key": "scene",
                "pano_position_m": [float(index), 0.0, 0.0],
            }
            for index in range(12)
        ]
        dataset.sample_indices = list(range(len(dataset.items)))
        dataset.indices_by_scene = {"scene": list(range(len(dataset.items)))}
        assert max(len(group) for group in dataset._build_groups()) == expected_max


def test_stanford_common_mask_excludes_only_black_polar_fill():
    image = torch.full((3, 16, 32), 0.5)
    image[:, :4] = 0.0
    image[:, -4:] = 0.0
    depth = torch.ones(1, 16, 32)

    stanford_mask = _rgb_depth_common_valid_mask(image, depth, "Stanford2D3DS")
    matterport_mask = _rgb_depth_common_valid_mask(image, depth, "Matterport3D")

    assert not bool(stanford_mask[:, :4].any())
    assert not bool(stanford_mask[:, -4:].any())
    assert bool(stanford_mask[:, 6:10].all())
    assert bool(matterport_mask.all())


def test_minimal_multipano_fallback_stays_inside_scene():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "minimal"
        _write_matterport_minimal_bundle(root)
        (root / "Matterport3D" / "scan_a" / "pano_depth" / "pano_1.png").unlink()
        dataset = PanoMinimalDataset(
            root=root,
            datasets="matterport3d",
            pano_size=(16, 32),
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
        )
        sample = dataset[0]
        assert all(name.startswith("scan_a_room0_") for name in sample["scene_name"].split("|"))


def test_minimal_matterport_groups_use_official_room_membership():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "minimal"
        _write_matterport_minimal_bundle(root)
        dataset = PanoMinimalDataset(
            root=root,
            datasets="matterport3d",
            pano_size=(16, 32),
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
        )
        keys = {str(item["scene_group_key"]) for item in dataset.items}
        assert keys == {
            "Matterport3D:scan_a:room0",
            "Matterport3D:scan_b:room0",
        }
        for group in dataset.groups:
            assert len({_scene_key(dataset, item_index) for item_index in group}) == 1
        assert all(not bool(item["pano_translation_valid"]) for item in dataset.items)
        assert all(not bool(item["pano_rotation_valid"]) for item in dataset.items)


def test_minimal_matterport_pose_is_converted_to_opencv():
    with tempfile.TemporaryDirectory() as tmp:
        pose_path = Path(tmp) / "pose.txt"
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        np.savetxt(pose_path, pose)
        position, position_valid, rotation, rotation_valid = _read_pose_position_rotation(pose_path)
        assert position_valid
        assert rotation_valid
        np.testing.assert_allclose(position, [1.0, -3.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(rotation),
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32),
            atol=1e-6,
        )


def test_minimal_panocity_groups_use_official_part_id_from_legacy_cache():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "minimal"
        _write_panocity_legacy_cache_bundle(root)
        dataset = PanoMinimalDataset(
            root=root,
            datasets="panocity",
            pano_size=(16, 32),
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
        )
        keys = [str(item["scene_group_key"]) for item in dataset.items]
        assert keys == [
            "Panocity:city_a:block_a:0",
            "Panocity:city_a:block_a:0",
            "Panocity:city_a:block_a:1",
            "Panocity:city_a:block_a:1",
        ]
        for group in dataset.groups:
            group_keys = {_scene_key(dataset, item_index) for item_index in group}
            assert len(group_keys) == 1


def test_minimal_structured3d_position_is_converted_to_opencv():
    with tempfile.TemporaryDirectory() as tmp:
        position_path = Path(tmp) / "camera_xyz.txt"
        position_path.write_text("1000 2000 3000\n", encoding="utf-8")
        position, position_valid = _read_structured3d_position(position_path)
        assert position_valid
        np.testing.assert_allclose(position, [1.0, -3.0, 2.0], atol=1e-6)


def test_structured3d_keeps_positions_but_disables_translation_supervision():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "minimal"
        dataset_root = root / "Structured3D"
        cache_dir = dataset_root / "cache"
        cache_dir.mkdir(parents=True)
        rows = [["scene_00000", ["00", "01"], [32, 64]]]
        (cache_dir / "structured3d_train_index.json").write_text(json.dumps(rows), encoding="utf-8")
        for pano_index, pano_id in enumerate(("00", "01")):
            pano_dir = dataset_root / "scene_00000" / "2D_rendering" / pano_id / "panorama"
            (pano_dir / "full").mkdir(parents=True)
            Image.new("RGB", (64, 32), (80 + pano_index * 30, 100, 160)).save(
                pano_dir / "full" / "rgb_rawlight.png"
            )
            Image.fromarray(np.full((32, 64), 4000, dtype=np.uint16)).save(
                pano_dir / "full" / "depth.png"
            )
            (pano_dir / "camera_xyz.txt").write_text(
                f"{pano_index * 1000} 0 0\n",
                encoding="utf-8",
            )
        dataset = PanoMinimalDataset(
            root=root,
            datasets="structured3d",
            pano_size=(16, 32),
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
        )
        sample = dataset[0]
        assert sample["pano_position_valid"].all()
        assert not sample["pano_translation_valid"].any()


def test_camera_supervision_dataset_allowlist():
    allowed = {"Panocity", "Stanford2D3DS"}
    for dataset_name in ("Panocity", "Stanford2D3DS", "Matterport3D", "Structured3D"):
        item = _item(
            dataset_name,
            f"{dataset_name}_sample",
            Path("rgb.png"),
            Path("depth.png"),
            [1.0, 2.0, 3.0],
            1000.0,
            position_valid=True,
            translation_valid=True,
            rotation_c2w=np.eye(3, dtype=np.float32).tolist(),
            rotation_valid=True,
        )
        expected = dataset_name in allowed
        assert bool(item["pano_translation_valid"]) is expected
        assert bool(item["pano_rotation_valid"]) is expected
        assert bool(item["pano_rotation_raw_valid"])


def test_eval_scene_neighborhood_selects_one_group_per_scene():
    class DummyDataset:
        items = [
            {"scene_group_key": "scene_a"},
            {"scene_group_key": "scene_a"},
            {"scene_group_key": "scene_b"},
            {"scene_group_key": "scene_b"},
        ]
        groups = [[0, 1], [1, 0], [2, 3], [3, 2]]

        def __len__(self):
            return len(self.groups)

    indices, info = select_eval_indices(DummyDataset(), limit=0, seed=7, sample_policy="scene_neighborhood")
    assert info["policy"] == "scene_neighborhood"
    assert info["scene_group_count"] == 2
    assert len(indices) == 2
    assert {0, 1} & set(indices)
    assert {2, 3} & set(indices)


def test_eval_scene_neighborhood_limit_fraction_applies_to_scene_groups():
    class DummyDataset:
        items = [
            {"scene_group_key": "scene_a"},
            {"scene_group_key": "scene_a"},
            {"scene_group_key": "scene_b"},
            {"scene_group_key": "scene_b"},
            {"scene_group_key": "scene_c"},
            {"scene_group_key": "scene_c"},
        ]
        groups = [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4]]

        def __len__(self):
            return len(self.groups)

    indices, info = select_eval_indices(
        DummyDataset(),
        limit=0,
        seed=7,
        sample_policy="scene_neighborhood",
        limit_fraction=0.5,
    )
    assert info["policy"] == "scene_neighborhood"
    assert info["scene_group_count"] == 3
    assert info["selected_scene_groups"] == 2
    assert len(indices) == 2


def test_eval_max_panos_clamps_min_and_max_counts():
    args = SimpleNamespace(pano_min_count=8, pano_max_count=10)
    apply_eval_max_panos(args, 6)
    assert args.pano_min_count == 6
    assert args.pano_max_count == 6


def test_eval_scene_neighborhood_rejects_cross_scene_group():
    class DummyDataset:
        items = [
            {"scene_group_key": "scene_a"},
            {"scene_group_key": "scene_b"},
        ]
        groups = [[0, 1]]

        def __len__(self):
            return len(self.groups)

    try:
        select_eval_indices(DummyDataset(), limit=0, seed=7, sample_policy="scene_neighborhood")
    except RuntimeError as exc:
        assert "crosses scenes" in str(exc)
    else:
        raise AssertionError("Expected cross-scene eval group to be rejected")


def _scene_key(dataset: PanoMinimalDataset, item_index: int) -> str:
    return str(dataset.items[item_index].get("scene_group_key"))


def _write_matterport_minimal_bundle(root: Path) -> None:
    cache_dir = root / "Matterport3D" / "cache"
    cache_dir.mkdir(parents=True)
    rows = [
        ["scan_a", "all", "all", ["pano_0", "pano_1"], [32, 64]],
        ["scan_b", "all", "all", ["pano_0", "pano_1"], [32, 64]],
    ]
    (cache_dir / "matterport3d_train_index.json").write_text(json.dumps(rows), encoding="utf-8")
    parsed_dir = root / "Matterport3D" / "parsed_json"
    parsed_dir.mkdir(parents=True)
    for scan_index, scan in enumerate(("scan_a", "scan_b")):
        (parsed_dir / f"{scan}.json").write_text(
            json.dumps(
                {
                    "room0": {
                        "room_name": "test room",
                        "panoramas": ["pano_0", "pano_1"],
                    }
                }
            ),
            encoding="utf-8",
        )
        for subdir in ("pano_skybox_color", "pano_depth", "pano_poses"):
            (root / "Matterport3D" / scan / subdir).mkdir(parents=True, exist_ok=True)
        for pano_index, pano_id in enumerate(("pano_0", "pano_1")):
            color = (80 + scan_index * 50, 100 + pano_index * 40, 160)
            Image.new("RGB", (64, 32), color).save(
                root / "Matterport3D" / scan / "pano_skybox_color" / f"{pano_id}.jpg"
            )
            depth = np.full((32, 64), 4000 + pano_index * 100, dtype=np.uint16)
            Image.fromarray(depth).save(root / "Matterport3D" / scan / "pano_depth" / f"{pano_id}.png")
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(scan_index * 100 + pano_index)
            np.savetxt(root / "Matterport3D" / scan / "pano_poses" / f"{pano_id}.txt", pose)


def _write_panocity_legacy_cache_bundle(root: Path) -> None:
    dataset_root = root / "Panocity"
    cache_dir = dataset_root / "cache"
    image_dir = dataset_root / "city_a" / "block_a" / "pano_images"
    depth_dir = dataset_root / "city_a" / "block_a" / "panodepth_images"
    cache_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    rows = []
    # Positions make a cross-part pano closer than the same-part pano. The
    # loader must still keep groups inside official 24-frame trajectory parts.
    for frame_index, x in ((0, 0.0), (1, 100.0), (24, 1.0), (25, 2.0)):
        rgb_name = f"pano_{frame_index:07d}.png"
        depth_name = f"pano_depth_{frame_index:07d}.png"
        Image.new("RGB", (64, 32), (80 + frame_index % 30, 100, 160)).save(image_dir / rgb_name)
        Image.fromarray(np.full((32, 64), 4000, dtype=np.uint16)).save(depth_dir / depth_name)
        rows.append(
            {
                "dataset": "Panocity",
                "city": "city_a",
                "block": "block_a",
                "scene_name": f"city_a_block_a_pano_{frame_index:07d}",
                "scene_group_key": "Panocity:city_a:block_a",
                "rgb_path": str(Path("city_a") / "block_a" / "pano_images" / rgb_name),
                "depth_path": str(Path("city_a") / "block_a" / "panodepth_images" / depth_name),
                "pano_position_m": [x, 0.0, 0.0],
                "pano_position_valid": True,
                "pano_rotation_c2w": np.eye(3, dtype=np.float32).tolist(),
                "pano_rotation_valid": True,
            }
        )
    (cache_dir / "panocity_train_index.json").write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    test_single_pano_sampling_returns_one_erp()
    test_variable_neighborhood_sampling_returns_anchor_first_sequence()
    test_minimal_multipano_groups_stay_inside_scene()
    test_minimal_multipano_fallback_stays_inside_scene()
    test_minimal_matterport_groups_use_official_room_membership()
    test_minimal_matterport_pose_is_converted_to_opencv()
    test_minimal_panocity_groups_use_official_part_id_from_legacy_cache()
    test_minimal_structured3d_position_is_converted_to_opencv()
    test_structured3d_keeps_positions_but_disables_translation_supervision()
    test_eval_scene_neighborhood_selects_one_group_per_scene()
    test_eval_scene_neighborhood_limit_fraction_applies_to_scene_groups()
    test_eval_max_panos_clamps_min_and_max_counts()
    test_eval_scene_neighborhood_rejects_cross_scene_group()
    print("pano dataset sampling (omega) ok")
