"""Tests for single and variable-neighborhood pano sampling."""

import os
import sys
import tempfile
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from training.data import PanoVKittiOmegaDataset  # noqa: E402
from training.train_pano_omega import write_smoke_dataset  # noqa: E402


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


if __name__ == "__main__":
    test_single_pano_sampling_returns_one_erp()
    test_variable_neighborhood_sampling_returns_anchor_first_sequence()
    print("pano dataset sampling (omega) ok")
