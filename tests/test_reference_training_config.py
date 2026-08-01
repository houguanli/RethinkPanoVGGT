from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_config():
    with (REPO_ROOT / "configs" / "train_multipano.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_reference_recipe_matches_paper_geometry_and_optimizer():
    config = _reference_config()
    sampler = config["sampler"]
    optimization = config["optimization"]
    stages = optimization["training_stages"]

    assert config["data"]["pano_min_count"] == 2
    assert config["data"]["pano_max_count"] == 10
    assert sampler["num_yaw"] == 4
    assert sampler["pitch_degrees"] == "-15"
    assert sampler["fov_degrees"] == 60.0
    assert optimization["optimizer_type"] == "adamw"

    assert [(stage["pano_min_count"], stage["pano_max_count"]) for stage in stages] == [
        (2, 2),
        (2, 6),
        (2, 10),
    ]
    assert [stage["num_yaw"] for stage in stages] == [4, 4, 6]
    assert [stage["window_size"] for stage in stages] == [384, 384, 512]
    assert all(stage["pitch_degrees"] == "-15" for stage in stages)
    assert all(stage["fov_degrees"] == 60.0 for stage in stages)
    assert all(stage["optimizer_type"] == "adamw" for stage in stages)


def test_reference_recipe_keeps_documented_training_schedule():
    stages = _reference_config()["optimization"]["training_stages"]

    assert [stage.get("until_minutes") for stage in stages] == [120.0, 300.0, None]
    assert [stage["lr"] for stage in stages] == [5.0e-5, 2.0e-5, 1.0e-5]
    assert [stage["camera_loss_weight"] for stage in stages] == [0.15, 0.25, 0.25]
    assert [stage["global_point_loss_weight"] for stage in stages] == [0.0, 0.05, 0.10]
    assert all(stage["trainable"] == "luna_residual_tail_heads" for stage in stages)
