from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "config"


def _compose(config_name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name)


def test_no_patch_bank_config_inherits_full_experiment():
    config = _compose("pano_luna_ablate_patch_bank")

    assert config.exp_name == "pano_luna_ablate_patch_bank"
    assert config.model.enable_luna is True
    assert config.model.luna_patch_bank_mode == "none"
    assert config.model.sampler.num_yaw == 8
    assert config.data.train.dataset.dataset_configs[0].split == "train"


def test_no_geora_config_disables_both_adapters():
    config = _compose("pano_luna_ablate_geora")

    assert config.exp_name == "pano_luna_ablate_geora"
    assert config.model.enable_luna is False
    assert config.model.luna_patch_layers == "last_half"
    assert config.model.luna_camera_layers == "last_half"


def test_shuffled_patch_bank_config_is_reproducible():
    config = _compose("pano_luna_ablate_patch_bank_shuffle")

    assert config.exp_name == "pano_luna_ablate_patch_bank_shuffle"
    assert config.model.enable_luna is True
    assert config.model.luna_patch_bank_mode == "shuffled"
    assert config.model.luna_patch_bank_shuffle_seed == config.seed_value


if __name__ == "__main__":
    test_no_patch_bank_config_inherits_full_experiment()
    test_no_geora_config_disables_both_adapters()
    test_shuffled_patch_bank_config_is_reproducible()
    print("ablation config composition tests ok")
