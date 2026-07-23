from pathlib import Path

import torch

from training.train_pano_omega import parse_args
from vggt_omega.models.layers.luna_patch import LunaPatchAdapter, scatter_mean_by_global_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "multipano_rtx5000x4_mixed4_pano_low_to_high_luna_after_full_warmup_9h.yaml"


def _config(name: str):
    return parse_args(["--config", str(CONFIG_DIR / name)])


def test_ablation_configs_inherit_original_4x5000_schedule():
    base = parse_args(["--config", str(BASE_CONFIG)])
    configs = {
        "full": _config("ablation_rtx5000x4_full.yaml"),
        "no_patch": _config("ablation_rtx5000x4_no_patch_bank.yaml"),
        "no_geora": _config("ablation_rtx5000x4_no_geora.yaml"),
        "shuffle": _config("ablation_rtx5000x4_shuffle_patch_bank.yaml"),
    }

    for config in configs.values():
        assert config.dataset_format == base.dataset_format == "pano_minimal"
        assert config.dataset_sampling_weights == base.dataset_sampling_weights
        assert config.pano_sample_mode == base.pano_sample_mode == "variable_neighborhood"
        assert config.pano_min_count == base.pano_min_count == 2
        assert config.pano_max_count == base.pano_max_count == 8
        assert config.training_stages == base.training_stages
        assert config.window_size == base.window_size == 384
        assert config.num_yaw == base.num_yaw == 4
        assert config.max_duration_minutes == base.max_duration_minutes == 540.0
        assert config.dataset_root == Path("/mnt/e/PanoVGGT_minimal_datasets/datasets")
        assert config.randomize_pano_order is False

    assert configs["full"].luna_patch_bank_mode == "aligned"
    assert configs["no_patch"].luna_patch_bank_mode == "none"
    assert configs["shuffle"].luna_patch_bank_mode == "shuffled"
    assert configs["shuffle"].luna_patch_bank_shuffle_seed == 43
    assert configs["no_geora"].disable_geora is True


def test_patch_bank_never_aggregates_across_panoramas():
    tokens = torch.tensor([[[[1.0]], [[3.0]], [[10.0]], [[14.0]]]])
    global_ids = torch.zeros(1, 4, 1, dtype=torch.long)
    pano_ids = torch.tensor([[[0], [0], [1], [1]]])

    context = scatter_mean_by_global_id(tokens, global_ids, pano_ids=pano_ids)

    expected = torch.tensor([[[[2.0]], [[2.0]], [[12.0]], [[12.0]]]])
    assert torch.equal(context, expected)


def test_shuffled_patch_bank_preserves_each_pano_feature_set():
    tokens = torch.tensor(
        [[[[1.0], [3.0]], [[5.0], [7.0]], [[20.0], [30.0]], [[40.0], [50.0]]]]
    )
    global_ids = torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]])
    pano_ids = torch.tensor([[[0, 0], [0, 0], [1, 1], [1, 1]]])
    aligned = scatter_mean_by_global_id(tokens, global_ids, pano_ids=pano_ids)
    shuffled = scatter_mean_by_global_id(
        tokens,
        global_ids,
        pano_ids=pano_ids,
        shuffle=True,
        shuffle_seed=43,
    )

    assert not torch.equal(shuffled, aligned)
    for pano_id in (0, 1):
        mask = pano_ids == pano_id
        assert torch.equal(
            torch.sort(shuffled[..., 0][mask]).values,
            torch.sort(aligned[..., 0][mask]).values,
        )


def test_no_patch_bank_is_parameter_matched():
    aligned = LunaPatchAdapter(dim=8, sphere_dim=7, hidden_dim=16, patch_bank_mode="aligned")
    no_bank = LunaPatchAdapter(dim=8, sphere_dim=7, hidden_dim=16, patch_bank_mode="none")

    aligned_params = sum(parameter.numel() for parameter in aligned.parameters())
    no_bank_params = sum(parameter.numel() for parameter in no_bank.parameters())
    tokens = torch.randn(1, 2, 4, 8)
    context = no_bank._get_patch_bank_context(tokens, {"sphere_encoding": torch.randn(1, 2, 4, 7)})

    assert aligned_params == no_bank_params
    assert torch.equal(context, torch.zeros_like(tokens))


def test_all_patch_bank_modes_support_backward():
    global_ids = torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]])
    pano_ids = torch.tensor([[[0, 0], [0, 0], [1, 1], [1, 1]]])
    sphere_encoding = torch.randn(1, 4, 2, 7)
    for mode in ("aligned", "none", "shuffled"):
        tokens = torch.randn(1, 4, 2, 8, requires_grad=True)
        adapter = LunaPatchAdapter(
            dim=8,
            sphere_dim=7,
            hidden_dim=16,
            patch_bank_mode=mode,
            patch_bank_shuffle_seed=43,
        )
        adapter.alpha.data.fill_(1.0)
        output = adapter(
            tokens,
            {
                "global_patch_id": global_ids,
                "pano_id": pano_ids,
                "sphere_encoding": sphere_encoding,
            },
        )
        output.square().mean().backward()
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()


if __name__ == "__main__":
    test_ablation_configs_inherit_original_4x5000_schedule()
    test_patch_bank_never_aggregates_across_panoramas()
    test_shuffled_patch_bank_preserves_each_pano_feature_set()
    test_no_patch_bank_is_parameter_matched()
    test_all_patch_bank_modes_support_backward()
    print("omega multipano ablation tests ok")
