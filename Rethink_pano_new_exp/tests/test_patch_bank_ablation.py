import torch

from vggt.layers.luna_patch import LunaPatchAdapter, scatter_mean_by_global_id
from vggt.models.aggregator import Aggregator


def _toy_patch_data():
    tokens = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]]]])
    global_ids = torch.tensor([[[0, 1], [0, 1]]])
    return tokens, global_ids


def test_aligned_patch_bank_uses_erp_correspondence():
    tokens, global_ids = _toy_patch_data()

    context = scatter_mean_by_global_id(tokens, global_ids)

    expected = torch.tensor([[[[3.0], [5.0]], [[3.0], [5.0]]]])
    assert torch.equal(context, expected)


def test_shuffled_patch_bank_preserves_features_but_breaks_alignment():
    tokens, global_ids = _toy_patch_data()
    aligned = scatter_mean_by_global_id(tokens, global_ids)

    shuffled_a = scatter_mean_by_global_id(tokens, global_ids, shuffle=True, shuffle_seed=0)
    shuffled_b = scatter_mean_by_global_id(tokens, global_ids, shuffle=True, shuffle_seed=0)

    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.equal(shuffled_a, aligned)
    assert torch.equal(torch.sort(shuffled_a.flatten()).values, torch.sort(aligned.flatten()).values)


def test_no_patch_bank_is_parameter_matched_zero_context():
    tokens, global_ids = _toy_patch_data()
    token_meta = {
        "global_patch_id": global_ids,
        "sphere_encoding": torch.randn(1, 2, 2, 7),
    }
    aligned = LunaPatchAdapter(dim=1, sphere_dim=7, hidden_dim=4, patch_bank_mode="aligned")
    no_bank = LunaPatchAdapter(dim=1, sphere_dim=7, hidden_dim=4, patch_bank_mode="none")

    aligned_params = sum(parameter.numel() for parameter in aligned.parameters())
    no_bank_params = sum(parameter.numel() for parameter in no_bank.parameters())
    context = no_bank._get_patch_bank_context(tokens, token_meta)

    assert aligned_params == no_bank_params
    assert torch.equal(context, torch.zeros_like(tokens))
    assert torch.equal(no_bank(tokens, token_meta), tokens)


def test_aggregator_propagates_patch_bank_ablation_settings():
    aggregator = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=16,
        depth=2,
        num_heads=2,
        num_register_tokens=1,
        patch_embed="conv",
        enable_luna=True,
        luna_patch_layers=[0, 1],
        luna_camera_layers=[],
        luna_patch_bank_mode="shuffled",
        luna_patch_bank_shuffle_seed=17,
    )

    assert aggregator.luna_patch_adapters["0"].patch_bank_mode == "shuffled"
    assert aggregator.luna_patch_adapters["1"].patch_bank_mode == "shuffled"
    assert aggregator.luna_patch_adapters["0"].patch_bank_shuffle_seed == 17
    assert aggregator.luna_patch_adapters["1"].patch_bank_shuffle_seed == 18


def test_patch_bank_modes_support_backward():
    global_ids = torch.tensor([[[0, 1], [0, 1]]])
    sphere_encoding = torch.randn(1, 2, 2, 7)
    for mode in ("aligned", "none", "shuffled"):
        tokens = torch.randn(1, 2, 2, 4, requires_grad=True)
        adapter = LunaPatchAdapter(
            dim=4,
            sphere_dim=7,
            hidden_dim=8,
            patch_bank_mode=mode,
            patch_bank_shuffle_seed=5,
        )
        adapter.alpha.data.fill_(1.0)
        output = adapter(
            tokens,
            {
                "global_patch_id": global_ids,
                "sphere_encoding": sphere_encoding,
            },
        )
        output.square().mean().backward()

        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()


if __name__ == "__main__":
    test_aligned_patch_bank_uses_erp_correspondence()
    test_shuffled_patch_bank_preserves_features_but_breaks_alignment()
    test_no_patch_bank_is_parameter_matched_zero_context()
    test_aggregator_propagates_patch_bank_ablation_settings()
    test_patch_bank_modes_support_backward()
    print("patch bank ablation tests ok")
