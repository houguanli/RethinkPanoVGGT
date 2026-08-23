"""Smoke tests for the VGGT-Omega LUNA experiment.

These mirror ``Rethink_pano_new_exp/tests/test_pano_sampler_luna.py`` but speak
the omega API (``patch_token_start`` instead of ``patch_start_idx``, omega's
``Aggregator`` constructor, etc.). Run from this folder so that
``vggt_omega`` resolves to the local fork::

    cd Rethink_pano_new_exp_omega
    python tests/test_pano_sampler_luna_omega.py

The aggregator smoke test bypasses the DinoVisionTransformer patch-embed (which
needs the released checkpoint) by monkey-patching the patch_embed with a tiny
Conv2d-based PatchEmbed. The aim is to verify shapes and the LUNA wiring, not
recover trained accuracy.
"""

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# Ensure local package wins over any globally installed vggt_omega.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from vggt_omega.data.pano_sampler import PanoWindowSampler, resolve_fov_degrees  # noqa: E402
from vggt_omega.models.aggregator import Aggregator, _resolve_luna_layers  # noqa: E402
from vggt_omega.models.layers import LunaCameraAdapter, LunaPatchAdapter, PatchEmbed  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays, rays_to_equirectangular  # noqa: E402
from vggt_omega.models.layers.luna_patch import scatter_mean_by_global_id  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402
from training.train_pano_omega import (  # noqa: E402
    apply_checkpoint_training_defaults,
    apply_stage_sampler_overrides,
    build_pitch_window_loss_weights,
    build_parser,
    capture_default_sampler_args,
    current_sampler_status,
    parse_args as parse_training_args,
    parse_pitch_loss_weights,
)
from scripts.evaluate_depth_checkpoint import apply_eval_sampler_overrides  # noqa: E402


def test_pano_window_sampler_shapes():
    pano = torch.rand(1, 3, 32, 64)
    # window_size=32 → 2x2 patches at patch_size=16
    sampler = PanoWindowSampler(window_size=32, patch_size=16, fov_degrees=75.0, num_yaw=4)

    output = sampler(pano)

    assert output.windows.shape == (1, 4, 3, 32, 32), output.windows.shape
    assert output.token_meta["global_patch_id"].shape == (1, 4, 4), output.token_meta["global_patch_id"].shape
    assert output.token_meta["sphere_encoding"].shape == (1, 4, 4, 7), output.token_meta["sphere_encoding"].shape
    assert output.camera_meta["camera_encoding"].shape == (1, 4, 16), output.camera_meta["camera_encoding"].shape
    assert output.camera_meta["view_params"].shape == (1, 4, 4), output.camera_meta["view_params"].shape


def test_pano_window_sampler_anisotropic_fov_is_backward_compatible():
    pano = torch.rand(1, 3, 32, 64)
    legacy = PanoWindowSampler(window_size=32, patch_size=16, fov_degrees=75.0, num_yaw=4)
    explicit = PanoWindowSampler(
        window_size=32,
        patch_size=16,
        fov_degrees=75.0,
        num_yaw=4,
        fov_x_degrees=75.0,
        fov_y_degrees=75.0,
    )

    legacy_output = legacy(pano)
    explicit_output = explicit(pano)

    assert legacy.get_fov_degrees() == explicit.get_fov_degrees() == (75.0, 75.0)
    assert torch.equal(legacy_output.windows, explicit_output.windows)
    assert torch.equal(
        legacy_output.token_meta["global_patch_id"],
        explicit_output.token_meta["global_patch_id"],
    )


def test_pano_window_sampler_uses_independent_horizontal_and_vertical_fov():
    pano = torch.rand(1, 3, 64, 128)
    sampler = PanoWindowSampler(
        window_size=64,
        patch_size=16,
        fov_degrees=75.0,
        num_yaw=4,
        pitch_degrees=(-15.0,),
        fov_x_degrees=95.0,
        fov_y_degrees=75.0,
    )

    output = sampler(pano)
    fov_x = torch.rad2deg(output.camera_meta["fov_x"])
    fov_y = torch.rad2deg(output.camera_meta["fov_y"])

    assert sampler.get_fov_degrees() == (95.0, 75.0)
    assert torch.allclose(fov_x, torch.full_like(fov_x, 95.0), atol=1e-5)
    assert torch.allclose(fov_y, torch.full_like(fov_y, 75.0), atol=1e-5)
    assert resolve_fov_degrees(75.0, 95.0, None) == (95.0, 75.0)


def test_training_stage_preserves_anisotropic_fov_defaults():
    args = build_parser().parse_args(
        [
            "--window-size",
            "32",
            "--num-yaw",
            "4",
            "--pitch-degrees=-15",
            "--fov-degrees",
            "75",
            "--fov-x-degrees",
            "95",
            "--fov-y-degrees",
            "75",
        ]
    )
    capture_default_sampler_args(args)
    model = nn.Module()
    model.pano_sampler = PanoWindowSampler(window_size=32, patch_size=16, num_yaw=4)
    model.aggregator = SimpleNamespace(patch_size=16)

    status = apply_stage_sampler_overrides(model, args, stage=None)

    assert status["fov_degrees"] == 75.0
    assert status["fov_x_degrees"] == 95.0
    assert status["fov_y_degrees"] == 75.0
    assert model.pano_sampler.get_fov_degrees() == (95.0, 75.0)


def test_pitch_minus15_keeps_75_degree_vertical_domain():
    rays = pinhole_rays(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([math.radians(-15.0)], dtype=torch.float64),
        torch.tensor([math.radians(95.0)], dtype=torch.float64),
        torch.tensor([math.radians(75.0)], dtype=torch.float64),
        65,
        65,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    _, phi, _, _ = rays_to_equirectangular(rays)
    center_column = torch.rad2deg(phi[0, :, 32])
    assert math.isclose(float(center_column.min()), -52.5, abs_tol=1e-6)
    assert math.isclose(float(center_column.max()), 22.5, abs_tol=1e-6)


def test_fov95_increases_coverage_and_shared_sphere_tokens():
    pano = torch.zeros(1, 3, 512, 1024)

    def token_stats(fov_x_degrees: float) -> tuple[int, int]:
        output = PanoWindowSampler(
            window_size=384,
            patch_size=16,
            fov_degrees=75.0,
            fov_x_degrees=fov_x_degrees,
            fov_y_degrees=75.0,
            num_yaw=4,
            pitch_degrees=(-15.0,),
        )(pano)
        ids = output.token_meta["global_patch_id"][0]
        view_sets = [set(row.tolist()) for row in ids]
        covered = set().union(*view_sets)
        shared: set[int] = set()
        for view_index in range(4):
            shared.update(view_sets[view_index] & view_sets[(view_index + 1) % 4])
        return len(covered), len(shared)

    covered75, shared75 = token_stats(75.0)
    covered95, shared95 = token_stats(95.0)
    assert covered95 >= covered75 + 40, (covered75, covered95)
    assert shared95 >= shared75 + 80, (shared75, shared95)


def test_checkpoint_sampler_geometry_round_trip_handles_multiple_pitch_rings():
    saved_args = build_parser().parse_args(
        [
            "--num-yaw", "4",
            "--pitch-degrees=-20,55,-72",
            "--fov-degrees", "75",
            "--fov-x-degrees", "95",
            "--fov-y-degrees", "75",
        ]
    )
    model = nn.Module()
    model.pano_sampler = PanoWindowSampler(
        num_yaw=4,
        pitch_degrees=(-20.0, 55.0, -72.0),
        fov_degrees=75.0,
        fov_x_degrees=95.0,
        fov_y_degrees=75.0,
    )
    status = current_sampler_status(model, saved_args)
    assert status["num_yaw"] == 4

    resumed_args = build_parser().parse_args([])
    apply_checkpoint_training_defaults(resumed_args, {"args": status})
    assert resumed_args.num_yaw == 4
    assert resumed_args.pitch_degrees == "-20,55,-72"
    assert resumed_args.fov_degrees == 75.0
    assert resumed_args.fov_x_degrees == 95.0
    assert resumed_args.fov_y_degrees == 75.0


def test_canonical_eval_override_resets_checkpoint_anisotropic_fov():
    train_args = SimpleNamespace(
        window_size=384,
        num_yaw=12,
        pitch_degrees="-20,55,-72",
        fov_degrees=75.0,
        fov_x_degrees=95.0,
        fov_y_degrees=75.0,
    )
    eval_args = SimpleNamespace(
        window_size=384,
        num_yaw=4,
        pitch_degrees="-15",
        fov_degrees=75.0,
        fov_x_degrees=None,
        fov_y_degrees=None,
    )
    apply_eval_sampler_overrides(train_args, eval_args)
    assert train_args.num_yaw == 4
    assert train_args.pitch_degrees == "-15"
    assert train_args.fov_x_degrees == train_args.fov_y_degrees == 75.0


def test_m1_ab_configs_round_trip_and_only_change_horizontal_fov():
    project_root = Path(THIS_DIR).parent
    control = parse_training_args(
        ["--config", str(project_root / "configs/multipano_4090_mixed4_m1_fov75x75_pitch15_luna_3h.yaml")]
    )
    treatment = parse_training_args(
        ["--config", str(project_root / "configs/multipano_4090_mixed4_m1_fov95x75_pitch15_luna_3h.yaml")]
    )
    for args in (control, treatment):
        assert args.num_yaw == 4
        assert args.pitch_degrees == "-15"
        assert args.fov_degrees == 75.0
        assert args.fov_y_degrees == 75.0
        assert args.pano_min_count == args.pano_max_count == 2
        assert args.seed == 47
        assert args.inherit_checkpoint_training_defaults is False
    assert control.fov_x_degrees == 75.0
    assert treatment.fov_x_degrees == 95.0


def test_pitch_ring_loss_weights_follow_yaw_major_sampler_order():
    weights = build_pitch_window_loss_weights(
        total_views=12,
        pano_count=1,
        num_yaw=4,
        pitch_degrees="-20,55,-72",
        pitch_loss_weights="1.0,0.45,0.25",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(weights, torch.tensor([[1.0, 0.45, 0.25] * 4]))
    assert parse_pitch_loss_weights("", 3) == (1.0, 1.0, 1.0)


def test_pitch_ring_loss_weights_repeat_per_panorama_and_validate_count():
    weights = build_pitch_window_loss_weights(
        total_views=16,
        pano_count=2,
        num_yaw=4,
        pitch_degrees="-20,55",
        pitch_loss_weights="1.0,0.5",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(weights, torch.tensor([[1.0, 0.5] * 8]))
    try:
        parse_pitch_loss_weights("1.0,0.5", 3)
    except ValueError as exc:
        assert "one value per pitch ring" in str(exc)
    else:
        raise AssertionError("Expected invalid pitch_loss_weights to fail")


def test_luna_adapters_are_zero_init_residuals():
    tokens = torch.randn(2, 3, 4, 8)
    token_meta = {
        "global_patch_id": torch.tensor([[[0, 1, 0, 2], [0, 1, 3, 2], [4, 4, 3, 2]]] * 2),
        "sphere_encoding": torch.randn(2, 3, 4, 7),
    }
    patch_adapter = LunaPatchAdapter(dim=8, sphere_dim=7)
    patch_out = patch_adapter(tokens, token_meta)
    assert torch.allclose(patch_out, tokens), "LUNA-Patch is not a zero-init residual"

    camera_tokens = torch.randn(2, 3, 8)
    camera_meta = torch.randn(2, 3, 16)
    camera_adapter = LunaCameraAdapter(dim=8, camera_meta_dim=16)
    camera_out = camera_adapter(camera_tokens, camera_meta)
    assert torch.allclose(camera_out, camera_tokens), "LUNA-Camera is not a zero-init residual"


def test_luna_patch_pooling_is_isolated_by_pano_id():
    # Repeated windows from pano 0 share a mean. Pano 1 has the same ERP grid
    # ID but another camera center, so it must remain in a separate pool.
    tokens = torch.tensor([[[[1.0]], [[3.0]], [[100.0]]]])
    global_ids = torch.zeros(1, 3, 1, dtype=torch.long)
    pano_ids = torch.tensor([[[0], [0], [1]]], dtype=torch.long)

    pooled = scatter_mean_by_global_id(tokens, global_ids, pano_ids=pano_ids)

    assert pooled[:, :2].tolist() == [[[[2.0]], [[2.0]]]]
    assert pooled[:, 2:].tolist() == [[[[100.0]]]]


def _swap_in_conv_patch_embed(aggregator: Aggregator, img_size: int, embed_dim: int) -> None:
    """Replace the omega DinoVisionTransformer patch_embed with a tiny Conv2d patch_embed.

    This avoids needing the released omega checkpoint to instantiate the model.
    The returned module produces ``[B*S, P, C]`` patch tokens directly.
    """
    aggregator.patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=aggregator.patch_size,
        in_chans=3,
        embed_dim=embed_dim,
    )


def test_luna_aggregator_smoke():
    # Build a tiny aggregator: depth=2, 1 register token, no register-attn layers.
    embed_dim = 64
    aggregator = Aggregator(
        patch_size=16,
        embed_dim=embed_dim,
        depth=2,
        num_heads=2,
        num_register_tokens=1,
        register_attention_block_indices=(),
        cached_layer_indices=(0, 1),
        enable_pano_global_token=True,
        enable_luna=True,
        luna_patch_layers=[1],
        luna_camera_layers=[1],
        luna_sphere_dim=7,
        luna_camera_meta_dim=16,
    )
    _swap_in_conv_patch_embed(aggregator, img_size=32, embed_dim=embed_dim)
    aggregator.eval()

    images = torch.rand(1, 2, 3, 32, 32)
    pano_geometry = torch.randn(1, 2, 6)
    token_meta = {
        "global_patch_id": torch.tensor([[[0, 1, 2, 3], [0, 4, 2, 5]]]),
        "sphere_encoding": torch.randn(1, 2, 4, 7),
    }
    camera_meta = torch.randn(1, 2, 16)

    with torch.no_grad():
        outputs, patch_token_start = aggregator(
            images,
            pano_geometry=pano_geometry,
            pano_token_meta=token_meta,
            pano_camera_meta=camera_meta,
        )

    # camera(1) + pano_global(1) + register(1) = 3
    assert patch_token_start == 3, patch_token_start
    # window 32x32 with patch_size 16 → 2x2 = 4 patches; plus 3 special = 7 tokens.
    assert len(outputs) == 2 and all(o is not None for o in outputs)
    assert outputs[-1].shape == (1, 2, 7, 2 * embed_dim), outputs[-1].shape


def test_luna_wrapper_can_skip_default_checkpoint_load():
    model = VGGTOmega_LUNA(
        embed_dim=64,
        checkpoint_path=None,
        aggregator_kwargs={
            "depth": 2,
            "num_heads": 2,
            "num_register_tokens": 1,
            "register_attention_block_indices": (),
            "cached_layer_indices": (0, 1),
        },
    )
    assert model.pano_sampler.window_size == 512


def test_luna_tail_layer_count_supports_full_depth():
    assert _resolve_luna_layers("last3", 24) == {21, 22, 23}
    assert _resolve_luna_layers("last24", 24) == set(range(24))


if __name__ == "__main__":
    test_pano_window_sampler_shapes()
    test_pano_window_sampler_anisotropic_fov_is_backward_compatible()
    test_pano_window_sampler_uses_independent_horizontal_and_vertical_fov()
    test_training_stage_preserves_anisotropic_fov_defaults()
    test_pitch_minus15_keeps_75_degree_vertical_domain()
    test_fov95_increases_coverage_and_shared_sphere_tokens()
    test_checkpoint_sampler_geometry_round_trip_handles_multiple_pitch_rings()
    test_canonical_eval_override_resets_checkpoint_anisotropic_fov()
    test_m1_ab_configs_round_trip_and_only_change_horizontal_fov()
    test_pitch_ring_loss_weights_follow_yaw_major_sampler_order()
    test_pitch_ring_loss_weights_repeat_per_panorama_and_validate_count()
    test_luna_adapters_are_zero_init_residuals()
    test_luna_patch_pooling_is_isolated_by_pano_id()
    test_luna_aggregator_smoke()
    test_luna_wrapper_can_skip_default_checkpoint_load()
    test_luna_tail_layer_count_supports_full_depth()
    print("pano sampler + LUNA aggregator smoke (omega) ok")
