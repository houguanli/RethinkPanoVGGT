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

import torch
import torch.nn as nn

# Ensure local package wins over any globally installed vggt_omega.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from vggt_omega.data.pano_sampler import PanoWindowSampler  # noqa: E402
from vggt_omega.models.aggregator import Aggregator  # noqa: E402
from vggt_omega.models.layers import LunaCameraAdapter, LunaPatchAdapter, PatchEmbed  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402


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


if __name__ == "__main__":
    test_pano_window_sampler_shapes()
    test_luna_adapters_are_zero_init_residuals()
    test_luna_aggregator_smoke()
    test_luna_wrapper_can_skip_default_checkpoint_load()
    print("pano sampler + LUNA aggregator smoke (omega) ok")
