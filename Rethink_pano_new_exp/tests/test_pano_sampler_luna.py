import torch

from vggt.data.pano_sampler import PanoWindowSampler
from vggt.layers.luna_camera import LunaCameraAdapter
from vggt.layers.luna_patch import LunaPatchAdapter
from vggt.models.aggregator import Aggregator


def test_pano_window_sampler_shapes():
    pano = torch.rand(1, 3, 32, 64)
    sampler = PanoWindowSampler(window_size=28, patch_size=14, fov_degrees=75.0, num_yaw=4)

    output = sampler(pano)

    assert output.windows.shape == (1, 4, 3, 28, 28)
    assert output.token_meta["global_patch_id"].shape == (1, 4, 4)
    assert output.token_meta["sphere_encoding"].shape == (1, 4, 4, 7)
    assert output.camera_meta["camera_encoding"].shape == (1, 4, 16)
    assert output.camera_meta["view_params"].shape == (1, 4, 4)


def test_luna_adapters_are_zero_init_residuals():
    tokens = torch.randn(2, 3, 4, 8)
    token_meta = {
        "global_patch_id": torch.tensor([[[0, 1, 0, 2], [0, 1, 3, 2], [4, 4, 3, 2]]] * 2),
        "sphere_encoding": torch.randn(2, 3, 4, 7),
    }
    patch_adapter = LunaPatchAdapter(dim=8, sphere_dim=7)
    patch_out = patch_adapter(tokens, token_meta)
    assert torch.allclose(patch_out, tokens)

    camera_tokens = torch.randn(2, 3, 8)
    camera_meta = torch.randn(2, 3, 16)
    camera_adapter = LunaCameraAdapter(dim=8, camera_meta_dim=16)
    camera_out = camera_adapter(camera_tokens, camera_meta)
    assert torch.allclose(camera_out, camera_tokens)


def test_luna_aggregator_smoke():
    images = torch.rand(1, 2, 3, 28, 28)
    pano_geometry = torch.randn(1, 2, 6)
    token_meta = {
        "global_patch_id": torch.tensor([[[0, 1, 2, 3], [0, 4, 2, 5]]]),
        "sphere_encoding": torch.randn(1, 2, 4, 7),
    }
    camera_meta = torch.randn(1, 2, 16)

    aggregator = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=16,
        depth=2,
        num_heads=2,
        num_register_tokens=1,
        patch_embed="conv",
        enable_pano_global_token=True,
        enable_luna=True,
        luna_patch_layers=[1],
        luna_camera_layers=[1],
        luna_sphere_dim=7,
        luna_camera_meta_dim=16,
    )
    aggregator.eval()

    outputs, patch_start_idx = aggregator(
        images,
        pano_geometry=pano_geometry,
        pano_token_meta=token_meta,
        pano_camera_meta=camera_meta,
    )

    assert patch_start_idx == 3
    assert len(outputs) == 2
    assert outputs[-1].shape == (1, 2, 7, 32)


if __name__ == "__main__":
    test_pano_window_sampler_shapes()
    test_luna_adapters_are_zero_init_residuals()
    test_luna_aggregator_smoke()
    print("pano sampler + luna adapter smoke ok")
