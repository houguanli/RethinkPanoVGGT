import torch

from vggt.models.vggt import build_pano_geometry


def test_build_pano_geometry_shape():
    params = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.5, 0.75]]])
    geometry = build_pano_geometry(pano_view_params=params)
    assert geometry.shape == (1, 2, 6)
    assert torch.allclose(geometry[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]))


if __name__ == "__main__":
    test_build_pano_geometry_shape()
    print("b2 pano geometry smoke ok")
