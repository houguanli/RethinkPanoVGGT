import torch

from training.loss import MultitaskLoss
from training.pano_loss import compute_pano_geometry_loss


def test_pano_geometry_loss_smoke():
    batch_size, num_views = 2, 4
    pose = torch.randn(batch_size, num_views, 9, requires_grad=True)
    pose.data[..., 3:7] = torch.nn.functional.normalize(pose.data[..., 3:7], dim=-1)

    batch = {
        "images": torch.rand(batch_size, num_views, 3, 32, 32),
        "extrinsics": torch.eye(4)[:3].repeat(batch_size, num_views, 1, 1),
        "intrinsics": torch.eye(3).repeat(batch_size, num_views, 1, 1),
        "point_masks": torch.ones(batch_size, num_views, 8, 8, dtype=torch.bool),
        "is_pano": torch.ones(batch_size, dtype=torch.bool),
    }
    predictions = {"pose_enc_list": [pose], "pose_enc": pose}

    loss_dict = compute_pano_geometry_loss(predictions, batch)
    loss_dict["loss_pano"].backward()
    assert pose.grad is not None

    criterion = MultitaskLoss(camera=None, depth=None, point=None, track=None, pano={"weight": 1.0})
    multitask_loss = criterion(predictions, batch)
    assert "objective" in multitask_loss
    assert "loss_pano_rel_rot" in multitask_loss


if __name__ == "__main__":
    test_pano_geometry_loss_smoke()
    print("b1 pano loss smoke ok")
