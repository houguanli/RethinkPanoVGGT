import torch

from training.loss import MultitaskLoss
from training.luna_loss import compute_luna_consistency_loss


def test_luna_consistency_loss_smoke():
    B, S, H, W = 1, 2, 28, 28
    world_points = torch.randn(B, S, H, W, 3, requires_grad=True)
    pose = torch.randn(B, S, 9, requires_grad=True)
    predictions = {
        "world_points": world_points,
        "pose_enc": pose,
        "pose_enc_list": [pose],
    }
    token_meta = {
        "global_patch_id": torch.tensor([[[0, 1, 2, 3], [0, 4, 2, 5]]]),
        "is_seam_region": torch.tensor([[[True, False, False, True], [True, False, False, True]]]),
    }
    batch = {"pano_token_meta": token_meta}

    loss_dict = compute_luna_consistency_loss(predictions, batch)
    loss_dict["loss_luna"].backward()

    assert world_points.grad is not None
    assert pose.grad is not None

    criterion = MultitaskLoss(camera=None, depth=None, point=None, track=None, luna={"weight": 1.0})
    multitask_loss = criterion(predictions, batch)
    assert "objective" in multitask_loss
    assert "loss_luna_point" in multitask_loss


if __name__ == "__main__":
    test_luna_consistency_loss_smoke()
    print("luna loss smoke ok")
