import unittest

import torch

from training.train_erp_completion import completion_losses
from vggt_omega.models.erp_completion import ERPRemainingBandHead, spherical_pixel_weights


class ERPCompletionTest(unittest.TestCase):
    def test_core_is_preserved_and_remaining_is_finite(self):
        torch.manual_seed(1)
        head = ERPRemainingBandHead(width=8)
        rgb = torch.rand(2, 3, 64, 128)
        omega = torch.zeros(2, 1, 64, 128)
        coverage = torch.zeros_like(omega, dtype=torch.bool)
        omega[:, :, 18:47] = torch.rand(2, 1, 29, 128) + 1.0
        coverage[:, :, 18:47] = True
        output = head(rgb, omega, coverage, blend_width_pixels=4)
        self.assertTrue(torch.equal(output["depth"][coverage], omega[coverage]))
        self.assertTrue(torch.isfinite(output["depth"]).all())
        self.assertTrue((output["depth"] > 0).all())

    def test_spherical_weights_downweight_poles(self):
        weight = spherical_pixel_weights(64, 128, torch.device("cpu"), torch.float32)
        self.assertLess(float(weight[0, 0, 0, 0]), float(weight[0, 0, 32, 0]))

    def test_loss_backpropagates_only_small_head(self):
        head = ERPRemainingBandHead(width=8)
        rgb = torch.rand(1, 3, 64, 128)
        omega = torch.zeros(1, 1, 64, 128)
        coverage = torch.zeros_like(omega, dtype=torch.bool)
        omega[:, :, 20:44] = 2.0
        coverage[:, :, 20:44] = True
        target = torch.rand_like(omega) * 4.0 + 1.0
        output = head(rgb, omega, coverage, blend_width_pixels=4)
        losses = completion_losses(output, target, torch.ones_like(coverage), rgb, "main", 4)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))


if __name__ == "__main__":
    unittest.main()
