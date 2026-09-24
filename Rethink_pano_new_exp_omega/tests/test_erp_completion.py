import unittest
from types import SimpleNamespace

import torch

from training.train_erp_completion import completion_losses
from vggt_omega.models.erp_completion import (
    ERPRemainingBandHead, spherical_pixel_weights, splat_omega_window_depth_to_erp,
    apply_completion_sampler_args, validate_completion_sampler_args, canonical_alignment_mask,
)
from vggt_omega.data.pano_sampler import PanoWindowSampler


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

    def test_remaining_scale_error_cannot_align_itself_away(self):
        target = torch.ones(2, 1, 32, 64) * 2
        coverage = torch.zeros_like(target, dtype=torch.bool)
        coverage[:, :, 14:18] = True
        pred = torch.where(coverage, target, target * 4)
        output = {"depth": pred, "completion_depth": pred, "base_depth": target, "coverage": coverage}
        losses = completion_losses(output, target, torch.ones_like(coverage),
                                   torch.zeros(2, 3, 32, 64), "main", 2, num_panos=2)
        self.assertGreater(losses["loss_remaining"].item(), 1)

    def test_belt60_coverage_and_legacy_splat(self):
        torch.set_num_threads(2)
        sampler = PanoWindowSampler(window_size=384, num_yaw=6, pitch_degrees=[-25, 25], fov_degrees=75)
        meta = sampler(torch.zeros(1, 3, 8, 16)).camera_meta
        depth = torch.ones(1, 12, 384, 384, 1)
        default = splat_omega_window_depth_to_erp(depth, meta, num_panos=1, erp_height=512, erp_width=1024)
        legacy = splat_omega_window_depth_to_erp(depth, meta, num_panos=1, erp_height=512, erp_width=1024,
                                                core_latitude_degrees=90)
        self.assertTrue(torch.equal(default.depth, legacy.depth))
        belt = splat_omega_window_depth_to_erp(depth, meta, num_panos=1, erp_height=512, erp_width=1024,
                                              core_latitude_degrees=60)
        lat = 90 - (torch.arange(512) + .5) * 180 / 512
        within = (lat.abs() <= 60).view(1, 1, 512, 1).expand_as(belt.valid_mask)
        self.assertFalse(belt.valid_mask[~within].any())
        self.assertGreater((belt.valid_mask & within).sum().item() / within.sum().item(), .998)
        self.assertEqual(canonical_alignment_mask(512, 1024).shape, (1, 1, 512, 1024))

    def test_sampler_metadata_overrides_parent_but_rejects_conflicting_cli(self):
        args = SimpleNamespace(num_yaw=4, pitch_degrees="-15")
        apply_completion_sampler_args(args, {})
        self.assertEqual(args.num_yaw, 4)
        payload = {"sampler_args": {"num_yaw": 6, "pitch_degrees": "-25,25"}}
        apply_completion_sampler_args(args, payload)
        validate_completion_sampler_args(args, payload)
        self.assertEqual(args.num_yaw, 6)
        args.num_yaw = 4
        with self.assertRaises(ValueError):
            validate_completion_sampler_args(args, payload)

    def test_eval_reports_unobserved_caps_as_missing_not_perfect(self):
        from scripts.evaluate_depth_checkpoint import compute_erp_completion_depth_metrics
        sampler = PanoWindowSampler(window_size=64, num_yaw=6, pitch_degrees=[-25, 25], fov_degrees=75)
        meta = sampler(torch.zeros(1, 3, 32, 64)).camera_meta
        valid = torch.ones(1, 1, 1, 32, 64, dtype=torch.bool)
        valid[..., :6, :] = False
        valid[..., -6:, :] = False
        result = compute_erp_completion_depth_metrics(
            pred_window_z=torch.ones(1, 12, 64, 64, 1),
            pano_rgb=torch.rand(1, 1, 3, 32, 64), gt_erp_depth=torch.ones(1, 1, 1, 32, 64),
            source_depth_semantics="range", max_range_depth=80, camera_meta=meta,
            completion_head=ERPRemainingBandHead(width=8),
            completion_head_args={"height":32,"width_erp":64,"core_latitude_degrees":60,
                                  "alignment_domain":"canonical_pitch15_fov75x75","gt_validity":"rgb_depth_common"},
            gt_common_mask=valid,
        )
        self.assertEqual(result["erp_cap60_valid_pixels"], 0)
        self.assertTrue(torch.isnan(torch.tensor(result["erp_cap60_abs_rel"])))
        self.assertGreater(result["erp_evaluated_gt_fraction"], .99)


if __name__ == "__main__":
    unittest.main()
