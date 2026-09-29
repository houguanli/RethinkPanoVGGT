import math

import torch

from scripts.evaluate_depth_checkpoint import (
    ERP_POLAR_DEPTH_PRIORS_M,
    build_covered_sphere_weights,
    compute_depth_metrics,
    compute_erp_prior_depth_metrics,
)
from evaluation_common.erp_depth import _window_rays


def camera_meta(yaw_values: list[float]) -> dict[str, torch.Tensor]:
    yaw = torch.tensor([yaw_values], dtype=torch.float32)
    zeros = torch.zeros_like(yaw)
    fov = torch.full_like(yaw, math.pi / 2.0)
    return {"yaw": yaw, "pitch": zeros, "fov_x": fov, "fov_y": fov}


def test_duplicate_window_does_not_increase_total_sphere_weight() -> None:
    single = build_covered_sphere_weights(camera_meta([0.0]), height=17, width=17, num_panos=1)
    duplicate = build_covered_sphere_weights(camera_meta([0.0, 0.0]), height=17, width=17, num_panos=1)

    torch.testing.assert_close(duplicate[:, 0], single[:, 0] * 0.5, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(duplicate[:, 1], single[:, 0] * 0.5, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(duplicate.sum(), single.sum(), rtol=1e-5, atol=1e-6)


def test_depth_metrics_use_supplied_spherical_weights() -> None:
    pred = torch.tensor([[[[[2.0], [1.0]]]]])
    target = torch.ones_like(pred)
    valid = torch.ones_like(pred, dtype=torch.bool)
    weights = torch.tensor([[[[[1.0], [3.0]]]]])

    metrics = compute_depth_metrics(pred, target, valid, spherical_weights=weights)

    assert abs(metrics["depth_abs_rel"] - 0.25) < 1e-6
    assert abs(metrics["depth_rmse"] - 0.5) < 1e-6


def test_alignment_mask_does_not_fit_scale_from_completed_region() -> None:
    pred = torch.tensor([2.0, 100.0])
    target = torch.tensor([1.0, 10.0])
    valid = torch.ones(2, dtype=torch.bool)
    alignment_valid = torch.tensor([True, False])
    exempt = torch.tensor([False, True])

    metrics = compute_depth_metrics(
        pred,
        target,
        valid,
        alignment_valid=alignment_valid,
        alignment_scale_exempt=exempt,
    )

    assert abs(metrics["depth_irls_scale"] - 0.5) < 1e-6
    # The second value is exempt from scale because it represents a metric prior.
    assert abs(metrics["depth_irls_abs_rel"] - 4.5) < 1e-6


def test_erp_metric_adds_dataset_polar_prior_without_changing_window_scale() -> None:
    height = width = 24
    yaw = torch.tensor([[0.0]], dtype=torch.float32)
    pitch = torch.zeros_like(yaw)
    fov = torch.full_like(yaw, math.pi / 2.0)
    _, z_factor = _window_rays(yaw, pitch, fov, fov, height, width, align_corners=False)
    pred_window_z = 2.0 * z_factor
    erp_height, erp_width = 48, 96
    gt = torch.full((1, 1, erp_height, erp_width), 4.0)
    latitudes = 90.0 - (torch.arange(erp_height) + 0.5) * 180.0 / erp_height
    south = latitudes <= -75.0
    gt[:, :, south] = ERP_POLAR_DEPTH_PRIORS_M["Stanford2D3DS"]["south"]

    metrics = compute_erp_prior_depth_metrics(
        pred_window_z=pred_window_z,
        gt_erp_depth=gt,
        source_depth_semantics="range",
        max_range_depth=1000.0,
        camera_meta={"yaw": yaw, "pitch": pitch, "fov_x": fov, "fov_y": fov},
        dataset_name="Stanford2D3DS",
        align_corners=False,
    )

    assert abs(metrics["erp_prior_depth_irls_scale"] - 2.0) < 1e-3
    assert metrics["erp_prior_fill_fraction"] > 0.05
    assert metrics["erp_evaluated_gt_fraction"] > metrics["erp_window_coverage_fraction"]
    assert metrics["erp_prior_depth_irls_abs_rel"] < 1e-3
