#!/usr/bin/env python3
"""Diagnose base Omega pano-relative pose through source-to-target depth reprojection.

For reproducible Matterport diagnostics, this script reads the same official
parsed room metadata as the training loader and records the selected room/pair
explicitly in its outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_untrained_omega_camera_cases import (  # noqa: E402
    build_base_omega,
    depth_metrics,
    finite_mean,
    finite_median,
    move_sample_to_device,
    rotation_metrics,
    rgb_tile,
    safe_name,
    squeeze_depth,
    tensor_to_list,
    translation_metrics,
)
from training.data.pano_minimal import PanoMinimalDataset  # noqa: E402
from training.train_pano_omega import (  # noqa: E402
    build_relative_pano_pose_targets,
    estimate_sample_depth_alignment_scale,
    load_checkpoint,
    sample_depth_targets,
)
from vggt_omega.models.layers.pano_position import pinhole_rays  # noqa: E402
from vggt_omega.utils.pano_pose import (  # noqa: E402
    omega_y_up_pose_to_official_y_down,
    omega_y_up_vectors_to_official_y_down,
)
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="matterport3d,structured3d")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cases-per-dataset", type=int, default=2)
    parser.add_argument("--panos-per-case", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-yaw", type=int, default=4)
    parser.add_argument("--pitch-degrees", default="-15")
    parser.add_argument("--fov-degrees", type=float, default=75.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-range-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale-alignment", default="sample_lstsq")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--min-pair-distance-m", type=float, default=0.25)
    parser.add_argument("--max-pair-distance-m", type=float, default=2.0)
    parser.add_argument("--gt-reproject-stride", type=int, default=2)
    parser.add_argument(
        "--matterport-pair",
        default="",
        help="Optional explicit scan:pano_a:pano_b pair for reproducible diagnostics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_base_omega(args).to(device).eval()
    load_checkpoint(model, args.checkpoint, strict=False)

    rows: list[dict[str, Any]] = []
    for dataset_name in [name.strip().lower() for name in args.datasets.split(",") if name.strip()]:
        dataset = PanoMinimalDataset(
            root=args.dataset_root,
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=args.panos_per_case,
            pano_max_count=args.panos_per_case,
            split=args.split,
            datasets=dataset_name,
            grouping="nearest",
            bad_sample_list=PROJECT_ROOT / "configs" / "structured3d_bad_scenes.txt",
        )
        cases = select_cases(args, dataset, dataset_name)
        for case_id, case in enumerate(cases):
            row = evaluate_case(args, model, dataset, dataset_name, case, case_id, device)
            rows.append(row)
            print(
                f"[case] {dataset_name}#{case_id} {row['case_label']} "
                f"pair_dist={row['gt_pair_distance_m']:.3f}m "
                f"t_raw={row['translation_raw_basis_l2_mean_m']:.3f}m "
                f"t_official={row['translation_l2_mean_m']:.3f}m "
                f"t_scaled={row['translation_depthscaled_l2_mean_m']:.3f}m "
                f"r_raw={row['rotation_raw_basis_deg_mean']} "
                f"r_official={row['rotation_deg_mean']} "
                f"gt_reproj_abslog={row['gt_reproj_abs_log']:.4f} "
                f"pred_pose_abslog={row['pred_predpose_reproj_abs_log']:.4f}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "per_case_metrics.csv", rows)
    summary = summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {args.output_dir / 'summary.json'}")


def select_cases(args: argparse.Namespace, dataset: PanoMinimalDataset, dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name == "matterport3d":
        return select_matterport_room_cases(args, dataset)
    if dataset_name == "structured3d":
        return select_scene_cases(args, dataset, rotation_required=False)
    raise ValueError(f"Unsupported dataset for this diagnostic: {dataset_name}")


def select_matterport_room_cases(args: argparse.Namespace, dataset: PanoMinimalDataset) -> list[dict[str, Any]]:
    by_scan_pano: dict[tuple[str, str], int] = {}
    for index, item in enumerate(dataset.items):
        path = Path(str(item["rgb_path"]))
        if len(path.parts) < 3:
            continue
        scan = path.parts[-3]
        pano_id = path.stem
        by_scan_pano[(scan, pano_id)] = index

    if args.matterport_pair:
        parts = args.matterport_pair.split(":")
        if len(parts) != 3:
            raise ValueError("--matterport-pair must be scan:pano_a:pano_b")
        scan, first_pano, second_pano = parts
        missing = [pano for pano in (first_pano, second_pano) if (scan, pano) not in by_scan_pano]
        if missing:
            raise ValueError(f"Explicit Matterport pano(s) not indexed: {missing}")
        indices = [by_scan_pano[(scan, first_pano)], by_scan_pano[(scan, second_pano)]]
        first = np.asarray(dataset.items[indices[0]]["pano_position_m"], dtype=np.float32)
        second = np.asarray(dataset.items[indices[1]]["pano_position_m"], dtype=np.float32)
        return [
            {
                "indices": indices,
                "case_label": f"{scan}/explicit_{first_pano[:8]}_{second_pano[:8]}",
                "scan": scan,
                "gt_pair_distance_m": float(np.linalg.norm(first - second)),
            }
        ]

    cases: list[dict[str, Any]] = []
    parsed_dir = args.dataset_root / "Matterport3D" / "parsed_json"
    for json_path in sorted(parsed_dir.glob("*.json")):
        scan = json_path.stem
        room_payload = json.loads(json_path.read_text(encoding="utf-8"))
        for room_id, room in sorted(room_payload.items(), key=lambda kv: str(kv[0])):
            pano_ids = [str(pid) for pid in room.get("panoramas", [])]
            indices = [by_scan_pano[(scan, pid)] for pid in pano_ids if (scan, pid) in by_scan_pano]
            if len(indices) < args.panos_per_case:
                continue
            pair = choose_nearest_pair(dataset, indices, args.min_pair_distance_m, args.max_pair_distance_m)
            if pair is None:
                continue
            cases.append(
                {
                    "indices": list(pair["indices"]),
                    "case_label": f"{scan}/room_{room_id}_{safe_name(str(room.get('room_name', 'room')))}",
                    "room_id": str(room_id),
                    "room_name": str(room.get("room_name", "")),
                    "scan": scan,
                    "gt_pair_distance_m": float(pair["distance"]),
                }
            )
    if not cases:
        raise RuntimeError("No readable same-room Matterport pairs found.")
    rng = np.random.default_rng(int(args.seed) + 17)
    order = rng.permutation(len(cases))
    return [cases[int(index)] for index in order[: args.cases_per_dataset]]


def select_scene_cases(
    args: argparse.Namespace,
    dataset: PanoMinimalDataset,
    rotation_required: bool,
) -> list[dict[str, Any]]:
    scene_to_indices: dict[str, list[int]] = {}
    for index, item in enumerate(dataset.items):
        key = str(item.get("scene_group_key") or item.get("scene_name", ""))
        scene_to_indices.setdefault(key, []).append(index)

    cases: list[dict[str, Any]] = []
    for scene_key, indices in sorted(scene_to_indices.items()):
        if len(indices) < args.panos_per_case:
            continue
        if rotation_required:
            indices = [idx for idx in indices if bool(dataset.items[idx].get("pano_rotation_valid", False))]
        pair = choose_nearest_pair(dataset, indices, args.min_pair_distance_m, args.max_pair_distance_m)
        if pair is None:
            continue
        cases.append(
            {
                "indices": list(pair["indices"]),
                "case_label": scene_key,
                "gt_pair_distance_m": float(pair["distance"]),
            }
        )
    if not cases:
        raise RuntimeError("No readable same-scene pairs found.")
    rng = np.random.default_rng(int(args.seed) + (31 if rotation_required else 43))
    order = rng.permutation(len(cases))
    return [cases[int(index)] for index in order[: args.cases_per_dataset]]


def choose_nearest_pair(
    dataset: PanoMinimalDataset,
    indices: list[int],
    min_distance: float,
    max_distance: float,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, tuple[int, int]]] = []
    for left, right in combinations(indices, 2):
        a = np.asarray(dataset.items[left].get("pano_position_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        b = np.asarray(dataset.items[right].get("pano_position_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        distance = float(np.linalg.norm(a - b))
        if min_distance <= distance <= max_distance:
            candidates.append((distance, (left, right)))
    if not candidates:
        return None
    distance, pair = sorted(candidates, key=lambda value: value[0])[0]
    return {"distance": distance, "indices": pair}


def build_official_relative_pose_targets(
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pano-0-relative targets directly from official loader poses."""
    centers = batch["pano_position_m"].to(device=device, dtype=dtype).unsqueeze(0)
    rotations_c2w = batch["pano_rotation_c2w"].to(device=device, dtype=dtype).unsqueeze(0)
    position_valid = batch["pano_position_valid"].to(device=device).bool().unsqueeze(0)
    rotation_valid = batch["pano_rotation_valid"].to(device=device).bool().unsqueeze(0)
    anchor_w2c = rotations_c2w[:, :1].transpose(-1, -2).contiguous()
    relative_centers = (anchor_w2c @ (centers - centers[:, :1])[..., None])[..., 0]
    relative_c2w = anchor_w2c @ rotations_c2w
    relative_w2c = relative_c2w.transpose(-1, -2).contiguous()
    relative_quat = F.normalize(mat_to_quat(relative_w2c), dim=-1, eps=1e-6)
    return (
        relative_centers,
        relative_quat,
        position_valid & position_valid[:, :1],
        rotation_valid & rotation_valid[:, :1],
    )


def quaternion_angle_deg(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, dim=-1, eps=1e-6)
    right = F.normalize(right, dim=-1, eps=1e-6)
    cosine = (left * right).sum(dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(cosine))


def evaluate_case(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataset: PanoMinimalDataset,
    dataset_name: str,
    case: dict[str, Any],
    case_id: int,
    device: torch.device,
) -> dict[str, Any]:
    sample = read_explicit_group(dataset, case["indices"])
    batch = move_sample_to_device(sample, device)
    pano_images = batch["pano_image"].unsqueeze(0)
    pano_depth = batch["pano_depth"].unsqueeze(0)
    pano_count = int(pano_images.shape[1])

    target_depth_z, target_valid_z = sample_depth_targets(
        model,
        pano_depth,
        source_depth_semantics="range",
        max_range_depth=float(args.max_range_depth),
    )
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float32
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        predictions = model(
            pano_images=pano_images,
            return_sampler_output=True,
            return_window_pose=True,
        )

    pred_center_init = predictions.get("pano_camera_center_init", predictions["pano_camera_center"]).float()
    pred_quat_init = F.normalize(
        predictions.get("pano_rotation_quat_w2c_init", predictions["pano_rotation_quat_w2c"]).float(),
        dim=-1,
        eps=1e-6,
    )
    pred_center_residual = predictions.get("pano_camera_center_residual")
    pred_quat_residual = predictions.get("pano_rotation_quat_w2c_residual")
    pred_center = predictions.get("pano_camera_center", pred_center_init).float()
    pred_quat = F.normalize(
        predictions.get("pano_rotation_quat_w2c", pred_quat_init).float(),
        dim=-1,
        eps=1e-6,
    )
    training_target_center, training_target_quat, _, _ = build_relative_pano_pose_targets(
        batch={key: value.unsqueeze(0) if torch.is_tensor(value) else value for key, value in batch.items()},
        position_mode="relative_anchor",
        device=device,
        dtype=pred_center.dtype,
    )
    target_center, target_quat, position_valid, rotation_valid = build_official_relative_pose_targets(
        batch,
        device=device,
        dtype=pred_center.dtype,
    )
    target_center_delta = float((training_target_center - target_center).abs().max().detach().cpu())
    target_rotation_delta_deg = float(
        quaternion_angle_deg(training_target_quat, target_quat).max().detach().cpu()
    )
    if target_center_delta > 1e-5 or target_rotation_delta_deg > 1e-3:
        raise RuntimeError(
            "Training camera target does not match the direct official-pose target: "
            f"center_delta={target_center_delta:.6g}, rotation_delta_deg={target_rotation_delta_deg:.6g}"
        )

    pred_center_init_official, pred_quat_init_official = omega_y_up_pose_to_official_y_down(
        pred_center_init,
        pred_quat_init,
    )
    pred_center_official, pred_quat_official = omega_y_up_pose_to_official_y_down(pred_center, pred_quat)

    translation_init_raw_basis = translation_metrics(pred_center_init, target_center, position_valid)
    rotation_init_raw_basis = rotation_metrics(pred_quat_init, target_quat, rotation_valid)
    translation_raw_basis = translation_metrics(pred_center, target_center, position_valid)
    rotation_raw_basis = rotation_metrics(pred_quat, target_quat, rotation_valid)
    translation_init = translation_metrics(pred_center_init_official, target_center, position_valid)
    rotation_init = rotation_metrics(pred_quat_init_official, target_quat, rotation_valid)
    translation = translation_metrics(pred_center_official, target_center, position_valid)
    rotation = rotation_metrics(pred_quat_official, target_quat, rotation_valid)

    raw_pred_depth_z = squeeze_depth(predictions["depth"].float())
    target_depth_z_s = squeeze_depth(target_depth_z.float())
    target_valid_z_s = squeeze_depth(target_valid_z.float()).bool()
    scale = estimate_sample_depth_alignment_scale(
        raw_pred_depth_z,
        target_depth_z_s,
        target_valid_z_s,
        mode=args.depth_scale_alignment,
        min_scale=0.05,
        max_scale=50.0,
        eps=1e-6,
    )
    pred_depth_z = raw_pred_depth_z * scale.reshape(-1, *([1] * (raw_pred_depth_z.ndim - 1)))
    window_depth = depth_metrics(pred_depth_z, target_depth_z_s, target_valid_z_s)
    depth_scale_scalar = float(scale.detach().cpu().reshape(-1)[0])
    pred_center_depthscaled = pred_center_official * depth_scale_scalar
    translation_depthscaled = translation_metrics(pred_center_depthscaled, target_center, position_valid)
    gt_pair_distance = float(case["gt_pair_distance_m"])
    translation_depthscaled_norm_l2 = (
        None
        if translation_depthscaled["l2_mean_m"] is None or gt_pair_distance <= 1e-6
        else float(translation_depthscaled["l2_mean_m"]) / gt_pair_distance
    )

    meta = predictions["pano_camera_meta"]
    target_hw = tuple(int(v) for v in sample["pano_depth"].shape[-2:])
    views_per_pano = int(pred_depth_z.shape[0] // pano_count)
    source_slice = slice(0, views_per_pano)
    identity_center = torch.zeros(3, dtype=torch.float32)
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    pred_source_projected, pred_source_projected_valid = reproject_window_depth_to_target_erp_official(
        pred_depth_z[source_slice].detach().float().cpu(),
        slice_camera_meta(meta, source_slice),
        identity_center,
        identity_quat,
        target_hw=target_hw,
        max_range_depth=float(args.max_range_depth),
    )
    pred_gtpose_projected, pred_gtpose_projected_valid = reproject_window_depth_to_target_erp_official(
        pred_depth_z[source_slice].detach().float().cpu(),
        slice_camera_meta(meta, source_slice),
        target_center[0, 1].detach().float().cpu(),
        target_quat[0, 1].detach().float().cpu(),
        target_hw=target_hw,
        max_range_depth=float(args.max_range_depth),
    )
    pred_predpose_projected, pred_predpose_projected_valid = reproject_window_depth_to_target_erp_official(
        pred_depth_z[source_slice].detach().float().cpu(),
        slice_camera_meta(meta, source_slice),
        pred_center_depthscaled[0, 1].detach().float().cpu(),
        pred_quat_official[0, 1].detach().float().cpu(),
        target_hw=target_hw,
        max_range_depth=float(args.max_range_depth),
    )
    gt_projected, gt_projected_valid = reproject_source_erp_depth_to_target_erp_official(
        sample["pano_depth"][0, 0].detach().float().cpu(),
        target_center[0, 1].detach().float().cpu(),
        target_quat[0, 1].detach().float().cpu(),
        target_hw=target_hw,
        max_range_depth=float(args.max_range_depth),
        stride=max(1, int(args.gt_reproject_stride)),
    )

    source_erp = sample["pano_depth"][0, 0].detach().float().cpu().numpy()
    source_valid_erp = np.isfinite(source_erp) & (source_erp > 0) & (source_erp <= float(args.max_range_depth))
    target_erp = sample["pano_depth"][1, 0].detach().float().cpu().numpy()
    target_valid_erp = np.isfinite(target_erp) & (target_erp > 0) & (target_erp <= float(args.max_range_depth))
    pred_source_reproj = reprojection_metrics(
        pred_source_projected,
        pred_source_projected_valid,
        source_erp,
        source_valid_erp,
    )
    pred_gtpose_reproj = reprojection_metrics(
        pred_gtpose_projected,
        pred_gtpose_projected_valid,
        target_erp,
        target_valid_erp,
    )
    pred_predpose_reproj = reprojection_metrics(
        pred_predpose_projected,
        pred_predpose_projected_valid,
        target_erp,
        target_valid_erp,
    )
    gt_reproj = reprojection_metrics(gt_projected, gt_projected_valid, target_erp, target_valid_erp)

    case_dir = args.output_dir / f"{dataset_name}_case{case_id:02d}_{safe_name(case['case_label'])}"
    case_dir.mkdir(parents=True, exist_ok=True)
    panel_path = case_dir / "relative_reprojection_panel.png"
    make_reprojection_panel(
        panel_path,
        sample,
        gt_projected,
        gt_projected_valid,
        pred_source_projected,
        pred_source_projected_valid,
        pred_gtpose_projected,
        pred_gtpose_projected_valid,
        pred_predpose_projected,
        pred_predpose_projected_valid,
        source_erp,
        source_valid_erp,
        target_erp,
        target_valid_erp,
        gt_reproj,
        pred_source_reproj,
        pred_gtpose_reproj,
        pred_predpose_reproj,
        case,
    )

    details = {
        "dataset": dataset_name,
        "case_id": case_id,
        "case": case,
        "scene_name": sample["scene_name"],
        "sequence_name": sample["sequence_name"],
        "rgb_path": sample["rgb_path"],
        "depth_path": sample["depth_path"],
        "pano_position_m": tensor_to_list(batch["pano_position_m"]),
        "pano_position_valid": tensor_to_list(batch["pano_position_valid"]),
        "pano_rotation_valid": tensor_to_list(batch["pano_rotation_valid"]),
        "pred_center_init_relative": tensor_to_list(pred_center_init[0]),
        "pred_quat_init_w2c_relative_xyzw": tensor_to_list(pred_quat_init[0]),
        "pred_center_init_official_relative": tensor_to_list(pred_center_init_official[0]),
        "pred_quat_init_official_w2c_relative_xyzw": tensor_to_list(pred_quat_init_official[0]),
        "pred_center_residual": tensor_to_list(pred_center_residual[0]) if torch.is_tensor(pred_center_residual) else None,
        "pred_quat_residual_xyzw": tensor_to_list(pred_quat_residual[0]) if torch.is_tensor(pred_quat_residual) else None,
        "pred_center_relative": tensor_to_list(pred_center[0]),
        "pred_center_official_relative": tensor_to_list(pred_center_official[0]),
        "target_center_relative": tensor_to_list(target_center[0]),
        "pred_quat_w2c_relative_xyzw": tensor_to_list(pred_quat[0]),
        "pred_quat_official_w2c_relative_xyzw": tensor_to_list(pred_quat_official[0]),
        "target_quat_w2c_relative_xyzw": tensor_to_list(target_quat[0]),
        "training_target_center_max_abs_delta": target_center_delta,
        "training_target_rotation_max_delta_deg": target_rotation_delta_deg,
        "translation_init_raw_basis": translation_init_raw_basis,
        "rotation_init_raw_basis": rotation_init_raw_basis,
        "translation_raw_basis": translation_raw_basis,
        "rotation_raw_basis": rotation_raw_basis,
        "translation_init": translation_init,
        "rotation_init": rotation_init,
        "translation": translation,
        "translation_depthscaled": translation_depthscaled,
        "translation_depthscaled_norm_l2_mean": translation_depthscaled_norm_l2,
        "rotation": rotation,
        "window_depth": window_depth,
        "pred_source_reprojection": pred_source_reproj,
        "pred_gtpose_reprojection": pred_gtpose_reproj,
        "pred_predpose_reprojection": pred_predpose_reproj,
        "gt_reprojection": gt_reproj,
        "depth_scale": depth_scale_scalar,
        "visualization": str(panel_path.resolve()),
    }
    metrics_path = case_dir / "case_metrics.json"
    metrics_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "dataset": dataset_name,
        "case_id": case_id,
        "case_label": case["case_label"],
        "scene_name": sample["scene_name"],
        "pano_count": pano_count,
        "gt_pair_distance_m": float(case["gt_pair_distance_m"]),
        "training_target_center_max_abs_delta": target_center_delta,
        "training_target_rotation_max_delta_deg": target_rotation_delta_deg,
        "translation_init_raw_basis_l2_mean_m": translation_init_raw_basis["l2_mean_m"],
        "rotation_init_raw_basis_deg_mean": rotation_init_raw_basis["deg_mean"],
        "translation_raw_basis_l2_mean_m": translation_raw_basis["l2_mean_m"],
        "rotation_raw_basis_deg_mean": rotation_raw_basis["deg_mean"],
        "translation_init_l2_mean_m": translation_init["l2_mean_m"],
        "rotation_init_deg_mean": rotation_init["deg_mean"],
        "translation_l2_mean_m": translation["l2_mean_m"],
        "translation_l2_median_m": translation["l2_median_m"],
        "translation_norm_l2_mean": translation["norm_l2_mean"],
        "translation_depthscaled_l2_mean_m": translation_depthscaled["l2_mean_m"],
        "translation_depthscaled_l2_median_m": translation_depthscaled["l2_median_m"],
        "translation_depthscaled_norm_l2_mean": translation_depthscaled_norm_l2,
        "rotation_deg_mean": rotation["deg_mean"],
        "rotation_deg_median": rotation["deg_median"],
        "window_depth_abs_rel": window_depth["abs_rel"],
        "window_depth_delta_1p25": window_depth["delta_1p25"],
        "depth_scale": depth_scale_scalar,
        "gt_reproj_overlap": gt_reproj["overlap_ratio"],
        "gt_reproj_abs_log": gt_reproj["abs_log"],
        "gt_reproj_abs_rel": gt_reproj["abs_rel"],
        "gt_reproj_visible_log0p1": gt_reproj["visible_ratio_log0p1"],
        "pred_source_reproj_overlap": pred_source_reproj["overlap_ratio"],
        "pred_source_reproj_abs_log": pred_source_reproj["abs_log"],
        "pred_source_reproj_abs_rel": pred_source_reproj["abs_rel"],
        "pred_source_reproj_visible_log0p1": pred_source_reproj["visible_ratio_log0p1"],
        "pred_gtpose_reproj_overlap": pred_gtpose_reproj["overlap_ratio"],
        "pred_gtpose_reproj_abs_log": pred_gtpose_reproj["abs_log"],
        "pred_gtpose_reproj_abs_rel": pred_gtpose_reproj["abs_rel"],
        "pred_gtpose_reproj_visible_log0p1": pred_gtpose_reproj["visible_ratio_log0p1"],
        "pred_predpose_reproj_overlap": pred_predpose_reproj["overlap_ratio"],
        "pred_predpose_reproj_abs_log": pred_predpose_reproj["abs_log"],
        "pred_predpose_reproj_abs_rel": pred_predpose_reproj["abs_rel"],
        "pred_predpose_reproj_visible_log0p1": pred_predpose_reproj["visible_ratio_log0p1"],
        "visualization": str(panel_path.resolve()),
        "metrics_json": str(metrics_path.resolve()),
    }


def read_explicit_group(dataset: PanoMinimalDataset, indices: list[int]) -> dict[str, Any]:
    samples = [dataset._read_item(dataset.items[int(index)]) for index in indices]
    return {
        "pano_image": torch.stack([sample["pano_image"] for sample in samples], dim=0),
        "pano_depth": torch.stack([sample["pano_depth"] for sample in samples], dim=0),
        "sequence_name": [sample["sequence_name"] for sample in samples],
        "scene_name": "|".join(sample["scene_name"] for sample in samples),
        "rgb_path": [sample["rgb_path"] for sample in samples],
        "depth_path": [sample["depth_path"] for sample in samples],
        "pano_position_m": torch.stack([sample["pano_position_m"] for sample in samples], dim=0),
        "pano_position_valid": torch.stack([sample["pano_position_valid"] for sample in samples], dim=0),
        "pano_rotation_c2w": torch.stack([sample["pano_rotation_c2w"] for sample in samples], dim=0),
        "pano_rotation_valid": torch.stack([sample["pano_rotation_valid"] for sample in samples], dim=0),
        "sample_weight": torch.stack([sample["sample_weight"] for sample in samples], dim=0),
        "metadata_valid_ratio": torch.stack([sample["metadata_valid_ratio"] for sample in samples], dim=0),
        "metadata_structure_score": torch.stack([sample["metadata_structure_score"] for sample in samples], dim=0),
        "metadata_quality_bin": [sample["metadata_quality_bin"] for sample in samples],
    }


def slice_camera_meta(meta: dict[str, torch.Tensor], view_slice: slice) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in ("yaw", "pitch", "fov_x", "fov_y", "rotations"):
        value = meta[key]
        if value.ndim >= 2:
            value = value[0, view_slice]
        else:
            value = value[view_slice]
        result[key] = value.detach().float().cpu()
    return result


def reproject_window_depth_to_target_erp_official(
    depth_z: torch.Tensor,
    camera_meta: dict[str, torch.Tensor],
    target_center: torch.Tensor,
    target_quat_w2c: torch.Tensor,
    target_hw: tuple[int, int],
    max_range_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(depth_z.shape[-2]), int(depth_z.shape[-1])
    yaw = camera_meta["yaw"].reshape(-1)
    pitch = camera_meta["pitch"].reshape(-1)
    fov_x = camera_meta["fov_x"].reshape(-1)
    fov_y = camera_meta["fov_y"].reshape(-1)
    rays = pinhole_rays(yaw, pitch, fov_x, fov_y, height, width, device=depth_z.device, dtype=depth_z.dtype)
    forward = camera_meta["rotations"][..., :, 2].reshape(-1, 3)
    z_factor = (rays * forward[:, None, None, :]).sum(dim=-1).clamp_min(1e-6)
    radial = depth_z / z_factor
    valid = torch.isfinite(depth_z) & (depth_z > 0) & torch.isfinite(radial) & (radial > 0)
    if max_range_depth > 0:
        valid &= radial <= max_range_depth
    points_source_omega = rays * radial[..., None]
    points_source_official = omega_y_up_vectors_to_official_y_down(points_source_omega)
    return splat_points_to_target_erp_official(
        points_source_official,
        valid,
        target_center,
        target_quat_w2c,
        target_hw,
        max_range_depth,
    )


def reproject_source_erp_depth_to_target_erp_official(
    source_range_depth: torch.Tensor,
    target_center: torch.Tensor,
    target_quat_w2c: torch.Tensor,
    target_hw: tuple[int, int],
    max_range_depth: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth = source_range_depth[::stride, ::stride].contiguous()
    rays = erp_rays_official(depth.shape[0], depth.shape[1], device=depth.device, dtype=depth.dtype)
    valid = torch.isfinite(depth) & (depth > 0)
    if max_range_depth > 0:
        valid &= depth <= max_range_depth
    points_source = rays * depth[..., None]
    return splat_points_to_target_erp_official(
        points_source,
        valid,
        target_center,
        target_quat_w2c,
        target_hw,
        max_range_depth,
    )


def erp_rays_official(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ERP rays in the official OpenCV pano basis [right, down, forward]."""
    y = torch.arange(height, device=device, dtype=dtype) / float(max(height - 1, 1))
    x = torch.arange(width, device=device, dtype=dtype) / float(max(width - 1, 1))
    vv, uu = torch.meshgrid(y, x, indexing="ij")
    theta = (uu - 0.5) * (2.0 * math.pi)
    phi = (vv - 0.5) * math.pi
    cos_phi = torch.cos(phi)
    return torch.stack([torch.sin(theta) * cos_phi, torch.sin(phi), torch.cos(theta) * cos_phi], dim=-1)


def splat_points_to_target_erp_official(
    points_source: torch.Tensor,
    valid: torch.Tensor,
    target_center: torch.Tensor,
    target_quat_w2c: torch.Tensor,
    target_hw: tuple[int, int],
    max_range_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    pano_h, pano_w = target_hw
    points = points_source[valid]
    if points.numel() == 0:
        return np.zeros((pano_h, pano_w), dtype=np.float32), np.zeros((pano_h, pano_w), dtype=bool)
    rotation_w2c = quat_to_mat(F.normalize(target_quat_w2c.reshape(1, 4), dim=-1))[0].to(points.dtype)
    center = target_center.to(dtype=points.dtype)
    target_points = (rotation_w2c @ (points - center).T).T
    target_range = torch.linalg.vector_norm(target_points, dim=-1)
    keep = torch.isfinite(target_range) & (target_range > 1e-6)
    if max_range_depth > 0:
        keep &= target_range <= max_range_depth
    target_points = target_points[keep]
    target_range = target_range[keep]
    if target_points.numel() == 0:
        return np.zeros((pano_h, pano_w), dtype=np.float32), np.zeros((pano_h, pano_w), dtype=bool)
    dirs = F.normalize(target_points, dim=-1)
    theta = torch.atan2(dirs[:, 0], dirs[:, 2])
    phi = torch.asin(dirs[:, 1].clamp(-1.0, 1.0))
    u = torch.remainder(theta / (2.0 * math.pi) + 0.5, 1.0)
    v = (0.5 + phi / math.pi).clamp(0.0, 1.0)
    x = torch.clamp(torch.round(u * (pano_w - 1)).long(), 0, pano_w - 1).cpu().numpy()
    y = torch.clamp(torch.round(v * (pano_h - 1)).long(), 0, pano_h - 1).cpu().numpy()
    z = target_range.detach().cpu().numpy().astype(np.float32)
    depth = np.full((pano_h, pano_w), np.inf, dtype=np.float32)
    np.minimum.at(depth, (y, x), z)
    valid_out = np.isfinite(depth)
    depth[~valid_out] = 0.0
    return depth, valid_out


def reprojection_metrics(
    projected: np.ndarray,
    projected_valid: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, float]:
    mask = projected_valid & target_valid & np.isfinite(projected) & np.isfinite(target_depth) & (projected > 0) & (target_depth > 0)
    target_count = int(target_valid.sum())
    if not mask.any():
        return {
            "overlap_pixels": 0,
            "target_valid_pixels": target_count,
            "overlap_ratio": 0.0,
            "abs_log": float("nan"),
            "abs_rel": float("nan"),
            "visible_ratio_log0p1": 0.0,
            "visible_ratio_log0p2": 0.0,
        }
    log_error = np.abs(np.log(np.maximum(projected[mask], 1e-6)) - np.log(np.maximum(target_depth[mask], 1e-6)))
    abs_rel = np.abs(projected[mask] - target_depth[mask]) / np.maximum(target_depth[mask], 1e-6)
    return {
        "overlap_pixels": int(mask.sum()),
        "target_valid_pixels": target_count,
        "overlap_ratio": float(mask.sum() / max(target_count, 1)),
        "abs_log": float(np.mean(log_error)),
        "abs_rel": float(np.mean(abs_rel)),
        "visible_ratio_log0p1": float(np.mean(log_error < 0.1)),
        "visible_ratio_log0p2": float(np.mean(log_error < 0.2)),
    }


def make_reprojection_panel(
    path: Path,
    sample: dict[str, Any],
    gt_projected: np.ndarray,
    gt_projected_valid: np.ndarray,
    pred_source_projected: np.ndarray,
    pred_source_projected_valid: np.ndarray,
    pred_gtpose_projected: np.ndarray,
    pred_gtpose_projected_valid: np.ndarray,
    pred_predpose_projected: np.ndarray,
    pred_predpose_projected_valid: np.ndarray,
    source_depth_target: np.ndarray,
    source_valid: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    gt_metrics: dict[str, float],
    pred_source_metrics: dict[str, float],
    pred_gtpose_metrics: dict[str, float],
    pred_predpose_metrics: dict[str, float],
    case: dict[str, Any],
) -> None:
    source_rgb = rgb_tile(sample["pano_image"][0], size=(256, 128))
    target_rgb = rgb_tile(sample["pano_image"][1], size=(256, 128))
    source_depth = depth_tile(sample["pano_depth"][0, 0].detach().cpu().numpy(), size=(256, 128))
    target_depth_tile = depth_tile(target_depth, size=(256, 128))
    gt_tile = depth_tile(gt_projected, gt_projected_valid, size=(256, 128))
    pred_source_tile = depth_tile(pred_source_projected, pred_source_projected_valid, size=(256, 128))
    pred_gtpose_tile = depth_tile(pred_gtpose_projected, pred_gtpose_projected_valid, size=(256, 128))
    pred_predpose_tile = depth_tile(pred_predpose_projected, pred_predpose_projected_valid, size=(256, 128))
    gt_err = error_tile(gt_projected, target_depth, gt_projected_valid & target_valid, size=(256, 128))
    pred_source_err = error_tile(
        pred_source_projected,
        source_depth_target,
        pred_source_projected_valid & source_valid,
        size=(256, 128),
    )
    pred_gtpose_err = error_tile(
        pred_gtpose_projected,
        target_depth,
        pred_gtpose_projected_valid & target_valid,
        size=(256, 128),
    )
    pred_predpose_err = error_tile(
        pred_predpose_projected,
        target_depth,
        pred_predpose_projected_valid & target_valid,
        size=(256, 128),
    )
    gt_mask = mask_tile(gt_projected_valid & target_valid, size=(256, 128))
    pred_source_mask = mask_tile(pred_source_projected_valid & source_valid, size=(256, 128))
    pred_gtpose_mask = mask_tile(pred_gtpose_projected_valid & target_valid, size=(256, 128))
    pred_predpose_mask = mask_tile(pred_predpose_projected_valid & target_valid, size=(256, 128))

    rows = [
        tile_row_with_titles(
            [
                ("Source ERP RGB", source_rgb),
                ("Target ERP RGB", target_rgb),
                ("Source GT range depth", source_depth),
                ("Target GT range depth", target_depth_tile),
            ]
        ),
        tile_row_with_titles(
            [
                ("GT depth + official GT pose", gt_tile),
                (f"GT abs-log error ({gt_metrics['abs_log']:.3f})", gt_err),
                ("GT overlap mask", gt_mask),
                ("", blank_tile()),
            ]
        ),
        tile_row_with_titles(
            [
                ("Omega depth in official source ERP", pred_source_tile),
                (f"Source abs-log error ({pred_source_metrics['abs_log']:.3f})", pred_source_err),
                ("Source coverage mask", pred_source_mask),
                ("", blank_tile()),
            ]
        ),
        tile_row_with_titles(
            [
                ("Omega depth + official GT pose", pred_gtpose_tile),
                (f"GT-pose abs-log error ({pred_gtpose_metrics['abs_log']:.3f})", pred_gtpose_err),
                ("GT-pose overlap mask", pred_gtpose_mask),
                ("", blank_tile()),
            ]
        ),
        tile_row_with_titles(
            [
                ("Omega depth + officialized pred pose", pred_predpose_tile),
                (f"Pred-pose abs-log error ({pred_predpose_metrics['abs_log']:.3f})", pred_predpose_err),
                ("Pred-pose overlap mask", pred_predpose_mask),
                ("", blank_tile()),
            ]
        ),
    ]
    header = np.full((38, rows[0].shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        header,
        f"{case['case_label']} | pair_dist={case['gt_pair_distance_m']:.3f}m",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    row_gap = np.full((10, rows[0].shape[1], 3), 42, dtype=np.uint8)
    stacked = [header]
    for row in rows:
        stacked.extend([row_gap, row])
    cv2.imwrite(str(path), np.concatenate(stacked, axis=0))


def depth_tile(depth: np.ndarray, valid: np.ndarray | None = None, size: tuple[int, int] = (256, 128)) -> np.ndarray:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(depth) & (depth > 0) if valid is None else (valid & np.isfinite(depth) & (depth > 0))
    output = np.zeros(depth.shape, dtype=np.uint8)
    if mask.any():
        maximum = float(np.nanpercentile(depth[mask], 98))
        output[mask] = np.clip(np.log1p(depth[mask]) / np.log1p(max(maximum, 1e-3)) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(output, cv2.COLORMAP_TURBO)
    colored[~mask] = (18, 18, 18)
    return cv2.resize(colored, size, interpolation=cv2.INTER_AREA)


def error_tile(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, size: tuple[int, int] = (256, 128)) -> np.ndarray:
    mask = valid & np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    error = np.zeros(pred.shape, dtype=np.uint8)
    if mask.any():
        log_error = np.abs(np.log(np.maximum(pred[mask], 1e-6)) - np.log(np.maximum(target[mask], 1e-6)))
        error[mask] = np.clip(log_error / 0.7 * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(error, cv2.COLORMAP_INFERNO)
    colored[~mask] = (18, 18, 18)
    return cv2.resize(colored, size, interpolation=cv2.INTER_AREA)


def mask_tile(mask: np.ndarray, size: tuple[int, int] = (256, 128)) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = (80, 220, 80)
    image[~mask] = (18, 18, 18)
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def blank_tile(size: tuple[int, int] = (256, 128)) -> np.ndarray:
    return np.full((size[1], size[0], 3), 18, dtype=np.uint8)


def tile_row_with_titles(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    tiles = []
    for title, tile in items:
        header = np.full((26, tile.shape[1], 3), 16, dtype=np.uint8)
        if title:
            cv2.putText(header, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([header, tile], axis=0))
    gap = np.full((tiles[0].shape[0], 8, 3), 42, dtype=np.uint8)
    row = [tiles[0]]
    for tile in tiles[1:]:
        row.extend([gap, tile])
    return np.concatenate(row, axis=1)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "cases": len(selected),
            "training_target_center_max_abs_delta": finite_mean(
                [row["training_target_center_max_abs_delta"] for row in selected]
            ),
            "training_target_rotation_max_delta_deg": finite_mean(
                [row["training_target_rotation_max_delta_deg"] for row in selected]
            ),
            "translation_init_raw_basis_l2_mean_m": finite_mean(
                [row["translation_init_raw_basis_l2_mean_m"] for row in selected]
            ),
            "translation_raw_basis_l2_mean_m": finite_mean(
                [row["translation_raw_basis_l2_mean_m"] for row in selected]
            ),
            "translation_init_l2_mean_m": finite_mean([row["translation_init_l2_mean_m"] for row in selected]),
            "translation_l2_mean_m": finite_mean([row["translation_l2_mean_m"] for row in selected]),
            "translation_depthscaled_l2_mean_m": finite_mean(
                [row["translation_depthscaled_l2_mean_m"] for row in selected]
            ),
            "translation_depthscaled_norm_l2_mean": finite_mean(
                [row["translation_depthscaled_norm_l2_mean"] for row in selected]
            ),
            "rotation_init_deg_mean": finite_mean([row["rotation_init_deg_mean"] for row in selected if row["rotation_init_deg_mean"] is not None]),
            "rotation_deg_mean": finite_mean([row["rotation_deg_mean"] for row in selected if row["rotation_deg_mean"] is not None]),
            "gt_reproj_abs_log": finite_mean([row["gt_reproj_abs_log"] for row in selected]),
            "pred_source_reproj_abs_log": finite_mean([row["pred_source_reproj_abs_log"] for row in selected]),
            "pred_gtpose_reproj_abs_log": finite_mean([row["pred_gtpose_reproj_abs_log"] for row in selected]),
            "pred_predpose_reproj_abs_log": finite_mean([row["pred_predpose_reproj_abs_log"] for row in selected]),
            "gt_reproj_overlap": finite_mean([row["gt_reproj_overlap"] for row in selected]),
            "pred_source_reproj_overlap": finite_mean([row["pred_source_reproj_overlap"] for row in selected]),
            "pred_gtpose_reproj_overlap": finite_mean([row["pred_gtpose_reproj_overlap"] for row in selected]),
            "pred_predpose_reproj_overlap": finite_mean([row["pred_predpose_reproj_overlap"] for row in selected]),
        }
    return {"datasets": by_dataset, "cases": rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "case_id",
        "case_label",
        "scene_name",
        "pano_count",
        "gt_pair_distance_m",
        "training_target_center_max_abs_delta",
        "training_target_rotation_max_delta_deg",
        "translation_init_raw_basis_l2_mean_m",
        "rotation_init_raw_basis_deg_mean",
        "translation_raw_basis_l2_mean_m",
        "rotation_raw_basis_deg_mean",
        "translation_init_l2_mean_m",
        "rotation_init_deg_mean",
        "translation_l2_mean_m",
        "translation_l2_median_m",
        "translation_norm_l2_mean",
        "translation_depthscaled_l2_mean_m",
        "translation_depthscaled_l2_median_m",
        "translation_depthscaled_norm_l2_mean",
        "rotation_deg_mean",
        "rotation_deg_median",
        "window_depth_abs_rel",
        "window_depth_delta_1p25",
        "depth_scale",
        "gt_reproj_overlap",
        "gt_reproj_abs_log",
        "gt_reproj_abs_rel",
        "gt_reproj_visible_log0p1",
        "pred_source_reproj_overlap",
        "pred_source_reproj_abs_log",
        "pred_source_reproj_abs_rel",
        "pred_source_reproj_visible_log0p1",
        "pred_gtpose_reproj_overlap",
        "pred_gtpose_reproj_abs_log",
        "pred_gtpose_reproj_abs_rel",
        "pred_gtpose_reproj_visible_log0p1",
        "pred_predpose_reproj_overlap",
        "pred_predpose_reproj_abs_log",
        "pred_predpose_reproj_abs_rel",
        "pred_predpose_reproj_visible_log0p1",
        "visualization",
        "metrics_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    main()
