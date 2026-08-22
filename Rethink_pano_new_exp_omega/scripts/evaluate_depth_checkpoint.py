#!/usr/bin/env python3
"""Validate a VGGT-Omega/LUNA depth checkpoint on held-out pano samples."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GIT_ROOT) not in sys.path:
    sys.path.insert(0, str(GIT_ROOT))

from evaluation_common.erp_depth import splat_window_z_depth_to_erp  # noqa: E402

from training.train_pano_omega import (  # noqa: E402
    DepthPredictionAdapter,
    adjacent_edge_overlap_loss,
    build_dataset,
    build_model,
    build_relative_pano_pose_targets,
    camera_alignment_loss,
    estimate_sample_depth_alignment_scale,
    expand_sample_scale_like,
    load_checkpoint,
    masked_depth_loss,
    move_batch_to_device,
    normalize_camera_supervision_args,
    normalize_dense_head_frames_chunk_size,
    normalize_pano_sampling_args,
    normalize_pred_depth_scale_args,
    normalize_training_stages,
    parse_dataset_depth_scales,
    parse_args as parse_training_args,
    erp_depth_to_range_depth,
    resolve_device,
    sample_depth_targets,
    set_seed,
    shared_frame_point_loss,
    omega_y_up_pose_to_official_y_down,
    unwrap_model,
)
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays, yaw_pitch_to_axes  # noqa: E402
from vggt_omega.data.pano_sampler import resolve_fov_degrees  # noqa: E402


DEPTH_METRIC_PROTOCOL = "dual_covered_sphere_and_erp_stable_polar_prior_v2"
_SPHERICAL_WEIGHT_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}

# Median of per-panorama polar-cap medians from 100 deterministic samples per
# dataset (seed 20260728, |latitude| >= 75 degrees). A missing value means the
# cap was invalid/Inf or its median was not stable enough across panoramas.
ERP_POLAR_PRIOR_SOURCE = "mixed4_polar_depth_100_seed20260728"
ERP_POLAR_PRIOR_LATITUDE_DEG = 75.0
ERP_POLAR_DEPTH_PRIORS_M = {
    "Panocity": {"north": None, "south": 10.039999961853027},
    "Matterport3D": {"north": None, "south": None},
    "Stanford2D3DS": {"north": None, "south": 1.380859375},
    "Structured3D": {"north": None, "south": None},
}


def canonical_eval_dataset_name(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "panocity": "Panocity",
        "matterport3d": "Matterport3D",
        "stanford2d3ds": "Stanford2D3DS",
        "structured3d": "Structured3D",
    }
    return aliases.get(normalized, str(value or "unknown"))


def erp_polar_prior_metadata() -> dict[str, Any]:
    return {
        "source": ERP_POLAR_PRIOR_SOURCE,
        "sample_count_per_dataset": 100,
        "latitude_threshold_degrees": ERP_POLAR_PRIOR_LATITUDE_DEG,
        "values_m": ERP_POLAR_DEPTH_PRIORS_M,
        "selection_rule": (
            "cap valid fraction >= 0.95 and cross-panorama standard deviation of cap medians "
            "/ mean <= 0.15; otherwise the prior is null"
        ),
        "scale_fit_domain": "valid window-covered ERP pixels only",
        "fill_domain": "uncovered GT-valid polar pixels with a non-null dataset/cap prior",
        "pixel_weighting": "uniform ERP pixels",
    }


DEPTH_METRIC_KEYS = [
    "depth_mae",
    "depth_rmse",
    "depth_abs_rel",
    "depth_delta_1p25",
    "depth_delta_1p25_2",
    "depth_delta_1p25_3",
    "depth_irls_scale",
    "depth_irls_mae",
    "depth_irls_rmse",
    "depth_irls_abs_rel",
    "depth_irls_delta_1p25",
    "depth_irls_delta_1p25_2",
    "depth_irls_delta_1p25_3",
    "depth_valid_pixels",
]

DEPTH_ACCUMULATOR_KEYS = [
    "depth_abs_error_sum",
    "depth_sq_error_sum",
    "depth_abs_rel_sum",
    "depth_delta_1p25_count",
    "depth_delta_1p25_2_count",
    "depth_delta_1p25_3_count",
    "depth_irls_abs_error_sum",
    "depth_irls_sq_error_sum",
    "depth_irls_abs_rel_sum",
    "depth_irls_delta_1p25_count",
    "depth_irls_delta_1p25_2_count",
    "depth_irls_delta_1p25_3_count",
]

ERP_PRIOR_DEPTH_METRIC_KEYS = [f"erp_prior_{key}" for key in DEPTH_METRIC_KEYS]
ERP_PRIOR_DEPTH_ACCUMULATOR_KEYS = [f"erp_prior_{key}" for key in DEPTH_ACCUMULATOR_KEYS]
ERP_PRIOR_COVERAGE_KEYS = [
    "erp_window_coverage_fraction",
    "erp_prior_fill_fraction",
    "erp_evaluated_gt_fraction",
]
ERP_PRIOR_DIAGNOSTIC_KEYS = [
    "erp_window_only_depth_irls_abs_rel",
    "erp_window_only_depth_irls_delta_1p25",
    "erp_no_prior_depth_irls_abs_rel",
    "erp_no_prior_depth_irls_delta_1p25",
    "erp_prior_region_abs_rel",
    "erp_prior_north_abs_rel",
    "erp_prior_south_abs_rel",
]

PANOVGGT_PRIMARY_METRICS = [
    "depth_irls_abs_rel",
    "depth_irls_delta_1p25",
    "depth_irls_rmse",
    "depth_abs_rel",
    "depth_delta_1p25",
    "depth_rmse",
]

CAMERA_POSE_SAMPLE_KEYS = [
    "camera_pose_pair_count",
    "camera_pose_auc3",
    "camera_pose_auc5",
    "camera_pose_auc15",
    "camera_pose_auc30",
    "camera_pose_rotation_deg_mean",
    "camera_pose_rotation_deg_median",
    "camera_pose_translation_deg_mean",
    "camera_pose_translation_deg_median",
    "camera_pose_max_error_deg_mean",
    "camera_pose_max_error_deg_median",
    "camera_pose_gt_translation_norm_mean",
    "camera_pose_gt_translation_norm_median",
]

PER_SAMPLE_CSV_FIELDS = [
    "dataset",
    "split",
    "run",
    "dataset_index",
    "seq_name",
    "rgb_path",
    "depth_path",
    "quality_bin",
    "input_pano_count",
    "camera_eval_pano_count",
    "loss",
    "loss_depth",
    "loss_overlap",
    "loss_global_point",
    "global_point_valid_ratio",
    "loss_camera",
    "loss_camera_t",
    "loss_camera_r",
    "camera_translation_deg",
    "camera_rotation_deg",
    "camera_translation_valid_count",
    "camera_rotation_valid_count",
    *CAMERA_POSE_SAMPLE_KEYS,
    "valid_fraction",
    "pred_depth_scale",
    "metadata_valid_ratio",
    "metadata_structure_score",
    "sample_weight",
    "depth_metric_protocol",
    *DEPTH_METRIC_KEYS,
    *DEPTH_ACCUMULATOR_KEYS,
    *ERP_PRIOR_DEPTH_METRIC_KEYS,
    *ERP_PRIOR_DEPTH_ACCUMULATOR_KEYS,
    *ERP_PRIOR_COVERAGE_KEYS,
    *ERP_PRIOR_DIAGNOSTIC_KEYS,
]

CAMERA_PAIR_CSV_FIELDS = [
    "dataset",
    "split",
    "run",
    "dataset_index",
    "seq_name",
    "rgb_path",
    "depth_path",
    "batch_index",
    "pair_i",
    "pair_j",
    "camera_pose_rotation_deg",
    "camera_pose_translation_deg",
    "camera_pose_max_error_deg",
    "camera_pose_gt_translation_norm",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config used to build model/dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to validate.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument("--per-sample-csv", type=Path, default=None, help="Optional per-sample CSV path.")
    parser.add_argument("--train-loss-csv", type=Path, default=None, help="Optional training loss.csv for comparison.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--curriculum-bins", default="all", help="Bins for the main validation run. Use all/clean,normal/hard.")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples for the main validation run.")
    parser.add_argument("--limit-fraction", type=float, default=0.0, help="Fraction of scene groups/samples to evaluate when --limit <= 0.")
    parser.add_argument("--eval-max-panos", type=int, default=0, help="Clamp eval multi-pano input length. Use 0 to keep config pano_max_count.")
    parser.add_argument("--window-size", type=int, default=0, help="Override square window resolution; 0 keeps checkpoint/config.")
    parser.add_argument("--num-yaw", type=int, default=0, help="Override yaw windows per pano; 0 keeps checkpoint/config.")
    parser.add_argument("--pitch-degrees", type=str, default=None, help="Override checkpoint pitch list for a fixed evaluation domain.")
    parser.add_argument("--fov-degrees", type=float, default=None, help="Override legacy scalar FoV for evaluation.")
    parser.add_argument("--fov-x-degrees", type=float, default=None, help="Override horizontal FoV for evaluation.")
    parser.add_argument("--fov-y-degrees", type=float, default=None, help="Override vertical FoV for evaluation.")
    parser.add_argument("--hard-limit", type=int, default=100, help="Extra hard-bin val samples. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic sample seed.")
    parser.add_argument(
        "--sample-policy",
        choices=["scene_neighborhood", "anchor"],
        default="scene_neighborhood",
        help=(
            "How to choose held-out multi-pano samples. scene_neighborhood selects one "
            "nearest-neighborhood group per scene/room/trajectory, matching PanoVGGT's "
            "sequence-level evaluation. anchor preserves the older random-anchor behavior."
        ),
    )
    parser.add_argument("--camera-pair-csv", type=Path, default=None, help="Optional streaming PanoVGGT-style camera pair CSV path.")
    parser.add_argument("--camera-pose-trans-norm-thresh", type=float, default=1e-2, help="GT baseline threshold for PanoVGGT-style camera translation-angle eval.")
    parser.add_argument("--camera-eval-max-panos", type=int, default=3, help="Maximum pano views used for PanoVGGT Table-2 camera metrics.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default=None)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--print-each-sample", dest="print_each_sample", action="store_true", default=True)
    parser.add_argument("--no-print-each-sample", dest="print_each_sample", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device, {"distributed": False, "local_rank": 0})

    train_args = parse_training_args(["--config", str(args.config)])
    train_args.device = args.device
    train_args.distributed = "none"
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers
    if args.amp_dtype is not None:
        train_args.amp_dtype = args.amp_dtype
    train_args.checkpoint = args.checkpoint
    normalize_args_for_eval(train_args)

    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    apply_checkpoint_eval_defaults(train_args, checkpoint_payload)
    apply_eval_sampler_overrides(train_args, args)
    apply_eval_max_panos(train_args, args.eval_max_panos)
    model = build_eval_model(train_args, args.checkpoint, checkpoint_payload, device)
    model.eval()

    camera_pair_csv = args.camera_pair_csv
    if camera_pair_csv is None and args.per_sample_csv is not None:
        camera_pair_csv = args.per_sample_csv.with_name(f"{args.per_sample_csv.stem}_camera_pairs.csv")
    if args.per_sample_csv is not None:
        initialize_csv(args.per_sample_csv, PER_SAMPLE_CSV_FIELDS)
    if camera_pair_csv is not None:
        initialize_csv(camera_pair_csv, CAMERA_PAIR_CSV_FIELDS)

    runs: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    camera_pair_rows: list[dict[str, Any]] = []
    runs.append(
        evaluate_run(
            name=f"{args.split}_{normalize_bins_label(args.curriculum_bins)}_{args.limit}",
            base_args=train_args,
            model=model,
            device=device,
            split=args.split,
            curriculum_bins=args.curriculum_bins,
            limit=args.limit,
            limit_fraction=args.limit_fraction,
            seed=args.seed,
            sample_policy=args.sample_policy,
            num_workers=args.num_workers,
            progress=args.progress,
            per_sample_rows=per_sample_rows,
            per_sample_csv=args.per_sample_csv,
            camera_pair_rows=camera_pair_rows,
            camera_pair_csv=camera_pair_csv,
            row_context={"split": args.split},
            camera_pose_trans_norm_thresh=args.camera_pose_trans_norm_thresh,
            camera_eval_max_panos=args.camera_eval_max_panos,
            print_each_sample=args.print_each_sample,
        )
    )
    if args.hard_limit > 0:
        runs.append(
            evaluate_run(
                name=f"{args.split}_hard_{args.hard_limit}",
                base_args=train_args,
                model=model,
                device=device,
                split=args.split,
                curriculum_bins="hard",
                limit=args.hard_limit,
                limit_fraction=0.0,
                seed=args.seed + 17,
                sample_policy=args.sample_policy,
                num_workers=args.num_workers,
                progress=args.progress,
                per_sample_rows=per_sample_rows,
                per_sample_csv=args.per_sample_csv,
                camera_pair_rows=camera_pair_rows,
                camera_pair_csv=camera_pair_csv,
                row_context={"split": args.split},
                camera_pose_trans_norm_thresh=args.camera_pose_trans_norm_thresh,
                print_each_sample=args.print_each_sample,
            )
        )

    fov_x_degrees, fov_y_degrees = resolve_fov_degrees(
        train_args.fov_degrees,
        getattr(train_args, "fov_x_degrees", None),
        getattr(train_args, "fov_y_degrees", None),
    )
    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "dataset_root": str(train_args.dataset_root),
        "depth_evaluation_domain": "covered_sphere_sampled_pinhole_windows",
        "depth_evaluation_domains": {
            "covered_sphere": "sampled pinhole windows with solid-angle weights and overlap de-duplication",
            "erp_with_polar_prior": "uniform ERP pixels after window splat and stable-cap prior completion",
        },
        "depth_metric_protocol": DEPTH_METRIC_PROTOCOL,
        "depth_pixel_weighting": "pinhole_solid_angle_divided_by_same_pano_window_coverage_count",
        "erp_polar_prior": erp_polar_prior_metadata(),
        "sampler": {
            "window_size": int(train_args.window_size),
            "patch_size": int(train_args.patch_size),
            "num_yaw": int(train_args.num_yaw),
            "pano_min_count": int(getattr(train_args, "pano_min_count", 1)),
            "pano_max_count": int(getattr(train_args, "pano_max_count", 1)),
            "pitch_degrees": str(train_args.pitch_degrees),
            "fov_degrees": float(train_args.fov_degrees),
            "fov_x_degrees": fov_x_degrees,
            "fov_y_degrees": fov_y_degrees,
        },
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv),
        "runs": runs,
        "panovggt_camera_pose_summary": summarize_camera_pose_pairs(camera_pair_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_sample_csv is not None:
        write_per_sample_csv(args.per_sample_csv, per_sample_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def normalize_args_for_eval(args: argparse.Namespace) -> None:
    normalize_pano_sampling_args(args)
    normalize_camera_supervision_args(args)
    normalize_pred_depth_scale_args(args)
    # Evaluation keeps anchor/order deterministic. Order robustness is measured
    # explicitly instead of changing the sample beneath streaming/resume keys.
    args.randomize_pano_order = False
    args.training_stages = normalize_training_stages(args.training_stages)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {}


def apply_checkpoint_eval_defaults(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    ckpt_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    if payload.get("pred_depth_scale") is not None:
        args.pred_depth_scale = float(payload["pred_depth_scale"])
    elif ckpt_args.get("pred_depth_scale") is not None:
        args.pred_depth_scale = float(ckpt_args["pred_depth_scale"])
    # Restore fields that determine module topology before loading a trainable-delta
    # checkpoint. Training configs can evolve after a run; rebuilding from a newer
    # config would otherwise drop learned LUNA layers or create adapter size
    # mismatches. Runtime sampling overrides are applied by the caller afterwards.
    for key in (
        "learn_pred_depth_scale",
        "dataset_depth_scale_mode",
        "dataset_depth_scales",
        "depth_residual_mode",
        "depth_residual_hidden",
        "depth_residual_max_log",
        "depth_residual_frames_chunk_size",
        "depth_residual_use_checkpoint",
        "store_depth_residual_debug",
        "luna_patch_layers",
        "luna_camera_layers",
        "enable_pano_global_token",
        "enable_pano_geometry_residual",
        "enable_camera_head",
        "dense_head_frames_chunk_size",
        "dense_head_use_checkpoint",
        "dense_head_return_confidence",
    ):
        if key in ckpt_args and ckpt_args[key] is not None:
            setattr(args, key, ckpt_args[key])
    for key in (
        "window_size",
        "patch_size",
        "num_yaw",
        "pitch_degrees",
        "fov_degrees",
        "fov_x_degrees",
        "fov_y_degrees",
    ):
        if key in ckpt_args and ckpt_args[key] is not None:
            setattr(args, key, ckpt_args[key])


def apply_eval_sampler_overrides(args: argparse.Namespace, eval_args: argparse.Namespace) -> None:
    """Apply explicit evaluation geometry after checkpoint-native defaults.

    An explicit scalar FoV defines an isotropic canonical domain unless an
    axis-specific value is also supplied. This prevents a checkpoint's wider
    horizontal FoV from leaking into ``--fov-degrees 75`` evaluations.
    """
    if int(getattr(eval_args, "window_size", 0) or 0) > 0:
        args.window_size = int(eval_args.window_size)
    if int(getattr(eval_args, "num_yaw", 0) or 0) > 0:
        args.num_yaw = int(eval_args.num_yaw)
    if getattr(eval_args, "pitch_degrees", None) is not None:
        args.pitch_degrees = str(eval_args.pitch_degrees)

    scalar_fov = getattr(eval_args, "fov_degrees", None)
    fov_x = getattr(eval_args, "fov_x_degrees", None)
    fov_y = getattr(eval_args, "fov_y_degrees", None)
    if scalar_fov is not None:
        args.fov_degrees = float(scalar_fov)
        args.fov_x_degrees = float(scalar_fov if fov_x is None else fov_x)
        args.fov_y_degrees = float(scalar_fov if fov_y is None else fov_y)
    else:
        if fov_x is not None:
            args.fov_x_degrees = float(fov_x)
        if fov_y is not None:
            args.fov_y_degrees = float(fov_y)


def apply_eval_max_panos(args: argparse.Namespace, max_panos: int | None) -> None:
    max_value = int(max_panos or 0)
    if max_value <= 0:
        return
    current_max = int(getattr(args, "pano_max_count", max_value) or max_value)
    current_min = int(getattr(args, "pano_min_count", 1) or 1)
    capped = max(1, min(current_max, max_value))
    args.pano_max_count = capped
    if current_min > capped:
        args.pano_min_count = capped


def build_eval_model(
    args: argparse.Namespace,
    checkpoint: Path,
    payload: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    base_model = build_model(args).to(device)
    load_checkpoint(base_model, checkpoint, strict=False)

    adapter_state = payload.get("adapter_state")
    dataset_depth_scales = parse_dataset_depth_scales(getattr(args, "dataset_depth_scales", None))
    dataset_depth_scale_mode = str(getattr(args, "dataset_depth_scale_mode", "none") or "none")
    needs_adapter = (
        bool(args.learn_pred_depth_scale)
        or args.depth_residual_mode != "none"
        or dataset_depth_scale_mode != "none"
        or adapter_state is not None
    )
    if not needs_adapter:
        return base_model

    model = DepthPredictionAdapter(
        base_model,
        initial_scale=float(args.pred_depth_scale),
        learn_scale=bool(args.learn_pred_depth_scale),
        dataset_scale_mode=dataset_depth_scale_mode,
        dataset_scales=dataset_depth_scales,
        residual_mode=str(args.depth_residual_mode),
        residual_hidden=int(args.depth_residual_hidden),
        residual_max_log=float(args.depth_residual_max_log),
        residual_frames_chunk_size=normalize_dense_head_frames_chunk_size(
            args.depth_residual_frames_chunk_size
        ),
        residual_use_checkpoint=bool(args.depth_residual_use_checkpoint),
        store_residual_debug=bool(args.store_depth_residual_debug),
    ).to(device)
    if adapter_state is not None:
        adapter_state = {key: value.to(device) if torch.is_tensor(value) else value for key, value in adapter_state.items()}
        missing, unexpected = model.load_state_dict(adapter_state, strict=False)
        print(f"[INFO] loaded adapter_state from {checkpoint}")
        print(f"[INFO] adapter_state missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    return model


def evaluate_run(
    name: str,
    base_args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    curriculum_bins: str | None,
    limit: int,
    limit_fraction: float,
    seed: int,
    sample_policy: str,
    num_workers: int,
    progress: bool,
    per_sample_rows: list[dict[str, Any]],
    per_sample_csv: Path | None = None,
    camera_pair_rows: list[dict[str, Any]] | None = None,
    camera_pair_csv: Path | None = None,
    row_context: dict[str, Any] | None = None,
    shard_rank: int = 0,
    num_shards: int = 1,
    progress_file: Path | None = None,
    progress_every: int = 25,
    progress_context: dict[str, Any] | None = None,
    camera_pose_trans_norm_thresh: float = 1e-2,
    camera_eval_max_panos: int = 3,
    print_each_sample: bool = True,
    exact_group_manifest: list[dict[str, Any]] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if int(num_shards) < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if int(shard_rank) < 0 or int(shard_rank) >= int(num_shards):
        raise ValueError(f"shard_rank must be in [0, {int(num_shards) - 1}], got {shard_rank}")

    eval_args = copy.copy(base_args)
    eval_args.dataset_split = split
    eval_args.curriculum_bins = None if curriculum_bins in (None, "", "all") else str(curriculum_bins)
    eval_args.dataset_max_samples = None
    pano_size = (eval_args.pano_height, eval_args.pano_width) if eval_args.pano_height > 0 and eval_args.pano_width > 0 else None
    dataset = build_dataset(eval_args, pano_size)
    if exact_group_manifest is not None:
        apply_exact_group_manifest(dataset, exact_group_manifest)
    indices, sampling_info = select_eval_indices(dataset, limit, seed, sample_policy, limit_fraction=limit_fraction)
    selected_indices = indices[int(shard_rank) :: int(num_shards)]
    selected_index_set = set(selected_indices)
    existing_rows = [
        row
        for row in per_sample_rows
        if str(row.get("run", "")) == name
        and parse_int(row.get("dataset_index"), default=-1) in selected_index_set
    ] if resume else []
    completed_indices = {
        parse_int(row.get("dataset_index"), default=-1)
        for row in existing_rows
    }
    pending_indices = [index for index in selected_indices if index not in completed_indices]
    write_eval_progress(
        progress_file,
        {
            **(progress_context or {}),
            "state": "running",
            "run": name,
            "split": split,
            "dataset_size": len(dataset),
            "candidate_samples": len(indices),
            "sample_policy": sampling_info,
            "shard_samples": len(selected_indices),
            "processed_samples": len(existing_rows),
            "resumed_samples": len(existing_rows),
            "shard_rank": int(shard_rank),
            "num_shards": int(num_shards),
            "updated_at": time.time(),
        },
    )
    subset = Subset(dataset, pending_indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    rows: list[dict[str, Any]] = list(existing_rows)
    amp_enabled = device.type == "cuda" and eval_args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if eval_args.amp_dtype == "bfloat16" else torch.float32
    iterator = tqdm(loader, desc=f"validate {name}", dynamic_ncols=True) if progress else loader
    with torch.no_grad():
        for local_index, batch in enumerate(iterator):
            moved = move_batch_to_device(batch, device)
            sampler_model = unwrap_model(model)
            target_depth, target_valid, target_camera_meta = sample_depth_targets(
                sampler_model,
                moved["pano_depth"],
                source_depth_semantics=eval_args.gt_depth_semantics,
                max_range_depth=eval_args.depth_max_m,
                return_camera_meta=True,
                source_valid_mask=moved.get("pano_rgb_depth_common_mask"),
            )
            spherical_weights = build_covered_sphere_weights(
                target_camera_meta,
                height=target_depth.shape[-3],
                width=target_depth.shape[-2],
                num_panos=int(moved["pano_image"].shape[1]),
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                predictions = model(
                    pano_images=moved["pano_image"],
                    return_sampler_output=True,
                    return_window_pose=eval_args.camera_supervision_mode != "pano_relative",
                )
                pred_depth_scale = predictions.get(
                    "_pred_depth_scale",
                    predictions["depth"].new_tensor(float(eval_args.pred_depth_scale)),
                )
                base_depth_scale = (
                    pred_depth_scale.detach()
                    if eval_args.depth_scale_alignment != "none"
                    else pred_depth_scale
                )
                pred_depth_base = predictions["depth"] * base_depth_scale
                sample_depth_scale = estimate_sample_depth_alignment_scale(
                    pred_depth_base,
                    target_depth,
                    target_valid,
                    mode=eval_args.depth_scale_alignment,
                    min_scale=eval_args.depth_scale_alignment_min,
                    max_scale=eval_args.depth_scale_alignment_max,
                    eps=eval_args.depth_scale_alignment_eps,
                )
                pred_depth = pred_depth_base * expand_sample_scale_like(
                    sample_depth_scale,
                    pred_depth_base,
                )
                loss_depth = masked_depth_loss(
                    pred_depth,
                    target_depth,
                    target_valid,
                    mode=eval_args.depth_loss_mode,
                    huber_delta=eval_args.depth_log_huber_delta,
                    error_clip=eval_args.depth_log_error_clip,
                    sample_weight=moved.get("sample_weight") if eval_args.loss_sample_weighting else None,
                    min_window_valid_ratio=eval_args.min_window_valid_ratio,
                    valid_ratio_power=eval_args.valid_ratio_loss_power,
                    sample_weight_min=eval_args.sample_weight_min,
                    sample_weight_max=eval_args.sample_weight_max,
                )
                loss_overlap = adjacent_edge_overlap_loss(
                    pred_depth,
                    target_valid,
                    sample_weight=moved.get("sample_weight") if eval_args.loss_sample_weighting else None,
                    band_fraction=eval_args.overlap_band_fraction,
                )
                depth_metrics = compute_depth_metrics(
                    pred_depth_base,
                    target_depth,
                    target_valid,
                    spherical_weights=spherical_weights,
                )
                erp_prior_metrics = compute_erp_prior_depth_metrics(
                    pred_window_z=pred_depth_base,
                    gt_erp_depth=moved["pano_depth"],
                    source_depth_semantics=eval_args.gt_depth_semantics,
                    max_range_depth=eval_args.depth_max_m,
                    camera_meta=target_camera_meta,
                    dataset_name=canonical_eval_dataset_name(
                        (row_context or {}).get("dataset")
                        or getattr(eval_args, "minimal_datasets", "unknown")
                    ),
                    align_corners=False,
                )
                camera_scale = base_depth_scale.detach() * sample_depth_scale
                with torch.autocast(device_type=device.type, enabled=False):
                    camera_losses = camera_alignment_loss(
                        predictions=predictions,
                        batch=moved,
                        translation_weight=eval_args.camera_translation_weight,
                        rotation_weight=eval_args.camera_rotation_weight,
                        fov_weight=eval_args.camera_fov_weight,
                        position_mode=eval_args.camera_position_mode,
                        supervision_mode=eval_args.camera_supervision_mode,
                        pano_consistency_weight=eval_args.pano_translation_consistency_weight,
                        translation_normalization=eval_args.camera_translation_normalization,
                        translation_normalization_eps=eval_args.camera_translation_normalization_eps,
                        pred_translation_scale=(
                            (
                                camera_scale.float()
                                if eval_args.camera_depth_scale_alignment
                                and eval_args.depth_scale_alignment != "none"
                                else None
                            )
                        ),
                    )
                    pose_metrics, pose_pair_rows = compute_panovggt_camera_pose_metrics(
                        predictions=predictions,
                        batch=moved,
                        position_mode=eval_args.camera_position_mode,
                        pred_translation_scale=(
                            (
                                camera_scale.float()
                                if eval_args.camera_depth_scale_alignment
                                and eval_args.depth_scale_alignment != "none"
                                else None
                            )
                        ),
                        trans_norm_thresh=float(camera_pose_trans_norm_thresh),
                        max_panos=int(camera_eval_max_panos),
                    )
                if float(eval_args.global_point_loss_weight) > 0:
                    global_point_metrics = shared_frame_point_loss(
                        pred_depth=pred_depth,
                        target_depth=target_depth,
                        valid_mask=target_valid,
                        predictions=predictions,
                        batch=moved,
                        pred_translation_scale=(
                            camera_scale
                            if eval_args.camera_depth_scale_alignment
                            and eval_args.depth_scale_alignment != "none"
                            else None
                        ),
                        stride=eval_args.global_point_stride,
                    )
                else:
                    global_point_metrics = {
                        "loss_global_point": loss_depth.new_zeros(()),
                        "global_point_valid_ratio": loss_depth.new_zeros(()),
                    }
                loss = (
                    loss_depth
                    + float(eval_args.overlap_consistency_weight) * loss_overlap
                    + float(eval_args.global_point_loss_weight)
                    * global_point_metrics["loss_global_point"]
                    + float(eval_args.camera_loss_weight) * camera_losses["loss_camera"]
                )

            row = {
                **(row_context or {}),
                "run": name,
                "dataset_index": int(pending_indices[local_index]),
                "seq_name": scalar_string(batch.get("scene_name") or batch.get("sequence_name")),
                "rgb_path": scalar_string(batch.get("rgb_path")),
                "depth_path": scalar_string(batch.get("depth_path")),
                "quality_bin": scalar_string(batch.get("metadata_quality_bin"), default="unknown"),
                "input_pano_count": int(moved["pano_image"].shape[1]),
                "camera_eval_pano_count": min(
                    int(moved["pano_image"].shape[1]), int(camera_eval_max_panos)
                ),
                "loss": float(loss.detach().cpu()),
                "loss_depth": float(loss_depth.detach().cpu()),
                "loss_overlap": float(loss_overlap.detach().cpu()),
                "loss_global_point": float(global_point_metrics["loss_global_point"].detach().cpu()),
                "global_point_valid_ratio": float(
                    global_point_metrics["global_point_valid_ratio"].detach().cpu()
                ),
                "loss_camera": float(camera_losses["loss_camera"].detach().cpu()),
                "loss_camera_t": float(camera_losses["loss_camera_t"].detach().cpu()),
                "loss_camera_r": float(camera_losses["loss_camera_r"].detach().cpu()),
                "camera_rotation_deg": float(camera_losses["camera_rotation_deg"].detach().cpu()),
                "camera_translation_deg": float(camera_losses["camera_translation_deg"].detach().cpu()),
                "camera_translation_valid_count": float(
                    camera_losses["camera_translation_valid_count"].detach().cpu()
                ),
                "camera_rotation_valid_count": float(
                    camera_losses["camera_rotation_valid_count"].detach().cpu()
                ),
                "valid_fraction": float(target_valid.float().mean().detach().cpu()),
                "pred_depth_scale": float(pred_depth_scale.detach().float().cpu()),
                "metadata_valid_ratio": scalar_float(batch.get("metadata_valid_ratio")),
                "metadata_structure_score": scalar_float(batch.get("metadata_structure_score")),
                "sample_weight": scalar_float(batch.get("sample_weight"), default=1.0),
                "depth_metric_protocol": DEPTH_METRIC_PROTOCOL,
                **pose_metrics,
                **depth_metrics,
                **erp_prior_metrics,
            }
            for pair_row in pose_pair_rows:
                pair_row.update(
                    {
                        **(row_context or {}),
                        "run": name,
                        "dataset_index": int(pending_indices[local_index]),
                        "seq_name": row["seq_name"],
                        "rgb_path": row["rgb_path"],
                        "depth_path": row["depth_path"],
                    }
                )
            rows.append(row)
            per_sample_rows.append(row)
            if per_sample_csv is not None:
                append_csv_row(per_sample_csv, PER_SAMPLE_CSV_FIELDS, row)
            if camera_pair_rows is not None:
                camera_pair_rows.extend(pose_pair_rows)
            if camera_pair_csv is not None:
                append_csv_rows(camera_pair_csv, CAMERA_PAIR_CSV_FIELDS, pose_pair_rows)
            processed = len(existing_rows) + local_index + 1
            if print_each_sample:
                print(
                    format_eval_sample_line(
                        row,
                        processed=processed,
                        total=len(selected_indices),
                        shard_rank=int(shard_rank),
                    ),
                    flush=True,
                )
            if progress_file is not None and (processed == 1 or processed == len(selected_indices) or processed % max(int(progress_every), 1) == 0):
                partial_camera_pose_summary = summarize_camera_pose_pairs(camera_pair_rows or [])
                write_eval_progress(
                    progress_file,
                    {
                        **(progress_context or {}),
                        "state": "running",
                        "run": name,
                        "split": split,
                        "dataset_size": len(dataset),
                        "candidate_samples": len(indices),
                        "shard_samples": len(selected_indices),
                        "processed_samples": processed,
                        "last_dataset_index": int(selected_indices[local_index]),
                        "last_seq_name": row["seq_name"],
                        "last_loss": row["loss"],
                        "last_loss_depth": row["loss_depth"],
                        "last_loss_camera": row["loss_camera"],
                        "last_depth_abs_rel": row["depth_abs_rel"],
                        "last_depth_delta_1p25": row["depth_delta_1p25"],
                        "last_depth_irls_scale": row["depth_irls_scale"],
                        "last_depth_irls_abs_rel": row["depth_irls_abs_rel"],
                        "last_depth_irls_delta_1p25": row["depth_irls_delta_1p25"],
                        "last_depth_irls_rmse": row["depth_irls_rmse"],
                        "last_camera_pose_pair_count": row["camera_pose_pair_count"],
                        "last_camera_pose_translation_deg_median": row["camera_pose_translation_deg_median"],
                        "last_camera_pose_rotation_deg_median": row["camera_pose_rotation_deg_median"],
                        "camera_pose_pair_count": partial_camera_pose_summary.get("pair_count", 0),
                        "camera_pose_auc30": partial_camera_pose_summary.get("auc@30"),
                        "shard_rank": int(shard_rank),
                        "num_shards": int(num_shards),
                        "updated_at": time.time(),
                    },
                )

    run_camera_pair_rows = [
        row
        for row in (camera_pair_rows or [])
        if str(row.get("run", "")) == name
        and parse_int(row.get("dataset_index"), default=-1) in selected_index_set
    ]
    result = {
        "name": name,
        "split": split,
        "curriculum_bins": curriculum_bins or "all",
        "dataset_size": len(dataset),
        "eval_pano_min_count": int(getattr(eval_args, "pano_min_count", 1)),
        "eval_pano_max_count": int(getattr(eval_args, "pano_max_count", 1)),
        "requested_samples": int(limit),
        "candidate_samples": len(indices),
        "evaluated_samples": len(rows),
        "sample_policy": sampling_info,
        "shard_rank": int(shard_rank),
        "num_shards": int(num_shards),
        "summary": summarize_values([row["loss"] for row in rows]),
        "depth_summary": summarize_values([row["loss_depth"] for row in rows]),
        "overlap_summary": summarize_values([row["loss_overlap"] for row in rows]),
        "global_point_summary": summarize_values([row["loss_global_point"] for row in rows]),
        "camera_summary": summarize_values([row["loss_camera"] for row in rows]),
        "camera_translation_summary": summarize_values([row["loss_camera_t"] for row in rows]),
        "camera_translation_deg_summary": summarize_values(
            [row["camera_translation_deg"] for row in rows if row["camera_translation_valid_count"] > 0]
        ),
        "camera_rotation_rad_summary": summarize_values([row["loss_camera_r"] for row in rows]),
        "camera_rotation_deg_summary": summarize_values(
            [row["camera_rotation_deg"] for row in rows if row["camera_rotation_valid_count"] > 0]
        ),
        "valid_fraction_summary": summarize_values([row["valid_fraction"] for row in rows]),
        "depth_metric_summary": summarize_metric_rows(rows, DEPTH_METRIC_KEYS),
        "erp_prior_depth_metric_summary": summarize_metric_rows(rows, ERP_PRIOR_DEPTH_METRIC_KEYS),
        "erp_prior_coverage_summary": summarize_metric_rows(rows, ERP_PRIOR_COVERAGE_KEYS),
        "erp_prior_diagnostic_summary": summarize_metric_rows(rows, ERP_PRIOR_DIAGNOSTIC_KEYS),
        "panovggt_metric_summary": summarize_panovggt_rows(rows),
        "panovggt_camera_pose_summary": summarize_camera_pose_pairs(run_camera_pair_rows),
        "by_quality_bin": summarize_by_key(rows, "quality_bin", "loss"),
        "best_samples": {
            "by_loss": rank_samples(rows, "loss", reverse=False),
            "by_depth_irls_abs_rel": rank_samples(rows, "depth_irls_abs_rel", reverse=False),
            "by_depth_irls_delta_1p25": rank_samples(rows, "depth_irls_delta_1p25", reverse=True),
        },
        "worst_samples": {
            "by_loss": rank_samples(rows, "loss", reverse=True),
            "by_depth_irls_abs_rel": rank_samples(rows, "depth_irls_abs_rel", reverse=True),
            "by_depth_irls_delta_1p25": rank_samples(rows, "depth_irls_delta_1p25", reverse=False),
        },
    }
    write_eval_progress(
        progress_file,
        {
            **(progress_context or {}),
            "state": "done",
            "run": name,
            "split": split,
            "dataset_size": len(dataset),
            "candidate_samples": len(indices),
            "shard_samples": len(selected_indices),
            "processed_samples": len(rows),
            "shard_rank": int(shard_rank),
            "num_shards": int(num_shards),
            "updated_at": time.time(),
        },
    )
    return result


def apply_exact_group_manifest(dataset: Any, rows: list[dict[str, Any]]) -> None:
    """Replace generated groups with the exact pano IDs listed in a CSV manifest."""
    groups = getattr(dataset, "groups", None)
    items = getattr(dataset, "items", None)
    if groups is None or items is None:
        raise TypeError("Exact group manifests require a dataset exposing items and groups")

    pano_to_item: dict[str, int] = {}
    for item_index, item in enumerate(items):
        rgb_name = Path(str(item.get("rgb_path", ""))).name
        match = re.search(r"camera_([0-9a-fA-F]{32})_", rgb_name)
        if match:
            pano_to_item[match.group(1).lower()] = item_index

    exact_groups: list[list[int]] = []
    for row_index, row in enumerate(rows):
        pano_columns = sorted(
            (key for key in row if key.startswith("pano_id_") and row.get(key)),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        if not pano_columns:
            raise ValueError(f"Manifest row {row_index} has no pano_id_N columns")
        pano_ids = [str(row[key]).strip().lower() for key in pano_columns]
        missing = [pano_id for pano_id in pano_ids if pano_id not in pano_to_item]
        if missing:
            raise KeyError(f"Manifest row {row_index} references missing pano IDs: {missing}")
        group = [pano_to_item[pano_id] for pano_id in pano_ids]
        scene_keys = {_item_scene_group_key(items[item_index]) for item_index in group}
        if len(scene_keys) != 1:
            raise RuntimeError(
                f"Manifest row {row_index} crosses scenes: pano_ids={pano_ids} scene_keys={sorted(scene_keys)}"
            )
        exact_groups.append(group)

    dataset.groups = exact_groups
    dataset.preserve_exact_group_duplicates = True


def write_eval_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if "started_at" not in payload and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if "started_at" in previous:
                payload["started_at"] = previous["started_at"]
        except (OSError, json.JSONDecodeError):
            pass
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def format_eval_sample_line(
    row: dict[str, Any],
    *,
    processed: int,
    total: int,
    shard_rank: int,
) -> str:
    pair_count = int(float(row.get("camera_pose_pair_count", 0) or 0))
    t_deg = format_eval_number(row.get("camera_pose_translation_deg_median")) if pair_count else "NA"
    r_deg = format_eval_number(row.get("camera_pose_rotation_deg_median")) if pair_count else "NA"
    seq_name = "_".join(str(row.get("seq_name", "unknown")).split())
    return (
        f"[EVAL-SAMPLE] shard={shard_rank} run={row.get('run', 'unknown')} "
        f"step={processed}/{total} index={row.get('dataset_index', -1)} "
        f"panos={row.get('input_pano_count', 0)} "
        f"loss={format_eval_number(row.get('loss'))} "
        f"depth_loss={format_eval_number(row.get('loss_depth'))} "
        f"camera_loss={format_eval_number(row.get('loss_camera'))} "
        f"raw_absrel={format_eval_number(row.get('depth_abs_rel'))} "
        f"raw_delta1={format_eval_number(row.get('depth_delta_1p25'))} "
        f"irls_scale={format_eval_number(row.get('depth_irls_scale'))} "
        f"irls_absrel={format_eval_number(row.get('depth_irls_abs_rel'))} "
        f"irls_delta1={format_eval_number(row.get('depth_irls_delta_1p25'))} "
        f"irls_rmse={format_eval_number(row.get('depth_irls_rmse'))} "
        f"erp_absrel={format_eval_number(row.get('erp_prior_depth_irls_abs_rel'))} "
        f"erp_delta1={format_eval_number(row.get('erp_prior_depth_irls_delta_1p25'))} "
        f"erp_eval={format_eval_number(row.get('erp_evaluated_gt_fraction'))} "
        f"pose_pairs={pair_count} t_med_deg={t_deg} r_med_deg={r_deg} "
        f"seq={seq_name}"
    )


def format_eval_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.6g}" if math.isfinite(number) else "NA"


def build_covered_sphere_weights(
    camera_meta: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    num_panos: int,
) -> torch.Tensor:
    """Build equal-solid-angle weights without double-counting overlap."""
    yaw = camera_meta["yaw"]
    pitch = camera_meta["pitch"]
    fov_x = camera_meta["fov_x"]
    fov_y = camera_meta["fov_y"]
    if yaw.ndim != 2 or pitch.shape != yaw.shape or fov_x.shape != yaw.shape or fov_y.shape != yaw.shape:
        raise ValueError("Expected flattened camera metadata with shape [batch, total_views]")
    batch_size, total_views = yaw.shape
    if num_panos < 1 or total_views % num_panos != 0:
        raise ValueError(f"Cannot split {total_views} views across {num_panos} panoramas")
    views_per_pano = total_views // num_panos
    grouped = [value.reshape(batch_size, num_panos, views_per_pano) for value in (yaw, pitch, fov_x, fov_y)]
    output: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        pano_weights: list[torch.Tensor] = []
        for pano_index in range(num_panos):
            params = tuple(value[batch_index, pano_index] for value in grouped)
            pano_weights.append(_covered_sphere_weights_for_view_set(*params, height=height, width=width))
        output.append(torch.cat(pano_weights, dim=0))
    return torch.stack(output, dim=0)[..., None]


def _covered_sphere_weights_for_view_set(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    key = (
        str(yaw.device),
        str(yaw.dtype),
        int(height),
        int(width),
        tuple(round(float(value), 8) for value in yaw.detach().cpu()),
        tuple(round(float(value), 8) for value in pitch.detach().cpu()),
        tuple(round(float(value), 8) for value in fov_x.detach().cpu()),
        tuple(round(float(value), 8) for value in fov_y.detach().cpu()),
    )
    cached = _SPHERICAL_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    metric_dtype = torch.float32
    yaw = yaw.to(dtype=metric_dtype)
    pitch = pitch.to(dtype=metric_dtype)
    fov_x = fov_x.to(dtype=metric_dtype)
    fov_y = fov_y.to(dtype=metric_dtype)
    rays = pinhole_rays(
        yaw,
        pitch,
        fov_x,
        fov_y,
        height,
        width,
        device=yaw.device,
        dtype=metric_dtype,
    )
    forward, right, up = yaw_pitch_to_axes(yaw, pitch)

    # For the gnomonic plane used by pinhole_rays,
    # dOmega = dx dy / (1 + x^2 + y^2)^(3/2).
    local_forward = torch.einsum("vhwc,vc->vhw", rays, forward).clamp_min(0.0)
    tan_x = torch.tan(fov_x * 0.5)
    tan_y = torch.tan(fov_y * 0.5)
    solid_angle = tan_x[:, None, None] * tan_y[:, None, None] * local_forward.pow(3)

    ray_flat = rays.reshape(-1, 3)
    z = ray_flat @ forward.transpose(0, 1)
    x = ray_flat @ right.transpose(0, 1)
    y = ray_flat @ up.transpose(0, 1)
    inside = (
        (z > 1e-7)
        & (x.abs() <= z * tan_x[None, :] + 1e-6)
        & (y.abs() <= z * tan_y[None, :] + 1e-6)
    )
    coverage_count = inside.sum(dim=-1).reshape_as(solid_angle).clamp_min(1)
    weights = (solid_angle / coverage_count).detach()
    _SPHERICAL_WEIGHT_CACHE[key] = weights
    return weights


def compute_erp_prior_depth_metrics(
    *,
    pred_window_z: torch.Tensor,
    gt_erp_depth: torch.Tensor,
    source_depth_semantics: str,
    max_range_depth: float,
    camera_meta: dict[str, torch.Tensor],
    dataset_name: str,
    align_corners: bool,
) -> dict[str, float]:
    """Fuse windows into ERP and complete only statistically stable polar caps."""
    gt_range = erp_depth_to_range_depth(gt_erp_depth.detach().float(), source_depth_semantics)
    if gt_range.ndim == 5 and gt_range.shape[2] == 1:
        gt_range = gt_range[:, :, 0]
    if gt_range.ndim != 4:
        raise ValueError(f"Expected GT ERP depth [B,N,H,W], got {tuple(gt_range.shape)}")
    batch, num_panos, erp_height, erp_width = gt_range.shape
    views = int(pred_window_z.shape[1])
    if num_panos < 1 or views % num_panos != 0:
        raise ValueError(f"Cannot map {views} windows to {num_panos} panoramas")
    view_pano_index = (
        torch.arange(num_panos, device=pred_window_z.device)
        .repeat_interleave(views // num_panos)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    splatted = splat_window_z_depth_to_erp(
        pred_window_z,
        yaw=camera_meta["yaw"],
        pitch=camera_meta["pitch"],
        fov_x=camera_meta["fov_x"],
        fov_y=camera_meta["fov_y"],
        view_pano_index=view_pano_index,
        num_panos=num_panos,
        erp_height=erp_height,
        erp_width=erp_width,
        align_corners=align_corners,
    )
    gt_valid = torch.isfinite(gt_range) & (gt_range > 0.0)
    if float(max_range_depth) > 0:
        gt_valid &= gt_range <= float(max_range_depth)
    window_mask = splatted["valid_mask"] & gt_valid

    row_latitudes = 90.0 - (
        torch.arange(erp_height, device=gt_range.device, dtype=torch.float32) + 0.5
    ) * 180.0 / float(erp_height)
    north = (row_latitudes >= ERP_POLAR_PRIOR_LATITUDE_DEG).reshape(1, 1, erp_height, 1)
    south = (row_latitudes <= -ERP_POLAR_PRIOR_LATITUDE_DEG).reshape(1, 1, erp_height, 1)
    priors = ERP_POLAR_DEPTH_PRIORS_M.get(str(dataset_name), {"north": None, "south": None})
    prior_depth = torch.zeros_like(gt_range)
    prior_available = torch.zeros_like(gt_valid)
    for cap_mask, cap_name in ((north, "north"), (south, "south")):
        prior = priors.get(cap_name)
        if prior is None:
            continue
        expanded = cap_mask.expand_as(gt_valid)
        prior_depth = torch.where(expanded, prior_depth.new_tensor(float(prior)), prior_depth)
        prior_available |= expanded

    # Priors complete only geometry not observed by a valid window prediction.
    prior_fill_mask = prior_available & ~splatted["valid_mask"] & gt_valid
    completed_pred = torch.where(prior_fill_mask, prior_depth, splatted["depth"])
    eval_mask = window_mask | prior_fill_mask
    metrics = compute_depth_metrics(
        completed_pred,
        gt_range,
        eval_mask,
        alignment_valid=window_mask,
        alignment_scale_exempt=prior_fill_mask,
    )
    window_only = compute_depth_metrics(splatted["depth"], gt_range, window_mask)
    missing_as_invalid = torch.where(
        prior_fill_mask,
        torch.full_like(splatted["depth"], 1e-6),
        splatted["depth"],
    )
    no_prior = compute_depth_metrics(
        missing_as_invalid,
        gt_range,
        eval_mask,
        alignment_valid=window_mask,
    )

    def prior_abs_rel(mask: torch.Tensor) -> float:
        if not mask.any():
            return 0.0
        values = (prior_depth[mask] - gt_range[mask]).abs() / gt_range[mask].clamp_min(1e-6)
        return float(values.mean().cpu())

    gt_valid_count = gt_valid.sum().clamp_min(1).float()
    return {
        **{f"erp_prior_{key}": value for key, value in metrics.items()},
        "erp_window_coverage_fraction": float(window_mask.sum().float().div(gt_valid_count).cpu()),
        "erp_prior_fill_fraction": float(prior_fill_mask.sum().float().div(gt_valid_count).cpu()),
        "erp_evaluated_gt_fraction": float(eval_mask.sum().float().div(gt_valid_count).cpu()),
        "erp_window_only_depth_irls_abs_rel": window_only["depth_irls_abs_rel"],
        "erp_window_only_depth_irls_delta_1p25": window_only["depth_irls_delta_1p25"],
        "erp_no_prior_depth_irls_abs_rel": no_prior["depth_irls_abs_rel"],
        "erp_no_prior_depth_irls_delta_1p25": no_prior["depth_irls_delta_1p25"],
        "erp_prior_region_abs_rel": prior_abs_rel(prior_fill_mask),
        "erp_prior_north_abs_rel": prior_abs_rel(prior_fill_mask & north),
        "erp_prior_south_abs_rel": prior_abs_rel(prior_fill_mask & south),
    }


def compute_depth_metrics(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_valid: torch.Tensor,
    spherical_weights: torch.Tensor | None = None,
    alignment_valid: torch.Tensor | None = None,
    alignment_scale_exempt: torch.Tensor | None = None,
) -> dict[str, float]:
    pred = pred_depth.detach().float()
    target = target_depth.detach().float()
    valid = (
        target_valid.detach().bool()
        & torch.isfinite(pred)
        & torch.isfinite(target)
        & (pred > 0)
        & (target > 0)
    )
    if valid.sum().item() == 0:
        return {
            "depth_mae": 0.0,
            "depth_rmse": 0.0,
            "depth_abs_rel": 0.0,
            "depth_delta_1p25": 0.0,
            "depth_delta_1p25_2": 0.0,
            "depth_delta_1p25_3": 0.0,
            "depth_irls_scale": 0.0,
            "depth_irls_mae": 0.0,
            "depth_irls_rmse": 0.0,
            "depth_irls_abs_rel": 0.0,
            "depth_irls_delta_1p25": 0.0,
            "depth_irls_delta_1p25_2": 0.0,
            "depth_irls_delta_1p25_3": 0.0,
            "depth_valid_pixels": 0,
            **{key: 0.0 for key in DEPTH_ACCUMULATOR_KEYS},
        }
    pred_values = pred[valid].clamp_min(1e-6)
    target_values = target[valid].clamp_min(1e-6)
    if spherical_weights is None:
        metric_weights = torch.ones_like(pred_values)
    else:
        weights = spherical_weights.detach().float()
        if weights.shape != pred.shape:
            raise ValueError(f"Spherical weight shape {tuple(weights.shape)} does not match depth {tuple(pred.shape)}")
        metric_weights = weights[valid].clamp_min(0.0)
    # Existing micro summaries divide accumulator values by valid pixels. Keep
    # that denominator while preserving the relative spherical weights.
    metric_weights = metric_weights * (float(metric_weights.numel()) / metric_weights.sum().clamp_min(1e-12))
    raw_metrics = depth_metrics_from_values(pred_values, target_values, prefix="depth", weights=metric_weights)
    if alignment_valid is None:
        alignment_mask = valid
    else:
        alignment_mask = (
            alignment_valid.detach().bool()
            & torch.isfinite(pred)
            & torch.isfinite(target)
            & (pred > 0)
            & (target > 0)
        )
    if alignment_mask.any():
        alignment_pred = pred[alignment_mask].clamp_min(1e-6)
        alignment_target = target[alignment_mask].clamp_min(1e-6)
        if spherical_weights is None:
            alignment_weights = None
        else:
            alignment_weights = spherical_weights.detach().float()[alignment_mask].clamp_min(0.0)
        irls_scale = fit_irls_scale(
            alignment_pred,
            alignment_target,
            sample_weights=alignment_weights,
        )
    else:
        irls_scale = pred_values.new_tensor(1.0)
    aligned_pred_values = pred_values * irls_scale
    if alignment_scale_exempt is not None:
        exempt = alignment_scale_exempt.detach().bool()
        if exempt.shape != pred.shape:
            raise ValueError(
                f"Alignment scale exempt shape {tuple(exempt.shape)} does not match depth {tuple(pred.shape)}"
            )
        aligned_pred_values = torch.where(exempt[valid], pred_values, aligned_pred_values)
    aligned_metrics = depth_metrics_from_values(
        aligned_pred_values,
        target_values,
        prefix="depth_irls",
        weights=metric_weights,
    )
    return {
        **raw_metrics,
        **depth_metric_accumulators(pred_values, target_values, prefix="depth", weights=metric_weights),
        "depth_irls_scale": float(irls_scale.cpu()),
        **aligned_metrics,
        **depth_metric_accumulators(
            aligned_pred_values,
            target_values,
            prefix="depth_irls",
            weights=metric_weights,
        ),
        "depth_valid_pixels": int(valid.sum().item()),
    }


def compute_panovggt_camera_pose_metrics(
    predictions: dict[str, Any],
    batch: dict[str, Any],
    position_mode: str,
    pred_translation_scale: torch.Tensor | None,
    trans_norm_thresh: float = 1e-2,
    max_panos: int = 3,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """PanoVGGT-style pairwise relative-pose metrics for pano-level camera heads."""
    empty = empty_camera_pose_sample_metrics()
    pred_center = predictions.get("pano_camera_center")
    pred_quat = predictions.get("pano_rotation_quat_w2c")
    if pred_center is None or pred_quat is None:
        return empty, []

    targets = build_relative_pano_pose_targets(
        batch=batch,
        position_mode=position_mode,
        device=pred_center.device,
        dtype=torch.float32,
    )
    if targets is None:
        return empty, []
    target_center, target_quat, position_valid, rotation_valid = targets
    if pred_center.shape != target_center.shape or pred_quat.shape != target_quat.shape:
        return empty, []
    if int(max_panos) > 0:
        pano_limit = int(max_panos)
        pred_center = pred_center[:, :pano_limit]
        pred_quat = pred_quat[:, :pano_limit]
        target_center = target_center[:, :pano_limit]
        target_quat = target_quat[:, :pano_limit]
        position_valid = position_valid[:, :pano_limit]
        rotation_valid = rotation_valid[:, :pano_limit]

    pred_center = torch.nan_to_num(pred_center.float(), nan=0.0, posinf=0.0, neginf=0.0)
    pred_quat = F.normalize(
        torch.nan_to_num(pred_quat.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    pred_center, pred_quat = omega_y_up_pose_to_official_y_down(pred_center, pred_quat)
    if pred_translation_scale is not None:
        pred_center = pred_center * expand_sample_scale_like(pred_translation_scale, pred_center)

    target_center = torch.nan_to_num(target_center.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target_quat = F.normalize(
        torch.nan_to_num(target_quat.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    pred_w2c = camera_w2c_from_center_quat(pred_center, pred_quat)
    target_w2c = camera_w2c_from_center_quat(target_center, target_quat)

    batch_size, pano_count = pred_center.shape[:2]
    if pano_count < 2:
        return empty, []

    pair_index = torch.combinations(
        torch.arange(pano_count, device=pred_center.device),
        r=2,
        with_replacement=False,
    )
    if pair_index.numel() == 0:
        return empty, []
    pair_i = pair_index[:, 0]
    pair_j = pair_index[:, 1]

    all_r_err: list[torch.Tensor] = []
    all_t_err: list[torch.Tensor] = []
    all_gt_norm: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []
    valid_pano = position_valid.bool() & rotation_valid.bool()
    finite_pred = torch.isfinite(pred_w2c).all(dim=(-1, -2))
    finite_target = torch.isfinite(target_w2c).all(dim=(-1, -2))
    valid_pano = valid_pano & finite_pred & finite_target

    for batch_index in range(batch_size):
        pair_valid = valid_pano[batch_index, pair_i] & valid_pano[batch_index, pair_j]
        if not bool(pair_valid.any()):
            continue
        pi = pair_i[pair_valid]
        pj = pair_j[pair_valid]
        rel_gt = target_w2c[batch_index, pj].bmm(invert_se3(target_w2c[batch_index, pi]))
        rel_pred = pred_w2c[batch_index, pj].bmm(invert_se3(pred_w2c[batch_index, pi]))
        gt_translation = rel_gt[:, :3, 3]
        gt_translation_norm = torch.linalg.vector_norm(gt_translation, dim=-1)
        baseline_valid = (
            gt_translation_norm > float(trans_norm_thresh)
        ) & torch.isfinite(gt_translation_norm) & torch.isfinite(rel_pred[:, :3, 3]).all(dim=-1)
        if not bool(baseline_valid.any()):
            continue

        rel_gt = rel_gt[baseline_valid]
        rel_pred = rel_pred[baseline_valid]
        pi = pi[baseline_valid]
        pj = pj[baseline_valid]
        gt_translation_norm = gt_translation_norm[baseline_valid]
        r_err = rotation_angle_degrees(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
        t_err = translation_angle_degrees(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
        finite_pair = torch.isfinite(r_err) & torch.isfinite(t_err)
        if not bool(finite_pair.any()):
            continue
        r_err = r_err[finite_pair]
        t_err = t_err[finite_pair]
        pi = pi[finite_pair]
        pj = pj[finite_pair]
        gt_translation_norm = gt_translation_norm[finite_pair]
        all_r_err.append(r_err)
        all_t_err.append(t_err)
        all_gt_norm.append(gt_translation_norm)
        for idx in range(int(r_err.numel())):
            r_value = float(r_err[idx].detach().cpu())
            t_value = float(t_err[idx].detach().cpu())
            pair_rows.append(
                {
                    "batch_index": int(batch_index),
                    "pair_i": int(pi[idx].detach().cpu()),
                    "pair_j": int(pj[idx].detach().cpu()),
                    "camera_pose_rotation_deg": r_value,
                    "camera_pose_translation_deg": t_value,
                    "camera_pose_max_error_deg": max(r_value, t_value),
                    "camera_pose_gt_translation_norm": float(gt_translation_norm[idx].detach().cpu()),
                }
            )

    if not all_r_err:
        return empty, []
    r_all = torch.cat(all_r_err).detach().float().cpu()
    t_all = torch.cat(all_t_err).detach().float().cpu()
    gt_norm_all = torch.cat(all_gt_norm).detach().float().cpu()
    return camera_pose_sample_metrics(r_all, t_all, gt_norm_all), pair_rows


def empty_camera_pose_sample_metrics() -> dict[str, float]:
    return {key: 0.0 for key in CAMERA_POSE_SAMPLE_KEYS}


def camera_w2c_from_center_quat(center: torch.Tensor, quat_w2c: torch.Tensor) -> torch.Tensor:
    rotations_w2c = quat_to_mat(F.normalize(quat_w2c.float(), dim=-1, eps=1e-6))
    translation = -(rotations_w2c @ center.float()[..., None])[..., 0]
    eye = torch.eye(4, device=center.device, dtype=torch.float32)
    pose = eye.reshape(*((1,) * (rotations_w2c.ndim - 2)), 4, 4).repeat(*rotations_w2c.shape[:-2], 1, 1)
    pose[..., :3, :3] = rotations_w2c
    pose[..., :3, 3] = translation
    return pose


def invert_se3(pose: torch.Tensor) -> torch.Tensor:
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    inv_rotation = rotation.transpose(-1, -2).contiguous()
    inv_translation = -(inv_rotation @ translation[..., None])[..., 0]
    inv_pose = torch.zeros_like(pose)
    inv_pose[..., :3, :3] = inv_rotation
    inv_pose[..., :3, 3] = inv_translation
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def rotation_angle_degrees(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    q_gt = mat_to_quat(rot_gt.float())
    q_pred = mat_to_quat(rot_pred.float())
    dot = (q_pred * q_gt).sum(dim=-1)
    loss = (1.0 - dot.square()).clamp(min=eps, max=1.0)
    return torch.arccos((1.0 - 2.0 * loss).clamp(-1.0, 1.0)) * (180.0 / math.pi)


def translation_angle_degrees(
    t_gt: torch.Tensor,
    t_pred: torch.Tensor,
    eps: float = 1e-15,
    default_err: float = 1e6,
) -> torch.Tensor:
    t_gt = t_gt.float() / (torch.linalg.vector_norm(t_gt.float(), dim=-1, keepdim=True) + eps)
    t_pred = t_pred.float() / (torch.linalg.vector_norm(t_pred.float(), dim=-1, keepdim=True) + eps)
    dot2 = (t_gt * t_pred).sum(dim=-1).square()
    err = torch.arccos(torch.sqrt(1.0 - (1.0 - dot2).clamp(min=eps)))
    err = torch.where(torch.isfinite(err), err, err.new_full(err.shape, float(default_err)))
    deg = err * (180.0 / math.pi)
    return torch.minimum(deg, (180.0 - deg).abs())


def camera_pose_sample_metrics(
    r_error: torch.Tensor,
    t_error: torch.Tensor,
    gt_translation_norm: torch.Tensor,
) -> dict[str, float]:
    max_error = torch.maximum(r_error, t_error)
    return {
        "camera_pose_pair_count": float(r_error.numel()),
        "camera_pose_auc3": pose_auc(r_error, t_error, 3),
        "camera_pose_auc5": pose_auc(r_error, t_error, 5),
        "camera_pose_auc15": pose_auc(r_error, t_error, 15),
        "camera_pose_auc30": pose_auc(r_error, t_error, 30),
        "camera_pose_rotation_deg_mean": float(r_error.mean()),
        "camera_pose_rotation_deg_median": float(r_error.median()),
        "camera_pose_translation_deg_mean": float(t_error.mean()),
        "camera_pose_translation_deg_median": float(t_error.median()),
        "camera_pose_max_error_deg_mean": float(max_error.mean()),
        "camera_pose_max_error_deg_median": float(max_error.median()),
        "camera_pose_gt_translation_norm_mean": float(gt_translation_norm.mean()),
        "camera_pose_gt_translation_norm_median": float(gt_translation_norm.median()),
    }


def pose_auc(r_error: torch.Tensor, t_error: torch.Tensor, max_threshold: int) -> float:
    if r_error.numel() == 0 or t_error.numel() == 0:
        return 0.0
    errs = torch.maximum(r_error.float(), t_error.float()).detach().cpu().numpy()
    hist, _ = np.histogram(errs, bins=np.arange(int(max_threshold) + 1))
    norm = hist.astype(np.float64) / max(len(errs), 1)
    return float(np.mean(np.cumsum(norm)))


def depth_metrics_from_values(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    prefix: str,
    weights: torch.Tensor | None = None,
) -> dict[str, float]:
    diff = pred_values - target_values
    abs_diff = diff.abs()
    ratio = torch.maximum(pred_values / target_values, target_values / pred_values)
    weights = torch.ones_like(pred_values) if weights is None else weights
    weight_sum = weights.sum().clamp_min(1e-12)
    return {
        f"{prefix}_mae": float((weights * abs_diff).sum().div(weight_sum).cpu()),
        f"{prefix}_rmse": float(torch.sqrt((weights * diff.square()).sum().div(weight_sum)).cpu()),
        f"{prefix}_abs_rel": float((weights * abs_diff / target_values).sum().div(weight_sum).cpu()),
        f"{prefix}_delta_1p25": float((weights * (ratio < 1.25)).sum().div(weight_sum).cpu()),
        f"{prefix}_delta_1p25_2": float((weights * (ratio < 1.25**2)).sum().div(weight_sum).cpu()),
        f"{prefix}_delta_1p25_3": float((weights * (ratio < 1.25**3)).sum().div(weight_sum).cpu()),
    }


def depth_metric_accumulators(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    prefix: str,
    weights: torch.Tensor | None = None,
) -> dict[str, float]:
    diff = pred_values - target_values
    abs_diff = diff.abs()
    ratio = torch.maximum(pred_values / target_values, target_values / pred_values)
    weights = torch.ones_like(pred_values) if weights is None else weights
    return {
        f"{prefix}_abs_error_sum": float((weights * abs_diff).sum().cpu()),
        f"{prefix}_sq_error_sum": float((weights * diff.square()).sum().cpu()),
        f"{prefix}_abs_rel_sum": float((weights * abs_diff / target_values).sum().cpu()),
        f"{prefix}_delta_1p25_count": float((weights * (ratio < 1.25)).sum().cpu()),
        f"{prefix}_delta_1p25_2_count": float((weights * (ratio < 1.25**2)).sum().cpu()),
        f"{prefix}_delta_1p25_3_count": float((weights * (ratio < 1.25**3)).sum().cpu()),
    }


def fit_irls_scale(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    iterations: int = 10,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    sample_weights = torch.ones_like(pred_values) if sample_weights is None else sample_weights
    scale = torch.median(target_values / pred_values).clamp_min(1e-6)
    for _ in range(iterations):
        residual = scale * pred_values - target_values
        weights = sample_weights / residual.abs().clamp_min(1e-3)
        denom = (weights * pred_values.square()).sum().clamp_min(1e-6)
        scale = ((weights * pred_values * target_values).sum() / denom).clamp_min(1e-6)
    return scale


def sample_indices(length: int, limit: int, seed: int) -> list[int]:
    if length <= 0:
        return []
    if int(limit) <= 0:
        return list(range(length))
    limit = min(max(int(limit), 0), length)
    indices = list(range(length))
    random.Random(int(seed)).shuffle(indices)
    return indices[:limit]


def resolve_eval_limit(total: int, limit: int, limit_fraction: float) -> int:
    if total <= 0:
        return 0
    if int(limit) > 0:
        return min(int(limit), total)
    fraction = float(limit_fraction or 0.0)
    if fraction > 0.0:
        return max(1, min(total, int(math.ceil(total * min(fraction, 1.0)))))
    return total


def select_eval_indices(
    dataset: Any,
    limit: int,
    seed: int,
    sample_policy: str,
    limit_fraction: float = 0.0,
) -> tuple[list[int], dict[str, Any]]:
    policy = str(sample_policy or "scene_neighborhood")
    if policy == "anchor":
        resolved_limit = resolve_eval_limit(len(dataset), limit, limit_fraction)
        indices = sample_indices(len(dataset), resolved_limit, seed)
        return indices, {
            "policy": "anchor",
            "dataset_size": len(dataset),
            "limit": int(limit),
            "limit_fraction": float(limit_fraction or 0.0),
            "selected_samples": len(indices),
            "scene_group_count": None,
        }
    if policy != "scene_neighborhood":
        raise ValueError(f"Unknown eval sample policy: {sample_policy}")

    scene_to_indices = scene_neighborhood_indices(dataset)
    if not scene_to_indices:
        resolved_limit = resolve_eval_limit(len(dataset), limit, limit_fraction)
        indices = sample_indices(len(dataset), resolved_limit, seed)
        return indices, {
            "policy": "anchor_fallback",
            "requested_policy": policy,
            "dataset_size": len(dataset),
            "limit": int(limit),
            "limit_fraction": float(limit_fraction or 0.0),
            "selected_samples": len(indices),
            "scene_group_count": None,
        }

    rng = random.Random(int(seed))
    scene_keys = sorted(scene_to_indices)
    rng.shuffle(scene_keys)
    resolved_limit = resolve_eval_limit(len(scene_keys), limit, limit_fraction)
    scene_keys = scene_keys[:resolved_limit]

    selected = [rng.choice(scene_to_indices[scene_key]) for scene_key in scene_keys]
    return selected, {
        "policy": "scene_neighborhood",
        "dataset_size": len(dataset),
        "limit": int(limit),
        "limit_fraction": float(limit_fraction or 0.0),
        "scene_group_count": len(scene_to_indices),
        "selected_scene_groups": len(scene_keys),
        "selected_samples": len(selected),
    }


def scene_neighborhood_indices(dataset: Any) -> dict[str, list[int]]:
    groups = getattr(dataset, "groups", None)
    items = getattr(dataset, "items", None)
    if groups is None or items is None:
        return {}

    scene_to_indices: dict[str, list[int]] = {}
    for group_index, group in enumerate(groups):
        if not group:
            continue
        scene_keys = {_item_scene_group_key(items[int(item_index)]) for item_index in group}
        if len(scene_keys) != 1:
            raise RuntimeError(f"Eval multi-pano group crosses scenes: group={group_index} scene_keys={sorted(scene_keys)}")
        scene_key = next(iter(scene_keys))
        scene_to_indices.setdefault(scene_key, []).append(group_index)
    return scene_to_indices


def _item_scene_group_key(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("scene_group_key", "scene_name"):
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
        dataset = str(item.get("dataset") or item.get("sequence_name") or "unknown")
        rgb_path = str(item.get("rgb_path") or "")
        return f"{dataset}:{rgb_path}"
    return str(item)


def summarize_values(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "gt_0p1": int((arr > 0.1).sum()),
        "gt_0p2": int((arr > 0.2).sum()),
        "gt_0p3": int((arr > 0.3).sum()),
    }


def summarize_by_key(rows: list[dict[str, Any]], key: str, value_key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(float(row[value_key]))
    return {group: summarize_values(values) for group, values in sorted(grouped.items())}


def summarize_metric_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[key] = summarize_values(values)
    return summary


def summarize_panovggt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        # Backward-compatible aliases used by existing Table 3 summaries.
        "macro_by_sample": summarize_metric_rows(rows, PANOVGGT_PRIMARY_METRICS),
        "micro_by_valid_pixel": summarize_panovggt_micro(rows),
        "covered_sphere": {
        "metric_meaning": {
            "depth_irls_abs_rel": "Covered-sphere Abs Rel after solid-angle-weighted per-sample robust scale-only alignment; overlapping windows are counted once.",
            "depth_irls_delta_1p25": "Covered-sphere delta < 1.25 after the same weighted scale-only alignment.",
            "depth_irls_rmse": "Covered-sphere RMSE after the same weighted scale-only alignment.",
            "depth_abs_rel": "Raw-scale Abs Rel using the model/checkpoint predicted depth scale.",
            "depth_delta_1p25": "Raw-scale delta < 1.25 using the model/checkpoint predicted depth scale.",
            "depth_rmse": "Raw-scale RMSE using the model/checkpoint predicted depth scale.",
        },
        "macro_by_sample": summarize_metric_rows(rows, PANOVGGT_PRIMARY_METRICS),
        "micro_by_valid_pixel": summarize_panovggt_micro(rows),
        },
        "erp_with_polar_prior": {
            "metric_meaning": {
                "erp_prior_depth_irls_abs_rel": (
                    "Uniform-ERP-pixel Abs Rel after scale fitting only on valid window-covered ERP pixels; "
                    "uncovered stable polar caps use fixed dataset priors."
                ),
                "erp_prior_depth_irls_delta_1p25": "ERP delta < 1.25 under the same scale and prior protocol.",
                "erp_prior_depth_irls_rmse": "ERP RMSE under the same scale and prior protocol.",
            },
            "macro_by_sample": summarize_metric_rows(
                rows,
                [
                    "erp_prior_depth_irls_abs_rel",
                    "erp_prior_depth_irls_delta_1p25",
                    "erp_prior_depth_irls_rmse",
                    "erp_evaluated_gt_fraction",
                ],
            ),
            "micro_by_valid_pixel": summarize_erp_prior_micro(rows),
        },
    }


def summarize_camera_pose_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r_values = finite_row_values(rows, "camera_pose_rotation_deg")
    t_values = finite_row_values(rows, "camera_pose_translation_deg")
    gt_norm_values = finite_row_values(rows, "camera_pose_gt_translation_norm")
    if not r_values or not t_values:
        return {
            "metric_meaning": {
                "translation_deg": "PanoVGGT-style relative translation direction error over unordered pano pairs.",
                "rotation_deg": "PanoVGGT-style relative rotation geodesic error over unordered pano pairs.",
                "auc@k": "Mean CDF of max(rotation_deg, translation_deg) over thresholds [0, k).",
            },
            "pair_count": 0,
        }
    r_arr = np.asarray(r_values, dtype=np.float64)
    t_arr = np.asarray(t_values, dtype=np.float64)
    max_arr = np.maximum(r_arr, t_arr)
    summary = {
        "metric_meaning": {
            "translation_deg": "PanoVGGT-style relative translation direction error over unordered pano pairs.",
            "rotation_deg": "PanoVGGT-style relative rotation geodesic error over unordered pano pairs.",
            "auc@k": "Mean CDF of max(rotation_deg, translation_deg), matching PanoVGGT/VGGSfM pose AUC aggregation.",
        },
        "pair_count": int(len(r_values)),
        "rotation_deg": summarize_values(r_values),
        "translation_deg": summarize_values(t_values),
        "max_error_deg": summarize_values(max_arr.tolist()),
        "gt_translation_norm": summarize_values(gt_norm_values),
    }
    for threshold in (3, 5, 15, 30):
        hist, _ = np.histogram(max_arr, bins=np.arange(threshold + 1))
        norm = hist.astype(np.float64) / max(len(max_arr), 1)
        summary[f"auc@{threshold}"] = float(np.mean(np.cumsum(norm)))
    return summary


def finite_row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_panovggt_micro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_depth_micro(rows, field_prefix="")


def summarize_erp_prior_micro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_depth_micro(rows, field_prefix="erp_prior_")


def summarize_depth_micro(rows: list[dict[str, Any]], field_prefix: str) -> dict[str, Any]:
    depth_prefix = f"{field_prefix}depth"
    valid = sum(float(row.get(f"{depth_prefix}_valid_pixels", 0.0)) for row in rows)
    if valid <= 0:
        return {f"{depth_prefix}_valid_pixels": 0}
    summary: dict[str, Any] = {f"{depth_prefix}_valid_pixels": int(valid)}
    for prefix in (depth_prefix, f"{depth_prefix}_irls"):
        abs_error_sum = sum(float(row.get(f"{prefix}_abs_error_sum", 0.0)) for row in rows)
        sq_error_sum = sum(float(row.get(f"{prefix}_sq_error_sum", 0.0)) for row in rows)
        abs_rel_sum = sum(float(row.get(f"{prefix}_abs_rel_sum", 0.0)) for row in rows)
        summary[f"{prefix}_mae"] = abs_error_sum / valid
        summary[f"{prefix}_rmse"] = math.sqrt(max(sq_error_sum / valid, 0.0))
        summary[f"{prefix}_abs_rel"] = abs_rel_sum / valid
        for label in ("1p25", "1p25_2", "1p25_3"):
            count = sum(float(row.get(f"{prefix}_delta_{label}_count", 0.0)) for row in rows)
            summary[f"{prefix}_delta_{label}"] = count / valid
    return summary


def rank_samples(rows: list[dict[str, Any]], key: str, reverse: bool, limit: int = 10) -> list[dict[str, Any]]:
    candidates = [row for row in rows if key in row and math.isfinite(float(row[key]))]
    ranked = sorted(candidates, key=lambda row: float(row[key]), reverse=reverse)[:limit]
    fields = [
        "dataset",
        "split",
        "run",
        "dataset_index",
        "seq_name",
        "rgb_path",
        "depth_path",
        "loss",
        "loss_depth",
        "loss_camera",
        "loss_camera_t",
        "valid_fraction",
        "camera_pose_pair_count",
        "camera_pose_translation_deg_median",
        "camera_pose_rotation_deg_median",
        "camera_pose_auc30",
        "depth_irls_abs_rel",
        "depth_irls_delta_1p25",
        "depth_abs_rel",
        "depth_delta_1p25",
    ]
    return [{field: row.get(field) for field in fields if field in row} for row in ranked]


def read_train_loss_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("loss") or row.get("loss_objective") or row.get("total_loss")
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                rows.append(value)
    if not rows:
        return None
    return {
        "path": str(path),
        "all": summarize_values(rows),
        "last_1000": summarize_values(rows[-1000:]),
        "last_5000": summarize_values(rows[-5000:]),
    }


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def initialize_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    append_csv_rows(path, fieldnames, [row])


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def scalar_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return str(value.detach().cpu().item())
        return str(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return scalar_string(value[0], default=default) if value else default
    return str(value)


def scalar_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().cpu().reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return scalar_float(value[0], default=default) if value else float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def normalize_bins_label(value: str | None) -> str:
    if value in (None, "", "all"):
        return "all"
    return str(value).replace(",", "_").replace(" ", "")


if __name__ == "__main__":
    main()
