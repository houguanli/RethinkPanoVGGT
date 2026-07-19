#!/usr/bin/env python3
"""Validate a VGGT-Omega/LUNA depth checkpoint on held-out pano samples."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    resolve_device,
    sample_depth_targets,
    set_seed,
    shared_frame_point_loss,
    omega_y_up_pose_to_official_y_down,
    unwrap_model,
)
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat  # noqa: E402


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
    *DEPTH_METRIC_KEYS,
    *DEPTH_ACCUMULATOR_KEYS,
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
            )
        )

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "dataset_root": str(train_args.dataset_root),
        "sampler": {
            "window_size": int(train_args.window_size),
            "patch_size": int(train_args.patch_size),
            "num_yaw": int(train_args.num_yaw),
            "pano_min_count": int(getattr(train_args, "pano_min_count", 1)),
            "pano_max_count": int(getattr(train_args, "pano_max_count", 1)),
            "pitch_degrees": str(train_args.pitch_degrees),
            "fov_degrees": float(train_args.fov_degrees),
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
    for key in ("learn_pred_depth_scale", "depth_residual_mode", "depth_residual_hidden", "depth_residual_max_log"):
        if key in ckpt_args and ckpt_args[key] is not None:
            setattr(args, key, ckpt_args[key])
    for key in ("window_size", "patch_size", "num_yaw", "pitch_degrees", "fov_degrees"):
        if key in ckpt_args and ckpt_args[key] is not None:
            setattr(args, key, ckpt_args[key])


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
    indices, sampling_info = select_eval_indices(dataset, limit, seed, sample_policy, limit_fraction=limit_fraction)
    selected_indices = indices[int(shard_rank) :: int(num_shards)]
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
            "processed_samples": 0,
            "shard_rank": int(shard_rank),
            "num_shards": int(num_shards),
            "updated_at": time.time(),
        },
    )
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    rows: list[dict[str, Any]] = []
    camera_pair_start = len(camera_pair_rows) if camera_pair_rows is not None else 0
    amp_enabled = device.type == "cuda" and eval_args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if eval_args.amp_dtype == "bfloat16" else torch.float32
    iterator = tqdm(loader, desc=f"validate {name}", dynamic_ncols=True) if progress else loader
    with torch.no_grad():
        for local_index, batch in enumerate(iterator):
            moved = move_batch_to_device(batch, device)
            sampler_model = unwrap_model(model)
            target_depth, target_valid = sample_depth_targets(
                sampler_model,
                moved["pano_depth"],
                source_depth_semantics=eval_args.gt_depth_semantics,
                max_range_depth=eval_args.depth_max_m,
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
                depth_metrics = compute_depth_metrics(pred_depth_base, target_depth, target_valid)
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
                "dataset_index": int(selected_indices[local_index]),
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
                **pose_metrics,
                **depth_metrics,
            }
            for pair_row in pose_pair_rows:
                pair_row.update(
                    {
                        **(row_context or {}),
                        "run": name,
                        "dataset_index": int(selected_indices[local_index]),
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
            processed = local_index + 1
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
                        "last_loss": row["loss"],
                        "last_depth_irls_abs_rel": row["depth_irls_abs_rel"],
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

    run_camera_pair_rows = (
        camera_pair_rows[camera_pair_start:] if camera_pair_rows is not None else []
    )
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


def compute_depth_metrics(pred_depth: torch.Tensor, target_depth: torch.Tensor, target_valid: torch.Tensor) -> dict[str, float]:
    pred = pred_depth.detach().float()
    target = target_depth.detach().float()
    valid = target_valid.detach().bool() & torch.isfinite(pred) & torch.isfinite(target) & (target > 0)
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
    raw_metrics = depth_metrics_from_values(pred_values, target_values, prefix="depth")
    irls_scale = fit_irls_scale(pred_values, target_values)
    aligned_metrics = depth_metrics_from_values(pred_values * irls_scale, target_values, prefix="depth_irls")
    return {
        **raw_metrics,
        **depth_metric_accumulators(pred_values, target_values, prefix="depth"),
        "depth_irls_scale": float(irls_scale.cpu()),
        **aligned_metrics,
        **depth_metric_accumulators(pred_values * irls_scale, target_values, prefix="depth_irls"),
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


def depth_metrics_from_values(pred_values: torch.Tensor, target_values: torch.Tensor, prefix: str) -> dict[str, float]:
    diff = pred_values - target_values
    abs_diff = diff.abs()
    ratio = torch.maximum(pred_values / target_values, target_values / pred_values)
    return {
        f"{prefix}_mae": float(abs_diff.mean().cpu()),
        f"{prefix}_rmse": float(torch.sqrt((diff.square()).mean()).cpu()),
        f"{prefix}_abs_rel": float((abs_diff / target_values).mean().cpu()),
        f"{prefix}_delta_1p25": float((ratio < 1.25).float().mean().cpu()),
        f"{prefix}_delta_1p25_2": float((ratio < 1.25**2).float().mean().cpu()),
        f"{prefix}_delta_1p25_3": float((ratio < 1.25**3).float().mean().cpu()),
    }


def depth_metric_accumulators(pred_values: torch.Tensor, target_values: torch.Tensor, prefix: str) -> dict[str, float]:
    diff = pred_values - target_values
    abs_diff = diff.abs()
    ratio = torch.maximum(pred_values / target_values, target_values / pred_values)
    return {
        f"{prefix}_abs_error_sum": float(abs_diff.sum().cpu()),
        f"{prefix}_sq_error_sum": float(diff.square().sum().cpu()),
        f"{prefix}_abs_rel_sum": float((abs_diff / target_values).sum().cpu()),
        f"{prefix}_delta_1p25_count": float((ratio < 1.25).float().sum().cpu()),
        f"{prefix}_delta_1p25_2_count": float((ratio < 1.25**2).float().sum().cpu()),
        f"{prefix}_delta_1p25_3_count": float((ratio < 1.25**3).float().sum().cpu()),
    }


def fit_irls_scale(pred_values: torch.Tensor, target_values: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    scale = torch.median(target_values / pred_values).clamp_min(1e-6)
    for _ in range(iterations):
        residual = scale * pred_values - target_values
        weights = 1.0 / residual.abs().clamp_min(1e-3)
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
        "metric_meaning": {
            "depth_irls_abs_rel": "Abs Rel after per-sample robust scale-only alignment; this is the primary PanoVGGT-style depth comparison.",
            "depth_irls_delta_1p25": "delta < 1.25 after per-sample robust scale-only alignment; this is the primary PanoVGGT-style depth comparison.",
            "depth_irls_rmse": "RMSE after per-sample robust scale-only alignment.",
            "depth_abs_rel": "Raw-scale Abs Rel using the model/checkpoint predicted depth scale.",
            "depth_delta_1p25": "Raw-scale delta < 1.25 using the model/checkpoint predicted depth scale.",
            "depth_rmse": "Raw-scale RMSE using the model/checkpoint predicted depth scale.",
        },
        "macro_by_sample": summarize_metric_rows(rows, PANOVGGT_PRIMARY_METRICS),
        "micro_by_valid_pixel": summarize_panovggt_micro(rows),
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
    valid = sum(float(row.get("depth_valid_pixels", 0.0)) for row in rows)
    if valid <= 0:
        return {"depth_valid_pixels": 0}
    summary: dict[str, Any] = {"depth_valid_pixels": int(valid)}
    for prefix in ("depth", "depth_irls"):
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


def normalize_bins_label(value: str | None) -> str:
    if value in (None, "", "all"):
        return "all"
    return str(value).replace(",", "_").replace(" ", "")


if __name__ == "__main__":
    main()
