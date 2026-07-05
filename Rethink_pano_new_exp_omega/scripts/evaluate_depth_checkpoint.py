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
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    load_checkpoint,
    masked_depth_loss,
    move_batch_to_device,
    normalize_camera_supervision_args,
    normalize_pano_sampling_args,
    normalize_pred_depth_scale_args,
    normalize_training_stages,
    parse_args as parse_training_args,
    resolve_device,
    sample_depth_targets,
    set_seed,
    unwrap_model,
)


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
    parser.add_argument("--hard-limit", type=int, default=100, help="Extra hard-bin val samples. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic sample seed.")
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
    model = build_eval_model(train_args, args.checkpoint, checkpoint_payload, device)
    model.eval()

    runs: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    runs.append(
        evaluate_run(
            name=f"{args.split}_{normalize_bins_label(args.curriculum_bins)}_{args.limit}",
            base_args=train_args,
            model=model,
            device=device,
            split=args.split,
            curriculum_bins=args.curriculum_bins,
            limit=args.limit,
            seed=args.seed,
            num_workers=args.num_workers,
            progress=args.progress,
            per_sample_rows=per_sample_rows,
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
                seed=args.seed + 17,
                num_workers=args.num_workers,
                progress=args.progress,
                per_sample_rows=per_sample_rows,
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
            "pitch_degrees": str(train_args.pitch_degrees),
            "fov_degrees": float(train_args.fov_degrees),
        },
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv),
        "runs": runs,
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


def build_eval_model(
    args: argparse.Namespace,
    checkpoint: Path,
    payload: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    base_model = build_model(args).to(device)
    load_checkpoint(base_model, checkpoint, strict=False)

    adapter_state = payload.get("adapter_state")
    needs_adapter = bool(args.learn_pred_depth_scale) or args.depth_residual_mode != "none" or adapter_state is not None
    if not needs_adapter:
        return base_model

    model = DepthPredictionAdapter(
        base_model,
        initial_scale=float(args.pred_depth_scale),
        learn_scale=bool(args.learn_pred_depth_scale),
        residual_mode=str(args.depth_residual_mode),
        residual_hidden=int(args.depth_residual_hidden),
        residual_max_log=float(args.depth_residual_max_log),
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
    seed: int,
    num_workers: int,
    progress: bool,
    per_sample_rows: list[dict[str, Any]],
    shard_rank: int = 0,
    num_shards: int = 1,
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
    indices = sample_indices(len(dataset), limit, seed)
    selected_indices = indices[int(shard_rank) :: int(num_shards)]
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
                predictions = model(pano_images=moved["pano_image"], return_sampler_output=True)
                pred_depth_scale = predictions.get(
                    "_pred_depth_scale",
                    predictions["depth"].new_tensor(float(eval_args.pred_depth_scale)),
                )
                pred_depth = predictions["depth"] * pred_depth_scale
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
                loss = loss_depth + float(eval_args.overlap_consistency_weight) * loss_overlap

            depth_metrics = compute_depth_metrics(pred_depth, target_depth, target_valid)
            row = {
                "run": name,
                "dataset_index": int(selected_indices[local_index]),
                "seq_name": scalar_string(batch.get("scene_name") or batch.get("sequence_name")),
                "rgb_path": scalar_string(batch.get("rgb_path")),
                "depth_path": scalar_string(batch.get("depth_path")),
                "quality_bin": scalar_string(batch.get("metadata_quality_bin"), default="unknown"),
                "loss": float(loss.detach().cpu()),
                "loss_depth": float(loss_depth.detach().cpu()),
                "loss_overlap": float(loss_overlap.detach().cpu()),
                "valid_fraction": float(target_valid.float().mean().detach().cpu()),
                "pred_depth_scale": float(pred_depth_scale.detach().float().cpu()),
                "metadata_valid_ratio": scalar_float(batch.get("metadata_valid_ratio")),
                "metadata_structure_score": scalar_float(batch.get("metadata_structure_score")),
                "sample_weight": scalar_float(batch.get("sample_weight"), default=1.0),
                **depth_metrics,
            }
            rows.append(row)
            per_sample_rows.append(row)

    return {
        "name": name,
        "split": split,
        "curriculum_bins": curriculum_bins or "all",
        "dataset_size": len(dataset),
        "requested_samples": int(limit),
        "candidate_samples": len(indices),
        "evaluated_samples": len(rows),
        "shard_rank": int(shard_rank),
        "num_shards": int(num_shards),
        "summary": summarize_values([row["loss"] for row in rows]),
        "depth_summary": summarize_values([row["loss_depth"] for row in rows]),
        "overlap_summary": summarize_values([row["loss_overlap"] for row in rows]),
        "valid_fraction_summary": summarize_values([row["valid_fraction"] for row in rows]),
        "depth_metric_summary": summarize_metric_rows(rows, DEPTH_METRIC_KEYS),
        "panovggt_metric_summary": summarize_panovggt_rows(rows),
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
            "depth_irls_abs_rel": "Abs Rel after per-sample IRLS scale alignment; closest to PanoVGGT depth table protocol.",
            "depth_irls_delta_1p25": "delta < 1.25 after per-sample IRLS scale alignment; closest to PanoVGGT depth table protocol.",
            "depth_irls_rmse": "RMSE after per-sample IRLS scale alignment.",
            "depth_abs_rel": "Raw-scale Abs Rel using the model/checkpoint predicted depth scale.",
            "depth_delta_1p25": "Raw-scale delta < 1.25 using the model/checkpoint predicted depth scale.",
            "depth_rmse": "Raw-scale RMSE using the model/checkpoint predicted depth scale.",
        },
        "macro_by_sample": summarize_metric_rows(rows, PANOVGGT_PRIMARY_METRICS),
        "micro_by_valid_pixel": summarize_panovggt_micro(rows),
    }


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
        "valid_fraction",
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
    fieldnames = [
        "run",
        "dataset_index",
        "seq_name",
        "rgb_path",
        "depth_path",
        "quality_bin",
        "loss",
        "loss_depth",
        "loss_overlap",
        "valid_fraction",
        "pred_depth_scale",
        "metadata_valid_ratio",
        "metadata_structure_score",
        "sample_weight",
        *DEPTH_METRIC_KEYS,
        *DEPTH_ACCUMULATOR_KEYS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
