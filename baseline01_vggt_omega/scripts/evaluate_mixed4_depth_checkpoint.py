#!/usr/bin/env python3
"""Evaluate a baseline01 VGGT-Omega checkpoint on mixed4 pano test splits.

This is the baseline-side counterpart of the LUNA mixed4 evaluator. It keeps
the ablation intentionally plain: load VGGTOmega, run pinhole windows from the
minimal pano datasets, apply the same current depth loss, and report
PanoVGGT-style single-pano depth metrics per dataset.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
LUNA_ROOT = REPO_ROOT.parent / "Rethink_pano_new_exp_omega"

for path in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.datasets.pano_minimal import PanoMinimalPinholeDataset  # noqa: E402
from loss import compute_depth_loss  # noqa: E402


DATASETS = [
    ("Panocity", "panocity", "test"),
    ("Matterport3D", "matterport3d", "test"),
    ("Stanford2D3DS", "stanford2d3ds", "test"),
    ("Structured3D", "structured3d", "test"),
]

PANOVGGT_TABLE3_MONOCULAR = {
    "Matterport3D": {"abs_rel": 0.0884, "delta_1p25": 0.9157},
    "Stanford2D3DS": {"abs_rel": 0.0711, "delta_1p25": 0.9392},
    "Structured3D": {"abs_rel": 0.0438, "delta_1p25": 0.9728},
    "Panocity": {"abs_rel": 0.0312, "delta_1p25": 0.9713},
}

PANOVGGT_TABLE3_MULTIVIEW = {
    "Matterport3D": {"abs_rel": 0.0840, "delta_1p25": 0.9266},
    "Stanford2D3DS": {"abs_rel": 0.0778, "delta_1p25": 0.9323},
    "Structured3D": {"abs_rel": 0.0400, "delta_1p25": 0.9870},
    "Panocity": {"abs_rel": 0.0196, "delta_1p25": 0.9812},
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

PANOVGGT_PRIMARY_METRICS = [
    "depth_irls_abs_rel",
    "depth_irls_delta_1p25",
    "depth_irls_rmse",
    "depth_abs_rel",
    "depth_delta_1p25",
    "depth_rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Hydra config name or yaml path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="baseline01 checkpoint.pt to evaluate.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON output.")
    parser.add_argument("--per-sample-csv", type=Path, default=None)
    parser.add_argument("--train-loss-csv", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override mixed4 dataset root.")
    parser.add_argument("--datasets", default="all", help="Comma list or all.")
    parser.add_argument("--limit-per-dataset", type=int, default=100, help="0 means full split.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default=None)
    parser.add_argument("--pred-depth-scale", type=float, default=None, help="Override cfg.loss.depth.pred_depth_scale.")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    set_seed(args.seed)
    cfg, config_label = load_hydra_config(args.config)
    device = resolve_device(args.device)
    amp_dtype = args.amp_dtype or str(cfg.optim.amp.amp_dtype if cfg.optim.amp.enabled else "none")

    model = build_model(cfg, args.checkpoint, device)
    depth_conf = OmegaConf.to_container(cfg.loss.depth, resolve=True)
    if not isinstance(depth_conf, dict):
        raise TypeError("cfg.loss.depth must resolve to a mapping")
    if args.pred_depth_scale is not None:
        depth_conf["pred_depth_scale"] = float(args.pred_depth_scale)

    selected = select_datasets(args.datasets)
    metrics_lib = load_luna_metric_helpers()
    runs: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for dataset_index, (display_name, minimal_name, split) in enumerate(DATASETS):
        if minimal_name not in selected:
            continue
        dataset = build_eval_dataset(
            cfg=cfg,
            minimal_name=minimal_name,
            split=split,
            dataset_root=args.dataset_root,
        )
        run = evaluate_dataset(
            cfg=cfg,
            dataset=dataset,
            display_name=display_name,
            minimal_name=minimal_name,
            split=split,
            model=model,
            depth_conf=depth_conf,
            device=device,
            amp_dtype=amp_dtype,
            limit=args.limit_per_dataset,
            seed=args.seed + dataset_index * 1009,
            progress=args.progress,
            metrics_lib=metrics_lib,
            per_sample_rows=per_sample_rows,
            skipped_rows=skipped_rows,
        )
        runs.append(run)
        torch.cuda.empty_cache()

    result = {
        "config": config_label,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": int(args.seed),
        "limit_per_dataset": int(args.limit_per_dataset),
        "datasets": sorted(selected),
        "dataset_root": str(resolve_dataset_root(cfg, args.dataset_root)),
        "ablation": {
            "model": "baseline01 VGGTOmega full finetune",
            "uses_pano_or_luna_structure": False,
            "uses_camera_supervision": False,
            "loss": "current depth log_huber + valid/sample weighting + overlap consistency",
            "pred_depth_scale": float(depth_conf.get("pred_depth_scale", 1.0)),
        },
        "split_policy": {
            "Panocity": "test (PanoVGGT official split when cache was built with official split JSONs)",
            "Matterport3D": "test",
            "Stanford2D3DS": "test",
            "Structured3D": "test (official split when cache was built with official split files)",
        },
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv, metrics_lib),
        "runs": runs,
        "overall": summarize_runs(runs),
        "panovggt_depth_benchmark": summarize_panovggt_benchmark(runs, per_sample_rows, metrics_lib),
        "case_rankings": {
            "best_by_depth_irls_abs_rel": metrics_lib.rank_samples(per_sample_rows, "depth_irls_abs_rel", reverse=False, limit=20),
            "worst_by_depth_irls_abs_rel": metrics_lib.rank_samples(per_sample_rows, "depth_irls_abs_rel", reverse=True, limit=20),
            "best_by_depth_irls_delta_1p25": metrics_lib.rank_samples(per_sample_rows, "depth_irls_delta_1p25", reverse=True, limit=20),
            "worst_by_depth_irls_delta_1p25": metrics_lib.rank_samples(per_sample_rows, "depth_irls_delta_1p25", reverse=False, limit=20),
            "worst_by_loss": metrics_lib.rank_samples(per_sample_rows, "loss", reverse=True, limit=20),
        },
        "skipped_samples": skipped_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_sample_csv is not None:
        write_per_sample_csv(args.per_sample_csv, per_sample_rows, metrics_lib)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def load_hydra_config(config: Path) -> tuple[Any, str]:
    if config.suffix in {".yaml", ".yml"}:
        config_path = config.expanduser().resolve()
        config_dir = config_path.parent
        config_name = config_path.stem
        label = str(config_path)
    else:
        config_dir = REPO_ROOT / "training" / "config"
        config_name = str(config)
        label = str(config)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    return cfg, label


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(raw)


def build_model(cfg: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = instantiate(cfg.model, _recursive_=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = extract_model_state(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] loaded checkpoint = {checkpoint}")
    print(f"[INFO] missing_keys = {len(missing)}; unexpected_keys = {len(unexpected)}")
    if unexpected:
        print(f"[INFO] unexpected_keys_sample = {list(unexpected)[:20]}")
    model.eval()
    return model


def extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint did not contain a state dict")
    state = dict(payload)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def build_eval_dataset(cfg: Any, minimal_name: str, split: str, dataset_root: Path | None) -> PanoMinimalPinholeDataset:
    ds_conf = cfg.data.train.dataset.dataset_configs[0]
    common_conf = OmegaConf.create(OmegaConf.to_container(cfg.data.train.common_config, resolve=True))
    common_conf.training = False
    common_conf.inside_random = False

    root = resolve_dataset_root(cfg, dataset_root)
    return PanoMinimalPinholeDataset(
        common_conf=common_conf,
        split=split,
        root=str(root),
        datasets=minimal_name,
        dataset_sampling_weights=None,
        len_train=int(ds_conf.get("len_train", 1000000)),
        len_test=1000000,
        output_depth_scale=float(ds_conf.get("output_depth_scale", 1000.0)),
        invalid_depth_value=ds_conf.get("invalid_depth_value", 65535.0),
        depth_max_m=float(ds_conf.get("depth_max_m", 80.0)),
        num_yaw=int(ds_conf.get("num_yaw", 8)),
        pitch_degrees=float(ds_conf.get("pitch_degrees", -15.0)),
        fov_degrees=float(ds_conf.get("fov_degrees", 75.0)),
        train_split_fraction=float(ds_conf.get("train_split_fraction", 0.95)),
        split_seed=int(ds_conf.get("split_seed", 42)),
        metadata_path=None,
        bad_sample_list=ds_conf.get("bad_sample_list", None),
        curriculum_bins=None,
        use_metadata_weights=False,
    )


def resolve_dataset_root(cfg: Any, dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        root = dataset_root.expanduser()
    else:
        root = Path(str(cfg.data.train.dataset.dataset_configs[0].root)).expanduser()
    return root if root.is_absolute() else (REPO_ROOT / root).resolve()


def evaluate_dataset(
    *,
    cfg: Any,
    dataset: PanoMinimalPinholeDataset,
    display_name: str,
    minimal_name: str,
    split: str,
    model: torch.nn.Module,
    depth_conf: dict[str, Any],
    device: torch.device,
    amp_dtype: str,
    limit: int,
    seed: int,
    progress: bool,
    metrics_lib: Any,
    per_sample_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    item_count = len(getattr(dataset, "items", []))
    indices = metrics_lib.sample_indices(item_count, limit, seed)
    rows: list[dict[str, Any]] = []
    amp_enabled = device.type == "cuda" and amp_dtype != "none"
    torch_amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float32
    run_name = f"{display_name}_{split}_{int(limit)}"
    iterator = tqdm(indices, desc=f"validate {run_name}", dynamic_ncols=True) if progress else indices

    with torch.no_grad():
        for index in iterator:
            item = dataset.items[int(index)]
            try:
                sample = dataset.get_data(
                    seq_index=int(index),
                    img_per_seq=int(cfg.data.train.dataset.dataset_configs[0].get("num_yaw", 8)),
                    aspect_ratio=1.0,
                )
                batch = sample_to_batch(sample, device)
                with torch.autocast(device_type=device.type, dtype=torch_amp_dtype, enabled=amp_enabled):
                    predictions = model(images=batch["images"])
                    loss_dict = compute_depth_loss(predictions, batch, **depth_conf)
                pred_scale = float(depth_conf.get("pred_depth_scale", 1.0))
                pred_depth = predictions["depth"].detach().float() * pred_scale
                target_depth = batch["depths"].detach().float()[..., None]
                target_valid = batch["point_masks"].detach().bool()[..., None]
                depth_metrics = metrics_lib.compute_depth_metrics(pred_depth, target_depth, target_valid)
                loss_total = scalar_tensor(loss_dict.get("loss_reg_depth", 0.0))
                loss_depth = scalar_tensor(loss_dict.get("loss_log_l1_depth", loss_dict.get("loss_reg_depth", 0.0)))
                loss_overlap = scalar_tensor(loss_dict.get("loss_overlap_depth", 0.0))
                row = {
                    "dataset": display_name,
                    "minimal_dataset": minimal_name,
                    "split": split,
                    "run": run_name,
                    "dataset_index": int(index),
                    "seq_name": str(sample.get("seq_name", item.get("name", ""))),
                    "rgb_path": str(item.get("rgb_path", "")),
                    "depth_path": str(item.get("depth_path", "")),
                    "quality_bin": str(item.get("metadata_quality_bin", sample.get("metadata_quality_bin", "unknown"))),
                    "loss": loss_total,
                    "loss_depth": loss_depth,
                    "loss_overlap": loss_overlap,
                    "valid_fraction": float(target_valid.float().mean().detach().cpu()),
                    "pred_depth_scale": pred_scale,
                    "metadata_valid_ratio": sample_scalar(sample.get("metadata_valid_ratio"), 1.0),
                    "metadata_structure_score": sample_scalar(sample.get("metadata_structure_score"), 0.0),
                    "sample_weight": sample_scalar(sample.get("sample_weight"), 1.0),
                    **depth_metrics,
                }
                rows.append(row)
                per_sample_rows.append(row)
            except Exception as exc:
                skipped = {
                    "dataset": display_name,
                    "split": split,
                    "dataset_index": int(index),
                    "seq_name": str(item.get("name", "")),
                    "rgb_path": str(item.get("rgb_path", "")),
                    "depth_path": str(item.get("depth_path", "")),
                    "error": repr(exc),
                }
                skipped_rows.append(skipped)
                print(f"[WARN] skipped {display_name} index={index}: {exc}")

    return {
        "name": run_name,
        "dataset": display_name,
        "minimal_dataset": minimal_name,
        "split": split,
        "dataset_size": item_count,
        "requested_samples": int(limit),
        "candidate_samples": len(indices),
        "evaluated_samples": len(rows),
        "skipped_samples": sum(1 for row in skipped_rows if row.get("dataset") == display_name and row.get("split") == split),
        "summary": metrics_lib.summarize_values([row["loss"] for row in rows]),
        "depth_summary": metrics_lib.summarize_values([row["loss_depth"] for row in rows]),
        "overlap_summary": metrics_lib.summarize_values([row["loss_overlap"] for row in rows]),
        "valid_fraction_summary": metrics_lib.summarize_values([row["valid_fraction"] for row in rows]),
        "depth_metric_summary": metrics_lib.summarize_metric_rows(rows, metrics_lib.DEPTH_METRIC_KEYS),
        "panovggt_metric_summary": metrics_lib.summarize_panovggt_rows(rows),
        "best_samples": {
            "by_loss": metrics_lib.rank_samples(rows, "loss", reverse=False),
            "by_depth_irls_abs_rel": metrics_lib.rank_samples(rows, "depth_irls_abs_rel", reverse=False),
            "by_depth_irls_delta_1p25": metrics_lib.rank_samples(rows, "depth_irls_delta_1p25", reverse=True),
        },
        "worst_samples": {
            "by_loss": metrics_lib.rank_samples(rows, "loss", reverse=True),
            "by_depth_irls_abs_rel": metrics_lib.rank_samples(rows, "depth_irls_abs_rel", reverse=True),
            "by_depth_irls_delta_1p25": metrics_lib.rank_samples(rows, "depth_irls_delta_1p25", reverse=False),
        },
    }


def sample_to_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    images = torch.from_numpy(np.stack(sample["images"]).astype(np.float32)).contiguous()
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC images, got {tuple(images.shape)}")
    images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255.0).unsqueeze(0)
    depths = torch.from_numpy(np.stack(sample["depths"]).astype(np.float32)).unsqueeze(0)
    point_masks = torch.from_numpy(np.stack(sample["point_masks"]).astype(bool)).unsqueeze(0)
    frame_count = int(images.shape[1])
    batch = {
        "images": images.to(device, non_blocking=True),
        "depths": depths.to(device, non_blocking=True),
        "point_masks": point_masks.to(device, non_blocking=True),
    }
    for key in ("sample_weight", "metadata_valid_ratio", "metadata_structure_score"):
        value = sample.get(key)
        if value is None:
            value = np.ones((frame_count,), dtype=np.float32)
        batch[key] = torch.as_tensor(value, dtype=torch.float32).reshape(1, frame_count).to(device, non_blocking=True)
    return batch


def load_luna_metric_helpers() -> Any:
    return SimpleNamespace(
        DEPTH_METRIC_KEYS=DEPTH_METRIC_KEYS,
        DEPTH_ACCUMULATOR_KEYS=DEPTH_ACCUMULATOR_KEYS,
        PANOVGGT_PRIMARY_METRICS=PANOVGGT_PRIMARY_METRICS,
        compute_depth_metrics=compute_depth_metrics,
        rank_samples=rank_samples,
        sample_indices=sample_indices,
        summarize_metric_rows=summarize_metric_rows,
        summarize_panovggt_rows=summarize_panovggt_rows,
        summarize_values=summarize_values,
    )


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
    aligned_pred = pred_values * irls_scale
    aligned_metrics = depth_metrics_from_values(aligned_pred, target_values, prefix="depth_irls")
    return {
        **raw_metrics,
        **depth_metric_accumulators(pred_values, target_values, prefix="depth"),
        "depth_irls_scale": float(irls_scale.cpu()),
        **aligned_metrics,
        **depth_metric_accumulators(aligned_pred, target_values, prefix="depth_irls"),
        "depth_valid_pixels": int(valid.sum().item()),
    }


def depth_metrics_from_values(pred_values: torch.Tensor, target_values: torch.Tensor, prefix: str) -> dict[str, float]:
    diff = pred_values - target_values
    abs_diff = diff.abs()
    ratio = torch.maximum(pred_values / target_values, target_values / pred_values)
    return {
        f"{prefix}_mae": float(abs_diff.mean().cpu()),
        f"{prefix}_rmse": float(torch.sqrt(diff.square().mean()).cpu()),
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


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(run.get("evaluated_samples", 0)) for run in runs)
    if total <= 0:
        return {"evaluated_samples": 0}
    return {
        "evaluated_samples": total,
        "weighted_loss_mean": weighted_run_mean(runs, "summary"),
        "weighted_depth_loss_mean": weighted_run_mean(runs, "depth_summary"),
        "weighted_overlap_loss_mean": weighted_run_mean(runs, "overlap_summary"),
    }


def weighted_run_mean(runs: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denom = 0
    for run in runs:
        n = int(run.get("evaluated_samples", 0))
        mean = run.get(key, {}).get("mean")
        if n > 0 and mean is not None:
            numerator += float(mean) * n
            denom += n
    return numerator / denom if denom > 0 else None


def summarize_panovggt_benchmark(runs: list[dict[str, Any]], rows: list[dict[str, Any]], metrics_lib: Any) -> dict[str, Any]:
    per_dataset = {}
    for run in runs:
        dataset = str(run.get("dataset", "unknown"))
        metrics = run.get("panovggt_metric_summary", {})
        micro = metrics.get("micro_by_valid_pixel", {})
        macro = metrics.get("macro_by_sample", {})
        ours_abs_rel = micro.get("depth_irls_abs_rel")
        ours_delta = micro.get("depth_irls_delta_1p25")
        reference_mono = PANOVGGT_TABLE3_MONOCULAR.get(dataset)
        reference_multi = PANOVGGT_TABLE3_MULTIVIEW.get(dataset)
        per_dataset[dataset] = {
            "evaluated_samples": run.get("evaluated_samples", 0),
            "split": run.get("split"),
            "ours_micro_irls_scale_aligned": {
                "abs_rel": ours_abs_rel,
                "delta_1p25": ours_delta,
                "rmse": micro.get("depth_irls_rmse"),
            },
            "ours_macro_by_sample_irls_scale_aligned": {
                "abs_rel": macro.get("depth_irls_abs_rel", {}).get("mean"),
                "delta_1p25": macro.get("depth_irls_delta_1p25", {}).get("mean"),
                "rmse": macro.get("depth_irls_rmse", {}).get("mean"),
            },
            "ours_raw_scale": {
                "abs_rel": micro.get("depth_abs_rel"),
                "delta_1p25": micro.get("depth_delta_1p25"),
                "rmse": micro.get("depth_rmse"),
            },
            "panovggt_table3_monocular": reference_mono,
            "panovggt_table3_multiview": reference_multi,
            "beats_panovggt_monocular": compare_to_reference(ours_abs_rel, ours_delta, reference_mono),
            "beats_panovggt_multiview": compare_to_reference(ours_abs_rel, ours_delta, reference_multi),
        }
    return {
        "paper_protocol_note": (
            "PanoVGGT Table 3 reports Abs Rel and delta<1.25 after IRLS scale normalization. "
            "Use ours_micro_irls_scale_aligned for the closest automatic comparison; raw-scale metrics are also retained."
        ),
        "per_dataset": per_dataset,
        "overall_micro_irls_scale_aligned": metrics_lib.summarize_panovggt_rows(rows).get("micro_by_valid_pixel", {}),
        "overall_macro_by_sample": metrics_lib.summarize_metric_rows(rows, metrics_lib.PANOVGGT_PRIMARY_METRICS),
        "panovggt_table3_monocular_macro_reference": macro_reference(PANOVGGT_TABLE3_MONOCULAR),
        "panovggt_table3_multiview_macro_reference": macro_reference(PANOVGGT_TABLE3_MULTIVIEW),
    }


def compare_to_reference(ours_abs_rel: Any, ours_delta: Any, reference: dict[str, float] | None) -> dict[str, Any] | None:
    if reference is None or ours_abs_rel is None or ours_delta is None:
        return None
    return {
        "abs_rel": float(ours_abs_rel) < float(reference["abs_rel"]),
        "delta_1p25": float(ours_delta) > float(reference["delta_1p25"]),
        "both": float(ours_abs_rel) < float(reference["abs_rel"]) and float(ours_delta) > float(reference["delta_1p25"]),
    }


def macro_reference(reference: dict[str, dict[str, float]]) -> dict[str, float]:
    values = list(reference.values())
    return {
        "abs_rel": sum(item["abs_rel"] for item in values) / len(values),
        "delta_1p25": sum(item["delta_1p25"] for item in values) / len(values),
    }


def select_datasets(raw: str) -> set[str]:
    aliases = {
        "panocity": "panocity",
        "pano_city": "panocity",
        "matterport3d": "matterport3d",
        "matterport": "matterport3d",
        "mp3d": "matterport3d",
        "stanford2d3ds": "stanford2d3ds",
        "stanford": "stanford2d3ds",
        "2d3ds": "stanford2d3ds",
        "s2d3ds": "stanford2d3ds",
        "structured3d": "structured3d",
        "s3d": "structured3d",
    }
    if raw is None or str(raw).strip().lower() in {"", "all", "*"}:
        return {minimal for _, minimal, _ in DATASETS}
    selected: set[str] = set()
    for token in str(raw).split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown dataset {token!r}; expected all or one of {sorted(aliases)}")
        selected.add(aliases[key])
    return selected


def read_train_loss_reference(path: Path | None, metrics_lib: Any) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
        "all": metrics_lib.summarize_values(rows),
        "last_1000": metrics_lib.summarize_values(rows[-1000:]),
        "last_5000": metrics_lib.summarize_values(rows[-5000:]),
    }


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]], metrics_lib: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "minimal_dataset",
        "split",
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
        *metrics_lib.DEPTH_METRIC_KEYS,
        *metrics_lib.DEPTH_ACCUMULATOR_KEYS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar_tensor(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def sample_scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    main()
