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
import traceback
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
GIT_ROOT = REPO_ROOT.parent
TRAINING_ROOT = REPO_ROOT / "training"
LUNA_ROOT = REPO_ROOT.parent / "Rethink_pano_new_exp_omega"

for path in (str(GIT_ROOT), str(REPO_ROOT), str(TRAINING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation_common.erp_depth import splat_window_z_depth_to_erp  # noqa: E402

from data.datasets.pano_minimal import (  # noqa: E402
    PanoMinimalMultiPanoPinholeDataset,
    PanoMinimalPinholeDataset,
)
from loss import compute_camera_loss, compute_depth_loss  # noqa: E402
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch  # noqa: E402
from vggt_omega.utils.lora import apply_lora_to_model  # noqa: E402
from vggt_omega.utils.geometry import closed_form_inverse_se3  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.rotation import mat_to_quat  # noqa: E402


DATASETS = [
    ("Panocity", "panocity", "test"),
    ("Matterport3D", "matterport3d", "test"),
    ("Stanford2D3DS", "stanford2d3ds", "test"),
    ("Structured3D", "structured3d", "test"),
]

PANOVGGT_EVAL_PANO_COUNTS = {
    "panocity": 10,
    "matterport3d": 3,
    "stanford2d3ds": 3,
    "structured3d": 3,
}

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

ERP_DEPTH_METRIC_KEYS = [f"erp_{key}" for key in DEPTH_METRIC_KEYS]
ERP_DEPTH_ACCUMULATOR_KEYS = [f"erp_{key}" for key in DEPTH_ACCUMULATOR_KEYS]
ERP_COVERAGE_KEYS = ["erp_coverage_fraction", "erp_common_valid_fraction"]

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

CAMERA_PAIR_CSV_FIELDS = [
    "dataset",
    "minimal_dataset",
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

PANOVGGT_PRIMARY_METRICS = [
    "depth_irls_abs_rel",
    "depth_irls_delta_1p25",
    "depth_irls_rmse",
    "depth_abs_rel",
    "depth_delta_1p25",
    "depth_rmse",
]

PER_SAMPLE_CSV_FIELDS = [
    "dataset",
    "minimal_dataset",
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
    "loss_camera",
    "loss_T",
    "loss_R",
    "loss_FL",
    "camera_valid_fraction",
    *CAMERA_POSE_SAMPLE_KEYS,
    "valid_fraction",
    "pred_depth_scale",
    "metadata_valid_ratio",
    "metadata_structure_score",
    "sample_weight",
    *DEPTH_METRIC_KEYS,
    *DEPTH_ACCUMULATOR_KEYS,
    *ERP_COVERAGE_KEYS,
    *ERP_DEPTH_METRIC_KEYS,
    *ERP_DEPTH_ACCUMULATOR_KEYS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Hydra config name or yaml path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="baseline01 checkpoint.pt to evaluate.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON output.")
    parser.add_argument("--per-sample-csv", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-sample/camera-pair CSV files and skip completed dataset indices.",
    )
    parser.add_argument("--train-loss-csv", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override mixed4 dataset root.")
    parser.add_argument("--datasets", default="all", help="Comma list or all.")
    parser.add_argument("--limit-per-dataset", type=int, default=100, help="0 means full split.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sample-policy",
        choices=["anchor"],
        default="anchor",
        help="Evaluate each selected test item as an anchor-centered neighborhood.",
    )
    parser.add_argument("--eval-max-panos", type=int, default=0, help="Clamp eval multi-pano input length. Use 0 to keep config pano_max_count.")
    parser.add_argument(
        "--pano-count-policy",
        choices=["config", "panovggt"],
        default="config",
        help="Use config pano counts or PanoVGGT eval counts: Panocity=10, indoor datasets=3.",
    )
    parser.add_argument(
        "--dataset-pano-counts",
        default="",
        help="Optional comma list name:count overriding --pano-count-policy, e.g. panocity:10,matterport3d:3.",
    )
    parser.add_argument("--windows-per-pano", type=int, default=0, help="Override windows per pano for multi-pano eval. Use 0 to keep config.")
    parser.add_argument("--camera-pair-csv", type=Path, default=None, help="Optional streaming PanoVGGT-style camera pair CSV path.")
    parser.add_argument("--camera-pose-trans-norm-thresh", type=float, default=1e-2, help="GT baseline threshold for translation-angle camera eval.")
    parser.add_argument(
        "--erp-latitude-limit-deg",
        type=float,
        default=75.0,
        help="Covered-ERP metric latitude limit; polar caps outside +/- this value are excluded.",
    )
    parser.add_argument(
        "--camera-eval-max-panos",
        type=int,
        default=0,
        help="Maximum pano views used for PanoVGGT camera metrics. Use 0 to evaluate all input panos.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default=None)
    parser.add_argument("--pred-depth-scale", type=float, default=None, help="Override cfg.loss.depth.pred_depth_scale.")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--fail-fast", action="store_true", help="Raise the first sample error with a full traceback.")
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
    dataset_pano_counts = resolve_dataset_pano_counts(args.pano_count_policy, args.dataset_pano_counts)
    metrics_lib = load_luna_metric_helpers()
    runs: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    camera_pair_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    camera_pair_csv = args.camera_pair_csv
    if camera_pair_csv is None and args.per_sample_csv is not None:
        camera_pair_csv = args.per_sample_csv.with_name(f"{args.per_sample_csv.stem}_camera_pairs.csv")
    completed_sample_keys: set[tuple[str, str, int]] = set()
    if args.resume:
        per_sample_rows, camera_pair_rows, completed_sample_keys = load_resume_rows(
            args.per_sample_csv,
            camera_pair_csv,
        )
        print(
            f"[resume] loaded samples={len(per_sample_rows)} "
            f"camera_pairs={len(camera_pair_rows)} completed_keys={len(completed_sample_keys)}"
        )
    else:
        if args.per_sample_csv is not None:
            initialize_csv(args.per_sample_csv, PER_SAMPLE_CSV_FIELDS)
        if camera_pair_csv is not None:
            initialize_csv(camera_pair_csv, CAMERA_PAIR_CSV_FIELDS)

    for dataset_index, (display_name, minimal_name, split) in enumerate(DATASETS):
        if minimal_name not in selected:
            continue
        dataset = build_eval_dataset(
            cfg=cfg,
            minimal_name=minimal_name,
            split=split,
            dataset_root=args.dataset_root,
            eval_max_panos=args.eval_max_panos,
            dataset_pano_count=dataset_pano_counts.get(minimal_name),
            windows_per_pano=args.windows_per_pano,
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
            per_sample_csv=args.per_sample_csv,
            camera_pair_rows=camera_pair_rows,
            camera_pair_csv=camera_pair_csv,
            skipped_rows=skipped_rows,
            camera_pose_trans_norm_thresh=args.camera_pose_trans_norm_thresh,
            camera_eval_max_panos=args.camera_eval_max_panos,
            erp_latitude_limit_deg=args.erp_latitude_limit_deg,
            normalize_scene_scale=bool(cfg.get("normalize_scene_scale", False)),
            fail_fast=bool(args.fail_fast),
            completed_sample_keys=completed_sample_keys,
        )
        runs.append(run)
        torch.cuda.empty_cache()

    result = {
        "config": config_label,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": int(args.seed),
        "limit_per_dataset": int(args.limit_per_dataset),
        "resumed": bool(args.resume),
        "datasets": sorted(selected),
        "sample_policy": str(args.sample_policy),
        "pano_count_policy": str(args.pano_count_policy),
        "dataset_pano_counts": {key: int(value) for key, value in sorted(dataset_pano_counts.items())},
        "camera_eval_max_panos": int(args.camera_eval_max_panos),
        "erp_latitude_limit_deg": float(args.erp_latitude_limit_deg),
        "dataset_root": str(resolve_dataset_root(cfg, args.dataset_root)),
        "ablation": {
            "model": "baseline01 VGGTOmega full finetune on flattened multi-pano pinhole windows",
            "uses_pano_or_luna_structure": False,
            "uses_camera_supervision": True,
            "loss": "VGGT-Omega depth/camera losses on pano-cropped multi-view groups",
            "pred_depth_scale": float(depth_conf.get("pred_depth_scale", 1.0)),
            "normalize_scene_scale": bool(cfg.get("normalize_scene_scale", False)),
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
        "panovggt_covered_erp_depth_benchmark": summarize_erp_panovggt_benchmark(runs, per_sample_rows, metrics_lib),
        "panovggt_camera_pose_summary": summarize_camera_pose_pairs(camera_pair_rows),
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
    lora_conf = cfg.get("lora", None)
    if lora_conf is not None and lora_conf.get("enabled", False):
        replaced = apply_lora_to_model(
            model,
            target_modules=lora_conf.get("target_modules", None),
            rank=lora_conf.get("rank", 8),
            alpha=lora_conf.get("alpha", 16.0),
            dropout=lora_conf.get("dropout", 0.0),
        )
        print(f"[INFO] enabled LoRA modules for eval = {len(replaced)}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = extract_model_state(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # LoRA modules are injected after the base model is moved, so migrate the
    # completed module tree once more before inference.
    model.to(device)
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


def build_eval_dataset(
    cfg: Any,
    minimal_name: str,
    split: str,
    dataset_root: Path | None,
    eval_max_panos: int = 0,
    dataset_pano_count: int | None = None,
    windows_per_pano: int = 0,
) -> PanoMinimalPinholeDataset | PanoMinimalMultiPanoPinholeDataset:
    ds_conf = cfg.data.train.dataset.dataset_configs[0]
    common_conf = OmegaConf.create(OmegaConf.to_container(cfg.data.train.common_config, resolve=True))
    common_conf.training = False
    common_conf.inside_random = False

    root = resolve_dataset_root(cfg, dataset_root)
    target = str(ds_conf.get("_target_", ""))
    if target.endswith("PanoMinimalMultiPanoPinholeDataset"):
        pano_max_count = int(dataset_pano_count or ds_conf.get("pano_max_count", 2))
        if int(eval_max_panos or 0) > 0:
            pano_max_count = max(1, min(pano_max_count, int(eval_max_panos)))
        pano_min_count = pano_max_count if dataset_pano_count else min(int(ds_conf.get("pano_min_count", pano_max_count)), pano_max_count)
        return PanoMinimalMultiPanoPinholeDataset(
            common_conf=common_conf,
            split=split,
            root=str(root),
            datasets=minimal_name,
            dataset_sampling_weights=None,
            len_train=int(ds_conf.get("len_train", 1000000)),
            len_test=int(ds_conf.get("len_test", 1000000)),
            output_depth_scale=float(ds_conf.get("output_depth_scale", 1000.0)),
            invalid_depth_value=ds_conf.get("invalid_depth_value", 65535.0),
            depth_max_m=float(ds_conf.get("depth_max_m", 80.0)),
            windows_per_pano=int(windows_per_pano or ds_conf.get("windows_per_pano", 4)),
            pitch_degrees=float(ds_conf.get("pitch_degrees", -15.0)),
            fov_degrees=float(ds_conf.get("fov_degrees", 75.0)),
            train_split_fraction=float(ds_conf.get("train_split_fraction", 0.95)),
            split_seed=int(ds_conf.get("split_seed", 42)),
            bad_sample_list=ds_conf.get("bad_sample_list", None),
            pano_sample_mode=str(ds_conf.get("pano_sample_mode", "fixed_neighborhood")),
            pano_min_count=pano_min_count,
            pano_max_count=pano_max_count,
            dataset_pano_counts=f"{minimal_name}:{pano_max_count}" if dataset_pano_count else ds_conf.get("dataset_pano_counts", None),
            grouping=str(ds_conf.get("grouping", "nearest")),
            camera_supervised_datasets=ds_conf.get("camera_supervised_datasets", "panocity,stanford2d3ds"),
            main_dataset_path=ds_conf.get("main_dataset_path", None),
        )
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
    per_sample_csv: Path | None,
    camera_pair_rows: list[dict[str, Any]],
    camera_pair_csv: Path | None,
    skipped_rows: list[dict[str, Any]],
    camera_pose_trans_norm_thresh: float,
    camera_eval_max_panos: int,
    erp_latitude_limit_deg: float,
    normalize_scene_scale: bool,
    fail_fast: bool,
    completed_sample_keys: set[tuple[str, str, int]],
) -> dict[str, Any]:
    runtime_dataset = getattr(dataset, "pano_dataset", None)
    source_items = getattr(runtime_dataset, "items", None) if runtime_dataset is not None else None
    if source_items is None:
        source_items = getattr(dataset, "items", None)
    # PanoMinimalMultiPanoPinholeDataset exposes a large synthetic training
    # length and wraps real anchor groups modulo the runtime dataset length.
    # Evaluation must visit each real anchor exactly once.
    if runtime_dataset is not None:
        item_count = len(runtime_dataset)
    else:
        item_count = len(source_items) if source_items is not None else len(dataset)
    indices = metrics_lib.sample_indices(item_count, limit, seed)
    selected_indices = {int(index) for index in indices}
    rows = [
        row
        for row in per_sample_rows
        if row.get("minimal_dataset") == minimal_name
        and row.get("split") == split
        and int(row["dataset_index"]) in selected_indices
    ]
    unexpected_indices = sorted(
        int(row["dataset_index"])
        for row in per_sample_rows
        if row.get("minimal_dataset") == minimal_name
        and row.get("split") == split
        and int(row["dataset_index"]) not in selected_indices
    )
    if unexpected_indices:
        raise ValueError(
            "Resume CSV contains indices outside the current sample selection. "
            f"Keep the original --limit-per-dataset/--seed settings; first unexpected indices: {unexpected_indices[:10]}"
        )
    resumed_row_count = len(rows)
    amp_enabled = device.type == "cuda" and amp_dtype != "none"
    torch_amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float32
    run_name = f"{display_name}_{split}_{int(limit)}"
    iterator = tqdm(indices, desc=f"validate {run_name}", dynamic_ncols=True) if progress else indices
    run_camera_pair_rows = [
        row
        for row in camera_pair_rows
        if row.get("minimal_dataset") == minimal_name
        and row.get("split") == split
        and int(row["dataset_index"]) in selected_indices
    ]
    ds_conf = cfg.data.train.dataset.dataset_configs[0]
    target = str(ds_conf.get("_target_", ""))
    if target.endswith("PanoMinimalMultiPanoPinholeDataset"):
        img_per_seq = int(getattr(dataset, "pano_max_count", ds_conf.get("pano_max_count", 2))) * int(
            getattr(dataset, "windows_per_pano", ds_conf.get("windows_per_pano", 4))
        )
    else:
        img_per_seq = int(ds_conf.get("num_yaw", 8))
    expected_pano_count = int(getattr(dataset, "pano_max_count", 1))
    expected_camera_pano_count = (
        expected_pano_count
        if int(camera_eval_max_panos) <= 0
        else min(expected_pano_count, int(camera_eval_max_panos))
    )
    for row in rows:
        if int(row["input_pano_count"]) != expected_pano_count:
            raise ValueError(
                "Resume CSV pano-count mismatch: "
                f"dataset={minimal_name} index={row['dataset_index']} "
                f"stored={row['input_pano_count']} current={expected_pano_count}"
            )
        if int(row["camera_eval_pano_count"]) != expected_camera_pano_count:
            raise ValueError(
                "Resume CSV camera pano-count mismatch: "
                f"dataset={minimal_name} index={row['dataset_index']} "
                f"stored={row['camera_eval_pano_count']} current={expected_camera_pano_count}"
            )

    with torch.no_grad():
        for index in iterator:
            sample_key = (minimal_name, split, int(index))
            if sample_key in completed_sample_keys:
                continue
            item = source_items[int(index)] if source_items is not None else {}
            try:
                sample = dataset.get_data(
                    seq_index=int(index),
                    img_per_seq=img_per_seq,
                    aspect_ratio=1.0,
                )
                batch = sample_to_batch(sample, device, normalize_scene_scale=normalize_scene_scale)
                with torch.autocast(device_type=device.type, dtype=torch_amp_dtype, enabled=amp_enabled):
                    predictions = model(images=batch["images"])
                    loss_dict = compute_depth_loss(predictions, batch, **depth_conf)
                    camera_loss_dict = compute_camera_loss(predictions, batch)
                pred_scale = float(depth_conf.get("pred_depth_scale", 1.0))
                pred_depth = predictions["depth"].detach().float() * pred_scale
                target_depth = batch["depths"].detach().float()[..., None]
                target_valid = batch["point_masks"].detach().bool()[..., None]
                depth_metrics = metrics_lib.compute_depth_metrics(pred_depth, target_depth, target_valid)
                erp_metrics = compute_covered_erp_depth_metrics(
                    pred_window_z=pred_depth,
                    batch=batch,
                    metrics_lib=metrics_lib,
                    latitude_limit_deg=float(erp_latitude_limit_deg),
                )
                pose_metrics, pose_pair_rows = compute_window0_camera_pose_metrics(
                    predictions=predictions,
                    batch=batch,
                    trans_norm_thresh=float(camera_pose_trans_norm_thresh),
                    max_panos=int(camera_eval_max_panos),
                )
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
                    "input_pano_count": int(batch["images"].shape[1])
                    // max(int(getattr(dataset, "windows_per_pano", 1)), 1),
                    "camera_eval_pano_count": (
                        int(batch["images"].shape[1])
                        // max(int(getattr(dataset, "windows_per_pano", 1)), 1)
                        if int(camera_eval_max_panos) <= 0
                        else min(
                            int(batch["images"].shape[1])
                            // max(int(getattr(dataset, "windows_per_pano", 1)), 1),
                            int(camera_eval_max_panos),
                        )
                    ),
                    "loss": loss_total,
                    "loss_depth": loss_depth,
                    "loss_overlap": loss_overlap,
                    "loss_camera": scalar_tensor(camera_loss_dict.get("loss_camera", 0.0)),
                    "loss_T": scalar_tensor(camera_loss_dict.get("loss_T", 0.0)),
                    "loss_R": scalar_tensor(camera_loss_dict.get("loss_R", 0.0)),
                    "loss_FL": scalar_tensor(camera_loss_dict.get("loss_FL", 0.0)),
                    "camera_valid_fraction": scalar_tensor(camera_loss_dict.get("camera_valid_fraction", 0.0)),
                    **pose_metrics,
                    "valid_fraction": float(target_valid.float().mean().detach().cpu()),
                    "pred_depth_scale": pred_scale,
                    "metadata_valid_ratio": sample_scalar(sample.get("metadata_valid_ratio"), 1.0),
                    "metadata_structure_score": sample_scalar(sample.get("metadata_structure_score"), 0.0),
                    "sample_weight": sample_scalar(sample.get("sample_weight"), 1.0),
                    **depth_metrics,
                    **erp_metrics,
                }
                rows.append(row)
                per_sample_rows.append(row)
                completed_sample_keys.add(sample_key)
                if per_sample_csv is not None:
                    append_csv_row(per_sample_csv, PER_SAMPLE_CSV_FIELDS, row)
                for pair_row in pose_pair_rows:
                    pair_row.update(
                        {
                            "dataset": display_name,
                            "minimal_dataset": minimal_name,
                            "split": split,
                            "run": run_name,
                            "dataset_index": int(index),
                            "seq_name": row["seq_name"],
                            "rgb_path": row["rgb_path"],
                            "depth_path": row["depth_path"],
                        }
                    )
                camera_pair_rows.extend(pose_pair_rows)
                run_camera_pair_rows.extend(pose_pair_rows)
                if camera_pair_csv is not None:
                    append_csv_rows(camera_pair_csv, CAMERA_PAIR_CSV_FIELDS, pose_pair_rows)
            except Exception as exc:
                if fail_fast:
                    raise
                error_traceback = traceback.format_exc()
                skipped = {
                    "dataset": display_name,
                    "split": split,
                    "dataset_index": int(index),
                    "seq_name": str(item.get("name", "")),
                    "rgb_path": str(item.get("rgb_path", "")),
                    "depth_path": str(item.get("depth_path", "")),
                    "error": repr(exc),
                    "traceback": error_traceback,
                }
                skipped_rows.append(skipped)
                print(f"[WARN] skipped {display_name} index={index}: {exc}")
                print(error_traceback)

    return {
        "name": run_name,
        "dataset": display_name,
        "minimal_dataset": minimal_name,
        "split": split,
        "dataset_size": item_count,
        "requested_samples": int(limit),
        "candidate_samples": len(indices),
        "evaluated_samples": len(rows),
        "resumed_samples": resumed_row_count,
        "effective_pano_min_count": int(getattr(dataset, "pano_min_count", 1)),
        "effective_pano_max_count": int(getattr(dataset, "pano_max_count", 1)),
        "windows_per_pano": int(getattr(dataset, "windows_per_pano", 1)),
        "img_per_seq": int(img_per_seq),
        "skipped_samples": sum(1 for row in skipped_rows if row.get("dataset") == display_name and row.get("split") == split),
        "summary": metrics_lib.summarize_values([row["loss"] for row in rows]),
        "depth_summary": metrics_lib.summarize_values([row["loss_depth"] for row in rows]),
        "overlap_summary": metrics_lib.summarize_values([row["loss_overlap"] for row in rows]),
        "valid_fraction_summary": metrics_lib.summarize_values([row["valid_fraction"] for row in rows]),
        "camera_loss_summary": metrics_lib.summarize_values([row["loss_camera"] for row in rows]),
        "camera_T_summary": metrics_lib.summarize_values([row["loss_T"] for row in rows]),
        "camera_R_summary": metrics_lib.summarize_values([row["loss_R"] for row in rows]),
        "camera_pose_summary": summarize_camera_pose_pairs(run_camera_pair_rows),
        "depth_metric_summary": metrics_lib.summarize_metric_rows(rows, metrics_lib.DEPTH_METRIC_KEYS),
        "erp_depth_metric_summary": metrics_lib.summarize_metric_rows(rows, ERP_DEPTH_METRIC_KEYS),
        "erp_coverage_summary": metrics_lib.summarize_metric_rows(rows, ERP_COVERAGE_KEYS),
        "panovggt_metric_summary": metrics_lib.summarize_panovggt_rows(rows),
        "best_samples": {
            "by_loss": metrics_lib.rank_samples(rows, "loss", reverse=False),
            "by_depth_irls_abs_rel": metrics_lib.rank_samples(rows, "depth_irls_abs_rel", reverse=False),
            "by_depth_irls_delta_1p25": metrics_lib.rank_samples(rows, "depth_irls_delta_1p25", reverse=True),
            "by_erp_depth_irls_abs_rel": metrics_lib.rank_samples(rows, "erp_depth_irls_abs_rel", reverse=False),
        },
        "worst_samples": {
            "by_loss": metrics_lib.rank_samples(rows, "loss", reverse=True),
            "by_depth_irls_abs_rel": metrics_lib.rank_samples(rows, "depth_irls_abs_rel", reverse=True),
            "by_depth_irls_delta_1p25": metrics_lib.rank_samples(rows, "depth_irls_delta_1p25", reverse=False),
            "by_erp_depth_irls_abs_rel": metrics_lib.rank_samples(rows, "erp_depth_irls_abs_rel", reverse=True),
        },
    }


def sample_to_batch(
    sample: dict[str, Any],
    device: torch.device,
    normalize_scene_scale: bool = False,
) -> dict[str, torch.Tensor]:
    images = torch.from_numpy(np.stack(sample["images"]).astype(np.float32)).contiguous()
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC images, got {tuple(images.shape)}")
    images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255.0).unsqueeze(0)
    depths = torch.from_numpy(np.stack(sample["depths"]).astype(np.float32)).unsqueeze(0)
    point_masks = torch.from_numpy(np.stack(sample["point_masks"]).astype(bool)).unsqueeze(0)
    frame_count = int(images.shape[1])
    batch = {
        "images": images,
        "depths": depths,
        "point_masks": point_masks,
        "scene_norm_multiplier": torch.ones(1, dtype=torch.float32),
    }
    if sample.get("pano_range_depth_erp") is not None:
        batch["pano_range_depth_erp"] = torch.from_numpy(
            np.asarray(sample["pano_range_depth_erp"], dtype=np.float32)
        ).unsqueeze(0)
    if all(key in sample for key in ("extrinsics", "intrinsics", "cam_points", "world_points")):
        batch["extrinsics"] = torch.from_numpy(np.stack(sample["extrinsics"]).astype(np.float32)).unsqueeze(0)
        batch["intrinsics"] = torch.from_numpy(np.stack(sample["intrinsics"]).astype(np.float32)).unsqueeze(0)
        batch["cam_points"] = torch.from_numpy(np.stack(sample["cam_points"]).astype(np.float32)).unsqueeze(0)
        batch["world_points"] = torch.from_numpy(np.stack(sample["world_points"]).astype(np.float32)).unsqueeze(0)
        if normalize_scene_scale:
            raw_depths = batch["depths"].clone()
            (
                batch["extrinsics"],
                batch["cam_points"],
                batch["world_points"],
                batch["depths"],
            ) = normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )
            ratio_valid = batch["point_masks"] & torch.isfinite(raw_depths) & (raw_depths > 0)
            if bool(ratio_valid.any()):
                batch["scene_norm_multiplier"] = torch.median(
                    batch["depths"][ratio_valid] / raw_depths[ratio_valid]
                ).reshape(1)
    for key in ("sample_weight", "metadata_valid_ratio", "metadata_structure_score"):
        value = sample.get(key)
        if value is None:
            value = np.ones((frame_count,), dtype=np.float32)
        batch[key] = torch.as_tensor(value, dtype=torch.float32).reshape(1, frame_count)
    for key in ("camera_valid",):
        value = sample.get(key)
        if value is None:
            value = np.zeros((frame_count,), dtype=np.bool_)
        batch[key] = torch.as_tensor(value, dtype=torch.bool).reshape(1, frame_count)
    for key in ("camera_weight",):
        value = sample.get(key)
        if value is None:
            value = np.ones((frame_count,), dtype=np.float32)
        batch[key] = torch.as_tensor(value, dtype=torch.float32).reshape(1, frame_count)
    for key in ("view_pano_index", "view_window_index", "pano_count", "windows_per_pano"):
        value = sample.get(key)
        if value is None:
            value = np.zeros((frame_count,), dtype=np.int64)
        batch[key] = torch.as_tensor(value, dtype=torch.int64).reshape(1, frame_count)
    for key in ("view_yaw", "view_pitch", "view_fov_x", "view_fov_y"):
        value = sample.get(key)
        if value is None:
            value = np.zeros((frame_count,), dtype=np.float32)
        batch[key] = torch.as_tensor(value, dtype=torch.float32).reshape(1, frame_count)
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def compute_covered_erp_depth_metrics(
    *,
    pred_window_z: torch.Tensor,
    batch: dict[str, torch.Tensor],
    metrics_lib: Any,
    latitude_limit_deg: float,
) -> dict[str, float]:
    gt_erp = batch.get("pano_range_depth_erp")
    if gt_erp is None:
        empty = metrics_lib.compute_depth_metrics(
            pred_window_z.new_zeros(1, 1, 1, 1),
            pred_window_z.new_zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, device=pred_window_z.device, dtype=torch.bool),
        )
        return {
            "erp_coverage_fraction": 0.0,
            "erp_common_valid_fraction": 0.0,
            **{f"erp_{key}": value for key, value in empty.items()},
        }
    gt_erp = gt_erp.float() * batch.get("scene_norm_multiplier", gt_erp.new_ones(1)).reshape(-1, 1, 1, 1)
    batch_size, num_panos, erp_height, erp_width = gt_erp.shape
    splatted = splat_window_z_depth_to_erp(
        pred_window_z,
        yaw=batch["view_yaw"],
        pitch=batch["view_pitch"],
        fov_x=batch["view_fov_x"],
        fov_y=batch["view_fov_y"],
        view_pano_index=batch["view_pano_index"],
        num_panos=num_panos,
        erp_height=erp_height,
        erp_width=erp_width,
        align_corners=False,
    )
    lat_limit = min(max(float(latitude_limit_deg), 0.0), 90.0)
    row_latitudes = 90.0 - (torch.arange(erp_height, device=gt_erp.device) + 0.5) * 180.0 / erp_height
    latitude_mask = (row_latitudes.abs() <= lat_limit).reshape(1, 1, erp_height, 1)
    latitude_mask = latitude_mask.expand(batch_size, num_panos, erp_height, erp_width)
    gt_valid = torch.isfinite(gt_erp) & (gt_erp > 0.0)
    coverage_mask = splatted["coverage_mask"] & latitude_mask
    # This mask is shared across methods; invalid predictions are retained as
    # zero-depth failures instead of reducing the evaluated region.
    common_mask = coverage_mask & gt_valid
    metrics = metrics_lib.compute_depth_metrics(splatted["depth"], gt_erp, common_mask)
    return {
        "erp_coverage_fraction": float(
            coverage_mask.sum().float().div(latitude_mask.sum().clamp_min(1)).cpu()
        ),
        "erp_common_valid_fraction": float(
            common_mask.sum().float().div((gt_valid & latitude_mask).sum().clamp_min(1)).cpu()
        ),
        **{f"erp_{key}": value for key, value in metrics.items()},
    }


def compute_window0_camera_pose_metrics(
    predictions: dict[str, Any],
    batch: dict[str, torch.Tensor],
    trans_norm_thresh: float = 1e-2,
    max_panos: int = 3,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """PanoVGGT-style pair metrics from Omega window pose predictions.

    The baseline model has only pinhole-window camera predictions. For each
    pano we use window index 0 as the pano pose representative; because every
    pano uses the same window-0 crop transform, that fixed local transform
    cancels in relative pose.
    """

    pred_pose = predictions.get("pose_enc")
    if pred_pose is None or "extrinsics" not in batch:
        return empty_camera_pose_sample_metrics(), []

    pred_extrinsics, _ = encoding_to_camera(pred_pose.float(), batch["images"].shape[-2:])
    target_extrinsics = batch["extrinsics"].float()
    view_pano_index = batch.get("view_pano_index")
    view_window_index = batch.get("view_window_index")
    camera_valid = batch.get("camera_valid")
    if view_pano_index is None or view_window_index is None:
        return empty_camera_pose_sample_metrics(), []
    if camera_valid is None:
        camera_valid = torch.ones(view_pano_index.shape, dtype=torch.bool, device=view_pano_index.device)

    pred_w2c = extrinsics_3x4_to_4x4(pred_extrinsics)
    target_w2c = extrinsics_3x4_to_4x4(target_extrinsics)
    all_r_err: list[torch.Tensor] = []
    all_t_err: list[torch.Tensor] = []
    all_gt_norm: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []

    batch_size = int(pred_w2c.shape[0])
    for batch_index in range(batch_size):
        pano_ids = torch.unique(view_pano_index[batch_index]).detach().cpu().tolist()
        ref_indices: list[int] = []
        ref_panos: list[int] = []
        for pano_id in sorted(int(value) for value in pano_ids):
            pano_mask = view_pano_index[batch_index] == pano_id
            window0 = torch.nonzero(pano_mask & (view_window_index[batch_index] == 0), as_tuple=False).flatten()
            if window0.numel() == 0:
                window0 = torch.nonzero(pano_mask, as_tuple=False).flatten()
            if window0.numel() == 0:
                continue
            ref_idx = int(window0[0].detach().cpu())
            if not bool(camera_valid[batch_index, ref_idx].detach().cpu()):
                continue
            if not torch.isfinite(pred_w2c[batch_index, ref_idx]).all() or not torch.isfinite(target_w2c[batch_index, ref_idx]).all():
                continue
            ref_indices.append(ref_idx)
            ref_panos.append(pano_id)
        if len(ref_indices) < 2:
            continue
        if int(max_panos) > 0:
            ref_indices = ref_indices[: int(max_panos)]
            ref_panos = ref_panos[: int(max_panos)]

        for a in range(len(ref_indices)):
            for b in range(a + 1, len(ref_indices)):
                idx_i = ref_indices[a]
                idx_j = ref_indices[b]
                rel_gt = target_w2c[batch_index, idx_j] @ invert_se3(target_w2c[batch_index, idx_i])
                rel_pred = pred_w2c[batch_index, idx_j] @ invert_se3(pred_w2c[batch_index, idx_i])
                gt_translation_norm = torch.linalg.vector_norm(rel_gt[:3, 3])
                if (
                    not torch.isfinite(gt_translation_norm)
                    or float(gt_translation_norm.detach().cpu()) <= float(trans_norm_thresh)
                ):
                    continue
                r_err = rotation_angle_degrees(rel_gt[:3, :3].unsqueeze(0), rel_pred[:3, :3].unsqueeze(0))[0]
                t_err = translation_angle_degrees(rel_gt[:3, 3].unsqueeze(0), rel_pred[:3, 3].unsqueeze(0))[0]
                if not torch.isfinite(r_err) or not torch.isfinite(t_err):
                    continue
                all_r_err.append(r_err.reshape(1))
                all_t_err.append(t_err.reshape(1))
                all_gt_norm.append(gt_translation_norm.reshape(1))
                r_value = float(r_err.detach().cpu())
                t_value = float(t_err.detach().cpu())
                pair_rows.append(
                    {
                        "batch_index": int(batch_index),
                        "pair_i": int(ref_panos[a]),
                        "pair_j": int(ref_panos[b]),
                        "camera_pose_rotation_deg": r_value,
                        "camera_pose_translation_deg": t_value,
                        "camera_pose_max_error_deg": max(r_value, t_value),
                        "camera_pose_gt_translation_norm": float(gt_translation_norm.detach().cpu()),
                    }
                )

    if not all_r_err:
        return empty_camera_pose_sample_metrics(), []
    r_all = torch.cat(all_r_err).detach().float().cpu()
    t_all = torch.cat(all_t_err).detach().float().cpu()
    gt_norm_all = torch.cat(all_gt_norm).detach().float().cpu()
    return camera_pose_sample_metrics(r_all, t_all, gt_norm_all), pair_rows


def extrinsics_3x4_to_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    out = eye.reshape(*((1,) * (extrinsics.ndim - 2)), 4, 4).repeat(*extrinsics.shape[:-2], 1, 1)
    out[..., :3, :4] = extrinsics
    return out


def invert_se3(pose: torch.Tensor) -> torch.Tensor:
    return closed_form_inverse_se3(pose.unsqueeze(0)).squeeze(0)


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


def empty_camera_pose_sample_metrics() -> dict[str, float]:
    return {key: 0.0 for key in CAMERA_POSE_SAMPLE_KEYS}


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
            "This legacy block is computed on sampled pinhole windows. Prefer "
            "panovggt_covered_erp_depth_benchmark for the ERP-splat common-mask result."
        ),
        "per_dataset": per_dataset,
        "overall_micro_irls_scale_aligned": metrics_lib.summarize_panovggt_rows(rows).get("micro_by_valid_pixel", {}),
        "overall_macro_by_sample": metrics_lib.summarize_metric_rows(rows, metrics_lib.PANOVGGT_PRIMARY_METRICS),
        "panovggt_table3_monocular_macro_reference": macro_reference(PANOVGGT_TABLE3_MONOCULAR),
        "panovggt_table3_multiview_macro_reference": macro_reference(PANOVGGT_TABLE3_MULTIVIEW),
    }


def summarize_erp_panovggt_benchmark(
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metrics_lib: Any,
) -> dict[str, Any]:
    def unprefix(row: dict[str, Any]) -> dict[str, Any]:
        converted = dict(row)
        for key, value in row.items():
            if key.startswith("erp_depth_"):
                converted[key[4:]] = value
        return converted

    converted_rows = [unprefix(row) for row in rows]
    converted_runs = []
    for run in runs:
        converted = dict(run)
        dataset = str(run.get("dataset", "unknown"))
        dataset_rows = [row for row in converted_rows if str(row.get("dataset", "unknown")) == dataset]
        converted["panovggt_metric_summary"] = metrics_lib.summarize_panovggt_rows(dataset_rows)
        converted_runs.append(converted)
    result = summarize_panovggt_benchmark(converted_runs, converted_rows, metrics_lib)
    result["paper_protocol_note"] = (
        "Window Z-depth is converted to radial depth and splatted back to ERP. Metrics use the "
        "intersection of deterministic 8-window coverage, finite prediction, valid GT, and the "
        "configured non-polar latitude band. Coverage is reported separately."
    )
    result["coverage"] = metrics_lib.summarize_metric_rows(
        rows, ["erp_coverage_fraction", "erp_common_valid_fraction"]
    )
    return result


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


def resolve_dataset_pano_counts(policy: str, raw_counts: str | None) -> dict[str, int]:
    if raw_counts not in (None, ""):
        counts: dict[str, int] = {}
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
        for part in str(raw_counts).split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"Dataset pano count override must be name:count, got {part!r}")
            name, value = part.split(":", 1)
            key = aliases.get(name.strip().lower(), name.strip().lower())
            count = int(value)
            if count < 1:
                raise ValueError(f"Dataset pano count must be positive, got {part!r}")
            counts[key] = count
        return counts
    if str(policy) == "panovggt":
        return dict(PANOVGGT_EVAL_PANO_COUNTS)
    return {}


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


def initialize_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def load_resume_rows(
    per_sample_csv: Path | None,
    camera_pair_csv: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, str, int]]]:
    if per_sample_csv is None:
        raise ValueError("--resume requires --per-sample-csv")

    sample_exists = per_sample_csv.exists() and per_sample_csv.stat().st_size > 0
    camera_exists = camera_pair_csv is not None and camera_pair_csv.exists() and camera_pair_csv.stat().st_size > 0
    if sample_exists and camera_pair_csv is not None and not camera_exists:
        raise FileNotFoundError(
            f"Cannot resume safely: sample CSV exists but camera pair CSV is missing: {camera_pair_csv}"
        )
    if camera_exists and not sample_exists:
        raise FileNotFoundError(
            f"Cannot resume safely: camera pair CSV exists but sample CSV is missing: {per_sample_csv}"
        )
    if not sample_exists:
        initialize_csv(per_sample_csv, PER_SAMPLE_CSV_FIELDS)
        if camera_pair_csv is not None:
            initialize_csv(camera_pair_csv, CAMERA_PAIR_CSV_FIELDS)
        return [], [], set()

    sample_rows = read_typed_csv_rows(
        per_sample_csv,
        PER_SAMPLE_CSV_FIELDS,
        string_fields={"dataset", "minimal_dataset", "split", "run", "seq_name", "rgb_path", "depth_path", "quality_bin"},
        integer_fields={"dataset_index", "input_pano_count", "camera_eval_pano_count"},
    )
    camera_rows = read_typed_csv_rows(
        camera_pair_csv,
        CAMERA_PAIR_CSV_FIELDS,
        string_fields={"dataset", "minimal_dataset", "split", "run", "seq_name", "rgb_path", "depth_path"},
        integer_fields={"dataset_index", "batch_index", "pair_i", "pair_j"},
    ) if camera_pair_csv is not None else []

    completed_keys: set[tuple[str, str, int]] = set()
    for row in sample_rows:
        key = (str(row["minimal_dataset"]), str(row["split"]), int(row["dataset_index"]))
        if key in completed_keys:
            raise ValueError(f"Duplicate sample key in resume CSV: {key}")
        completed_keys.add(key)

    camera_keys: set[tuple[str, str, int, int, int, int]] = set()
    for row in camera_rows:
        sample_key = (str(row["minimal_dataset"]), str(row["split"]), int(row["dataset_index"]))
        if sample_key not in completed_keys:
            raise ValueError(f"Camera pair row has no matching completed sample: {sample_key}")
        key = (*sample_key, int(row["batch_index"]), int(row["pair_i"]), int(row["pair_j"]))
        if key in camera_keys:
            raise ValueError(f"Duplicate camera pair key in resume CSV: {key}")
        camera_keys.add(key)
    return sample_rows, camera_rows, completed_keys


def read_typed_csv_rows(
    path: Path | None,
    expected_fields: list[str],
    *,
    string_fields: set[str],
    integer_fields: set[str],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Resume CSV schema mismatch for {path}: expected {expected_fields}, got {reader.fieldnames}"
            )
        rows: list[dict[str, Any]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"Malformed resume CSV row at {path}:{line_number}")
            row: dict[str, Any] = {}
            for field in expected_fields:
                raw = raw_row.get(field, "")
                if field in string_fields:
                    row[field] = raw
                elif field in integer_fields:
                    if raw == "":
                        raise ValueError(f"Missing integer field {field!r} at {path}:{line_number}")
                    row[field] = int(raw)
                else:
                    row[field] = float(raw) if raw != "" else float("nan")
            rows.append(row)
    return rows


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]], metrics_lib: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_CSV_FIELDS, extrasaction="ignore")
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
