#!/usr/bin/env python3
"""Validate a mixed minimal-pano checkpoint per dataset split.

This is a thin wrapper around evaluate_depth_checkpoint.py. It keeps the
training loss definition unchanged, but reports separate held-out metrics for
Panocity, Matterport3D, Stanford2D3DS, and Structured3D.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_depth_checkpoint import (  # noqa: E402
    CAMERA_PAIR_CSV_FIELDS,
    DEPTH_ACCUMULATOR_KEYS,
    DEPTH_METRIC_PROTOCOL,
    DEPTH_METRIC_KEYS,
    ERP_PRIOR_COVERAGE_KEYS,
    ERP_PRIOR_DIAGNOSTIC_KEYS,
    ERP_PRIOR_DEPTH_METRIC_KEYS,
    PANOVGGT_PRIMARY_METRICS,
    PER_SAMPLE_CSV_FIELDS,
    apply_checkpoint_eval_defaults,
    build_eval_model,
    evaluate_run,
    erp_polar_prior_metadata,
    apply_eval_max_panos,
    initialize_csv,
    load_checkpoint_payload,
    normalize_args_for_eval,
    rank_samples,
    read_train_loss_reference,
    summarize_camera_pose_pairs,
    summarize_metric_rows,
    summarize_panovggt_rows,
    write_eval_progress,
)
from training.train_pano_omega import (  # noqa: E402
    parse_args as parse_training_args,
    resolve_device,
    set_seed,
)


DATASETS = [
    ("Stanford2D3DS", "stanford2d3ds", "test"),
    ("Matterport3D", "matterport3d", "test"),
    ("Structured3D", "structured3d", "test"),
    ("Panocity", "panocity", "test"),
]

DATASET_SEED_OFFSETS = {
    "panocity": 0,
    "matterport3d": 1,
    "stanford2d3ds": 2,
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

PANOVGGT_DATASET_PANO_COUNTS = {
    "panocity": 10,
    "matterport3d": 3,
    "stanford2d3ds": 3,
    "structured3d": 3,
}
SINGLE_PANO_DATASET_COUNTS = {name: 1 for name in PANOVGGT_DATASET_PANO_COUNTS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config used to build model/datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to validate.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset root from the training config.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument("--per-sample-csv", type=Path, default=None, help="Optional per-sample CSV path.")
    parser.add_argument("--train-loss-csv", type=Path, default=None, help="Optional training loss.csv for comparison.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset ids/names to evaluate: all, panocity, matterport3d, stanford2d3ds, structured3d.",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=100, help="Held-out samples per dataset. Use 0 for the full split.")
    parser.add_argument("--limit-fraction", type=float, default=0.0, help="Fraction of scene groups per dataset when --limit-per-dataset <= 0.")
    parser.add_argument("--eval-max-panos", type=int, default=0, help="Clamp eval multi-pano input length for all datasets. Use 0 to keep config pano_max_count.")
    parser.add_argument("--window-size", type=int, default=0, help="Override square window resolution; 0 keeps checkpoint/config.")
    parser.add_argument("--num-yaw", type=int, default=0, help="Override yaw windows per pano; 0 keeps checkpoint/config.")
    parser.add_argument("--panocity-max-panos", type=int, default=0, help="Optional Panocity-specific eval pano cap, overriding --eval-max-panos for Panocity.")
    parser.add_argument(
        "--pano-count-policy",
        choices=["config", "panovggt", "single"],
        default="panovggt",
        help=(
            "Pano count policy. Default panovggt uses Panocity=10 and indoor datasets=3; "
            "single forces every dataset to one panorama; config keeps checkpoint/config counts."
        ),
    )
    parser.add_argument(
        "--dataset-pano-counts",
        default="",
        help="Optional comma list name:count overriding the selected pano-count policy.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help="Optional CSV containing exact pano_id_0..N groups. Rows are filtered by dataset when present.",
    )
    parser.add_argument(
        "--sample-policy",
        choices=["scene_neighborhood", "anchor"],
        default="scene_neighborhood",
        help=(
            "scene_neighborhood evaluates one nearest-neighborhood group per "
            "scene/room/trajectory; anchor preserves the older random-anchor behavior."
        ),
    )
    parser.add_argument("--camera-pair-csv", type=Path, default=None, help="Optional streaming PanoVGGT-style camera pair CSV path.")
    parser.add_argument("--camera-pose-trans-norm-thresh", type=float, default=1e-2, help="GT baseline threshold for PanoVGGT-style camera translation-angle eval.")
    parser.add_argument("--camera-eval-max-panos", type=int, default=3, help="Maximum pano views used for PanoVGGT Table-2 camera metrics.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default=None)
    parser.add_argument("--num-shards", type=int, default=1, help="Split each dataset over this many independent eval workers.")
    parser.add_argument("--shard-rank", type=int, default=0, help="Shard id for this worker, in [0, num_shards).")
    parser.add_argument("--progress-file", type=Path, default=None, help="Optional JSON file updated periodically with shard progress.")
    parser.add_argument("--progress-every", type=int, default=25, help="Samples between progress-file updates.")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--print-each-sample", dest="print_each_sample", action="store_true", default=True)
    parser.add_argument("--no-print-each-sample", dest="print_each_sample", action="store_false")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed run+dataset_index rows from the streaming CSV and evaluate only missing samples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device, {"distributed": False, "local_rank": 0})
    write_eval_progress(
        args.progress_file,
        {
            "state": "initializing_model",
            "pid": os.getpid(),
            "shard_rank": int(args.shard_rank),
            "num_shards": int(args.num_shards),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_cuda_device_count": torch_cuda_device_count(),
            "requested_datasets": str(args.datasets),
            "updated_at": time.time(),
        },
    )

    train_args = parse_training_args(["--config", str(args.config)])
    if args.dataset_root is not None:
        train_args.dataset_root = args.dataset_root
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
    if args.window_size > 0:
        train_args.window_size = int(args.window_size)
    if args.num_yaw > 0:
        train_args.num_yaw = int(args.num_yaw)
    model = build_eval_model(train_args, args.checkpoint, checkpoint_payload, device)
    model.eval()

    camera_pair_csv = args.camera_pair_csv
    if camera_pair_csv is None and args.per_sample_csv is not None:
        camera_pair_csv = args.per_sample_csv.with_name(f"{args.per_sample_csv.stem}_camera_pairs.csv")
    loaded_sample_rows = read_csv_rows(args.per_sample_csv) if args.resume else []
    per_sample_rows = [
        row for row in loaded_sample_rows if str(row.get("depth_metric_protocol", "")) == DEPTH_METRIC_PROTOCOL
    ]
    dropped_legacy_rows = len(loaded_sample_rows) - len(per_sample_rows)
    camera_pair_rows = read_csv_rows(camera_pair_csv) if args.resume else []
    if dropped_legacy_rows:
        retained_keys = {
            (str(row.get("run", "")), int(float(row.get("dataset_index", -1)))) for row in per_sample_rows
        }
        camera_pair_rows = [
            row
            for row in camera_pair_rows
            if (str(row.get("run", "")), int(float(row.get("dataset_index", -1)))) in retained_keys
        ]
        print(
            f"[EVAL-RESUME] discarded {dropped_legacy_rows} rows from an older depth metric protocol; "
            "those samples will be evaluated again",
            flush=True,
        )
    if args.per_sample_csv is not None:
        if args.resume:
            write_per_sample_csv(args.per_sample_csv, per_sample_rows)
        else:
            initialize_csv(args.per_sample_csv, PER_SAMPLE_CSV_FIELDS)
    if camera_pair_csv is not None:
        if args.resume:
            write_camera_pair_csv(camera_pair_csv, camera_pair_rows)
        else:
            initialize_csv(camera_pair_csv, CAMERA_PAIR_CSV_FIELDS)

    selected_datasets = select_datasets(args.datasets)
    manifest_rows = load_sample_manifest(args.sample_manifest)
    dataset_pano_counts = resolve_dataset_pano_counts(args.pano_count_policy, args.dataset_pano_counts)
    write_eval_progress(
        args.progress_file,
        {
            "state": "model_ready",
            "pid": os.getpid(),
            "shard_rank": int(args.shard_rank),
            "num_shards": int(args.num_shards),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_cuda_device_count": torch_cuda_device_count(),
            "selected_datasets": sorted(selected_datasets),
            "updated_at": time.time(),
        },
    )
    runs: list[dict[str, Any]] = []
    for dataset_index, (display_name, minimal_name, split) in enumerate(DATASETS):
        if minimal_name not in selected_datasets:
            continue
        dataset_args = copy.copy(train_args)
        dataset_args.dataset_format = "pano_minimal"
        dataset_args.minimal_datasets = minimal_name
        dataset_pano_count = dataset_pano_counts.get(minimal_name)
        if dataset_pano_count is not None:
            apply_exact_eval_pano_count(dataset_args, dataset_pano_count, minimal_name)
            dataset_eval_max_panos = int(dataset_pano_count)
        else:
            dataset_eval_max_panos = int(args.eval_max_panos or 0)
            if minimal_name == "panocity" and int(args.panocity_max_panos or 0) > 0:
                dataset_eval_max_panos = int(args.panocity_max_panos)
            apply_eval_max_panos(dataset_args, dataset_eval_max_panos)
        run_name = f"{display_name}_{split}_{args.limit_per_dataset}"
        before_count = len(per_sample_rows)
        run = evaluate_run(
            name=run_name,
            base_args=dataset_args,
            model=model,
            device=device,
            split=split,
            curriculum_bins="all",
            limit=args.limit_per_dataset,
            limit_fraction=args.limit_fraction,
            seed=args.seed + DATASET_SEED_OFFSETS[minimal_name] * 1009,
            sample_policy=args.sample_policy,
            num_workers=args.num_workers,
            progress=args.progress,
            per_sample_rows=per_sample_rows,
            per_sample_csv=args.per_sample_csv,
            camera_pair_rows=camera_pair_rows,
            camera_pair_csv=camera_pair_csv,
            row_context={"dataset": display_name, "split": split},
            shard_rank=args.shard_rank,
            num_shards=args.num_shards,
            progress_file=args.progress_file,
            progress_every=args.progress_every,
            progress_context={
                "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "torch_cuda_device_count": torch_cuda_device_count(),
                "selected_datasets": sorted(selected_datasets),
            },
            camera_pose_trans_norm_thresh=args.camera_pose_trans_norm_thresh,
            camera_eval_max_panos=args.camera_eval_max_panos,
            print_each_sample=args.print_each_sample,
            exact_group_manifest=manifest_rows_for_dataset(manifest_rows, display_name, minimal_name),
            resume=args.resume,
        )
        run["dataset"] = display_name
        run["minimal_dataset"] = minimal_name
        run["requested_eval_max_panos"] = dataset_eval_max_panos
        run["effective_eval_pano_min_count"] = int(getattr(dataset_args, "pano_min_count", 1))
        run["effective_eval_pano_max_count"] = int(getattr(dataset_args, "pano_max_count", 1))
        runs.append(run)
        for row in per_sample_rows[before_count:]:
            row["dataset"] = display_name
            row["split"] = split
        write_dataset_shard_snapshot(
            args=args,
            train_args=train_args,
            run=run,
            minimal_name=minimal_name,
            selected_datasets=selected_datasets,
            dataset_pano_counts=dataset_pano_counts,
            per_sample_rows=per_sample_rows,
            camera_pair_rows=camera_pair_rows,
        )

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "limit_per_dataset": int(args.limit_per_dataset),
        "limit_fraction": float(args.limit_fraction or 0.0),
        "eval_max_panos": int(args.eval_max_panos or 0),
        "window_size": int(train_args.window_size),
        "num_yaw": int(train_args.num_yaw),
        "pitch_degrees": str(train_args.pitch_degrees),
        "fov_degrees": float(train_args.fov_degrees),
        "depth_evaluation_domain": "covered_sphere_sampled_pinhole_windows",
        "depth_evaluation_domains": {
            "covered_sphere": "sampled pinhole windows with solid-angle weights and overlap de-duplication",
            "erp_with_polar_prior": "uniform ERP pixels after window splat and stable-cap prior completion",
        },
        "depth_metric_protocol": DEPTH_METRIC_PROTOCOL,
        "depth_pixel_weighting": "pinhole_solid_angle_divided_by_same_pano_window_coverage_count",
        "erp_polar_prior": erp_polar_prior_metadata(),
        "panocity_max_panos": int(args.panocity_max_panos or 0),
        "pano_count_policy": str(args.pano_count_policy),
        "dataset_pano_counts": {key: int(value) for key, value in sorted(dataset_pano_counts.items())},
        "camera_eval_max_panos": int(args.camera_eval_max_panos),
        "datasets": sorted(selected_datasets),
        "shard_rank": int(args.shard_rank),
        "num_shards": int(args.num_shards),
        "dataset_root": str(train_args.dataset_root),
        "sample_policy": str(args.sample_policy),
        "sample_manifest": str(args.sample_manifest) if args.sample_manifest is not None else None,
        "resume": bool(args.resume),
        "split_policy": {
            "Panocity": "test (PanoVGGT official split when cache was built with official split JSONs)",
            "Matterport3D": "test",
            "Stanford2D3DS": "test",
            "Structured3D": "test (official split when cache was built with official split files)",
        },
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv),
        "runs": runs,
        "overall": summarize_runs(runs),
        "panovggt_depth_benchmark": summarize_panovggt_benchmark(runs, per_sample_rows),
        "panovggt_camera_benchmark": summarize_panovggt_camera_benchmark(runs, camera_pair_rows),
        "case_rankings": {
            "best_by_depth_irls_abs_rel": rank_samples(per_sample_rows, "depth_irls_abs_rel", reverse=False, limit=20),
            "worst_by_depth_irls_abs_rel": rank_samples(per_sample_rows, "depth_irls_abs_rel", reverse=True, limit=20),
            "best_by_depth_irls_delta_1p25": rank_samples(per_sample_rows, "depth_irls_delta_1p25", reverse=True, limit=20),
            "worst_by_depth_irls_delta_1p25": rank_samples(per_sample_rows, "depth_irls_delta_1p25", reverse=False, limit=20),
            "worst_by_loss": rank_samples(per_sample_rows, "loss", reverse=True, limit=20),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_sample_csv is not None:
        write_per_sample_csv(args.per_sample_csv, per_sample_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def load_sample_manifest(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Sample manifest is empty: {path}")
    return rows


def read_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def write_dataset_shard_snapshot(
    *,
    args: argparse.Namespace,
    train_args: argparse.Namespace,
    run: dict[str, Any],
    minimal_name: str,
    selected_datasets: set[str],
    dataset_pano_counts: dict[str, int],
    per_sample_rows: list[dict[str, Any]],
    camera_pair_rows: list[dict[str, Any]],
) -> None:
    if args.per_sample_csv is None:
        return
    run_name = str(run["name"])
    dataset_rows = [row for row in per_sample_rows if str(row.get("run", "")) == run_name]
    dataset_camera_rows = [row for row in camera_pair_rows if str(row.get("run", "")) == run_name]
    sample_path = args.per_sample_csv.with_name(f"{args.per_sample_csv.stem}_{minimal_name}.csv")
    camera_path = sample_path.with_name(f"{sample_path.stem}_camera_pairs.csv")
    json_path = args.output.with_name(f"{args.output.stem}_{minimal_name}.json")
    write_per_sample_csv(sample_path, dataset_rows)
    write_camera_pair_csv(camera_path, dataset_camera_rows)
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(args.device),
        "seed": int(args.seed),
        "limit_per_dataset": int(args.limit_per_dataset),
        "window_size": int(train_args.window_size),
        "num_yaw": int(train_args.num_yaw),
        "pitch_degrees": str(train_args.pitch_degrees),
        "fov_degrees": float(train_args.fov_degrees),
        "depth_evaluation_domain": "covered_sphere_sampled_pinhole_windows",
        "depth_evaluation_domains": {
            "covered_sphere": "sampled pinhole windows with solid-angle weights and overlap de-duplication",
            "erp_with_polar_prior": "uniform ERP pixels after window splat and stable-cap prior completion",
        },
        "depth_metric_protocol": DEPTH_METRIC_PROTOCOL,
        "depth_pixel_weighting": "pinhole_solid_angle_divided_by_same_pano_window_coverage_count",
        "erp_polar_prior": erp_polar_prior_metadata(),
        "num_shards": int(args.num_shards),
        "shard_rank": int(args.shard_rank),
        "dataset_root": str(train_args.dataset_root),
        "sample_policy": str(args.sample_policy),
        "pano_count_policy": str(args.pano_count_policy),
        "dataset_pano_counts": {key: int(value) for key, value in sorted(dataset_pano_counts.items())},
        "datasets": sorted(selected_datasets),
        "runs": [run],
    }
    atomic_write_json(json_path, payload)
    print(
        f"[EVAL-DATASET-DONE] dataset={minimal_name} samples={len(dataset_rows)} "
        f"json={json_path} csv={sample_path}",
        flush=True,
    )


def write_camera_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMERA_PAIR_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def manifest_rows_for_dataset(
    rows: list[dict[str, str]] | None,
    display_name: str,
    minimal_name: str,
) -> list[dict[str, str]] | None:
    if rows is None:
        return None
    dataset_columns = ("dataset", "dataset_name", "minimal_dataset")
    if not any(any(row.get(key) for key in dataset_columns) for row in rows):
        return rows
    aliases = {display_name.lower(), minimal_name.lower()}
    selected = [
        row
        for row in rows
        if str(next((row.get(key) for key in dataset_columns if row.get(key)), "")).lower() in aliases
    ]
    if not selected:
        raise ValueError(f"Sample manifest contains no rows for {display_name}")
    return selected


def select_datasets(raw: str) -> set[str]:
    aliases = {
        "panocity": "panocity",
        "pano_city": "panocity",
        "matterport3d": "matterport3d",
        "matterport": "matterport3d",
        "mp3d": "matterport3d",
        "stanford2d3ds": "stanford2d3ds",
        "stanford": "stanford2d3ds",
        "s2d3ds": "stanford2d3ds",
        "structured3d": "structured3d",
        "s3d": "structured3d",
    }
    if raw is None or str(raw).strip().lower() in {"", "all", "*"}:
        return {minimal_name for _, minimal_name, _ in DATASETS}
    selected: set[str] = set()
    for token in str(raw).split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            known = ", ".join(sorted(aliases))
            raise ValueError(f"Unknown dataset '{token}'. Expected one of: all, {known}")
        selected.add(aliases[key])
    if not selected:
        raise ValueError("No datasets selected")
    return selected


def resolve_dataset_pano_counts(policy: str, raw_counts: str | None) -> dict[str, int]:
    normalized_policy = str(policy)
    if normalized_policy == "panovggt":
        counts = dict(PANOVGGT_DATASET_PANO_COUNTS)
    elif normalized_policy == "single":
        counts = dict(SINGLE_PANO_DATASET_COUNTS)
    else:
        counts = {}
    for token in str(raw_counts or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Expected dataset:count in --dataset-pano-counts, got {token!r}")
        name, raw_count = token.split(":", 1)
        dataset_name = next(iter(select_datasets(name)))
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"Pano count must be positive for {dataset_name}: {count}")
        counts[dataset_name] = count
    return counts


def apply_exact_eval_pano_count(args: argparse.Namespace, count: int, dataset_name: str) -> None:
    count = max(1, int(count))
    args.pano_min_count = count
    args.pano_max_count = count
    args.dataset_pano_counts = f"{dataset_name}:{count}"


def torch_cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_loss = 0.0
    weighted_depth_loss = 0.0
    weighted_overlap_loss = 0.0
    weighted_global_point_loss = 0.0
    weighted_camera_loss = 0.0
    weighted_camera_translation_deg = 0.0
    weighted_camera_rotation_deg = 0.0
    camera_translation_samples = 0
    camera_rotation_samples = 0
    total_samples = 0
    for run in runs:
        n = int(run.get("evaluated_samples", 0))
        if n <= 0:
            continue
        total_samples += n
        weighted_loss += float(run.get("summary", {}).get("mean", 0.0)) * n
        weighted_depth_loss += float(run.get("depth_summary", {}).get("mean", 0.0)) * n
        weighted_overlap_loss += float(run.get("overlap_summary", {}).get("mean", 0.0)) * n
        weighted_global_point_loss += float(run.get("global_point_summary", {}).get("mean", 0.0)) * n
        weighted_camera_loss += float(run.get("camera_summary", {}).get("mean", 0.0)) * n
        translation_n = int(run.get("camera_translation_deg_summary", {}).get("n", 0))
        if translation_n > 0:
            weighted_camera_translation_deg += (
                float(run["camera_translation_deg_summary"]["mean"]) * translation_n
            )
            camera_translation_samples += translation_n
        rotation_n = int(run.get("camera_rotation_deg_summary", {}).get("n", 0))
        if rotation_n > 0:
            weighted_camera_rotation_deg += (
                float(run["camera_rotation_deg_summary"]["mean"]) * rotation_n
            )
            camera_rotation_samples += rotation_n
    denom = float(total_samples) if total_samples > 0 else 1.0
    return {
        "evaluated_samples": total_samples,
        "weighted_loss_mean": weighted_loss / denom if total_samples else None,
        "weighted_depth_loss_mean": weighted_depth_loss / denom if total_samples else None,
        "weighted_overlap_loss_mean": weighted_overlap_loss / denom if total_samples else None,
        "weighted_global_point_loss_mean": weighted_global_point_loss / denom if total_samples else None,
        "weighted_camera_loss_mean": weighted_camera_loss / denom if total_samples else None,
        "weighted_camera_translation_deg_mean": (
            weighted_camera_translation_deg / camera_translation_samples if camera_translation_samples else None
        ),
        "weighted_camera_rotation_deg_mean": (
            weighted_camera_rotation_deg / camera_rotation_samples if camera_rotation_samples else None
        ),
        "camera_rotation_samples": camera_rotation_samples,
        "camera_translation_samples": camera_translation_samples,
    }


def summarize_panovggt_benchmark(runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_dataset = {}
    for run in runs:
        dataset = str(run.get("dataset", "unknown"))
        metrics = run.get("panovggt_metric_summary", {})
        micro = metrics.get("micro_by_valid_pixel", {})
        macro = metrics.get("macro_by_sample", {})
        erp_metrics = metrics.get("erp_with_polar_prior", {})
        erp_micro = erp_metrics.get("micro_by_valid_pixel", {})
        erp_macro = erp_metrics.get("macro_by_sample", {})
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
            "ours_erp_with_polar_prior": {
                "micro_irls_scale_aligned": {
                    "abs_rel": erp_micro.get("erp_prior_depth_irls_abs_rel"),
                    "delta_1p25": erp_micro.get("erp_prior_depth_irls_delta_1p25"),
                    "rmse": erp_micro.get("erp_prior_depth_irls_rmse"),
                },
                "macro_by_sample_irls_scale_aligned": {
                    "abs_rel": erp_macro.get("erp_prior_depth_irls_abs_rel", {}).get("mean"),
                    "delta_1p25": erp_macro.get("erp_prior_depth_irls_delta_1p25", {}).get("mean"),
                    "rmse": erp_macro.get("erp_prior_depth_irls_rmse", {}).get("mean"),
                    "evaluated_gt_fraction": erp_macro.get("erp_evaluated_gt_fraction", {}).get("mean"),
                },
                "same_domain_diagnostics_macro": {
                    key: value.get("mean")
                    for key, value in summarize_metric_rows(
                        [row for row in rows if str(row.get("dataset", "unknown")) == dataset],
                        ERP_PRIOR_DIAGNOSTIC_KEYS,
                    ).items()
                },
            },
            "panovggt_table3_monocular": reference_mono,
            "panovggt_table3_multiview": reference_multi,
            "beats_panovggt_monocular": compare_to_reference(ours_abs_rel, ours_delta, reference_mono),
            "beats_panovggt_multiview": compare_to_reference(ours_abs_rel, ours_delta, reference_multi),
        }
    return {
        "paper_protocol_note": (
            "PanoVGGT Table 3 reports Abs Rel and delta<1.25 after IRLS scale normalization. "
            "Metrics here use solid-angle weighting over the sphere covered by sampled pinhole windows, "
            "with overlap divided by coverage count. PanoVGGT Table 3 uses a different spatial weighting, "
            "so the stored paper-reference comparisons are numeric context rather than protocol-identical claims."
        ),
        "per_dataset": per_dataset,
        "overall_micro_irls_scale_aligned": summarize_panovggt_rows(rows).get("micro_by_valid_pixel", {}),
        "overall_macro_by_sample": summarize_metric_rows(rows, PANOVGGT_PRIMARY_METRICS),
        "overall_erp_with_polar_prior": {
            "micro_by_valid_pixel": summarize_panovggt_rows(rows)
            .get("erp_with_polar_prior", {})
            .get("micro_by_valid_pixel", {}),
            "macro_by_sample": summarize_metric_rows(
                rows,
                ERP_PRIOR_DEPTH_METRIC_KEYS + ERP_PRIOR_COVERAGE_KEYS + ERP_PRIOR_DIAGNOSTIC_KEYS,
            ),
        },
        "erp_polar_prior": erp_polar_prior_metadata(),
        "panovggt_table3_monocular_macro_reference": macro_reference(PANOVGGT_TABLE3_MONOCULAR),
        "panovggt_table3_multiview_macro_reference": macro_reference(PANOVGGT_TABLE3_MULTIVIEW),
    }


def summarize_panovggt_camera_benchmark(runs: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_dataset = {}
    for run in runs:
        dataset = str(run.get("dataset", "unknown"))
        dataset_rows = [row for row in pair_rows if str(row.get("dataset", "unknown")) == dataset]
        per_dataset[dataset] = {
            "evaluated_samples": run.get("evaluated_samples", 0),
            "split": run.get("split"),
            "pose": summarize_camera_pose_pairs(dataset_rows),
        }
    return {
        "protocol_note": (
            "Camera pose metrics follow PanoVGGT's pairwise relative-pose evaluation: "
            "unordered pano pairs, translation direction error in degrees, rotation geodesic error in degrees, "
            "and AUC over max(R,T). Datasets without valid translation/rotation GT produce pair_count=0."
        ),
        "per_dataset": per_dataset,
        "overall": summarize_camera_pose_pairs(pair_rows),
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


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
