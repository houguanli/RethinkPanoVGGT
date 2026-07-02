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
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_depth_checkpoint import (  # noqa: E402
    DEPTH_METRIC_KEYS,
    apply_checkpoint_eval_defaults,
    build_eval_model,
    evaluate_run,
    load_checkpoint_payload,
    normalize_args_for_eval,
    read_train_loss_reference,
)
from training.train_pano_omega import (  # noqa: E402
    parse_args as parse_training_args,
    resolve_device,
    set_seed,
)


DATASETS = [
    ("Panocity", "panocity", "val"),
    ("Matterport3D", "matterport3d", "test"),
    ("Stanford2D3DS", "stanford2d3ds", "test"),
    ("Structured3D", "structured3d", "val"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config used to build model/datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to validate.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument("--per-sample-csv", type=Path, default=None, help="Optional per-sample CSV path.")
    parser.add_argument("--train-loss-csv", type=Path, default=None, help="Optional training loss.csv for comparison.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit-per-dataset", type=int, default=100, help="Held-out samples per dataset.")
    parser.add_argument("--seed", type=int, default=123)
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
    for dataset_index, (display_name, minimal_name, split) in enumerate(DATASETS):
        dataset_args = copy.copy(train_args)
        dataset_args.dataset_format = "pano_minimal"
        dataset_args.minimal_datasets = minimal_name
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
            seed=args.seed + dataset_index * 1009,
            num_workers=args.num_workers,
            progress=args.progress,
            per_sample_rows=per_sample_rows,
        )
        run["dataset"] = display_name
        run["minimal_dataset"] = minimal_name
        runs.append(run)
        for row in per_sample_rows[before_count:]:
            row["dataset"] = display_name
            row["split"] = split

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "limit_per_dataset": int(args.limit_per_dataset),
        "dataset_root": str(train_args.dataset_root),
        "split_policy": {
            "Panocity": "val (no test cache in current local official-layout bundle)",
            "Matterport3D": "test",
            "Stanford2D3DS": "test",
            "Structured3D": "val (no test cache in current local bundle)",
        },
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv),
        "runs": runs,
        "overall": summarize_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_sample_csv is not None:
        write_per_sample_csv(args.per_sample_csv, per_sample_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_loss = 0.0
    weighted_depth_loss = 0.0
    weighted_overlap_loss = 0.0
    total_samples = 0
    for run in runs:
        n = int(run.get("evaluated_samples", 0))
        if n <= 0:
            continue
        total_samples += n
        weighted_loss += float(run.get("summary", {}).get("mean", 0.0)) * n
        weighted_depth_loss += float(run.get("depth_summary", {}).get("mean", 0.0)) * n
        weighted_overlap_loss += float(run.get("overlap_summary", {}).get("mean", 0.0)) * n
    denom = float(total_samples) if total_samples > 0 else 1.0
    return {
        "evaluated_samples": total_samples,
        "weighted_loss_mean": weighted_loss / denom if total_samples else None,
        "weighted_depth_loss_mean": weighted_depth_loss / denom if total_samples else None,
        "weighted_overlap_loss_mean": weighted_overlap_loss / denom if total_samples else None,
    }


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "run",
        "dataset_index",
        "seq_name",
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
