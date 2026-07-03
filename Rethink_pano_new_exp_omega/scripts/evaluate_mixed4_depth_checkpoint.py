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
    PANOVGGT_PRIMARY_METRICS,
    apply_checkpoint_eval_defaults,
    build_eval_model,
    evaluate_run,
    load_checkpoint_payload,
    normalize_args_for_eval,
    rank_samples,
    read_train_loss_reference,
    summarize_metric_rows,
    summarize_panovggt_rows,
)
from training.train_pano_omega import (  # noqa: E402
    parse_args as parse_training_args,
    resolve_device,
    set_seed,
)


DATASETS = [
    ("Panocity", "panocity", "test"),
    ("Matterport3D", "matterport3d", "test"),
    ("Stanford2D3DS", "stanford2d3ds", "test"),
    ("Structured3D", "structured3d", "val"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config used to build model/datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to validate.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument("--per-sample-csv", type=Path, default=None, help="Optional per-sample CSV path.")
    parser.add_argument("--train-loss-csv", type=Path, default=None, help="Optional training loss.csv for comparison.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit-per-dataset", type=int, default=100, help="Held-out samples per dataset. Use 0 for the full split.")
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
            "Panocity": "test (PanoVGGT official split when cache was built with official split JSONs)",
            "Matterport3D": "test",
            "Stanford2D3DS": "test",
            "Structured3D": "val (no test cache in current local bundle)",
        },
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv),
        "runs": runs,
        "overall": summarize_runs(runs),
        "panovggt_depth_benchmark": summarize_panovggt_benchmark(runs, per_sample_rows),
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


def summarize_panovggt_benchmark(runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "overall_micro_irls_scale_aligned": summarize_panovggt_rows(rows).get("micro_by_valid_pixel", {}),
        "overall_macro_by_sample": summarize_metric_rows(rows, PANOVGGT_PRIMARY_METRICS),
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


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
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
        *DEPTH_METRIC_KEYS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
