#!/usr/bin/env python3
"""Merge independently sharded mixed4 validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_depth_checkpoint import (  # noqa: E402
    CAMERA_PAIR_CSV_FIELDS,
    DEPTH_ACCUMULATOR_KEYS,
    DEPTH_METRIC_KEYS,
    PANOVGGT_PRIMARY_METRICS,
    rank_samples,
    read_train_loss_reference,
    summarize_by_key,
    summarize_metric_rows,
    summarize_panovggt_rows,
    summarize_values,
)
from scripts.evaluate_mixed4_depth_checkpoint import (  # noqa: E402
    summarize_panovggt_camera_benchmark,
    summarize_panovggt_benchmark,
    summarize_runs,
    write_per_sample_csv,
)


NUMERIC_FIELDS = {
    "dataset_index",
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
    "valid_fraction",
    "pred_depth_scale",
    "metadata_valid_ratio",
    "metadata_structure_score",
    "sample_weight",
    *DEPTH_METRIC_KEYS,
    *DEPTH_ACCUMULATOR_KEYS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-json", type=Path, nargs="+", required=True, help="Per-shard summary JSON files.")
    parser.add_argument("--shard-csv", type=Path, nargs="+", required=True, help="Per-shard per-sample CSV files.")
    parser.add_argument("--shard-camera-csv", type=Path, nargs="+", default=None, help="Per-shard camera-pair CSV files.")
    parser.add_argument("--output", type=Path, required=True, help="Merged summary JSON path.")
    parser.add_argument("--per-sample-csv", type=Path, default=None, help="Merged per-sample CSV path.")
    parser.add_argument("--camera-pair-csv", type=Path, default=None, help="Merged camera-pair CSV path.")
    parser.add_argument("--train-loss-csv", type=Path, default=None, help="Optional training loss.csv for comparison.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payloads = [read_json(path) for path in args.shard_json]
    rows = read_rows(args.shard_csv)
    camera_pair_rows = read_camera_pair_rows(args.shard_camera_csv or [])
    run_meta = collect_run_meta(payloads)

    runs = []
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_run[str(row.get("run", "unknown"))].append(row)

    for name in run_order(run_meta, rows_by_run):
        run_rows = rows_by_run.get(name, [])
        meta = run_meta.get(name, {})
        if not run_rows:
            continue
        dataset = str(run_rows[0].get("dataset") or meta.get("dataset") or "unknown")
        split = str(run_rows[0].get("split") or meta.get("split") or "")
        runs.append(
            {
                "name": name,
                "split": split,
                "curriculum_bins": meta.get("curriculum_bins", "all"),
                "dataset_size": meta.get("dataset_size"),
                "requested_samples": meta.get("requested_samples", 0),
                "candidate_samples": meta.get("candidate_samples"),
                "evaluated_samples": len(run_rows),
                "num_shards": max(int(payload.get("num_shards", 1)) for payload in payloads),
                "summary": summarize_values([float(row["loss"]) for row in run_rows]),
                "depth_summary": summarize_values([float(row["loss_depth"]) for row in run_rows]),
                "overlap_summary": summarize_values([float(row["loss_overlap"]) for row in run_rows]),
                "global_point_summary": summarize_values(
                    optional_numeric_values(run_rows, "loss_global_point")
                ),
                "camera_summary": summarize_values(optional_numeric_values(run_rows, "loss_camera")),
                "camera_translation_summary": summarize_values(
                    optional_numeric_values(run_rows, "loss_camera_t")
                ),
                "camera_translation_deg_summary": summarize_values(
                    optional_numeric_values(
                        run_rows,
                        "camera_translation_deg",
                        valid_count_key="camera_translation_valid_count",
                    )
                ),
                "camera_rotation_rad_summary": summarize_values(
                    optional_numeric_values(run_rows, "loss_camera_r")
                ),
                "camera_rotation_deg_summary": summarize_values(
                    optional_numeric_values(
                        run_rows,
                        "camera_rotation_deg",
                        valid_count_key="camera_rotation_valid_count",
                    )
                ),
                "valid_fraction_summary": summarize_values([float(row["valid_fraction"]) for row in run_rows]),
                "depth_metric_summary": summarize_metric_rows(run_rows, DEPTH_METRIC_KEYS),
                "panovggt_metric_summary": summarize_panovggt_rows(run_rows),
                "by_quality_bin": summarize_by_key(run_rows, "quality_bin", "loss"),
                "best_samples": {
                    "by_loss": rank_samples(run_rows, "loss", reverse=False),
                    "by_depth_irls_abs_rel": rank_samples(run_rows, "depth_irls_abs_rel", reverse=False),
                    "by_depth_irls_delta_1p25": rank_samples(run_rows, "depth_irls_delta_1p25", reverse=True),
                },
                "worst_samples": {
                    "by_loss": rank_samples(run_rows, "loss", reverse=True),
                    "by_depth_irls_abs_rel": rank_samples(run_rows, "depth_irls_abs_rel", reverse=True),
                    "by_depth_irls_delta_1p25": rank_samples(run_rows, "depth_irls_delta_1p25", reverse=False),
                },
                "dataset": dataset,
                "minimal_dataset": meta.get("minimal_dataset"),
            }
        )

    first = payloads[0] if payloads else {}
    result = {
        "config": first.get("config"),
        "checkpoint": first.get("checkpoint"),
        "device": "merged_shards",
        "seed": first.get("seed"),
        "limit_per_dataset": first.get("limit_per_dataset"),
        "num_shards": max(int(payload.get("num_shards", 1)) for payload in payloads) if payloads else 1,
        "dataset_root": first.get("dataset_root"),
        "sample_policy": first.get("sample_policy"),
        "pano_count_policy": first.get("pano_count_policy"),
        "dataset_pano_counts": first.get("dataset_pano_counts"),
        "split_policy": first.get("split_policy"),
        "train_loss_reference": read_train_loss_reference(args.train_loss_csv) or first.get("train_loss_reference"),
        "runs": runs,
        "overall": summarize_runs(runs),
        "panovggt_depth_benchmark": summarize_panovggt_benchmark(runs, rows),
        "panovggt_camera_benchmark": summarize_panovggt_camera_benchmark(runs, camera_pair_rows),
        "case_rankings": {
            "best_by_depth_irls_abs_rel": rank_samples(rows, "depth_irls_abs_rel", reverse=False, limit=20),
            "worst_by_depth_irls_abs_rel": rank_samples(rows, "depth_irls_abs_rel", reverse=True, limit=20),
            "best_by_depth_irls_delta_1p25": rank_samples(rows, "depth_irls_delta_1p25", reverse=True, limit=20),
            "worst_by_depth_irls_delta_1p25": rank_samples(rows, "depth_irls_delta_1p25", reverse=False, limit=20),
            "worst_by_loss": rank_samples(rows, "loss", reverse=True, limit=20),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_sample_csv is not None:
        write_per_sample_csv(args.per_sample_csv, rows)
    if args.camera_pair_csv is not None:
        write_camera_pair_csv(args.camera_pair_csv, camera_pair_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = coerce_row(raw)
                key = (str(row.get("run", "")), int(row.get("dataset_index", -1)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("run", "")), int(row.get("dataset_index", -1))))
    return rows


def read_camera_pair_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()
    numeric_fields = {
        "dataset_index",
        "batch_index",
        "pair_i",
        "pair_j",
        "camera_pose_rotation_deg",
        "camera_pose_translation_deg",
        "camera_pose_max_error_deg",
        "camera_pose_gt_translation_norm",
    }
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for field in numeric_fields:
                    value = row.get(field)
                    if value in (None, ""):
                        continue
                    number = float(value)
                    row[field] = int(number) if field in {"dataset_index", "batch_index", "pair_i", "pair_j"} else number
                key = (
                    str(row.get("run", "")),
                    int(row.get("dataset_index", -1)),
                    int(row.get("pair_i", -1)),
                    int(row.get("pair_j", -1)),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("run", "")), int(row.get("dataset_index", -1)), int(row.get("pair_i", -1)), int(row.get("pair_j", -1))))
    return rows


def write_camera_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMERA_PAIR_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for key in NUMERIC_FIELDS:
        value = out.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        out[key] = int(number) if key in {"dataset_index", "depth_valid_pixels"} else number
    return out


def optional_numeric_values(
    rows: list[dict[str, Any]],
    key: str,
    valid_count_key: str | None = None,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if valid_count_key is not None:
            try:
                if float(row.get(valid_count_key, 0.0)) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def collect_run_meta(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for run in payload.get("runs", []):
            name = str(run.get("name", "unknown"))
            current = meta.setdefault(name, {})
            for key in (
                "name",
                "split",
                "curriculum_bins",
                "dataset_size",
                "requested_samples",
                "candidate_samples",
                "dataset",
                "minimal_dataset",
            ):
                if key in run and run[key] is not None:
                    current[key] = run[key]
    return meta


def run_order(run_meta: dict[str, dict[str, Any]], rows_by_run: dict[str, list[dict[str, Any]]]) -> list[str]:
    names = list(run_meta.keys())
    for name in rows_by_run:
        if name not in run_meta:
            names.append(name)
    return names


if __name__ == "__main__":
    main()
