#!/usr/bin/env python3
"""Merge sharded baseline no-split full-ERP evaluation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_mixed4_depth_checkpoint import (
    CAMERA_PAIR_CSV_FIELDS,
    ERP_COVERAGE_KEYS,
    ERP_DEPTH_METRIC_KEYS,
    PER_SAMPLE_CSV_FIELDS,
    append_csv_rows,
    initialize_csv,
    load_luna_metric_helpers,
    load_resume_rows,
    summarize_camera_pose_pairs,
    summarize_erp_panovggt_benchmark,
    summarize_panovggt_benchmark,
    summarize_runs,
    write_per_sample_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-json", type=Path, nargs="+", required=True)
    parser.add_argument("--shard-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--shard-camera-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-csv", type=Path, required=True)
    parser.add_argument("--camera-pair-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.shard_json) == len(args.shard_csv) == len(args.shard_camera_csv)):
        raise ValueError("Shard JSON/sample CSV/camera CSV counts must match")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.shard_json]
    validate_payloads(payloads)

    rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    seen_samples: set[tuple[str, str, int]] = set()
    seen_pairs: set[tuple[str, str, int, int, int, int]] = set()
    for sample_path, camera_path in zip(args.shard_csv, args.shard_camera_csv):
        shard_rows, shard_camera_rows, _ = load_resume_rows(sample_path, camera_path)
        for row in shard_rows:
            key = (str(row["minimal_dataset"]), str(row["split"]), int(row["dataset_index"]))
            if key in seen_samples:
                raise ValueError(f"Duplicate sample across shards: {key}")
            seen_samples.add(key)
            rows.append(row)
        for row in shard_camera_rows:
            key = (
                str(row["minimal_dataset"]),
                str(row["split"]),
                int(row["dataset_index"]),
                int(row["batch_index"]),
                int(row["pair_i"]),
                int(row["pair_j"]),
            )
            if key in seen_pairs:
                raise ValueError(f"Duplicate camera pair across shards: {key}")
            seen_pairs.add(key)
            camera_rows.append(row)

    rows.sort(key=lambda row: (str(row["minimal_dataset"]), int(row["dataset_index"])))
    camera_rows.sort(
        key=lambda row: (
            str(row["minimal_dataset"]),
            int(row["dataset_index"]),
            int(row["batch_index"]),
            int(row["pair_i"]),
            int(row["pair_j"]),
        )
    )
    metrics_lib = load_luna_metric_helpers()
    runs = merge_runs(payloads, rows, camera_rows, metrics_lib)
    first = dict(payloads[0])
    skipped = [item for payload in payloads for item in payload.get("skipped_samples", [])]
    result = {
        **first,
        "device": "merged_4gpu_shards",
        "resumed": any(bool(payload.get("resumed")) for payload in payloads),
        "num_shards": len(payloads),
        "shard_rank": None,
        "runs": runs,
        "overall": summarize_runs(runs),
        "panovggt_depth_benchmark": summarize_panovggt_benchmark(runs, rows, metrics_lib),
        "panovggt_covered_erp_depth_benchmark": summarize_erp_panovggt_benchmark(
            runs, rows, metrics_lib
        ),
        "panovggt_camera_pose_summary": summarize_camera_pose_pairs(camera_rows),
        "case_rankings": {
            "best_by_depth_irls_abs_rel": metrics_lib.rank_samples(
                rows, "depth_irls_abs_rel", reverse=False, limit=20
            ),
            "worst_by_depth_irls_abs_rel": metrics_lib.rank_samples(
                rows, "depth_irls_abs_rel", reverse=True, limit=20
            ),
            "best_by_depth_irls_delta_1p25": metrics_lib.rank_samples(
                rows, "depth_irls_delta_1p25", reverse=True, limit=20
            ),
            "worst_by_depth_irls_delta_1p25": metrics_lib.rank_samples(
                rows, "depth_irls_delta_1p25", reverse=False, limit=20
            ),
            "worst_by_loss": metrics_lib.rank_samples(rows, "loss", reverse=True, limit=20),
        },
        "skipped_samples": skipped,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_per_sample_csv(args.per_sample_csv, rows, metrics_lib)
    initialize_csv(args.camera_pair_csv, CAMERA_PAIR_CSV_FIELDS)
    append_csv_rows(args.camera_pair_csv, CAMERA_PAIR_CSV_FIELDS, camera_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def validate_payloads(payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        raise ValueError("At least one shard payload is required")
    compatibility_keys = (
        "config",
        "checkpoint",
        "seed",
        "limit_per_dataset",
        "datasets",
        "sample_policy",
        "pano_count_policy",
        "dataset_pano_counts",
        "latitude_band_deg",
        "depth_evaluation_domain",
        "depth_pixel_weighting",
    )
    first = payloads[0]
    for shard_index, payload in enumerate(payloads[1:], start=1):
        mismatched = [key for key in compatibility_keys if payload.get(key) != first.get(key)]
        if mismatched:
            raise ValueError(f"Shard {shard_index} protocol mismatch: {mismatched}")


def merge_runs(
    payloads: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    metrics_lib: Any,
) -> list[dict[str, Any]]:
    meta_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for payload in payloads:
        for run in payload.get("runs", []):
            name = str(run["name"])
            if name not in meta_by_name:
                order.append(name)
            meta_by_name[name].append(run)

    rows_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_name[str(row["run"])].append(row)
    cameras_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in camera_rows:
        cameras_by_name[str(row["run"])].append(row)

    merged: list[dict[str, Any]] = []
    for name in order:
        run_rows = rows_by_name.get(name, [])
        if not run_rows:
            continue
        metas = meta_by_name[name]
        base = metas[0]
        merged.append(
            {
                "name": name,
                "dataset": base.get("dataset"),
                "minimal_dataset": base.get("minimal_dataset"),
                "split": base.get("split"),
                "dataset_size": base.get("dataset_size"),
                "requested_samples": base.get("requested_samples"),
                "candidate_samples": sum(int(meta.get("candidate_samples", 0)) for meta in metas),
                "global_candidate_samples": max(
                    int(meta.get("global_candidate_samples", 0)) for meta in metas
                ),
                "evaluated_samples": len(run_rows),
                "resumed_samples": sum(int(meta.get("resumed_samples", 0)) for meta in metas),
                "effective_pano_min_count": base.get("effective_pano_min_count"),
                "effective_pano_max_count": base.get("effective_pano_max_count"),
                "windows_per_pano": base.get("windows_per_pano"),
                "img_per_seq": base.get("img_per_seq"),
                "num_shards": len(payloads),
                "shard_rank": None,
                "skipped_samples": sum(int(meta.get("skipped_samples", 0)) for meta in metas),
                "summary": metrics_lib.summarize_values([row["loss"] for row in run_rows]),
                "depth_summary": metrics_lib.summarize_values(
                    [row["loss_depth"] for row in run_rows]
                ),
                "overlap_summary": metrics_lib.summarize_values(
                    [row["loss_overlap"] for row in run_rows]
                ),
                "valid_fraction_summary": metrics_lib.summarize_values(
                    [row["valid_fraction"] for row in run_rows]
                ),
                "camera_loss_summary": metrics_lib.summarize_values(
                    [row["loss_camera"] for row in run_rows]
                ),
                "camera_T_summary": metrics_lib.summarize_values(
                    [row["loss_T"] for row in run_rows]
                ),
                "camera_R_summary": metrics_lib.summarize_values(
                    [row["loss_R"] for row in run_rows]
                ),
                "camera_pose_summary": summarize_camera_pose_pairs(cameras_by_name.get(name, [])),
                "depth_metric_summary": metrics_lib.summarize_metric_rows(
                    run_rows, metrics_lib.DEPTH_METRIC_KEYS
                ),
                "erp_depth_metric_summary": metrics_lib.summarize_metric_rows(
                    run_rows, ERP_DEPTH_METRIC_KEYS
                ),
                "erp_coverage_summary": metrics_lib.summarize_metric_rows(
                    run_rows, ERP_COVERAGE_KEYS
                ),
                "panovggt_metric_summary": metrics_lib.summarize_panovggt_rows(run_rows),
                "best_samples": {
                    "by_loss": metrics_lib.rank_samples(run_rows, "loss", reverse=False),
                    "by_depth_irls_abs_rel": metrics_lib.rank_samples(
                        run_rows, "depth_irls_abs_rel", reverse=False
                    ),
                    "by_depth_irls_delta_1p25": metrics_lib.rank_samples(
                        run_rows, "depth_irls_delta_1p25", reverse=True
                    ),
                    "by_erp_depth_irls_abs_rel": metrics_lib.rank_samples(
                        run_rows, "erp_depth_irls_abs_rel", reverse=False
                    ),
                },
                "worst_samples": {
                    "by_loss": metrics_lib.rank_samples(run_rows, "loss", reverse=True),
                    "by_depth_irls_abs_rel": metrics_lib.rank_samples(
                        run_rows, "depth_irls_abs_rel", reverse=True
                    ),
                    "by_depth_irls_delta_1p25": metrics_lib.rank_samples(
                        run_rows, "depth_irls_delta_1p25", reverse=False
                    ),
                    "by_erp_depth_irls_abs_rel": metrics_lib.rank_samples(
                        run_rows, "erp_depth_irls_abs_rel", reverse=True
                    ),
                },
            }
        )
    return merged


if __name__ == "__main__":
    main()
