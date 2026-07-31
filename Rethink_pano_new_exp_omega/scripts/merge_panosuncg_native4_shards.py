#!/usr/bin/env python3
"""Merge deterministic PanoSUNCG native-four-window evaluation shards."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_panosuncg_native4 import (  # noqa: E402
    CSV_FIELDS,
    summarize_rows,
)
from scripts.evaluate_panosuncg_zeroshot import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    if len(args.shard_dir) < 1:
        raise ValueError("At least one shard directory is required")

    rows: list[dict[str, str]] = []
    configs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard_dir in args.shard_dir:
        csv_path = shard_dir / "per_sample_metrics.csv"
        summary_path = shard_dir / "metrics_summary.json"
        config_path = shard_dir / "run_config.json"
        for path in (csv_path, summary_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = read_json(summary_path)
        if summary.get("status") != "completed":
            raise RuntimeError(f"Incomplete shard: {summary_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            shard_rows = list(csv.DictReader(handle))
        if len(shard_rows) != int(summary["completed_samples"]):
            raise RuntimeError(
                f"CSV/summary count mismatch in {shard_dir}: "
                f"{len(shard_rows)} vs {summary['completed_samples']}"
            )
        rows.extend(shard_rows)
        configs.append(read_json(config_path))
        summaries.append(summary)

    split_hashes = {config["split_sha256"] for config in configs}
    total_counts = {int(config["total_split_samples"]) for config in configs}
    num_shards = {int(config["num_shards"]) for config in configs}
    ranks = {int(config["shard_rank"]) for config in configs}
    if len(split_hashes) != 1 or len(total_counts) != 1 or len(num_shards) != 1:
        raise RuntimeError("Shard run configurations do not describe the same evaluation")
    expected_shards = num_shards.pop()
    if ranks != set(range(expected_shards)):
        raise RuntimeError(f"Expected shard ranks 0..{expected_shards - 1}, got {sorted(ranks)}")
    total_split_samples = total_counts.pop()

    by_index: dict[int, dict[str, str]] = {}
    by_rgb: set[str] = set()
    for row in rows:
        index = int(row["sample_index"])
        rgb = row["rgb_relative_path"]
        if index in by_index or rgb in by_rgb:
            raise RuntimeError(f"Duplicate merged sample: index={index} rgb={rgb}")
        by_index[index] = row
        by_rgb.add(rgb)
    rows = [by_index[index] for index in sorted(by_index)]
    if len(rows) != total_split_samples:
        raise RuntimeError(
            f"Merged sample count is {len(rows)}, expected {total_split_samples}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = args.output_dir / "per_sample_metrics.csv"
    with merged_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    merged_config = {
        **configs[0],
        "shard_rank": None,
        "shard_samples": None,
        "num_shards": expected_shards,
        "merged_shard_directories": [str(path.resolve()) for path in args.shard_dir],
    }
    final = summarize_rows(
        rows,
        total_split_samples,
        total_split_samples=total_split_samples,
    )
    final["run_config"] = merged_config
    final["shard_summaries"] = [
        {
            "shard_rank": int(config["shard_rank"]),
            "completed_samples": int(summary["completed_samples"]),
        }
        for config, summary in sorted(
            zip(configs, summaries),
            key=lambda pair: int(pair[0]["shard_rank"]),
        )
    ]
    final["completed_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(args.output_dir / "run_config.json", merged_config)
    atomic_write_json(args.output_dir / "metrics_summary.json", final)
    atomic_write_json(args.output_dir / "progress.json", final)

    for shard_dir in args.shard_dir:
        error_path = shard_dir / "errors.jsonl"
        if error_path.is_file() and error_path.stat().st_size:
            shutil.copy2(
                error_path,
                args.output_dir / f"errors_{shard_dir.name}.jsonl",
            )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
