#!/usr/bin/env python3
"""Select validation cases and export reconstructions for visualization."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME_MAP = {
    "panocity": "panocity",
    "matterport3d": "matterport3d",
    "stanford2d3ds": "stanford2d3ds",
    "structured3d": "structured3d",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pick good/bad validation samples from mixed4 validation outputs and optionally run "
            "reconstruct_pano_omega.py for each selected case."
        )
    )
    parser.add_argument("--summary", type=Path, required=True, help="Validation summary.json.")
    parser.add_argument(
        "--per-sample-csv",
        type=Path,
        default=None,
        help="Validation per_sample.csv. Defaults to summary parent/per_sample.csv.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint passed to reconstruct_pano_omega.py.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Mixed4 dataset root. Defaults to dataset_root stored in summary.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for selected_cases.* and recon outputs. Defaults to summary parent/reconstruct_cases_<metric>.",
    )
    parser.add_argument(
        "--metric",
        default="depth_irls_abs_rel",
        help="Metric used for ranking. Typical: depth_irls_abs_rel, depth_irls_delta_1p25, loss.",
    )
    parser.add_argument(
        "--order",
        choices=["auto", "min", "max"],
        default="auto",
        help="Ranking direction. auto minimizes loss/error/rmse/abs_rel and maximizes delta.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names to include, or all.",
    )
    parser.add_argument(
        "--per-dataset-k",
        type=int,
        default=2,
        help="Select this many cases per dataset. Set 0 to disable per-dataset selection.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Also select this many global cases across all datasets.",
    )
    parser.add_argument("--run", action="store_true", help="Run reconstruction for selected cases.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable for reconstruction.")
    parser.add_argument(
        "--reconstruct-script",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "reconstruct_pano_omega.py",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument(
        "--fit-pred-depth-scale-from-target",
        action="store_true",
        help="Pass through to reconstruct_pano_omega.py for GT-aligned visual comparison.",
    )
    parser.add_argument(
        "--mask-pred-by-target-valid",
        action="store_true",
        help="Pass through to reconstruct_pano_omega.py to suppress invalid GT regions in pred export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = read_json(args.summary)
    per_sample_csv = args.per_sample_csv or args.summary.parent / "per_sample.csv"
    rows = read_rows(per_sample_csv)
    if not rows:
        raise ValueError(f"No validation rows found in {per_sample_csv}")

    dataset_root = args.dataset_root or Path(str(summary.get("dataset_root", "")))
    if not str(dataset_root):
        raise ValueError("--dataset-root is required when summary.json does not contain dataset_root.")
    output_root = args.output_root or args.summary.parent / f"reconstruct_cases_{safe_name(args.metric)}"
    output_root.mkdir(parents=True, exist_ok=True)

    selected = select_rows(rows, args)
    if not selected:
        raise ValueError("No cases selected. Check --datasets, --metric, --per-dataset-k, and --top-k.")
    selected = add_reconstruction_commands(selected, args, dataset_root, output_root)

    write_selected_cases(output_root, selected, args, dataset_root)
    print(f"[INFO] selected cases = {len(selected)}")
    print(f"[INFO] wrote {output_root / 'selected_cases.json'}")
    print(f"[INFO] wrote {output_root / 'selected_cases.csv'}")

    if args.run:
        for index, row in enumerate(selected, 1):
            print(f"[INFO] reconstruct {index}/{len(selected)} {row['dataset']} {row['seq_name']}")
            subprocess.run(row["command"], check=True)
        print(f"[INFO] reconstruction output root = {output_root}")
    else:
        print("[INFO] add --run to export reconstruction files.")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    metric = args.metric
    requested_datasets = parse_datasets(args.datasets)
    filtered = [
        row
        for row in rows
        if row.get(metric, "") not in ("", "nan", "NaN", "None")
        and (requested_datasets is None or normalize_dataset(row.get("dataset", "")) in requested_datasets)
    ]
    reverse = ranking_reverse(metric, args.order)
    filtered.sort(key=lambda row: float(row[metric]), reverse=reverse)

    selected: list[dict] = []
    seen = set()
    if args.per_dataset_k > 0:
        by_dataset: dict[str, list[dict]] = {}
        for row in filtered:
            by_dataset.setdefault(normalize_dataset(row.get("dataset", "")), []).append(row)
        for dataset in sorted(by_dataset):
            for row in by_dataset[dataset][: args.per_dataset_k]:
                append_unique(selected, seen, row)
    if args.top_k > 0:
        for row in filtered[: args.top_k]:
            append_unique(selected, seen, row)
    return selected


def append_unique(selected: list[dict], seen: set[tuple[str, str, str]], row: dict) -> None:
    key = (row.get("dataset", ""), row.get("split", ""), row.get("rgb_path", row.get("seq_name", "")))
    if key in seen:
        return
    seen.add(key)
    selected.append(dict(row))


def parse_datasets(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {normalize_dataset(item) for item in value.split(",") if item.strip()}


def normalize_dataset(value: str) -> str:
    return str(value).strip().lower()


def ranking_reverse(metric: str, order: str) -> bool:
    if order == "max":
        return True
    if order == "min":
        return False
    metric_lower = metric.lower()
    return "delta" in metric_lower or metric_lower.endswith("accuracy")


def add_reconstruction_commands(
    rows: list[dict],
    args: argparse.Namespace,
    dataset_root: Path,
    output_root: Path,
) -> list[dict]:
    enriched = []
    for rank, row in enumerate(rows, 1):
        dataset_key = normalize_dataset(row.get("dataset", ""))
        minimal_dataset = DATASET_NAME_MAP.get(dataset_key)
        if minimal_dataset is None:
            raise ValueError(f"Unknown mixed4 dataset name: {row.get('dataset')!r}")
        split = row.get("split") or "val"
        seq_name = row.get("seq_name") or Path(row.get("rgb_path", f"case_{rank}")).stem
        case_dir = output_root / f"{rank:03d}_{minimal_dataset}_{safe_name(seq_name)}"
        command = [
            str(args.python),
            str(args.reconstruct_script),
            "--dataset-root",
            str(dataset_root),
            "--dataset-format",
            "pano_minimal",
            "--minimal-datasets",
            minimal_dataset,
            "--dataset-split",
            split,
            "--sample-rgb-path",
            row["rgb_path"],
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(case_dir),
            "--device",
            args.device,
            "--max-points",
            str(args.max_points),
        ]
        if args.fit_pred_depth_scale_from_target:
            command.append("--fit-pred-depth-scale-from-target")
        if args.mask_pred_by_target_valid:
            command.append("--mask-pred-by-target-valid")
        row = dict(row)
        row["rank"] = rank
        row["minimal_dataset"] = minimal_dataset
        row["reconstruction_dir"] = str(case_dir)
        row["command"] = command
        enriched.append(row)
    return enriched


def write_selected_cases(output_root: Path, rows: list[dict], args: argparse.Namespace, dataset_root: Path) -> None:
    metadata = {
        "summary": str(args.summary),
        "per_sample_csv": str(args.per_sample_csv or args.summary.parent / "per_sample.csv"),
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "metric": args.metric,
        "order": args.order,
        "datasets": args.datasets,
        "per_dataset_k": args.per_dataset_k,
        "top_k": args.top_k,
        "cases": rows,
    }
    with (output_root / "selected_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    fieldnames = sorted({key for row in rows for key in row.keys() if key != "command"})
    fieldnames.append("command")
    with (output_root / "selected_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["command"] = shell_join(row["command"])
            writer.writerow(csv_row)


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shell_quote(str(part)) for part in parts)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:")
    if all(ch in safe_chars for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def safe_name(value: str) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return safe[:140] or "case"


if __name__ == "__main__":
    main()
