#!/usr/bin/env python3
"""Profile north/south polar-cap depth validity and distributions for mixed4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.pano_minimal import _read_depth_tensor  # noqa: E402


DATASETS = ("panocity", "matterport3d", "stanford2d3ds", "structured3d")
SPLITS = ("train", "val", "test")
PER_SAMPLE_FIELDS = (
    "dataset",
    "source_split",
    "sample_index",
    "scene_name",
    "depth_path",
    "pole",
    "latitude_threshold_degrees",
    "pixel_count",
    "valid_count",
    "valid_fraction",
    "positive_inf_fraction",
    "negative_inf_fraction",
    "nan_fraction",
    "nonpositive_finite_fraction",
    "depth_min_m",
    "depth_p05_m",
    "depth_p25_m",
    "depth_median_m",
    "depth_p75_m",
    "depth_p95_m",
    "depth_max_m",
    "depth_mean_m",
    "depth_std_m",
    "depth_iqr_m",
    "depth_relative_iqr",
    "near_constant_rel_iqr_le_1pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--latitude-threshold-degrees", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.latitude_threshold_degrees < 90.0:
        raise ValueError("--latitude-threshold-degrees must be in (0, 90)")
    if args.samples_per_dataset < 1:
        raise ValueError("--samples-per-dataset must be positive")

    all_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "seed": int(args.seed),
        "samples_per_dataset": int(args.samples_per_dataset),
        "polar_cap_definition": f"absolute ERP pixel-center latitude >= {args.latitude_threshold_degrees:g} degrees",
        "loader_depth_semantics": "meters; source zero and configured invalid sentinel become positive infinity",
        "near_constant_definitions": "finite positive cap depth IQR / median <= 0.01, 0.02, or 0.05",
        "datasets": {},
    }
    for dataset_offset, dataset_name in enumerate(DATASETS):
        items = collect_unique_items(args.dataset_root, dataset_name)
        rng = random.Random(args.seed + dataset_offset * 1009)
        selected = rng.sample(items, min(args.samples_per_dataset, len(items)))
        dataset_rows: list[dict[str, Any]] = []
        aggregate_values: dict[str, list[np.ndarray]] = {"north": [], "south": []}
        for sample_index, item in enumerate(selected):
            depth = _read_depth_tensor(
                Path(item["depth_path"]),
                output_depth_scale=float(item["output_depth_scale"]),
                invalid_depth_value=65535.0,
            )[0].numpy()
            north_mask, south_mask = polar_masks(depth.shape[0], depth.shape[1], args.latitude_threshold_degrees)
            for pole, mask in (("north", north_mask), ("south", south_mask)):
                row, finite_values = summarize_cap(depth[mask], pole, args.latitude_threshold_degrees)
                row.update(
                    {
                        "dataset": dataset_name,
                        "source_split": item["source_split"],
                        "sample_index": sample_index,
                        "scene_name": item.get("scene_name", ""),
                        "depth_path": str(item["depth_path"]),
                    }
                )
                dataset_rows.append(row)
                all_rows.append(row)
                if finite_values.size:
                    aggregate_values[pole].append(finite_values)
        result["datasets"][dataset_name] = {
            "available_unique_panoramas": len(items),
            "sampled_panoramas": len(selected),
            "source_split_counts": dict(sorted(Counter(item["source_split"] for item in selected).items())),
            "north": summarize_dataset_cap(dataset_rows, aggregate_values["north"], "north"),
            "south": summarize_dataset_cap(dataset_rows, aggregate_values["south"], "south"),
        }
        print(
            f"[polar-depth] dataset={dataset_name} sampled={len(selected)} "
            f"north_valid={result['datasets'][dataset_name]['north']['aggregate_valid_fraction']:.4f} "
            f"south_valid={result['datasets'][dataset_name]['south']['aggregate_valid_fraction']:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.per_sample_csv or args.output.with_name(f"{args.output.stem}_per_sample.csv")
    write_rows(csv_path, all_rows)
    print(f"[polar-depth] summary={args.output} per_sample={csv_path}", flush=True)


def collect_unique_items(root: Path, dataset_name: str) -> list[dict[str, Any]]:
    by_depth_path: dict[str, dict[str, Any]] = {}
    dataset_dir_name = {
        "panocity": "Panocity",
        "matterport3d": "Matterport3D",
        "stanford2d3ds": "Stanford2D3DS",
        "structured3d": "Structured3D",
    }[dataset_name]
    dataset_root = root / dataset_dir_name
    if dataset_name == "panocity":
        split_paths = [("all", dataset_root / "cache" / "panocity_all_index.json")]
    else:
        prefix = {"matterport3d": "matterport3d", "stanford2d3ds": "2d3ds", "structured3d": "structured3d"}[
            dataset_name
        ]
        split_paths = [(split, dataset_root / "cache" / f"{prefix}_{split}_index.json") for split in SPLITS]
    for split, index_path in split_paths:
        if not index_path.exists():
            continue
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        for item in depth_items_from_cache(dataset_name, dataset_root, rows):
            key = str(item["depth_path"])
            if key not in by_depth_path:
                by_depth_path[key] = {**item, "source_split": split}
    items = list(by_depth_path.values())
    if not items:
        raise FileNotFoundError(f"No indexed samples found for {dataset_name} under {root}")
    return items


def depth_items_from_cache(dataset_name: str, dataset_root: Path, rows: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if dataset_name == "panocity":
        for row in rows:
            depth_path = dataset_root / str(row["depth_path"])
            items.append(
                {
                    "depth_path": depth_path,
                    "scene_name": row.get("scene_name", depth_path.stem),
                    "output_depth_scale": 100.0,
                }
            )
    elif dataset_name == "matterport3d":
        for scan, _room_id, _room_name, pano_ids, _size in rows:
            for pano_id in pano_ids:
                depth_path = dataset_root / str(scan) / "pano_depth" / f"{pano_id}.png"
                items.append(
                    {"depth_path": depth_path, "scene_name": f"{scan}_{pano_id}", "output_depth_scale": 4000.0}
                )
    elif dataset_name == "stanford2d3ds":
        for area, _room_id, room_name, pano_ids, _size in rows:
            depth_dir = dataset_root / str(area) / "pano" / "depth"
            for pano_id in pano_ids:
                matches = sorted(depth_dir.glob(f"camera_{pano_id}_*_depth.png"))
                if matches:
                    items.append(
                        {
                            "depth_path": matches[0],
                            "scene_name": f"{area}_{room_name}_{pano_id}",
                            "output_depth_scale": 512.0,
                        }
                    )
    elif dataset_name == "structured3d":
        for scene, pano_ids, _size in rows:
            for pano_id in pano_ids:
                depth_path = (
                    dataset_root
                    / str(scene)
                    / "2D_rendering"
                    / str(pano_id)
                    / "panorama"
                    / "full"
                    / "depth.png"
                )
                items.append(
                    {"depth_path": depth_path, "scene_name": f"{scene}_{pano_id}", "output_depth_scale": 1000.0}
                )
    return items


def polar_masks(height: int, width: int, threshold_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    latitude = (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * 180.0
    north = np.broadcast_to((latitude >= threshold_degrees)[:, None], (height, width))
    south = np.broadcast_to((latitude <= -threshold_degrees)[:, None], (height, width))
    return north, south


def summarize_cap(values: np.ndarray, pole: str, threshold_degrees: float) -> tuple[dict[str, Any], np.ndarray]:
    values = values.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    valid = finite & (values > 0)
    finite_values = values[valid]
    count = int(values.size)
    row: dict[str, Any] = {
        "pole": pole,
        "latitude_threshold_degrees": float(threshold_degrees),
        "pixel_count": count,
        "valid_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "positive_inf_fraction": float(np.isposinf(values).mean()),
        "negative_inf_fraction": float(np.isneginf(values).mean()),
        "nan_fraction": float(np.isnan(values).mean()),
        "nonpositive_finite_fraction": float((finite & (values <= 0)).mean()),
    }
    if finite_values.size:
        q05, q25, median, q75, q95 = np.quantile(finite_values, [0.05, 0.25, 0.5, 0.75, 0.95])
        iqr = q75 - q25
        relative_iqr = iqr / max(abs(median), 1e-12)
        row.update(
            {
                "depth_min_m": float(finite_values.min()),
                "depth_p05_m": float(q05),
                "depth_p25_m": float(q25),
                "depth_median_m": float(median),
                "depth_p75_m": float(q75),
                "depth_p95_m": float(q95),
                "depth_max_m": float(finite_values.max()),
                "depth_mean_m": float(finite_values.mean()),
                "depth_std_m": float(finite_values.std()),
                "depth_iqr_m": float(iqr),
                "depth_relative_iqr": float(relative_iqr),
                "near_constant_rel_iqr_le_1pct": bool(relative_iqr <= 0.01),
            }
        )
    else:
        row.update({key: None for key in PER_SAMPLE_FIELDS if key.startswith("depth_")})
        row["near_constant_rel_iqr_le_1pct"] = False
    return row, finite_values.astype(np.float32, copy=False)


def summarize_dataset_cap(
    rows: list[dict[str, Any]],
    value_chunks: list[np.ndarray],
    pole: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["pole"] == pole]
    total_pixels = sum(int(row["pixel_count"]) for row in selected)
    total_valid = sum(int(row["valid_count"]) for row in selected)
    values = np.concatenate(value_chunks) if value_chunks else np.empty(0, dtype=np.float32)
    valid_rows = [row for row in selected if int(row["valid_count"]) > 0]
    result: dict[str, Any] = {
        "sample_count": len(selected),
        "aggregate_pixel_count": total_pixels,
        "aggregate_valid_count": total_valid,
        "aggregate_valid_fraction": total_valid / max(total_pixels, 1),
        "aggregate_positive_inf_fraction": weighted_row_fraction(selected, "positive_inf_fraction"),
        "aggregate_negative_inf_fraction": weighted_row_fraction(selected, "negative_inf_fraction"),
        "aggregate_nan_fraction": weighted_row_fraction(selected, "nan_fraction"),
        "aggregate_nonpositive_finite_fraction": weighted_row_fraction(selected, "nonpositive_finite_fraction"),
        "all_invalid_sample_fraction": sum(int(row["valid_count"]) == 0 for row in selected) / max(len(selected), 1),
        "at_least_95pct_invalid_sample_fraction": sum(float(row["valid_fraction"]) <= 0.05 for row in selected)
        / max(len(selected), 1),
        "near_constant_sample_fraction_among_valid_rel_iqr_le_1pct": near_constant_fraction(valid_rows, 0.01),
        "near_constant_sample_fraction_among_valid_rel_iqr_le_2pct": near_constant_fraction(valid_rows, 0.02),
        "near_constant_sample_fraction_among_valid_rel_iqr_le_5pct": near_constant_fraction(valid_rows, 0.05),
        "per_sample_valid_fraction": numeric_summary([float(row["valid_fraction"]) for row in selected]),
        "per_sample_median_depth_m": numeric_summary(
            [float(row["depth_median_m"]) for row in valid_rows if row.get("depth_median_m") is not None]
        ),
        "per_sample_relative_iqr": numeric_summary(
            [float(row["depth_relative_iqr"]) for row in valid_rows if row.get("depth_relative_iqr") is not None]
        ),
    }
    if values.size:
        rounded = np.round(values.astype(np.float64), 3)
        unique, counts = np.unique(rounded, return_counts=True)
        top = int(np.argmax(counts))
        result["valid_depth_m"] = numeric_summary(values.tolist())
        result["most_common_depth_rounded_1mm_m"] = float(unique[top])
        result["most_common_depth_rounded_1mm_fraction"] = float(counts[top] / values.size)
    else:
        result["valid_depth_m"] = {}
        result["most_common_depth_rounded_1mm_m"] = None
        result["most_common_depth_rounded_1mm_fraction"] = 0.0
    return result


def weighted_row_fraction(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["pixel_count"]) for row in rows)
    return sum(float(row[key]) * int(row["pixel_count"]) for row in rows) / max(total, 1)


def near_constant_fraction(rows: list[dict[str, Any]], threshold: float) -> float:
    eligible = [row for row in rows if row.get("depth_relative_iqr") is not None]
    return sum(float(row["depth_relative_iqr"]) <= threshold for row in eligible) / max(len(eligible), 1)


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
