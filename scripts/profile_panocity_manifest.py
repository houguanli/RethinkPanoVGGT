#!/usr/bin/env python3
"""Profile PanoCity paired RGB/depth data and write a curriculum manifest.

The output JSONL is intentionally simple so both the LUNA and baseline01
datasets can consume it without extra dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "PanoCity_paired")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-depth-scale", type=float, default=1000.0)
    parser.add_argument("--invalid-depth-value", type=float, default=65535.0)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    parser.add_argument("--structure-size", type=int, default=256)
    parser.add_argument("--bad-valid-ratio", type=float, default=0.10)
    parser.add_argument("--hard-valid-ratio", type=float, default=0.35)
    parser.add_argument("--clean-valid-ratio", type=float, default=0.65)
    parser.add_argument("--hard-far-ratio", type=float, default=0.60)
    return parser


def iter_image_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    entries = []
    with os.scandir(folder) as scan:
        for entry in scan:
            if entry.is_file():
                path = Path(entry.path)
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    entries.append(path)
    return sorted(entries)


def depth_path_for_rgb(rgb_path: Path, depth_dir: Path) -> Path | None:
    token = rgb_path.name.split("_", 1)[0]
    candidates = [
        depth_dir / rgb_path.name,
        depth_dir / rgb_path.name.replace("_rgb_", "_depth_"),
        depth_dir / f"{token}_depth_{token}.png",
        depth_dir / f"{token}_pano_{token}.png",
        depth_dir / f"{rgb_path.stem}.png",
    ]
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def build_pairs(root: Path, max_samples: int | None) -> list[tuple[Path, Path]]:
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    pairs = []
    seen_depth = set()
    for rgb_path in iter_image_files(rgb_dir):
        depth_path = depth_path_for_rgb(rgb_path, depth_dir)
        if depth_path is None:
            continue
        depth_key = str(depth_path.resolve())
        if depth_key in seen_depth:
            continue
        seen_depth.add(depth_key)
        pairs.append((rgb_path, depth_path))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    return pairs


def finite_quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.quantile(values, q))


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def profile_pair(payload: tuple[str, str, str, float, float, float, int]) -> dict[str, Any]:
    rgb_raw, depth_raw, root_raw, output_depth_scale, invalid_depth_value, depth_max_m, structure_size = payload
    rgb_path = Path(rgb_raw)
    depth_path = Path(depth_raw)
    root = Path(root_raw)
    name = rgb_path.stem
    entry: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "rgb_rel": safe_rel(rgb_path, root),
        "depth_rel": safe_rel(depth_path, root),
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
    }

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        entry.update({"readable": False, "error": f"Cannot read depth: {depth_path}"})
        return entry
    if depth.ndim == 3:
        depth = depth[..., 0]
    raw = depth.astype(np.float32)
    depth_m = raw / float(output_depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if invalid_depth_value is not None:
        valid &= raw < float(invalid_depth_value)
    if depth_max_m > 0:
        valid &= depth_m <= float(depth_max_m)

    height, width = raw.shape[:2]
    valid_count = int(valid.sum())
    total = int(valid.size)
    valid_ratio = float(valid_count / max(total, 1))
    valid_depth = depth_m[valid]
    far_ratio = float(((valid_depth > 60.0).sum() / max(valid_count, 1))) if valid_count else 0.0
    near_ratio = float(((valid_depth < 0.5).sum() / max(valid_count, 1))) if valid_count else 0.0

    third = max(1, height // 3)
    top_valid_ratio = float(valid[:third].mean()) if height > 0 else 0.0
    middle_valid_ratio = float(valid[third : 2 * third].mean()) if height >= 3 else valid_ratio
    bottom_valid_ratio = float(valid[2 * third :].mean()) if height >= 3 else valid_ratio

    structure_score = 0.0
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    if rgb is not None:
        scale = float(structure_size) / max(float(rgb.shape[1]), 1.0)
        resized = cv2.resize(
            rgb,
            (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        grad_x = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        structure_score = float(np.sqrt(grad_x * grad_x + grad_y * grad_y).mean() / 255.0)

    entry.update(
        {
            "readable": True,
            "width": int(width),
            "height": int(height),
            "valid_ratio": valid_ratio,
            "invalid_ratio": 1.0 - valid_ratio,
            "valid_pixels": valid_count,
            "total_pixels": total,
            "top_valid_ratio": top_valid_ratio,
            "middle_valid_ratio": middle_valid_ratio,
            "bottom_valid_ratio": bottom_valid_ratio,
            "near_ratio_lt_0p5m": near_ratio,
            "far_ratio_gt_60m": far_ratio,
            "mean_depth_m": float(valid_depth.mean()) if valid_count else 0.0,
            "p50_depth_m": finite_quantile(valid_depth, 0.50),
            "p90_depth_m": finite_quantile(valid_depth, 0.90),
            "p95_depth_m": finite_quantile(valid_depth, 0.95),
            "p99_depth_m": finite_quantile(valid_depth, 0.99),
            "structure_score": structure_score,
        }
    )
    return entry


def numeric_values(entries: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def assign_curriculum(entries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    valid_entries = [entry for entry in entries if entry.get("readable")]
    structure_q = quantiles(numeric_values(valid_entries, "structure_score"))
    structure_q10 = structure_q.get("q10", 0.0)
    structure_q50 = structure_q.get("q50", 0.0)

    for entry in entries:
        if not entry.get("readable"):
            quality_bin = "bad"
            quality_score = 0.0
            sample_weight = 0.0
        else:
            valid_ratio = float(entry.get("valid_ratio", 0.0))
            structure = float(entry.get("structure_score", 0.0))
            far_ratio = float(entry.get("far_ratio_gt_60m", 0.0))
            low_structure = structure < structure_q10

            if valid_ratio < args.bad_valid_ratio:
                quality_bin = "bad"
            elif valid_ratio < args.hard_valid_ratio or low_structure or far_ratio > args.hard_far_ratio:
                quality_bin = "hard"
            elif valid_ratio >= args.clean_valid_ratio and structure >= structure_q50 and far_ratio <= 0.35:
                quality_bin = "clean"
            else:
                quality_bin = "normal"

            structure_norm = 0.5
            if structure_q50 > 0:
                structure_norm = min(max(structure / (2.0 * structure_q50), 0.0), 1.0)
            quality_score = (
                0.65 * min(max(valid_ratio, 0.0), 1.0)
                + 0.25 * structure_norm
                + 0.10 * (1.0 - min(max(far_ratio, 0.0), 1.0))
            )
            weights = {"bad": 0.0, "hard": 0.50, "normal": 1.0, "clean": 1.10}
            sample_weight = weights[quality_bin]

        entry["quality_bin"] = quality_bin
        entry["quality_score"] = float(quality_score)
        entry["sample_weight"] = float(sample_weight)

    return {
        "structure_score": structure_q,
        "structure_q10": structure_q10,
        "structure_q50": structure_q50,
    }


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(path: Path, root: Path, entries: list[dict[str, Any]], thresholds: dict[str, Any], elapsed: float) -> None:
    bins = Counter(str(entry.get("quality_bin", "unknown")) for entry in entries)
    summary = {
        "schema_version": 1,
        "root": str(root),
        "created_at_unix": time.time(),
        "elapsed_seconds": elapsed,
        "total_entries": len(entries),
        "readable_entries": sum(1 for entry in entries if entry.get("readable")),
        "quality_bins": dict(sorted(bins.items())),
        "thresholds": thresholds,
        "metrics": {
            key: quantiles(numeric_values(entries, key))
            for key in [
                "valid_ratio",
                "invalid_ratio",
                "top_valid_ratio",
                "middle_valid_ratio",
                "bottom_valid_ratio",
                "far_ratio_gt_60m",
                "mean_depth_m",
                "p95_depth_m",
                "structure_score",
                "quality_score",
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    root = args.root
    output_jsonl = args.output_jsonl or (root / "metadata" / "panocity_manifest.jsonl")
    summary_json = args.summary_json or (root / "metadata" / "panocity_manifest_summary.json")

    started = time.time()
    pairs = build_pairs(root, args.max_samples)
    if not pairs:
        raise SystemExit(f"No paired RGB/depth files found under {root}")

    payloads = [
        (
            str(rgb),
            str(depth),
            str(root),
            float(args.output_depth_scale),
            float(args.invalid_depth_value),
            float(args.depth_max_m),
            int(args.structure_size),
        )
        for rgb, depth in pairs
    ]

    entries: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = map(profile_pair, payloads)
        for entry in tqdm(iterator, total=len(payloads), desc="Profiling PanoCity"):
            entries.append(entry)
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for entry in tqdm(pool.map(profile_pair, payloads, chunksize=32), total=len(payloads), desc="Profiling PanoCity"):
                entries.append(entry)

    thresholds = assign_curriculum(entries, args)
    elapsed = time.time() - started
    write_jsonl(output_jsonl, entries)
    write_summary(summary_json, root, entries, thresholds, elapsed)

    bins = Counter(str(entry.get("quality_bin", "unknown")) for entry in entries)
    print(f"[INFO] wrote manifest = {output_jsonl}")
    print(f"[INFO] wrote summary = {summary_json}")
    print(f"[INFO] entries = {len(entries)} readable = {sum(1 for e in entries if e.get('readable'))}")
    print(f"[INFO] quality_bins = {dict(sorted(bins.items()))}")
    print(f"[INFO] elapsed_seconds = {elapsed:.1f}")


if __name__ == "__main__":
    main()
