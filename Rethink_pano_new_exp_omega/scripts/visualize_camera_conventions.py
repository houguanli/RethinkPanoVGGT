#!/usr/bin/env python3
"""Render native-vs-canonical GT-depth camera reprojection diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_camera_conventions import (  # noqa: E402
    DATASET_NAMES,
    erp_rays,
    load_pairs,
    prepare_pair_samples,
    reproject_sample_depth,
    score_candidate,
)
from training.data.pano_minimal import (  # noqa: E402
    PanoMinimalDataset,
    _CAMERA_CANONICAL_TO_NATIVE,
    _WORLD_NATIVE_TO_CANONICAL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    panels: list[np.ndarray] = []
    for offset, dataset_name in enumerate(DATASET_NAMES):
        dataset = PanoMinimalDataset(
            root=args.root,
            pano_sample_mode="fixed_neighborhood",
            pano_min_count=2,
            pano_max_count=2,
            grouping="nearest",
            split=args.split,
            split_seed=args.seed,
            datasets=dataset_name,
            canonicalize_camera=False,
            strict=True,
        )
        pairs = load_pairs(
            dataset,
            count=1,
            pairs_per_scene=1,
            height=args.height,
            width=args.width,
            depth_semantics="range",
            seed=args.seed + offset * 1009,
        )
        if not pairs:
            raise RuntimeError(f"No camera pair available for {dataset_name}")
        pair = pairs[0]
        samples = prepare_pair_samples(
            pairs,
            erp_rays(args.height, args.width),
            points_per_direction=args.height * args.width,
            max_depth_m=80.0,
            seed=args.seed + offset * 3001,
        )
        sample = samples[0]
        identity = np.eye(3, dtype=np.float64)
        official = _CAMERA_CANONICAL_TO_NATIVE[dataset_name]
        native_view = reproject_sample_depth(sample, identity, 1.0)
        canonical_view = reproject_sample_depth(sample, official, 1.0)
        native_metrics = score_candidate(samples, identity, 1.0)
        canonical_metrics = score_candidate(samples, official, 1.0)
        panel = make_panel(dataset_name, pair, native_view, canonical_view)
        output_path = args.output_dir / f"{dataset_name}_gt_depth_reprojection.png"
        cv2.imwrite(str(output_path), panel)
        panels.append(panel)

        world_basis = _WORLD_NATIVE_TO_CANONICAL[dataset_name]
        camera_basis = _CAMERA_CANONICAL_TO_NATIVE[dataset_name]
        pair_summary = []
        for record in (pair.first, pair.second):
            center_canonical = world_basis @ record.center
            rotation_canonical = world_basis @ record.rotation_c2w @ camera_basis
            center_roundtrip = world_basis.T @ center_canonical
            rotation_roundtrip = world_basis.T @ rotation_canonical @ camera_basis.T
            pair_summary.append(
                {
                    "name": record.name,
                    "center_native": record.center.tolist(),
                    "center_canonical": center_canonical.tolist(),
                    "rotation_c2w_native": record.rotation_c2w.tolist(),
                    "rotation_c2w_canonical": rotation_canonical.tolist(),
                    "center_roundtrip_max_error": float(np.max(np.abs(center_roundtrip - record.center))),
                    "rotation_roundtrip_max_error": float(
                        np.max(np.abs(rotation_roundtrip - record.rotation_c2w))
                    ),
                }
            )
        summaries[dataset_name] = {
            "scene": pair.scene,
            "first": pair.first.name,
            "second": pair.second.name,
            "world_native_to_canonical": world_basis.tolist(),
            "camera_canonical_to_native": camera_basis.tolist(),
            "native_identity_metrics": native_metrics,
            "official_canonical_metrics": canonical_metrics,
            "cameras": pair_summary,
            "visualization": str(output_path.resolve()),
        }
        print(f"[camera-vis] wrote {output_path.resolve()}")

    max_width = max(panel.shape[1] for panel in panels)
    combined = np.concatenate([pad_width(panel, max_width) for panel in panels], axis=0)
    combined_path = args.output_dir / "mixed4_gt_depth_camera_conversion.png"
    cv2.imwrite(str(combined_path), combined)
    summary_path = args.output_dir / "camera_transform_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[camera-vis] wrote {combined_path.resolve()}")
    print(f"[camera-vis] wrote {summary_path.resolve()}")


def make_panel(dataset_name, pair, native_view, canonical_view) -> np.ndarray:
    target_depth = native_view["target_depth"]
    source_depth = pair.first.depth
    finite_depth = np.concatenate(
        [source_depth[np.isfinite(source_depth)], target_depth[np.isfinite(target_depth)]]
    )
    depth_max = float(np.quantile(finite_depth, 0.98)) if finite_depth.size else 10.0
    tiles = [
        depth_tile(source_depth, depth_max, "Source GT depth"),
        depth_tile(target_depth, depth_max, "Target GT depth"),
        depth_tile(native_view["reprojected_depth"], depth_max, "Native identity reprojection"),
        depth_tile(canonical_view["reprojected_depth"], depth_max, "Official canonical reprojection"),
        error_tile(native_view["abs_log_error"], "Native abs-log error"),
        error_tile(canonical_view["abs_log_error"], "Canonical abs-log error"),
        mask_tile(native_view["visible_mask"], "Native visible surface"),
        mask_tile(canonical_view["visible_mask"], "Canonical visible surface"),
    ]
    top = np.concatenate(tiles[:4], axis=1)
    bottom = np.concatenate(tiles[4:], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    title = f"{dataset_name} | {pair.first.name} -> {pair.second.name}"
    return add_header(panel, title)


def depth_tile(depth: np.ndarray, maximum: float, label: str) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = np.log1p(np.clip(depth[valid], 0.0, maximum)) / np.log1p(maximum)
        normalized[valid] = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (24, 24, 24)
    return add_label(colored, label)


def error_tile(error: np.ndarray, label: str) -> np.ndarray:
    valid = np.isfinite(error)
    normalized = np.zeros(error.shape, dtype=np.uint8)
    normalized[valid] = np.clip(error[valid] / 0.5 * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    colored[~valid] = (24, 24, 24)
    return add_label(colored, label)


def mask_tile(mask: np.ndarray, label: str) -> np.ndarray:
    colored = np.full((*mask.shape, 3), 24, dtype=np.uint8)
    colored[mask] = (80, 210, 100)
    return add_label(colored, label)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (18, 18, 18), thickness=-1)
    cv2.putText(output, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return output


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((34, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def pad_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    return cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18))


if __name__ == "__main__":
    main()
