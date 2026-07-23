#!/usr/bin/env python3
"""Read one real mixed4 multi-pano sample before launching an ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.train_pano_omega import build_dataset, normalize_pano_sampling_args, parse_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--probe-dataset",
        default="matterport3d",
        choices=["matterport3d", "stanford2d3ds", "structured3d", "panocity"],
        help="Dataset used for the actual two-panorama read.",
    )
    cli = parser.parse_args()

    train_argv = [
        "--config",
        str(cli.config),
        "--dataset-max-samples",
        str(cli.max_samples),
        "--minimal-datasets",
        cli.probe_dataset,
        "--pano-sample-mode",
        "fixed_neighborhood",
        "--pano-min-count",
        "2",
        "--pano-max-count",
        "2",
        "--no-randomize-pano-order",
    ]
    if cli.dataset_root is not None:
        train_argv.extend(["--dataset-root", str(cli.dataset_root)])
    args = parse_args(train_argv)
    normalize_pano_sampling_args(args)
    root = Path(args.dataset_root)
    expected_indexes = {
        "matterport3d": root / "Matterport3D" / "cache" / "matterport3d_train_index.json",
        "stanford2d3ds": root / "Stanford2D3DS" / "cache" / "2d3ds_train_index.json",
        "structured3d": root / "Structured3D" / "cache" / "structured3d_train_index.json",
        "panocity": root / "Panocity" / "cache" / "panocity_train_index.json",
    }
    missing = [str(path) for path in expected_indexes.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing mixed4 indexes: {missing}")

    dataset = build_dataset(args, pano_size=None)
    sample = dataset[0]

    image = sample["pano_image"]
    depth = sample["pano_depth"]
    dataset_counts = {}
    for item in getattr(dataset, "items", []):
        name = str(item.get("dataset") or item.get("dataset_name") or "unknown").lower()
        dataset_counts[name] = dataset_counts.get(name, 0) + 1

    print(f"dataset_root={Path(args.dataset_root).resolve()}")
    print(f"mixed4_indexes_ok={list(expected_indexes)}")
    print(f"probe_dataset={cli.probe_dataset}")
    print(f"dataset_length={len(dataset)}")
    print(f"indexed_datasets={dataset_counts}")
    print(f"pano_image_shape={tuple(image.shape)}")
    print(f"pano_depth_shape={tuple(depth.shape)}")
    print(f"depth_finite_ratio={float(torch.isfinite(depth).float().mean()):.6f}")
    print(f"scene_name={sample['scene_name']}")


if __name__ == "__main__":
    main()
