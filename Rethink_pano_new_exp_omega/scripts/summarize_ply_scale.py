#!/usr/bin/env python3
"""Print radius statistics for one or more ASCII PLY point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.ply:
        xyz = load_xyz(path)
        r = np.linalg.norm(xyz, axis=1)
        print(path)
        print(
            f"  points={len(r)} min={np.min(r):.4f} p10={np.percentile(r, 10):.4f} "
            f"median={np.median(r):.4f} p90={np.percentile(r, 90):.4f} max={np.max(r):.4f}"
        )


def load_xyz(path: Path) -> np.ndarray:
    data = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not data:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(data, dtype=np.float32)


if __name__ == "__main__":
    main()
