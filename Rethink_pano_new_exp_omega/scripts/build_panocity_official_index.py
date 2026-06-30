#!/usr/bin/env python3
"""Build train/val indexes for the official Panocity layout.

The expected root is:

  Panocity/<city>/<block>/pano_images/pano_*.png
  Panocity/<city>/<block>/panodepth_images/pano_depth_*.png
  Panocity/<city>/<block>/*_poses.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.pano_minimal import build_panocity_official_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Official Panocity dataset root.")
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Panocity root does not exist: {root}")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError(f"--train-fraction must be in (0, 1), got {args.train_fraction}")

    rows = build_panocity_official_rows(root)
    rows.sort(key=lambda row: (row.get("city", ""), row.get("block", ""), row.get("scene_name", "")))
    if not rows:
        raise RuntimeError(f"No official Panocity samples found under {root}")

    order = list(range(len(rows)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    train_count = int(round(len(order) * args.train_fraction))
    train_indices = set(order[:train_count])
    train_rows = [row for index, row in enumerate(rows) if index in train_indices]
    val_rows = [row for index, row in enumerate(rows) if index not in train_indices]

    cache_dir = root / "cache"
    _write_json(cache_dir / "panocity_all_index.json", rows)
    _write_json(cache_dir / "panocity_train_index.json", train_rows)
    _write_json(cache_dir / "panocity_val_index.json", val_rows)
    _write_json(
        cache_dir / "panocity_index_summary.json",
        {
            "root": str(root),
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "train_fraction": args.train_fraction,
            "seed": args.seed,
        },
    )
    print(
        f"[OK] wrote Panocity official indexes under {cache_dir}: "
        f"total={len(rows)} train={len(train_rows)} val={len(val_rows)}"
    )


if __name__ == "__main__":
    main()
