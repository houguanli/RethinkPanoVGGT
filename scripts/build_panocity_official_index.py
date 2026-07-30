#!/usr/bin/env python3
"""Build train/val/test indexes for the official Panocity layout.

The expected root is:

  Panocity/<city>/<block>/pano_images/pano_*.png
  Panocity/<city>/<block>/panodepth_images/pano_depth_*.png
  Panocity/<city>/<block>/*_poses.json

By default this script uses the explicit PanoCity split JSONs released in the
PanoVGGT repository and expands each trajectory into single-pano rows. Use
--panocity-split-source generated only for custom subsets without the official
split files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_mixed4_official_indexes import DEFAULT_PANOCITY_SPLIT_DIR, build_panocity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Official Panocity dataset root.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.90,
        help="Generated fallback train fraction. Official splits are used by default when present.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--panocity-split-source",
        choices=("auto", "official", "generated"),
        default="auto",
        help="Use PanoVGGT official split JSONs or generate a fallback split.",
    )
    parser.add_argument(
        "--panocity-split-dir",
        type=Path,
        default=DEFAULT_PANOCITY_SPLIT_DIR,
        help="Directory containing panocity_{train,val,test}_index.json official split files.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Panocity root does not exist: {root}")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError(f"--train-fraction must be in (0, 1), got {args.train_fraction}")

    summary = build_panocity(
        root,
        train_fraction=args.train_fraction,
        seed=args.seed,
        split_source=args.panocity_split_source,
        split_dir=args.panocity_split_dir.expanduser().resolve(),
    )
    print(
        f"[OK] wrote Panocity indexes under {root / 'cache'}: "
        f"source={summary.get('split_source')} "
        f"total={summary['all']['expanded']} "
        f"train={summary['train']['expanded']} "
        f"val={summary['val']['expanded']} "
        f"test={summary['test']['expanded']}"
    )


if __name__ == "__main__":
    main()
