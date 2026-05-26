#!/usr/bin/env python3
"""Run PanoVGGT baseline on the RethinkPanoVGGT comparison pano sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPARE_ROOT = METHOD_ROOT.parent
PROJECT_ROOT = COMPARE_ROOT.parent

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from inference import main as inference_main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "panovggt_compare_sample" / "images",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "panovggt_compare_sample" / "masks",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=COMPARE_ROOT / "panovggt_ckpt" / "model.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=METHOD_ROOT / "training" / "config" / "default.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Rethink_pano_new_exp_omega" / "outputs" / "panovggt_baseline_sample0",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-log-depth", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inference_args = argparse.Namespace(
        config=str(args.config),
        checkpoint=str(args.checkpoint),
        image_dir=str(args.image_dir),
        mask_dir=str(args.mask_dir) if args.mask_dir is not None else None,
        output_dir=str(args.output_dir),
        device=args.device,
        log_depth=not args.no_log_depth,
    )
    inference_main(inference_args)


if __name__ == "__main__":
    main()
