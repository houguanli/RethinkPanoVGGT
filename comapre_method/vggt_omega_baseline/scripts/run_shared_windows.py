#!/usr/bin/env python3
"""Run official VGGT-Omega on pre-sampled windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vggt_omega.models import VGGTOmega


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.windows, map_location="cpu", weights_only=False)
    windows = payload["windows"].to(device)

    model = VGGTOmega().to(device).eval()
    with torch.no_grad():
        pred = model(windows)
    keep = {}
    for key in ("depth", "depth_conf", "pose_enc", "camera_and_register_tokens"):
        if key in pred:
            keep[key] = pred[key].detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, args.output)
    print(f"[INFO] wrote official window output = {args.output}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


if __name__ == "__main__":
    main()
