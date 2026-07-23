#!/usr/bin/env python3
"""Export shared pano windows and local pano-wrapper outputs for equivalence checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import PanoVKittiOmegaDataset  # noqa: E402
from training.train_pano_omega import build_model, load_checkpoint, sample_depth_targets, set_seed  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT.parent / "ckpt" / "vggt_omega_1b_512.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-depth-semantics", choices=["range", "cubemap_z", "double_cubemap_z"], default="range")
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = SimpleNamespace(
        patch_size=16,
        window_size=512,
        num_yaw=8,
        pitch_degrees="0",
        fov_degrees=75.0,
        pano_height=0,
        pano_width=0,
        enable_camera_head=True,
        smoke=False,
    )

    dataset = PanoVKittiOmegaDataset(root=args.dataset_root)
    sample = dataset[args.sample_index]
    pano = sample["pano_image"][None].to(device)

    current_model = build_model(model_args).to(device).eval()
    load_checkpoint(current_model, args.checkpoint, strict=False)
    with torch.no_grad():
        current_pred = current_model(pano_images=pano, return_sampler_output=True)
        target_depth, target_valid = sample_depth_targets(
            current_model,
            sample["pano_depth"][None].to(device),
            source_depth_semantics=args.gt_depth_semantics,
            max_range_depth=args.depth_max_m,
        )

    windows = current_pred["pano_windows"].detach().cpu()
    camera_meta = {key: value.detach().cpu() for key, value in current_pred["pano_camera_meta"].items()}
    torch.save(
        {
            "windows": windows,
            "camera_meta": camera_meta,
            "target_depth": target_depth.detach().cpu(),
            "target_valid": target_valid.detach().cpu(),
            "gt_depth_semantics": args.gt_depth_semantics,
            "sample": {
                "scene_name": sample["scene_name"],
                "rgb_path": sample["rgb_path"],
                "depth_path": sample["depth_path"],
            },
        },
        output_dir / "shared_windows.pt",
    )
    save_prediction(output_dir / "pano_current_setting.pt", current_pred)

    pure_model = VGGTOmega_LUNA(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        enable_pano_global_token=False,
        enable_luna=False,
        sampler={
            "window_size": 512,
            "patch_size": 16,
            "fov_degrees": 75.0,
            "num_yaw": 8,
            "pitch_degrees": (0.0,),
        },
        checkpoint_path=None,
    ).to(device).eval()
    load_checkpoint(pure_model, args.checkpoint, strict=True)
    with torch.no_grad():
        pure_pred = pure_model(pano_images=pano, return_sampler_output=True)
    save_prediction(output_dir / "pano_pure_sampler.pt", pure_pred)
    print(f"[INFO] exported pano equivalence inputs = {output_dir}")


def save_prediction(path: Path, pred: dict) -> None:
    keep = {}
    for key in ("depth", "depth_conf", "pose_enc", "camera_and_register_tokens"):
        if key in pred:
            keep[key] = pred[key].detach().cpu()
    torch.save(keep, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


if __name__ == "__main__":
    main()
