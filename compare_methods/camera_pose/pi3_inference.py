"""Checkpoint-explicit batch inference adapter for the official Pi3 checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_UPSTREAM = THIS_DIR / "Pi3"
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt" / "Pi3" / "model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-path", dest="data_paths", action="append", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "camera_methods" / "pi3",
    )
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--pixel-limit", type=int, default=50176)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _safe_name(path: Path, used: set[str]) -> str:
    base = path.stem if path.is_file() else path.name
    base = base.replace(" ", "_") or "sample"
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def main() -> int:
    args = parse_args()
    upstream = args.upstream_dir.expanduser().resolve()
    if not (upstream / "pi3" / "models" / "pi3.py").is_file():
        raise FileNotFoundError(f"Pi3 submodule is not initialized: {upstream}")
    sys.path.insert(0, str(upstream))
    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor, write_ply
    from pi3.utils.geometry import depth_normal_edge

    if args.check_only:
        print("ok pi3 inference imports")
        return 0

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pi3 checkpoint not found: {checkpoint}")
    if args.max_frames < 2:
        raise ValueError("Pi3 camera inference requires at least two frames")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Pi3().to(device).eval()
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    del state

    data_paths = args.data_paths or [
        upstream / "examples" / "house",
        upstream / "examples" / "parkour",
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    used: set[str] = set()
    records = []
    for raw_path in data_paths:
        data_path = raw_path.expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Pi3 input not found: {data_path}")
        images = load_images_as_tensor(str(data_path), interval=1, PIXEL_LIMIT=args.pixel_limit)
        images = images[: args.max_frames]
        if images.shape[0] < 2:
            raise RuntimeError(f"Pi3 input has fewer than two usable frames: {data_path}")
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            result = model(images[None])

        masks = torch.sigmoid(result["conf"][..., 0]) > args.confidence_threshold
        masks = torch.logical_and(
            masks,
            ~depth_normal_edge(result["local_points"], rtol=0.03, mask=masks),
        )[0]
        name = _safe_name(data_path, used)
        ply_path = output_dir / f"{name}.ply"
        pose_path = output_dir / f"{name}_camera_poses.npz"
        points = result["points"][0][masks].float().cpu()
        colors = images.permute(0, 2, 3, 1)[masks].float().cpu()
        write_ply(points, colors, str(ply_path))
        camera_poses = result["camera_poses"][0].float().cpu().numpy()
        np.savez_compressed(pose_path, camera_poses=camera_poses)
        peak_gib = None
        if device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        records.append(
            {
                "input": str(data_path),
                "frames": int(images.shape[0]),
                "input_shape": list(images.shape),
                "points_written": int(points.shape[0]),
                "camera_pose_shape": list(camera_poses.shape),
                "peak_cuda_memory_gib": peak_gib,
                "point_cloud": str(ply_path),
                "camera_poses": str(pose_path),
            }
        )
        del images, result, points, colors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "method": "Pi3",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "dtype": str(dtype),
        "pixel_limit": args.pixel_limit,
        "samples": records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
