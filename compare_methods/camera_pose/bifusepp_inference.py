"""Headless multi-image inference adapter for the official BiFuse++ checkout."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tv_models
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_UPSTREAM = THIS_DIR / "BiFusePlusPlus"
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt" / "BiFusePlusPlus" / "pretrain" / "supervised_pretrain.pkl"


def _load_bifuse_module(upstream: Path):
    package_dir = upstream / "BiFusev2"
    if not (package_dir / "BiFuse.py").is_file():
        raise FileNotFoundError(f"BiFuse++ submodule is not initialized: {upstream}")

    # Upstream imports training-only PyTorch3D and Lightning modules from its
    # package __init__. Official depth inference only needs these projection
    # classes, so expose a narrow package without the unrelated training stack.
    package = types.ModuleType("BiFusev2")
    package.__path__ = [str(package_dir)]
    package.__package__ = "BiFusev2"
    sys.modules["BiFusev2"] = package

    projection_dir = package_dir / "Projection"
    projection = types.ModuleType("BiFusev2.Projection")
    projection.__path__ = [str(projection_dir)]
    projection.__package__ = "BiFusev2.Projection"
    sys.modules["BiFusev2.Projection"] = projection
    for module_name in ("Equirec2Cube", "Cube2Equirec"):
        module = importlib.import_module(f"BiFusev2.Projection.{module_name}")
        setattr(projection, module_name, getattr(module, module_name))

    return importlib.import_module("BiFusev2.BiFuse")


def _build_model(module, mode: str):
    network_args = {
        "save_path": str(Path(tempfile.gettempdir()) / "bifusepp_adapter_state"),
        "dnet_args": {"layers": 34, "CE_equi_h": [8, 16, 32, 64, 128, 256, 512]},
        "pnet_args": {"layers": 18, "nb_tgts": 2},
    }

    # The released checkpoint already contains both ResNet encoders.
    original_resnet18 = tv_models.resnet18
    original_resnet34 = tv_models.resnet34

    def without_download(builder):
        def wrapped(*args, **kwargs):
            kwargs["weights"] = None
            return builder(*args, **kwargs)

        return wrapped

    tv_models.resnet18 = without_download(original_resnet18)
    tv_models.resnet34 = without_download(original_resnet34)
    try:
        if mode == "supervised":
            return module.SupervisedCombinedModel(**network_args)
        return module.SelfSupervisedCombinedModel(**network_args)
    finally:
        tv_models.resnet18 = original_resnet18
        tv_models.resnet34 = original_resnet34


def _load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((1024, 512), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _prepare_released_state_dict(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop unused, untrained fusion layers retained by the release checkpoint."""
    prepared = {}
    dropped = []
    for key, value in state.items():
        if ".conv_equi." in key or ".conv_cube." in key:
            dropped.append(key)
            continue
        prepared[key] = value

    tracked = [
        key
        for key in dropped
        if key.endswith("num_batches_tracked") and int(state[key].item()) != 0
    ]
    if tracked:
        raise RuntimeError(f"BiFuse++ legacy fusion layers were trained unexpectedly: {tracked}")
    return prepared, dropped


def _is_runtime_projection_buffer(key: str) -> bool:
    leaf = key.rsplit(".", 1)[-1]
    return leaf.startswith(("grid_", "mask_", "XY_"))


def _safe_stem(path: Path, used: set[str]) -> str:
    stem = path.stem.replace(" ", "_")
    candidate = stem
    counter = 2
    while candidate in used:
        candidate = f"{stem}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", dest="inputs", action="append", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "camera_methods" / "bifusepp",
    )
    parser.add_argument("--mode", choices=("supervised", "selfsupervised"), default="supervised")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = _load_bifuse_module(args.upstream_dir.resolve())
    if args.check_only:
        print("ok bifusepp inference imports")
        return 0

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BiFuse++ checkpoint not found: {checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model(module, args.mode)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, dropped_legacy = _prepare_released_state_dict(state)
    incompatible = model.load_state_dict(state, strict=False)
    missing_learned = [key for key in incompatible.missing_keys if not _is_runtime_projection_buffer(key)]
    if missing_learned or incompatible.unexpected_keys:
        raise RuntimeError(
            "BiFuse++ checkpoint is not compatible after key migration: "
            f"missing_learned={missing_learned}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device).eval()

    inputs = args.inputs or [
        args.upstream_dir / "data" / "mp3d.jpg",
        args.upstream_dir / "data" / "panosuncg.jpg",
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    records = []
    loaded_batches = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input panorama not found: {path}")
        batch = _load_image(path)
        loaded_batches.append(batch)
        with torch.inference_mode():
            device_batch = batch.to(device)
            if args.mode == "supervised":
                # Training calls the combined model, whose forward normalizes
                # RGB before DepthNet. The released demo bypasses this step.
                depth = model(device_batch)[0]
            else:
                normalized = model.preprocess(device_batch)
                raw_inverse_depth = model.dnet(normalized)[0]
                depth = 1.0 / (10.0 * torch.sigmoid(raw_inverse_depth) + 0.01)
        depth_np = depth[0, 0].float().cpu().numpy().clip(0.0, 10.0)
        stem = _safe_stem(path, used)
        npy_path = output_dir / f"{stem}_depth.npy"
        png_path = output_dir / f"{stem}_depth_mm.png"
        np.save(npy_path, depth_np)
        Image.fromarray(np.rint(depth_np * 1000.0).astype(np.uint16)).save(png_path)
        records.append(
            {
                "input": str(path),
                "shape": list(depth_np.shape),
                "depth_min_m": float(depth_np.min()),
                "depth_max_m": float(depth_np.max()),
                "depth_mean_m": float(depth_np.mean()),
                "depth_npy": str(npy_path),
                "depth_png_mm": str(png_path),
            }
        )

    pose_record = None
    if args.mode == "selfsupervised":
        if len(loaded_batches) < 3:
            raise ValueError("Self-supervised BiFuse++ camera inference requires three --input panoramas")
        ref = loaded_batches[1].to(device)
        targets = [loaded_batches[0].to(device), loaded_batches[2].to(device)]
        with torch.inference_mode():
            _, _, relative_pose = model(ref, targets)
        pose_path = output_dir / "relative_pose_6dof.npy"
        pose_np = relative_pose[0].float().cpu().numpy()
        np.save(pose_path, pose_np)
        pose_record = {
            "input_order": [str(Path(value).expanduser().resolve()) for value in inputs[:3]],
            "reference_index": 1,
            "target_indices": [0, 2],
            "relative_pose_shape": list(pose_np.shape),
            "relative_pose_6dof": str(pose_path),
        }

    summary = {
        "method": "BiFuse++",
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "missing_learned_keys": missing_learned,
        "regenerated_projection_buffers": len(incompatible.missing_keys),
        "dropped_untrained_legacy_fusion_keys": len(dropped_legacy),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "pose": pose_record,
        "samples": records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
