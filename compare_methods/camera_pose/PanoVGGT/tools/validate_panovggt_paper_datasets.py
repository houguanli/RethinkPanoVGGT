#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PAPER_DEPTH = {
    "Matterport3D": {
        "monocular": {"Abs Rel": 0.0884, "delta1": 0.9157},
        "multi": {"Abs Rel": 0.0840, "delta1": 0.9266},
    },
    "Stanford2D3DS": {
        "monocular": {"Abs Rel": 0.0711, "delta1": 0.9392},
        "multi": {"Abs Rel": 0.0778, "delta1": 0.9323},
    },
    "Structured3D": {
        "monocular": {"Abs Rel": 0.0438, "delta1": 0.9728},
        "multi": {"Abs Rel": 0.0400, "delta1": 0.9870},
    },
}

PAPER_POSE = {
    "Matterport3D": {"AUC@30": 0.459, "R_mean": 21.394, "R_med": 23.466, "T_mean": 18.900, "T_med": 18.100},
    "Stanford2D3DS": {"AUC@30": 0.556, "R_mean": 18.801, "R_med": 24.170, "T_mean": 10.999, "T_med": 9.762},
}

PAPER_POINT = {
    "Matterport3D": {
        "global_point": {"Acc-mean": 0.1743, "Acc-med": 0.1231, "Comp-mean": 0.1530, "Comp-med": 0.0890, "overall_mean": 0.1636, "overall_med": 0.1060},
        "local_point": {"Acc-mean": 0.1750, "Acc-med": 0.1242, "Comp-mean": 0.1530, "Comp-med": 0.0932, "overall_mean": 0.1640, "overall_med": 0.1087},
    },
    "Stanford2D3DS": {
        "global_point": {"Acc-mean": 0.2087, "Acc-med": 0.1752, "Comp-mean": 0.2624, "Comp-med": 0.1943, "overall_mean": 0.2355, "overall_med": 0.1848},
        "local_point": {"Acc-mean": 0.2109, "Acc-med": 0.1786, "Comp-mean": 0.2590, "Comp-med": 0.1943, "overall_mean": 0.2349, "overall_med": 0.1865},
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_panovggt(repo: Path, ckpt: Path, device: torch.device, config: str):
    sys.path.insert(0, str(repo))
    from evaluation.eval_allpano import build_model_from_hydra_config, load_checkpoint

    model = build_model_from_hydra_config(config, device)
    load_info = load_checkpoint(model, str(ckpt), device, strict=False)
    return model, {"missing": load_info["missing"], "unexpected": load_info["unexpected"]}


def build_dataset(repo: Path, name: str, root: Path, split: str, frames: int, img_size: int):
    sys.path.insert(0, str(repo))
    from training.data.datasets.panocity import PanoCityDataset
    from training.data.datasets.matterport3d import Matterport3DDataset
    from training.data.datasets.stanford2d3ds import Stanford2D3DSDataset
    from training.data.datasets.structured3d import Structured3DDataset

    if name == "Structured3D" and not any(root.glob("scene_*")):
        nested = root / "Structured3D"
        if nested.is_dir():
            root = nested

    common = SimpleNamespace(
        img_size=img_size,
        patch_size=14,
        rescale=None,
        rescale_aug=None,
        landscape_check=False,
        training=False,
        get_nearby=True,
        inside_random=False,
        allow_duplicate_img=True,
        augs=None,
        debug=False,
    )
    if name == "PanoCity":
        return PanoCityDataset(
            common_conf=common,
            split=split,
            PanoCity_DIR=str(root),
            min_num_images=max(2, frames),
            len_test=10**9,
            get_nearby=True,
        )
    if name == "Matterport3D":
        return Matterport3DDataset(
            common_conf=common,
            split=split,
            Matterport3D_DIR=str(root),
            min_num_images=max(2, frames),
            len_test=10**9,
            get_nearby=True,
        )
    if name == "Stanford2D3DS":
        return Stanford2D3DSDataset(
            common_conf=common,
            split=split,
            Stanford2D3DS_DIR=str(root),
            min_num_images=max(2, frames),
            len_test=10**9,
            get_nearby=True,
        )
    if name == "Structured3D":
        return Structured3DDataset(
            common_conf=common,
            split=split,
            Structured3D_DIR=str(root),
            min_num_rooms=max(2, frames),
            len_test=10**9,
            get_nearby=True,
        )
    raise ValueError(name)


def compare_metrics(dataset: str, observed: dict, frames: int) -> dict:
    regime = "monocular" if frames == 1 else "multi"
    out = {"depth_target": PAPER_DEPTH.get(dataset, {}).get(regime), "pose_target": PAPER_POSE.get(dataset), "point_target": PAPER_POINT.get(dataset)}
    depth = observed.get("depth", {})
    if out["depth_target"]:
        out["depth_delta"] = {
            "Abs Rel": depth.get("Abs Rel", 0) - out["depth_target"]["Abs Rel"],
            "delta1": depth.get("δ < 1.25", 0) - out["depth_target"]["delta1"],
        }
    pose = observed.get("pose", {})
    if out["pose_target"] and pose:
        out["pose_delta"] = {k: pose.get(k, 0) - v for k, v in out["pose_target"].items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/compare_methods/camera_pose/PanoVGGT")
    parser.add_argument("--ckpt", default="/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/compare_methods/camera_pose/PanoVGGT/checkpoints/model_lowres.pt")
    parser.add_argument("--config", default="default")
    parser.add_argument("--datasets", nargs="+", default=["panocity", "matterport", "stanford", "structured3d"])
    parser.add_argument("--panocity-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/PanoCity")
    parser.add_argument("--matterport-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Matterport3D")
    parser.add_argument("--stanford-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Stanford2D3DS")
    parser.add_argument("--structured3d-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Structured3D")
    parser.add_argument("--split", default="test", choices=["val", "test", "test_final"])
    parser.add_argument("--num-seqs", type=int, default=10, help="-1 means all indexed sequences")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--depth-align", default="irls-absrel", choices=["median-scale", "robust-scale", "scale&shift", "irls-absrel", "metric"])
    parser.add_argument("--depth-lat-min", type=float, default=-15.0)
    parser.add_argument("--depth-lat-max", type=float, default=60.0)
    parser.add_argument("--depth-irls-iters", type=int, default=100)
    parser.add_argument("--eval-unit", default="sample", choices=["sample", "sequence"])
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument("--with-pointcloud", action="store_true", help="Run expensive dense point-cloud metrics.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="/mnt/e/PanoVGGT_minimal_datasets/logs/validate_results.json")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    ckpt = Path(args.ckpt).resolve()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp_dtype)

    sys.path.insert(0, str(repo))
    import evaluation.eval_allpano as eval_mod
    if not args.with_pointcloud:
        eval_mod.eval_pointcloud = lambda *_, **__: {}
    eval_one_dataset = eval_mod.eval_one_dataset

    model, load_info = load_panovggt(repo, ckpt, device, args.config)
    name_map = {
        "panocity": ("PanoCity", Path(args.panocity_root)),
        "matterport": ("Matterport3D", Path(args.matterport_root)),
        "stanford": ("Stanford2D3DS", Path(args.stanford_root)),
        "structured3d": ("Structured3D", Path(args.structured3d_root)),
    }

    json_root = Path(args.output).resolve().parent / "per_dataset"
    json_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in args.datasets:
        dataset_name, root = name_map[key.lower()]
        print(f"[validate] dataset={dataset_name} root={root}")
        ds = build_dataset(repo, dataset_name, root, args.split, args.frames, args.img_size)
        if getattr(ds, "sequence_list_len", 0) <= 0:
            raise RuntimeError(
                f"{dataset_name} has zero indexed sequences under {root}. "
                "Check the dataset root, split files, and cache index before trusting validation output."
            )
        metrics = eval_one_dataset(
            dataset_name,
            ds,
            model,
            device,
            amp_dtype,
            num_seqs=args.num_seqs,
            num_frames=args.frames,
            depth_align=args.depth_align,
            depth_lat_min=args.depth_lat_min,
            depth_lat_max=args.depth_lat_max,
            depth_irls_iters=args.depth_irls_iters,
            json_root=str(json_root),
            skip_pointcloud=not args.with_pointcloud,
            eval_unit=args.eval_unit,
            sample_stride=args.sample_stride,
        )
        results[dataset_name] = {
            "metrics": metrics,
            "paper_compare": compare_metrics(dataset_name, metrics, args.frames),
        }

    payload = {
        "config": vars(args),
        "ckpt": str(ckpt),
        "model_load": {"missing": load_info["missing"], "unexpected": load_info["unexpected"]},
        "results": results,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[validate] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
