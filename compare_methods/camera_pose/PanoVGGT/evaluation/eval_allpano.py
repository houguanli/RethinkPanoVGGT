#!/usr/bin/env python3
"""
Evaluate a panoramic 3D reconstruction model across all supported datasets.

Computes three families of metrics per dataset:
  • **Pose**  — pairwise rotation / translation AUC
  • **Depth** — AbsRel, RMSE, δ-thresholds, …
  • **Point cloud** — accuracy, completion, normal consistency (multi-view fusion)

Both ``world_points`` (= local_points × predicted_pose) and ``global_points``
(= direct network prediction) are evaluated when available.

Usage
-----
    python -m evaluation.eval_allpano \\
        --ckpt  path/to/checkpoint.pt \\
        --json_root results/

    # quick smoke-test on 10 sequences per dataset
    python -m evaluation.eval_allpano \\
        --num_seqs_panocity 10 \\
        --num_seqs_matterport 10 \\
        --json_root results/quick
"""

import os
import glob
import json
import random
import argparse
import importlib
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# dataset loaders
from training.data.datasets.panocity import PanoCityDataset
from training.data.datasets.matterport3d import Matterport3DDataset
from training.data.datasets.stanford2d3ds import Stanford2D3DSDataset
from training.data.datasets.structured3d import Structured3DDataset

# evaluation utilities
from evaluation.utils.depth import depth_evaluation
from evaluation.utils.pointcloud import eval_pointcloud
from evaluation.utils.pose import (
    se3_to_relative_pose_error,
    calculate_auc_np,
)
from evaluation.utils.geometry import closed_form_inverse_se3


# =========================================================================
#  Per-dataset thresholds for ICP / normal estimation
# =========================================================================

POINTCLOUD_CONFIG = {
    # outdoor — large scale
    "PanoCity":     {"icp_threshold": 1.0, "normal_radius": 2.0},
    # indoor — room scale
    "Matterport3D": {"icp_threshold": 0.1, "normal_radius": 0.2},
    "Stanford2D3DS":{"icp_threshold": 0.1, "normal_radius": 0.2},
    "Structured3D": {"icp_threshold": 0.1, "normal_radius": 0.2},
}

_DEFAULT_PC = {"icp_threshold": 0.1, "normal_radius": 0.2}

DATASET_KEYS = {
    "panocity": "PanoCity",
    "matterport": "Matterport3D",
    "matterport3d": "Matterport3D",
    "stanford": "Stanford2D3DS",
    "stanford2d3ds": "Stanford2D3DS",
    "structured3d": "Structured3D",
}

PAPER_DEPTH_TARGETS = {
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


# =========================================================================
#  Helpers
# =========================================================================

def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_se3_4x4(mat: torch.Tensor) -> torch.Tensor:
    """Pad ``(N, 3, 4)`` → ``(N, 4, 4)`` by appending ``[0 0 0 1]``."""
    if mat.shape[-2:] == (4, 4):
        return mat
    N = mat.shape[0]
    row = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
    return torch.cat([mat, row.view(1, 1, 4).expand(N, -1, -1)], dim=1)


def _safe_mean(values: list) -> float:
    return float(np.mean(values)) if values else 0.0


def _ensure_nhw3(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape a point-map tensor to ``(N, H, W, 3)``."""
    if tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 4 and tensor.shape[1] == 3:
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_lowres_ckpt() -> Path:
    root = _repo_root()
    candidates = [
        root / "checkpoints" / "model_lowres.pt",
        root / "../../../ckpt/PanoVGGT/model_lowres.pt",
        root / "../../../ckpt/PanoVGGT/model.pt",
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _cfg_select(cfg, key: str, default):
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def build_model_from_hydra_config(config_name: str, device: torch.device) -> torch.nn.Module:
    """Build PanoVGGT with the repository Hydra model config."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from panovggt.models.panovggt_model import PanoVGGTModel

    config_dir = _repo_root() / "training" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)

    model_cfg = cfg.model
    aggregator_cfg = OmegaConf.to_container(
        _cfg_select(model_cfg, "aggregator", {}), resolve=True
    )
    if isinstance(aggregator_cfg, dict):
        aggregator_cfg.setdefault("load_dinov2_pretrained", False)
        aggregator_cfg.setdefault("allow_dinov2_download", False)

    model = PanoVGGTModel(
        img_size=int(cfg.img_size),
        patch_size=int(cfg.patch_size),
        embed_dim=int(_cfg_select(cfg, "embed_dim", 1024)),
        enable_camera=bool(_cfg_select(model_cfg, "enable_camera", True)),
        enable_depth=bool(_cfg_select(model_cfg, "enable_depth", True)),
        enable_point=bool(_cfg_select(model_cfg, "enable_point", True)),
        enable_global_points=bool(_cfg_select(model_cfg, "enable_global_points", True)),
        geometry_output_scale=float(_cfg_select(model_cfg, "geometry_output_scale", 1.0)),
        train_geometry_output_scale=bool(_cfg_select(model_cfg, "train_geometry_output_scale", False)),
        aggregator=aggregator_cfg,
    )
    return model.to(device).eval()


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, strict: bool = False) -> dict:
    ckpt = Path(ckpt_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"PanoVGGT checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
            break
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in payload.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(
        f"Model loaded from {ckpt} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}, strict={strict})"
    )
    return {
        "checkpoint": str(ckpt),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "strict": strict,
    }


def _latitude_band_mask(height: int, width: int, lat_min: float, lat_max: float) -> np.ndarray:
    """Return an equirectangular latitude mask where top row is +90 degrees."""
    rows = np.arange(height, dtype=np.float32) + 0.5
    lat = 90.0 - rows * (180.0 / float(height))
    lo, hi = sorted((float(lat_min), float(lat_max)))
    row_mask = (lat >= lo) & (lat <= hi)
    return np.broadcast_to(row_mask[:, None], (height, width)).copy()


def _depth_align_kwargs(depth_align: str, irls_iters: int) -> dict:
    if depth_align == "metric":
        return {"metric_scale": True}
    if depth_align == "scale&shift":
        return {"align_with_lad2": True, "max_iters": irls_iters}
    if depth_align == "irls-absrel":
        return {"align_with_irls_absrel": True, "max_iters": irls_iters}
    if depth_align == "robust-scale":
        return {"align_with_scale": True}
    return {}


def _normalize_dataset_root(name: str, root: str) -> str:
    path = Path(root).expanduser()
    if name == "Structured3D" and not any(path.glob("scene_*")):
        nested = path / "Structured3D"
        if nested.is_dir():
            path = nested
    return str(path)


def _dataset_root_from_base(base_root: str, dataset_dir: str) -> str:
    return str(Path(base_root).expanduser() / dataset_dir)


def _sequence_item_ids(name: str, dataset, seq_index: int) -> list[int]:
    """Return deterministic item ids inside one top-level sequence."""
    if name == "PanoCity":
        traj = dataset.trajectories[seq_index % dataset.sequence_list_len]
        pano_images = traj["pano_images"]
        poses_file = traj.get("poses_file")
        if poses_file:
            poses = dataset._read_poses(os.path.join(dataset.PanoCity_DIR, poses_file))
            valid = []
            for i, rgb_rel in enumerate(pano_images):
                pose = poses.get(os.path.basename(rgb_rel))
                if pose is not None and np.isfinite(pose).all():
                    valid.append(i)
            return valid
        return list(range(len(pano_images)))

    if name == "Matterport3D":
        return list(range(len(dataset.room_trajectories[seq_index % dataset.sequence_list_len][3])))

    if name == "Stanford2D3DS":
        area, _, _, panoramas, _ = dataset.trajectories[seq_index % dataset.sequence_list_len]
        valid = []
        for i, pano_id in enumerate(panoramas):
            pose_pattern = os.path.join(
                dataset.Stanford2D3DS_DIR,
                area,
                "pano",
                "pose",
                f"camera_{pano_id}_*_frame_equirectangular_domain_pose.json",
            )
            if glob.glob(pose_pattern):
                valid.append(i)
        return valid

    if name == "Structured3D":
        return list(range(len(dataset.scene_trajectories[seq_index % dataset.sequence_list_len][1])))

    raise ValueError(f"Unsupported dataset: {name}")


def _iter_eval_samples(
    name: str,
    dataset,
    indices: list[int],
    num_frames: int,
    eval_unit: str,
    sample_stride: int,
) -> list[tuple[int, list[int] | None]]:
    if eval_unit == "sequence":
        return [(si, None) for si in indices]

    stride = max(1, int(sample_stride))
    window = max(1, int(num_frames))
    samples: list[tuple[int, list[int] | None]] = []
    for si in indices:
        item_ids = _sequence_item_ids(name, dataset, si)
        if len(item_ids) < window:
            continue
        for start in range(0, len(item_ids) - window + 1, stride):
            samples.append((si, item_ids[start:start + window]))
    return samples


# =========================================================================
#  Single-sequence evaluation
# =========================================================================

def evaluate_sequence(
    model: torch.nn.Module,
    data: Dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    depth_max: float = 10.0,
    depth_align: str = "median-scale",
    depth_lat_min: float | None = None,
    depth_lat_max: float | None = None,
    depth_irls_iters: int = 100,
    dataset_name: str = "",
    skip_pointcloud: bool = False,
) -> tuple:
    """Run inference on one sequence and compute all metrics.

    Returns:
        ``(pose_metrics, depth_metrics, world_point_metrics, global_point_metrics)``
    """
    # ── unpack ground truth ──────────────────────────────────────────
    images = torch.stack(data["images"], dim=0).to(device)       # (N, 3, H, W)
    N, _, H, W = images.shape

    gt_extri = torch.from_numpy(np.stack(data["extrinsics"], 0)).to(device)   # (N, 3, 4) w2c
    gt_world = torch.stack(data["world_points"], 0).permute(0, 3, 1, 2).to(device)  # (N, 3, H, W)
    gt_depth = torch.stack(data["depths"], 0).to(device)                              # (N, H, W)
    gt_mask = torch.stack(data["point_masks"], 0).bool().to(device)                   # (N, H, W)

    # ── forward pass ─────────────────────────────────────────────────
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=(amp_dtype is not None), dtype=amp_dtype or torch.float32
    ):
        pred = model(images=images)

    # ── parse predictions ────────────────────────────────────────────
    def _squeeze_batch(t):
        return t[0] if (t is not None and t.dim() >= 4 and t.shape[0] == 1) else t

    pred_extri = _squeeze_batch(pred.get("camera_poses"))          # (N, 3|4, 4) c2w
    pred_depth = _squeeze_batch(pred.get("depth"))                 # (N, H, W[, 1])
    pred_world = _squeeze_batch(pred.get("world_points"))          # (N, H, W, 3)
    pred_global = _squeeze_batch(pred.get("global_points"))        # (N, H, W, 3)

    if pred_depth is not None:
        if pred_depth.dim() == 4 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if pred_depth.shape[-2:] != (H, W):
            pred_depth = torch.nn.functional.interpolate(
                pred_depth[:, None], size=(H, W), mode="bilinear", align_corners=False
            )[:, 0]
        pred_depth = pred_depth.clamp_min(0)

    if pred_world is not None:
        pred_world = _ensure_nhw3(pred_world)
    if pred_global is not None:
        pred_global = _ensure_nhw3(pred_global)

    # ── 1. Pose metrics ──────────────────────────────────────────────
    pose_metrics: dict = {}
    if pred_extri is not None and pred_extri.shape[0] == N:
        pred_se3 = _to_se3_4x4(pred_extri)
        gt_se3 = _to_se3_4x4(gt_extri)
        pred_w2c = closed_form_inverse_se3(pred_se3)

        r_err, t_err = se3_to_relative_pose_error(pred_w2c, gt_se3, N)

        if r_err.numel() > 0:
            rn, tn = r_err.cpu().numpy(), t_err.cpu().numpy()
            pose_metrics = {
                "AUC@30": calculate_auc_np(rn, tn, 30),
                "AUC@15": calculate_auc_np(rn, tn, 15),
                "AUC@5":  calculate_auc_np(rn, tn, 5),
                "AUC@3":  calculate_auc_np(rn, tn, 3),
                "R_mean": float(r_err.mean()), "R_med": float(r_err.median()),
                "T_mean": float(t_err.mean()), "T_med": float(t_err.median()),
            }
        else:
            pose_metrics = {k: 0.0 for k in
                           ["AUC@30", "AUC@15", "AUC@5", "AUC@3",
                            "R_mean", "R_med", "T_mean", "T_med"]}

    # ── 2. Depth metrics ─────────────────────────────────────────────
    depth_metrics: dict = {}
    if pred_depth is not None:
        per_frame: Dict[str, list] = {}
        align_kw = _depth_align_kwargs(depth_align, depth_irls_iters)
        lat_mask = None
        if depth_lat_min is not None and depth_lat_max is not None:
            lat_mask = _latitude_band_mask(H, W, depth_lat_min, depth_lat_max)
        for i in range(N):
            custom_mask = gt_mask[i].cpu().numpy()
            if lat_mask is not None:
                custom_mask = custom_mask & lat_mask
            res, *_ = depth_evaluation(
                pred_depth[i].cpu().numpy(),
                gt_depth[i].cpu().numpy(),
                max_depth=depth_max,
                custom_mask=custom_mask,
                **align_kw,
            )
            for k, v in res.items():
                per_frame.setdefault(k, []).append(v)
        depth_metrics = {k: _safe_mean(v) for k, v in per_frame.items()}

    # ── 3. Point-cloud metrics ───────────────────────────────────────
    if skip_pointcloud:
        return pose_metrics, depth_metrics, {}, {}

    pc_cfg = POINTCLOUD_CONFIG.get(dataset_name, _DEFAULT_PC)
    gt_world_np = gt_world.cpu().float().numpy().transpose(0, 2, 3, 1)  # (N,H,W,3)
    mask_np = gt_mask.cpu().numpy()

    def _eval_points(pred_pts_tensor):
        if pred_pts_tensor is None:
            return {}
        pred_np = pred_pts_tensor.cpu().float().numpy()
        all_pred, all_gt = [], []
        for i in range(N):
            m = mask_np[i]
            all_pred.append(pred_np[i][m])
            all_gt.append(gt_world_np[i][m])
        p = np.concatenate(all_pred) if all_pred else np.empty((0, 3))
        g = np.concatenate(all_gt) if all_gt else np.empty((0, 3))
        return eval_pointcloud(p, g, **pc_cfg)

    world_pt_metrics = _eval_points(pred_world)
    global_pt_metrics = _eval_points(pred_global)

    return pose_metrics, depth_metrics, world_pt_metrics, global_pt_metrics


# =========================================================================
#  Dataset-level loop
# =========================================================================

def eval_one_dataset(
    name: str,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    amp_dtype,
    num_seqs: int,
    num_frames: int,
    depth_align: str,
    depth_lat_min: float | None,
    depth_lat_max: float | None,
    depth_irls_iters: int,
    json_root: str,
    skip_pointcloud: bool = False,
    eval_unit: str = "sample",
    sample_stride: int = 1,
) -> dict:
    """Evaluate *model* on *dataset* and save per-dataset JSON."""
    pc_cfg = POINTCLOUD_CONFIG.get(name, _DEFAULT_PC)
    print(f"\n{'=' * 72}")
    print(f"  Dataset : {name}")
    print(f"  Depth   : align={depth_align} | latitude=[{depth_lat_min}, {depth_lat_max}]")
    if skip_pointcloud:
        print("  Point-cloud metrics: skipped")
    else:
        print(f"  ICP th  : {pc_cfg['icp_threshold']}m  |  normal radius : {pc_cfg['normal_radius']}m")
    print(f"{'=' * 72}")

    indices = list(range(dataset.sequence_list_len))
    if 0 < num_seqs < len(indices):
        indices = indices[:num_seqs]
    samples = _iter_eval_samples(name, dataset, indices, num_frames, eval_unit, sample_stride)
    print(f"  Eval    : unit={eval_unit} | sequences={len(indices)} | samples={len(samples)} | stride={sample_stride}")

    agg = {"pose": {}, "depth": {}, "world_point": {}, "global_point": {}}
    successful_samples = 0

    for sample_idx, (si, ids) in enumerate(tqdm(samples, desc=name)):
        try:
            data = dataset.get_data(seq_index=si, img_per_seq=num_frames, ids=ids, aspect_ratio=1.0)
            pose_m, depth_m, wpt_m, gpt_m = evaluate_sequence(
                model, data, device, amp_dtype,
                depth_max=dataset.depth_max,
                depth_align=depth_align,
                depth_lat_min=depth_lat_min,
                depth_lat_max=depth_lat_max,
                depth_irls_iters=depth_irls_iters,
                dataset_name=name,
                skip_pointcloud=skip_pointcloud,
            )
            for src, bucket in [(pose_m, "pose"), (depth_m, "depth"),
                                (wpt_m, "world_point"), (gpt_m, "global_point")]:
                for k, v in src.items():
                    agg[bucket].setdefault(k, []).append(v)
            successful_samples += 1

        except Exception as exc:
            print(f"\n  [ERROR] sample {sample_idx} seq {si} ids={ids}: {exc}")
            traceback.print_exc()

    result = {cat: {k: _safe_mean(v) for k, v in metrics.items()}
              for cat, metrics in agg.items()}
    regime = "monocular" if num_frames == 1 else "multi"
    paper_depth_target = PAPER_DEPTH_TARGETS.get(name, {}).get(regime)
    if paper_depth_target and result.get("depth"):
        result["paper_depth_target"] = paper_depth_target
        result["paper_depth_delta"] = {
            "Abs Rel": result["depth"].get("Abs Rel", 0.0) - paper_depth_target["Abs Rel"],
            "delta1": result["depth"].get("δ < 1.25", 0.0) - paper_depth_target["delta1"],
        }

    # print summary
    for cat in ("pose", "depth", "world_point", "global_point", "paper_depth_delta"):
        if result.get(cat):
            rounded = {k: round(v, 5) for k, v in result[cat].items()}
            print(f"  {cat:>14s}: {rounded}")

    # save
    os.makedirs(json_root, exist_ok=True)
    path = os.path.join(json_root, f"eval_{name.lower()}.json")
    with open(path, "w") as f:
        json.dump({**result, "num_sequences": len(indices),
                   "num_eval_samples": len(samples),
                   "num_successful_samples": successful_samples,
                   "eval_unit": eval_unit,
                   "sample_stride": sample_stride,
                   "frames_per_seq": num_frames,
                   "depth_align": depth_align,
                   "depth_eval_latitude_band": [depth_lat_min, depth_lat_max],
                   "depth_irls_iters": depth_irls_iters,
                   "pointcloud_enabled": not skip_pointcloud,
                   "pointcloud_config": pc_cfg}, f, indent=2)
    print(f"  → {path}")
    return result


# =========================================================================
#  CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate on all panoramic datasets (pose / depth / point-cloud).")

    # paths
    p.add_argument("--datasets_root", type=str,
                   default=os.environ.get("PANOVGGT_DATASETS_ROOT", "/mnt/e/PanoVGGT_minimal_datasets/datasets"),
                   help="Root containing PanoCity, Matterport3D, Stanford2D3DS, and Structured3D subdirectories.")
    p.add_argument("--panocity_root", type=str,
                   default=os.environ.get("PANOCITY_ROOT"))
    p.add_argument("--matterport_root", type=str,
                   default=os.environ.get("MATTERPORT3D_ROOT"))
    p.add_argument("--stanford_root", type=str,
                   default=os.environ.get("STANFORD2D3DS_ROOT"))
    p.add_argument("--structured3d_root", type=str,
                   default=os.environ.get("STRUCTURED3D_ROOT"))
    p.add_argument("--datasets", nargs="+",
                   default=["panocity", "matterport", "stanford", "structured3d"],
                   choices=sorted(DATASET_KEYS.keys()),
                   help="Datasets to evaluate. Defaults to all four paper datasets.")

    # model
    p.add_argument("--ckpt",  type=str, default=str(_default_lowres_ckpt()),
                   help="Path to model checkpoint. Defaults to checkpoints/model_lowres.pt when present.")
    p.add_argument("--model_build_mode", type=str, default="hydra", choices=["hydra", "direct"],
                   help="Use repository Hydra config or direct module:Class construction.")
    p.add_argument("--config", type=str, default="default",
                   help="Hydra config name when --model_build_mode=hydra.")
    p.add_argument("--strict_load", action="store_true",
                   help="Require exact checkpoint/model key match.")
    p.add_argument("--model", type=str, default="panovggt.models.panovggt_model:PanoVGGTModel",
                   help="module:ClassName")
    p.add_argument("--model_kwargs", type=str, default=None,
                   help="JSON string of model constructor kwargs")
    p.add_argument("--model_kwargs_file", type=str, default=None,
                   help="Path to a JSON file with model constructor kwargs")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test", "test_final"])

    # per-dataset sequence counts (-1 = all)
    p.add_argument("--num_seqs_panocity",       type=int, default=-1)
    p.add_argument("--num_seqs_matterport",   type=int, default=-1)
    p.add_argument("--num_seqs_stanford",     type=int, default=-1)
    p.add_argument("--num_seqs_structured3d", type=int, default=-1)

    # per-dataset frame counts
    p.add_argument("--frames_panocity",       type=int, default=10)
    p.add_argument("--frames_matterport",   type=int, default=3)
    p.add_argument("--frames_stanford",     type=int, default=3)
    p.add_argument("--frames_structured3d", type=int, default=3)

    # evaluation
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--amp_dtype",   type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--depth_align", type=str, default="irls-absrel",
                   choices=["median-scale", "robust-scale", "scale&shift", "irls-absrel", "metric"])
    p.add_argument("--depth_lat_min", type=float, default=-15.0,
                   help="Minimum latitude in degrees for depth metric pixels after full-image alignment.")
    p.add_argument("--depth_lat_max", type=float, default=60.0,
                   help="Maximum latitude in degrees for depth metric pixels after full-image alignment.")
    p.add_argument("--depth_irls_iters", type=int, default=100,
                   help="IRLS iterations for scale+shift depth alignment.")
    p.add_argument("--eval_unit", type=str, default="sample", choices=["sample", "sequence"],
                   help="sample enumerates every deterministic sliding window inside each sequence; sequence keeps one call per sequence.")
    p.add_argument("--sample_stride", type=int, default=1,
                   help="Sliding-window stride when --eval_unit=sample.")
    p.add_argument("--no_pointcloud", action="store_true",
                   help="Skip expensive point-cloud ICP metrics and report pose/depth only.")
    p.add_argument("--json_root",   type=str, default="eval_results")

    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp_dtype)

    # ── load model ───────────────────────────────────────────────────
    if args.model_build_mode == "hydra":
        model = build_model_from_hydra_config(args.config, device)
    else:
        mod_path, cls_name = args.model.split(":")
        ModelClass = getattr(importlib.import_module(mod_path), cls_name)
        if args.model_kwargs_file:
            with open(args.model_kwargs_file, "r") as f:
                kwargs = json.load(f)
        else:
            kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
        model = ModelClass(**kwargs).to(device).eval()
    load_info = load_checkpoint(model, args.ckpt, device, strict=args.strict_load)
    model_img_size = int(getattr(getattr(model, "aggregator", None), "img_size", 518))

    # ── common dataset config ────────────────────────────────────────
    common = SimpleNamespace(
        img_size=model_img_size, patch_size=14, rescale=None, rescale_aug=None,
        landscape_check=False, training=False, get_nearby=True,
        inside_random=False, allow_duplicate_img=True, augs=None, debug=False,
    )

    all_results = {}

    # ── datasets ─────────────────────────────────────────────────────
    panocity_root = args.panocity_root or _dataset_root_from_base(args.datasets_root, "PanoCity")
    matterport_root = args.matterport_root or _dataset_root_from_base(args.datasets_root, "Matterport3D")
    stanford_root = args.stanford_root or _dataset_root_from_base(args.datasets_root, "Stanford2D3DS")
    structured3d_root = args.structured3d_root or _dataset_root_from_base(args.datasets_root, "Structured3D")

    DATASETS = [
        ("PanoCity", PanoCityDataset, {
            "PanoCity_DIR": _normalize_dataset_root("PanoCity", panocity_root),
            "min_num_images": max(2, args.frames_panocity),
        }, args.num_seqs_panocity, args.frames_panocity),

        ("Matterport3D", Matterport3DDataset, {
            "Matterport3D_DIR": _normalize_dataset_root("Matterport3D", matterport_root),
            "min_num_images": max(2, args.frames_matterport),
        }, args.num_seqs_matterport, args.frames_matterport),

        ("Stanford2D3DS", Stanford2D3DSDataset, {
            "Stanford2D3DS_DIR": _normalize_dataset_root("Stanford2D3DS", stanford_root),
            "min_num_images": max(2, args.frames_stanford),
        }, args.num_seqs_stanford, args.frames_stanford),

        ("Structured3D", Structured3DDataset, {
            "Structured3D_DIR": _normalize_dataset_root("Structured3D", structured3d_root),
            "min_num_rooms": max(2, args.frames_structured3d),
        }, args.num_seqs_structured3d, args.frames_structured3d),
    ]

    selected_datasets = {DATASET_KEYS[key] for key in args.datasets}
    for name, DatasetClass, ds_kwargs, n_seqs, n_frames in DATASETS:
        if name not in selected_datasets:
            continue
        if n_seqs == 0:
            print(f"Skipping {name} because requested sequence count is 0")
            continue
        ds = DatasetClass(
            common_conf=common, split=args.split,
            len_train=10**9, len_test=10**9,
            expand_ratio=3, augmentation=None,
            **ds_kwargs,
        )
        all_results[name] = eval_one_dataset(
            name, ds, model, device, amp_dtype,
            num_seqs=n_seqs, num_frames=n_frames,
            depth_align=args.depth_align,
            depth_lat_min=args.depth_lat_min,
            depth_lat_max=args.depth_lat_max,
            depth_irls_iters=args.depth_irls_iters,
            json_root=args.json_root,
            skip_pointcloud=args.no_pointcloud,
            eval_unit=args.eval_unit,
            sample_stride=args.sample_stride,
        )

    # ── global summary ───────────────────────────────────────────────
    summary_path = os.path.join(args.json_root, "eval_all_datasets.json")
    with open(summary_path, "w") as f:
        json.dump({
            "results": all_results,
            "config": vars(args),
            "resolved_dataset_roots": {
                "PanoCity": _normalize_dataset_root("PanoCity", panocity_root),
                "Matterport3D": _normalize_dataset_root("Matterport3D", matterport_root),
                "Stanford2D3DS": _normalize_dataset_root("Stanford2D3DS", stanford_root),
                "Structured3D": _normalize_dataset_root("Structured3D", structured3d_root),
            },
            "model_load": load_info,
            "model_img_size": model_img_size,
        }, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"  All done — summary saved to {summary_path}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
