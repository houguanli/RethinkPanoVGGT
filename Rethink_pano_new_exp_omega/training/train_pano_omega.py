"""Train VGGT-Omega LUNA on converted panorama data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import MixedPanoDataset, PanoCityPairedOmegaDataset, PanoMinimalDataset, PanoVKittiOmegaDataset  # noqa: E402
from vggt_omega.data.pano_sampler import make_default_view_grid  # noqa: E402
from vggt_omega.models.heads.dense_head import DenseHead  # noqa: E402
from vggt_omega.models.layers import PatchEmbed  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402
from vggt_omega.utils.pano_pose import (  # noqa: E402
    omega_y_up_pose_to_official_y_down,
    omega_y_up_vectors_to_official_y_down,
)
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat  # noqa: E402


DEFAULT_DATASET_ROOT = Path("whitehole/AOKI/datasets/PANO_LUNA_omega")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ckpt" / "vggt_omega_1b_512.pt"
CONFIG_PATH_KEYS = {
    "dataset_root",
    "checkpoint",
    "base_checkpoint",
    "output_dir",
    "log_csv",
    "loss_plot",
    "tensorboard_dir",
    "debug_dir",
    "metadata_path",
    "bad_sample_list",
    "dataset_roots",
}
DEFAULT_PANOCITY_PRED_DEPTH_SCALE = 5.491308212280273


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VGGT-Omega LUNA on converted pano VKitti-style data.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config; explicit CLI values override it.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-format", choices=["vkitti", "panocity_paired", "pano_minimal", "mixed_pano"], default="vkitti")
    parser.add_argument("--dataset-roots", nargs="+", type=Path, default=None)
    parser.add_argument("--minimal-datasets", type=str, default="all")
    parser.add_argument(
        "--dataset-sampling-weights",
        type=str,
        default=None,
        help="Optional mixed dataset sampling weights, e.g. panocity:0.5,matterport3d:0.3,structured3d:0.15,stanford2d3ds:0.05.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Optional original Omega checkpoint loaded before a warmup/fine-tune checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pano_omega_luna")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--distributed", choices=["auto", "none", "ddp"], default="auto")
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--find-unused-parameters", action="store_true", default=True)
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default="none")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pano-sample-mode",
        choices=["single", "fixed_neighborhood", "variable_neighborhood"],
        default="single",
    )
    parser.add_argument("--pano-min-count", type=int, default=1)
    parser.add_argument("--pano-max-count", type=int, default=1)
    parser.add_argument("--panos-per-sample", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pano-grouping", choices=["nearest", "sequential"], default="nearest")
    parser.add_argument("--dataset-max-samples", type=int, default=None, help="Optional dataset cap for debugging.")
    parser.add_argument("--dataset-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--train-split-fraction", type=float, default=0.95)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--bad-sample-list",
        type=Path,
        default=None,
        help="Optional newline-delimited bad PanoCity stems/paths to skip before training.",
    )
    parser.add_argument("--curriculum-bins", type=str, default=None)
    parser.add_argument("--use-metadata-weights", dest="use_metadata_weights", action="store_true", default=True)
    parser.add_argument("--no-metadata-weights", dest="use_metadata_weights", action="store_false")
    parser.add_argument(
        "--output-depth-scale",
        type=float,
        default=100.0,
        help="Scale factor used by flat paired datasets to convert stored depth values to meters.",
    )
    parser.add_argument(
        "--invalid-depth-value",
        type=float,
        default=None,
        help="Stored depth values greater than or equal to this are treated as invalid for paired datasets.",
    )
    parser.add_argument(
        "--pano-position-step-m",
        type=float,
        default=1.0,
        help="Synthetic center spacing used when a dataset does not provide pano poses.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer-type", choices=["adamw", "adafactor"], default="adamw")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--num-yaw", type=int, default=8)
    parser.add_argument("--pitch-degrees", type=str, default="0")
    parser.add_argument("--fov-degrees", type=float, default=75.0)
    parser.add_argument("--pano-height", type=int, default=0, help="Optional resize height before sampling.")
    parser.add_argument("--pano-width", type=int, default=0, help="Optional resize width before sampling.")
    parser.add_argument(
        "--luna-patch-layers",
        type=str,
        default="second_half",
        help="LUNA patch adapter layers: second_half, final, none, or comma-separated layer indices.",
    )
    parser.add_argument(
        "--luna-camera-layers",
        type=str,
        default="23",
        help="LUNA camera adapter layers: final, none, or comma-separated layer indices.",
    )
    parser.add_argument("--enable-pano-global-token", dest="enable_pano_global_token", action="store_true", default=True)
    parser.add_argument("--disable-pano-global-token", dest="enable_pano_global_token", action="store_false")
    parser.add_argument(
        "--enable-pano-geometry-residual",
        dest="enable_pano_geometry_residual",
        action="store_true",
        default=False,
        help="Add pano geometry to the existing camera token through an alpha=0 gated residual path.",
    )
    parser.add_argument(
        "--disable-pano-geometry-residual",
        dest="enable_pano_geometry_residual",
        action="store_false",
    )
    parser.add_argument(
        "--aggregator-checkpoint",
        dest="aggregator_use_checkpoint",
        action="store_true",
        default=False,
        help="Checkpoint each aggregator attention layer to reduce activation memory at the cost of recompute.",
    )
    parser.add_argument("--no-aggregator-checkpoint", dest="aggregator_use_checkpoint", action="store_false")
    parser.add_argument(
        "--training-stages",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--trainable",
        choices=[
            "luna",
            "dense",
            "camera",
            "heads",
            "luna_dense",
            "luna_heads",
            "luna_dense_tail",
            "luna_residual",
            "luna_residual_dense",
            "luna_residual_dense_tail",
            "luna_residual_tail_heads",
            "luna_residual_heads",
            "all",
        ],
        default="luna_heads",
    )
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument(
        "--inherit-checkpoint-training-defaults",
        dest="inherit_checkpoint_training_defaults",
        action="store_true",
        default=False,
        help=(
            "Adopt training defaults stored in a checkpoint payload. Disabled by default so "
            "stage handoff checkpoints do not override the next config's depth residual or "
            "scale-alignment policy."
        ),
    )
    parser.add_argument(
        "--no-inherit-checkpoint-training-defaults",
        dest="inherit_checkpoint_training_defaults",
        action="store_false",
    )
    parser.add_argument("--enable-camera-head", dest="enable_camera_head", action="store_true", default=True)
    parser.add_argument("--disable-camera-head", dest="enable_camera_head", action="store_false")
    parser.add_argument("--camera-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--camera-comparable-weight",
        type=float,
        default=0.25,
        help="Fixed camera weight used only for a cross-stage comparable loss metric.",
    )
    parser.add_argument("--camera-translation-weight", type=float, default=1.0)
    parser.add_argument("--camera-rotation-weight", type=float, default=1.0)
    parser.add_argument("--camera-fov-weight", type=float, default=0.1)
    parser.add_argument(
        "--global-point-loss-weight",
        type=float,
        default=0.0,
        help="Weight for sparse shared-frame point supervision built from predicted depth and pano pose.",
    )
    parser.add_argument(
        "--global-point-stride",
        type=int,
        default=16,
        help="Spatial stride for shared-frame point supervision; larger values reduce memory use.",
    )
    parser.add_argument(
        "--camera-translation-normalization",
        choices=["none", "target_rms", "target_mean_norm", "target_max_norm"],
        default="none",
        help="Normalize camera translation loss by the current sample's target relative pose scale.",
    )
    parser.add_argument(
        "--camera-translation-normalization-eps",
        type=float,
        default=1.0,
        help="Minimum translation normalization scale in meters.",
    )
    parser.add_argument(
        "--camera-supervision-mode",
        choices=["auto", "none", "window_pose", "pano_relative"],
        default="window_pose",
        help=(
            "Camera loss target. 'none' disables it; 'window_pose' supervises every sampled "
            "virtual window; 'pano_relative' supervises pano centers relative to pano0."
        ),
    )
    parser.add_argument(
        "--pano-translation-consistency-weight",
        type=float,
        default=0.1,
        help="Extra within-pano center consistency term used by camera-supervision-mode=pano_relative.",
    )
    parser.add_argument(
        "--camera-position-mode",
        choices=["none", "local_zero", "relative_anchor", "relative_mean", "world"],
        default="local_zero",
    )
    parser.add_argument(
        "--pred-depth-scale",
        type=float,
        default=1.0,
        help="Fixed metric calibration applied to predicted Z-depth before loss/export.",
    )
    parser.add_argument(
        "--learn-pred-depth-scale",
        action="store_true",
        help="Learn a positive scalar multiplier on predicted depth during training.",
    )
    parser.add_argument(
        "--pred-depth-scale-lr",
        type=float,
        default=None,
        help="Optional learning rate for --learn-pred-depth-scale; defaults to --lr.",
    )
    parser.add_argument(
        "--depth-residual-mode",
        choices=["none", "log_conv"],
        default="none",
        help="Optional residual depth adapter: final_depth = raw_depth * exp(delta_log_depth).",
    )
    parser.add_argument(
        "--depth-residual-hidden",
        type=int,
        default=32,
        help="Hidden channels for --depth-residual-mode=log_conv.",
    )
    parser.add_argument(
        "--depth-residual-max-log",
        type=float,
        default=0.5,
        help="Clamp residual log-depth correction to +/- this value.",
    )
    parser.add_argument(
        "--depth-residual-frames-chunk-size",
        type=int,
        default=8,
        help="Number of sampled windows processed by the depth residual head at once; set <=0 to disable chunking.",
    )
    parser.add_argument(
        "--depth-residual-checkpoint",
        dest="depth_residual_use_checkpoint",
        action="store_true",
        default=False,
        help="Checkpoint depth residual chunks to reduce activation memory.",
    )
    parser.add_argument("--no-depth-residual-checkpoint", dest="depth_residual_use_checkpoint", action="store_false")
    parser.add_argument(
        "--store-depth-residual-debug",
        action="store_true",
        default=False,
        help="Store raw_depth and depth_log_residual tensors in predictions for debugging.",
    )
    parser.add_argument(
        "--dense-head-frames-chunk-size",
        type=int,
        default=8,
        help=(
            "Number of sampled windows decoded by DenseHead at once. Smaller values reduce "
            "activation memory while preserving the full pano/window context in the aggregator; "
            "set <=0 to disable DenseHead chunking."
        ),
    )
    parser.add_argument(
        "--dense-head-checkpoint",
        dest="dense_head_use_checkpoint",
        action="store_true",
        default=False,
        help="Recompute DenseHead chunks during backward to reduce activation memory.",
    )
    parser.add_argument("--no-dense-head-checkpoint", dest="dense_head_use_checkpoint", action="store_false")
    parser.add_argument(
        "--dense-head-return-confidence",
        dest="dense_head_return_confidence",
        action="store_true",
        default=True,
        help="Return DenseHead confidence maps. Disable for training losses that do not consume depth_conf.",
    )
    parser.add_argument("--no-dense-head-return-confidence", dest="dense_head_return_confidence", action="store_false")
    parser.add_argument(
        "--gt-depth-semantics",
        choices=["range", "cubemap_z", "double_cubemap_z"],
        default="range",
        help="Interpretation of saved ERP GT depth before sampling virtual windows.",
    )
    parser.add_argument("--depth-max-m", type=float, default=80.0, help="Maximum valid radial GT depth in meters.")
    parser.add_argument(
        "--depth-loss-mode",
        choices=["log_l1", "log_huber", "clipped_log_l1"],
        default="log_l1",
        help="Robust depth loss variant applied in log-depth space.",
    )
    parser.add_argument(
        "--depth-log-huber-delta",
        type=float,
        default=0.2,
        help="SmoothL1 beta for --depth-loss-mode=log_huber.",
    )
    parser.add_argument(
        "--depth-log-error-clip",
        type=float,
        default=0.5,
        help="Maximum per-pixel absolute log-depth error for --depth-loss-mode=clipped_log_l1.",
    )
    parser.add_argument(
        "--depth-scale-alignment",
        choices=["none", "sample_lstsq", "sample_log_median"],
        default="none",
        help=(
            "Per-sample scale alignment for depth supervision. sample_lstsq follows the "
            "PanoVGGT/VGGT scale-normalized geometry protocol by fitting one optimal "
            "stop-gradient scale per training sample before computing depth loss."
        ),
    )
    parser.add_argument("--depth-scale-alignment-min", type=float, default=0.05)
    parser.add_argument("--depth-scale-alignment-max", type=float, default=50.0)
    parser.add_argument("--depth-scale-alignment-eps", type=float, default=1e-6)
    parser.add_argument(
        "--no-camera-depth-scale-alignment",
        dest="camera_depth_scale_alignment",
        action="store_false",
        default=True,
        help="Do not apply the same per-sample depth scale to camera translation predictions.",
    )
    parser.add_argument(
        "--camera-depth-scale-alignment",
        dest="camera_depth_scale_alignment",
        action="store_true",
    )
    parser.add_argument("--loss-sample-weighting", dest="loss_sample_weighting", action="store_true", default=True)
    parser.add_argument("--no-loss-sample-weighting", dest="loss_sample_weighting", action="store_false")
    parser.add_argument("--min-window-valid-ratio", type=float, default=0.05)
    parser.add_argument("--valid-ratio-loss-power", type=float, default=0.5)
    parser.add_argument("--sample-weight-min", type=float, default=0.25)
    parser.add_argument("--sample-weight-max", type=float, default=1.25)
    parser.add_argument("--overlap-consistency-weight", type=float, default=0.0)
    parser.add_argument("--overlap-band-fraction", type=float, default=0.20)
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--no-save-last", dest="save_last", action="store_false")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument(
        "--checkpoint-format",
        choices=["trainable_delta", "full"],
        default="trainable_delta",
        help="Use trainable_delta on 24GB GPUs; full stores every model tensor.",
    )
    parser.add_argument("--max-duration-minutes", type=float, default=0.0)
    parser.add_argument("--progress-bar", dest="progress_bar", action="store_true", default=True)
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument("--progress-log-every", type=int, default=50)
    parser.add_argument("--log-csv", type=Path, default=None)
    parser.add_argument("--camera-stats-json", type=Path, default=None)
    parser.add_argument("--camera-stats-every", type=int, default=50)
    parser.add_argument("--camera-error-thresholds-deg", type=str, default="5,10,15,30,45,60,90")
    parser.add_argument("--camera-error-example-limit", type=int, default=64)
    parser.add_argument("--loss-plot", type=Path, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=True)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--debug-depth-dump", dest="debug_depth_dump", action="store_true", default=True)
    parser.add_argument("--no-debug-depth-dump", dest="debug_depth_dump", action="store_false")
    parser.add_argument(
        "--debug-depth-threshold",
        type=float,
        default=0.3,
        help="Dump pred/GT depth windows whose per-window log-L1 depth loss exceeds this value.",
    )
    parser.add_argument("--debug-dir", type=Path, default=Path("logs/debug"))
    parser.add_argument("--debug-depth-max-dumps-per-step", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny generated-data training step.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="pano_omega_smoke_") as tmp:
            smoke_root = Path(tmp) / "converted_pano4vggt_omega"
            write_smoke_dataset(smoke_root)
            args.dataset_root = smoke_root
            args.dataset_format = "vkitti"
            args.checkpoint = None
            args.output_dir = Path(tmp) / "outputs"
            args.device = "cpu"
            args.max_steps = 1
            args.epochs = 1
            args.batch_size = 1
            args.num_workers = 0
            args.window_size = 32
            args.num_yaw = 2
            args.pano_height = 32
            args.pano_width = 64
            train(args)
        return

    train(args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv_list)
    parser = build_parser()
    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(config_args.config, parser))
    return parser.parse_args(argv_list)


def load_config_defaults(path: Path, parser: argparse.ArgumentParser) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    flattened: Dict = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    allowed = {action.dest for action in parser._actions}
    unknown = sorted(set(flattened) - allowed)
    if unknown:
        raise ValueError(f"Unknown training config keys in {path}: {unknown}")
    for key in CONFIG_PATH_KEYS:
        if key in flattened and flattened[key] is not None:
            if key == "dataset_roots":
                flattened[key] = [Path(value) for value in flattened[key]]
            else:
                flattened[key] = Path(flattened[key])
    flattened["config"] = path
    return flattened


def train(args: argparse.Namespace) -> None:
    dist_state = setup_distributed(args)
    device = resolve_device(args.device, dist_state)
    pano_size = (args.pano_height, args.pano_width) if args.pano_height > 0 and args.pano_width > 0 else None
    normalize_pano_sampling_args(args)
    normalize_camera_supervision_args(args)
    normalize_pred_depth_scale_args(args)
    capture_default_sampler_args(args)
    capture_default_loss_args(args)
    args.training_stages = normalize_training_stages(args.training_stages)
    if args.pano_sample_mode == "variable_neighborhood" and args.batch_size != 1:
        raise ValueError("variable_neighborhood uses variable-length inputs and currently requires batch_size=1.")
    try:
        dataset = build_dataset(args, pano_size)
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_state["world_size"],
            rank=dist_state["rank"],
            shuffle=True,
            drop_last=False,
        ) if dist_state["distributed"] else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        model = build_model(args).to(device)
        checkpoint_payload = load_checkpoint_payload(args.checkpoint) if args.checkpoint is not None else {}
        if args.inherit_checkpoint_training_defaults:
            apply_checkpoint_training_defaults(args, checkpoint_payload)
        if args.base_checkpoint is not None and (
            args.checkpoint is None or args.base_checkpoint.resolve() != args.checkpoint.resolve()
        ):
            load_checkpoint(model, args.base_checkpoint, strict=False)
        if args.checkpoint is not None:
            load_checkpoint(model, args.checkpoint, strict=args.strict_checkpoint)

        if args.learn_pred_depth_scale or args.depth_residual_mode != "none":
            model = DepthPredictionAdapter(
                model,
                initial_scale=args.pred_depth_scale,
                learn_scale=args.learn_pred_depth_scale,
                residual_mode=args.depth_residual_mode,
                residual_hidden=args.depth_residual_hidden,
                residual_max_log=args.depth_residual_max_log,
                residual_frames_chunk_size=normalize_dense_head_frames_chunk_size(
                    args.depth_residual_frames_chunk_size
                ),
                residual_use_checkpoint=args.depth_residual_use_checkpoint,
                store_residual_debug=args.store_depth_residual_debug,
            ).to(device)
            load_adapter_state_if_present(model, checkpoint_payload, args.checkpoint)

        active_stage_index = None
        active_stage_name = "default"
        active_stage = None
        if args.training_stages:
            active_stage_index, active_stage = select_training_stage(
                args.training_stages,
                next_step=1,
                elapsed_minutes=0.0,
            )
            active_stage_name = str(active_stage.get("name", f"stage{active_stage_index + 1}"))
            apply_stage_sampler_overrides(model, args, active_stage)
            apply_stage_loss_overrides(args, active_stage)
            trainable_count, frozen_count = configure_trainable_for_stage(model, args, active_stage)
        else:
            if isinstance(model, DepthPredictionAdapter):
                configure_trainable(model.model, args.trainable)
                if isinstance(model.pred_depth_log_scale, torch.nn.Parameter):
                    model.pred_depth_log_scale.requires_grad_(bool(args.learn_pred_depth_scale))
                if model.depth_residual_head is not None:
                    for param in model.depth_residual_head.parameters():
                        param.requires_grad_(True)
                trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
                frozen_count = sum(param.numel() for param in model.parameters() if not param.requires_grad)
            else:
                trainable_count, frozen_count = configure_trainable(model, args.trainable)
        if dist_state["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist_state["local_rank"]] if device.type == "cuda" else None,
                output_device=dist_state["local_rank"] if device.type == "cuda" else None,
                find_unused_parameters=args.find_unused_parameters,
            )
        optimizer = build_optimizer_for_stage(model, args, active_stage)

        rank0_print(f"[INFO] dataset_root = {dataset.root}", dist_state)
        rank0_print(f"[INFO] dataset_format = {args.dataset_format}", dist_state)
        if hasattr(dataset, "split"):
            rank0_print(
                f"[INFO] dataset_split = {dataset.split} "
                f"train_fraction={getattr(dataset, 'train_split_fraction', 'n/a')} "
                f"split_seed={getattr(dataset, 'split_seed', 'n/a')}",
                dist_state,
            )
        rank0_print(f"[INFO] samples = {len(dataset)}", dist_state)
        sampling_summary = getattr(dataset, "dataset_sampling_summary", None)
        if sampling_summary is not None:
            rank0_print(f"[INFO] dataset_sampling = {json.dumps(sampling_summary, sort_keys=True)}", dist_state)
        rank0_print(
        "[INFO] pano_sampling = "
        f"{args.pano_sample_mode} min={args.pano_min_count} max={args.pano_max_count} grouping={args.pano_grouping}",
        dist_state,
        )
        rank0_print(
            f"[INFO] luna_layers = patch:{args.luna_patch_layers} camera:{args.luna_camera_layers}",
            dist_state,
        )
        rank0_print(f"[INFO] enable_pano_global_token = {args.enable_pano_global_token}", dist_state)
        rank0_print(f"[INFO] enable_pano_geometry_residual = {args.enable_pano_geometry_residual}", dist_state)
        rank0_print(f"[INFO] aggregator_checkpoint = {args.aggregator_use_checkpoint}", dist_state)
        rank0_print(
            "[INFO] camera_supervision = "
            f"{args.camera_supervision_mode} position={args.camera_position_mode} "
            f"weight={args.camera_loss_weight} "
            f"translation_norm={args.camera_translation_normalization}",
            dist_state,
        )
        if args.camera_supervision_mode == "pano_relative":
            rank0_print(
                "[INFO] camera_prediction_basis = omega_y_up_to_official_y_down",
                dist_state,
            )
            rank0_print(
                "[INFO] camera_pair_graph = all_ordered_pairs_after_joint_forward "
                "(N input panos -> N*(N-1) directed camera edges)",
                dist_state,
            )
            rank0_print(
                f"[INFO] shared_frame_points = weight:{args.global_point_loss_weight} "
                f"stride:{args.global_point_stride}",
                dist_state,
            )
        rank0_print(f"[INFO] pred_depth_scale = {args.pred_depth_scale}", dist_state)
        rank0_print(f"[INFO] learn_pred_depth_scale = {args.learn_pred_depth_scale}", dist_state)
        rank0_print(
            "[INFO] depth_scale_alignment = "
            f"{args.depth_scale_alignment} "
            f"range=[{args.depth_scale_alignment_min}, {args.depth_scale_alignment_max}] "
            f"camera={args.camera_depth_scale_alignment}",
            dist_state,
        )
        rank0_print(
            "[INFO] depth_residual = "
            f"{args.depth_residual_mode} hidden={args.depth_residual_hidden} "
            f"max_log={args.depth_residual_max_log} "
            f"frames_chunk_size={args.depth_residual_frames_chunk_size} "
            f"checkpoint={args.depth_residual_use_checkpoint} "
            f"store_debug={args.store_depth_residual_debug}",
            dist_state,
        )
        rank0_print(
            "[INFO] depth_loss = "
            f"{args.depth_loss_mode} huber_delta={args.depth_log_huber_delta} "
            f"clip={args.depth_log_error_clip}",
            dist_state,
        )
        rank0_print(
            "[INFO] dense_head = "
            f"frames_chunk_size={args.dense_head_frames_chunk_size} "
            f"checkpoint={args.dense_head_use_checkpoint} "
            f"return_confidence={args.dense_head_return_confidence}",
            dist_state,
        )
        rank0_print(
            f"[INFO] distributed = {dist_state['distributed']} "
            f"rank={dist_state['rank']} world_size={dist_state['world_size']} local_rank={dist_state['local_rank']}",
            dist_state,
        )
        rank0_print(f"[INFO] device = {device}", dist_state)
        rank0_print(f"[INFO] trainable_params = {trainable_count:,}; frozen_params = {frozen_count:,}", dist_state)
        if args.training_stages:
            rank0_print(f"[INFO] training_stages = {format_training_stages(args.training_stages)}", dist_state)
            rank0_print(
                f"[INFO] active_stage = {format_stage_status(active_stage_index, active_stage, trainable_count, frozen_count)}",
                dist_state,
            )

        if is_main_process(dist_state):
            args.output_dir.mkdir(parents=True, exist_ok=True)
        barrier(dist_state)
        log_csv = args.log_csv or (args.output_dir / "loss.csv")
        camera_stats_json = args.camera_stats_json or (args.output_dir / "camera_error_stats.json")
        camera_stats = CameraErrorStats(
            thresholds_deg=parse_float_list(args.camera_error_thresholds_deg),
            example_limit=args.camera_error_example_limit,
        )
        loss_plot = args.loss_plot or (args.output_dir / "loss_curve.png")
        tensorboard_dir = args.tensorboard_dir or (args.output_dir / "tensorboard")
        tensorboard_writer = create_tensorboard_writer(tensorboard_dir, dist_state, enabled=args.tensorboard)
        rank0_print(f"[INFO] camera_stats_json = {camera_stats_json}", dist_state)
        max_duration_seconds = args.max_duration_minutes * 60.0 if args.max_duration_minutes > 0 else None
        started_at = time.time()
        metrics_history = []

        model.train()
        global_step = 0
        stop_reason = "max_steps"
        progress = None
        last_progress_elapsed = 0.0
        if is_main_process(dist_state) and args.progress_bar:
            if max_duration_seconds is not None:
                progress = tqdm(
                    total=int(max_duration_seconds),
                    desc="LUNA train",
                    unit="s",
                    dynamic_ncols=True,
                    leave=True,
                )
            else:
                progress = tqdm(total=int(args.max_steps), desc="LUNA train", unit="step", dynamic_ncols=True, leave=True)
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                if should_stop_for_duration(started_at, max_duration_seconds, device, dist_state):
                    stop_reason = "max_duration"
                    break

                elapsed_before_step = time.time() - started_at
                if args.training_stages:
                    next_stage_index, next_stage = select_training_stage(
                        args.training_stages,
                        next_step=global_step + 1,
                        elapsed_minutes=elapsed_before_step / 60.0,
                    )
                    if next_stage_index != active_stage_index:
                        active_stage_index = next_stage_index
                        active_stage = next_stage
                        active_stage_name = str(active_stage.get("name", f"stage{active_stage_index + 1}"))
                        apply_stage_sampler_overrides(unwrap_model(model), args, active_stage)
                        apply_stage_loss_overrides(args, active_stage)
                        trainable_count, frozen_count = configure_trainable_for_stage(
                            unwrap_model(model),
                            args,
                            active_stage,
                        )
                        optimizer, optimizer_preserved = transition_optimizer_for_stage(
                            optimizer,
                            model,
                            args,
                            active_stage,
                        )
                        rank0_print(
                            f"[INFO] active_stage = {format_stage_status(active_stage_index, active_stage, trainable_count, frozen_count)}",
                            dist_state,
                        )
                        rank0_print(
                            f"[INFO] optimizer_state_preserved = {optimizer_preserved}",
                            dist_state,
                        )

                global_step += 1
                batch = limit_batch_panos(batch, active_stage)
                batch = move_batch_to_device(batch, device)
                try:
                    loss_dict = train_step(model, batch, optimizer, args, global_step=global_step)
                except torch.cuda.OutOfMemoryError:
                    print_cuda_memory(f"[OOM] rank={dist_state['rank']} step={global_step}", device)
                    raise
                local_camera_records = camera_diag_records_from_batch(
                    loss_dict=loss_dict,
                    batch=batch,
                    global_step=global_step,
                    stage_name=active_stage_name,
                    dist_state=dist_state,
                )
                camera_records = gather_camera_diag_records(local_camera_records, dist_state)
                loss_dict = strip_camera_diag_tensors(loss_dict)
                loss_dict = reduce_loss_dict(loss_dict, dist_state)
                elapsed_seconds = time.time() - started_at
                sampler_status = current_sampler_status(unwrap_model(model), args)
                metrics = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "elapsed_seconds": elapsed_seconds,
                    "loss": float(loss_dict["loss"].item()),
                    "loss_comparable": float(loss_dict.get("loss_comparable", loss_dict["loss"]).item()),
                    "loss_depth": float(loss_dict["loss_depth"].item()),
                    "loss_overlap": float(loss_dict.get("loss_overlap", torch.tensor(0.0)).item()),
                    "loss_overlap_weighted": float(
                        loss_dict.get("loss_overlap_weighted", torch.tensor(0.0)).item()
                    ),
                    "loss_global_point": float(
                        loss_dict.get("loss_global_point", torch.tensor(0.0)).item()
                    ),
                    "loss_global_point_weighted": float(
                        loss_dict.get("loss_global_point_weighted", torch.tensor(0.0)).item()
                    ),
                    "loss_camera": float(loss_dict["loss_camera"].item()),
                    "loss_camera_weighted": float(
                        loss_dict.get("loss_camera_weighted", torch.tensor(0.0)).item()
                    ),
                    "loss_camera_t": float(loss_dict.get("loss_camera_t", torch.tensor(0.0)).item()),
                    "loss_camera_r": float(loss_dict.get("loss_camera_r", torch.tensor(0.0)).item()),
                    "loss_camera_fov": float(loss_dict.get("loss_camera_fov", torch.tensor(0.0)).item()),
                    "loss_camera_consistency": float(
                        loss_dict.get("loss_camera_consistency", torch.tensor(0.0)).item()
                    ),
                    "camera_rotation_deg": float(
                        loss_dict.get("camera_rotation_deg", torch.tensor(0.0)).item()
                    ),
                    "camera_translation_deg": float(
                        loss_dict.get("camera_translation_deg", torch.tensor(0.0)).item()
                    ),
                    "camera_translation_valid_count": float(
                        loss_dict.get("camera_translation_valid_count", torch.tensor(0.0)).item()
                    ),
                    "camera_rotation_valid_count": float(
                        loss_dict.get("camera_rotation_valid_count", torch.tensor(0.0)).item()
                    ),
                    "global_point_valid_ratio": float(
                        loss_dict.get("global_point_valid_ratio", torch.tensor(0.0)).item()
                    ),
                    "global_point_finite_ratio": float(
                        loss_dict.get("global_point_finite_ratio", torch.tensor(0.0)).item()
                    ),
                    "global_point_geometry_finite_ratio": float(
                        loss_dict.get("global_point_geometry_finite_ratio", torch.tensor(0.0)).item()
                    ),
                    "pano_count": float(loss_dict.get("pano_count", torch.tensor(1.0)).item()),
                    "depth_valid_ratio": float(loss_dict.get("depth_valid_ratio", torch.tensor(0.0)).item()),
                    "depth_window_keep_ratio": float(
                        loss_dict.get("depth_window_keep_ratio", torch.tensor(0.0)).item()
                    ),
                    "depth_loss_valid_ratio": float(loss_dict.get("depth_loss_valid_ratio", torch.tensor(0.0)).item()),
                    "depth_loss_window_keep_ratio": float(
                        loss_dict.get("depth_loss_window_keep_ratio", torch.tensor(0.0)).item()
                    ),
                    "pred_depth_finite_ratio": float(loss_dict.get("pred_depth_finite_ratio", torch.tensor(0.0)).item()),
                    "loss_depth_unfiltered": float(loss_dict.get("loss_depth_unfiltered", torch.tensor(0.0)).item()),
                    "depth_sample_scale_median": float(
                        loss_dict.get("depth_sample_scale_median", torch.tensor(1.0)).item()
                    ),
                    "depth_sample_scale_min": float(loss_dict.get("depth_sample_scale_min", torch.tensor(1.0)).item()),
                    "depth_sample_scale_max": float(loss_dict.get("depth_sample_scale_max", torch.tensor(1.0)).item()),
                    "pred_depth_scale": float(loss_dict["pred_depth_scale"].item()),
                    "stage": active_stage_name,
                    "stage_index": int(active_stage_index + 1) if active_stage_index is not None else 0,
                    "window_size": sampler_status["window_size"],
                    "patch_size": sampler_status["patch_size"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if is_main_process(dist_state):
                    if camera_records:
                        camera_stats.update(camera_records)
                    if (
                        args.camera_stats_every > 0
                        and global_step % args.camera_stats_every == 0
                        and camera_stats.total_records > 0
                    ):
                        camera_stats.write(camera_stats_json)
                    metrics_history.append(metrics)
                    append_loss_csv(log_csv, metrics)
                    write_tensorboard_metrics(tensorboard_writer, metrics)
                    message = (
                        f"[TRAIN] epoch={epoch + 1} step={global_step} "
                        f"stage={active_stage_name} elapsed={elapsed_seconds / 60.0:.2f}m "
                        f"loss={metrics['loss']:.6f} depth={metrics['loss_depth']:.6f} "
                        f"overlap={metrics['loss_overlap']:.6f} global={metrics['loss_global_point']:.6f} "
                        f"camera={metrics['loss_camera']:.6f} "
                        f"camera_t={metrics['loss_camera_t']:.6f} "
                        f"camera_t_deg={metrics['camera_translation_deg']:.3f} "
                        f"camera_r_deg={metrics['camera_rotation_deg']:.3f} "
                        f"camera_valid=t{metrics['camera_translation_valid_count']:.1f}/"
                        f"r{metrics['camera_rotation_valid_count']:.1f} "
                        f"panos={metrics['pano_count']:.1f} "
                        f"depth_valid={metrics['depth_valid_ratio']:.4f} "
                        f"depth_loss_valid={metrics['depth_loss_valid_ratio']:.4f} "
                        f"depth_keep={metrics['depth_loss_window_keep_ratio']:.4f} "
                        f"pred_finite={metrics['pred_depth_finite_ratio']:.4f} "
                        f"sample_scale={metrics['depth_sample_scale_median']:.4f} "
                        f"scale={metrics['pred_depth_scale']:.6f}"
                    )
                    if progress is not None:
                        if max_duration_seconds is not None:
                            update = max(0, int(elapsed_seconds) - int(last_progress_elapsed))
                            if update:
                                progress.update(update)
                                last_progress_elapsed = elapsed_seconds
                        else:
                            progress.update(1)
                        progress.set_postfix(
                            step=global_step,
                            stage=active_stage_name,
                            window=metrics["window_size"],
                            loss=f"{metrics['loss']:.4f}",
                            comp=f"{metrics['loss_comparable']:.4f}",
                            depth=f"{metrics['loss_depth']:.4f}",
                            global_pt=f"{metrics['loss_global_point']:.4f}",
                            gfinite=f"{metrics['global_point_finite_ratio']:.3f}",
                            ggeom=f"{metrics['global_point_geometry_finite_ratio']:.3f}",
                            camera=f"{metrics['loss_camera']:.4f}",
                            tdeg=f"{metrics['camera_translation_deg']:.1f}",
                            rdeg=f"{metrics['camera_rotation_deg']:.1f}",
                            valid=f"{metrics['depth_valid_ratio']:.3f}",
                            lvalid=f"{metrics['depth_loss_valid_ratio']:.3f}",
                            keep=f"{metrics['depth_loss_window_keep_ratio']:.3f}",
                            pfinite=f"{metrics['pred_depth_finite_ratio']:.3f}",
                            sscale=f"{metrics['depth_sample_scale_median']:.3f}",
                            scale=f"{metrics['pred_depth_scale']:.3f}",
                        )
                        if args.progress_log_every > 0 and global_step % args.progress_log_every == 0:
                            progress.write(message)
                    else:
                        print(message, flush=True)

                if (
                    is_main_process(dist_state)
                    and args.save_every_steps > 0
                    and global_step % args.save_every_steps == 0
                    and not args.smoke
                ):
                    save_checkpoint(args.output_dir / f"step_{global_step:06d}.pt", unwrap_model(model), args, global_step)

                if global_step >= args.max_steps:
                    stop_reason = "max_steps"
                    break

            if global_step >= args.max_steps or stop_reason == "max_duration":
                break
        else:
            stop_reason = "epochs_complete"
    finally:
        if "progress" in locals() and progress is not None:
            progress.close()
        if "tensorboard_writer" in locals() and tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()
        if "metrics_history" in locals() and is_main_process(dist_state):
            if metrics_history:
                save_loss_plot(loss_plot, metrics_history)
            if "camera_stats" in locals() and camera_stats.total_records > 0:
                camera_stats.write(camera_stats_json)
                print(f"[INFO] saved camera stats = {camera_stats_json}")
            if args.save_last and not args.smoke:
                ckpt_path = args.output_dir / "last.pt"
                save_checkpoint(ckpt_path, unwrap_model(model), args, global_step)
                print(f"[INFO] saved checkpoint = {ckpt_path}")
            print(f"[INFO] stop_reason = {stop_reason}; steps = {global_step}")
        cleanup_distributed(dist_state)


def build_dataset(args: argparse.Namespace, pano_size: Tuple[int, int] | None):
    common_kwargs = {
        "root": args.dataset_root,
        "pano_size": pano_size,
        "pano_sample_mode": args.pano_sample_mode,
        "pano_min_count": args.pano_min_count,
        "pano_max_count": args.pano_max_count,
        "grouping": args.pano_grouping,
        "max_samples": args.dataset_max_samples,
    }
    if args.dataset_format == "vkitti":
        return PanoVKittiOmegaDataset(**common_kwargs)
    if args.dataset_format == "panocity_paired":
        return PanoCityPairedOmegaDataset(
            **common_kwargs,
            output_depth_scale=args.output_depth_scale,
            invalid_depth_value=args.invalid_depth_value,
            position_step_m=args.pano_position_step_m,
            split=args.dataset_split,
            train_split_fraction=args.train_split_fraction,
            split_seed=args.split_seed,
            metadata_path=args.metadata_path,
            bad_sample_list=args.bad_sample_list,
            curriculum_bins=args.curriculum_bins,
            use_metadata_weights=args.use_metadata_weights,
        )
    if args.dataset_format == "pano_minimal":
        return PanoMinimalDataset(
            **common_kwargs,
            split=args.dataset_split,
            train_split_fraction=args.train_split_fraction,
            split_seed=args.split_seed,
            datasets=args.minimal_datasets,
            bad_sample_list=args.bad_sample_list,
            dataset_sampling_weights=args.dataset_sampling_weights,
            output_depth_scale=args.output_depth_scale,
            invalid_depth_value=args.invalid_depth_value,
        )
    if args.dataset_format == "mixed_pano":
        roots = args.dataset_roots or [args.dataset_root]
        datasets = []
        for root in roots:
            root = Path(root)
            if (root / "rgb").exists() and (root / "depth").exists():
                datasets.append(
                    PanoCityPairedOmegaDataset(
                        **{**common_kwargs, "root": root},
                        output_depth_scale=args.output_depth_scale,
                        invalid_depth_value=args.invalid_depth_value,
                        position_step_m=args.pano_position_step_m,
                        split=args.dataset_split,
                        train_split_fraction=args.train_split_fraction,
                        split_seed=args.split_seed,
                        metadata_path=args.metadata_path,
                        bad_sample_list=args.bad_sample_list,
                        curriculum_bins=args.curriculum_bins,
                        use_metadata_weights=args.use_metadata_weights,
                    )
                )
            else:
                datasets.append(
                    PanoMinimalDataset(
                        **{**common_kwargs, "root": root},
                        split=args.dataset_split,
                        train_split_fraction=args.train_split_fraction,
                        split_seed=args.split_seed,
                        datasets=args.minimal_datasets,
                        bad_sample_list=args.bad_sample_list,
                        dataset_sampling_weights=args.dataset_sampling_weights,
                        output_depth_scale=args.output_depth_scale,
                        invalid_depth_value=args.invalid_depth_value,
                    )
                )
        return MixedPanoDataset(datasets)
    raise ValueError(f"Unknown dataset_format: {args.dataset_format}")


class DepthPredictionAdapter(torch.nn.Module):
    """Wrap VGGT-Omega with optional positive scale and log-depth residual.

    The residual path is initialized to identity:

        final_depth = raw_depth * exp(clamp(delta_log_depth)).

    It uses sampled pano windows plus raw predicted log depth, so it learns a
    small correction around the pretrained Omega depth instead of replacing it.
    """

    def __init__(
        self,
        model: VGGTOmega_LUNA,
        initial_scale: float,
        learn_scale: bool = False,
        residual_mode: str = "none",
        residual_hidden: int = 32,
        residual_max_log: float = 0.5,
        residual_frames_chunk_size: int | None = 8,
        residual_use_checkpoint: bool = False,
        store_residual_debug: bool = False,
    ) -> None:
        super().__init__()
        if initial_scale <= 0:
            raise ValueError(f"initial pred_depth_scale must be positive, got {initial_scale}")
        self.model = model
        self.learn_scale = bool(learn_scale)
        self.residual_mode = residual_mode
        self.residual_max_log = float(residual_max_log)
        self.depth_residual_enabled = residual_mode != "none"
        self.residual_frames_chunk_size = (
            None if residual_frames_chunk_size is None or int(residual_frames_chunk_size) <= 0
            else int(residual_frames_chunk_size)
        )
        self.residual_use_checkpoint = bool(residual_use_checkpoint)
        self.store_residual_debug = bool(store_residual_debug)
        log_scale = torch.tensor(math.log(float(initial_scale)), dtype=torch.float32)
        self.register_buffer("initial_pred_depth_log_scale", log_scale.clone())
        if self.learn_scale:
            self.pred_depth_log_scale = torch.nn.Parameter(log_scale)
        else:
            self.register_buffer("pred_depth_log_scale", log_scale)

        if residual_mode == "none":
            self.depth_residual_head = None
        elif residual_mode == "log_conv":
            hidden = max(4, int(residual_hidden))
            self.depth_residual_head = torch.nn.Sequential(
                torch.nn.Conv2d(4, hidden, kernel_size=3, padding=1),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv2d(hidden, 1, kernel_size=1),
            )
            final = self.depth_residual_head[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
        else:
            raise ValueError(f"Unknown depth residual mode: {residual_mode}")

    def forward(self, *args, **kwargs) -> Dict:
        predictions = dict(self.model(*args, **kwargs))
        predictions["_pred_depth_scale"] = self.pred_depth_scale()
        raw_depth = torch.nan_to_num(
            predictions["depth"].float(),
            nan=1e-4,
            posinf=1e4,
            neginf=1e-4,
        ).clamp_min(1e-4)
        predictions["depth"] = raw_depth
        if self.depth_residual_head is None:
            return predictions
        if not self.depth_residual_enabled:
            return predictions
        windows = predictions.get("pano_windows")
        if windows is None:
            return predictions
        batch_size, num_views, _, height, width = windows.shape
        chunk_size = self.residual_frames_chunk_size or num_views
        delta_chunks = []
        for start_idx in range(0, num_views, chunk_size):
            end_idx = min(start_idx + chunk_size, num_views)
            rgb = windows[:, start_idx:end_idx].reshape(-1, 3, height, width).float()
            log_depth = torch.log(raw_depth[:, start_idx:end_idx]).permute(0, 1, 4, 2, 3)
            log_depth = log_depth.reshape(-1, 1, height, width)
            if self.residual_use_checkpoint and self.training and torch.is_grad_enabled():
                delta_chunk = checkpoint(
                    self._run_depth_residual_head,
                    rgb,
                    log_depth,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                delta_chunk = self._run_depth_residual_head(rgb, log_depth)
            delta_chunks.append(delta_chunk)
        delta = torch.cat(delta_chunks, dim=0)
        delta = delta.reshape(batch_size, num_views, 1, height, width).permute(0, 1, 3, 4, 2)
        delta_limit = float(self.residual_max_log) if self.residual_max_log > 0 else 10.0
        delta = torch.nan_to_num(
            delta.float(),
            nan=0.0,
            posinf=delta_limit,
            neginf=-delta_limit,
        ).clamp(min=-delta_limit, max=delta_limit)
        if self.store_residual_debug:
            predictions["raw_depth"] = raw_depth
            predictions["depth_log_residual"] = delta
        final_depth = raw_depth * torch.exp(delta)
        predictions["depth"] = torch.nan_to_num(
            final_depth,
            nan=1e-4,
            posinf=1e4,
            neginf=1e-4,
        ).clamp_min(1e-4)
        return predictions

    def _run_depth_residual_head(self, rgb: torch.Tensor, log_depth: torch.Tensor) -> torch.Tensor:
        if self.depth_residual_head is None:
            raise RuntimeError("Depth residual head is not initialized.")
        residual_input = torch.cat([rgb, log_depth], dim=1)
        delta = self.depth_residual_head(residual_input)
        if self.residual_max_log > 0:
            delta = torch.tanh(delta) * self.residual_max_log
        return delta

    def pred_depth_scale(self) -> torch.Tensor:
        log_scale = torch.nan_to_num(
            self.pred_depth_log_scale.float(),
            nan=float(self.initial_pred_depth_log_scale.item()),
            posinf=8.0,
            neginf=-8.0,
        ).clamp(min=-8.0, max=8.0)
        return log_scale.exp()

    @torch.no_grad()
    def sanitize_parameters(self) -> None:
        if isinstance(self.pred_depth_log_scale, torch.nn.Parameter):
            data = self.pred_depth_log_scale.data
            fallback = self.initial_pred_depth_log_scale.to(device=data.device, dtype=data.dtype)
            data.copy_(torch.where(torch.isfinite(data), data, fallback).clamp(min=-8.0, max=8.0))

    def sample_pano_windows(self, *args, **kwargs):
        return self.model.sample_pano_windows(*args, **kwargs)

    @property
    def pano_sampler(self):
        return self.model.pano_sampler


LearnablePredDepthScale = DepthPredictionAdapter


def load_checkpoint_payload(checkpoint_path: Path | None) -> Dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    return payload if isinstance(payload, dict) else {}


def apply_checkpoint_training_defaults(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    if not payload:
        return
    ckpt_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    if payload.get("pred_depth_scale") is not None:
        args.pred_depth_scale = float(payload["pred_depth_scale"])
    elif ckpt_args.get("pred_depth_scale") is not None:
        args.pred_depth_scale = float(ckpt_args["pred_depth_scale"])
    for key in ("learn_pred_depth_scale", "depth_residual_mode", "depth_residual_hidden", "depth_residual_max_log"):
        if key in ckpt_args and ckpt_args[key] is not None:
            setattr(args, key, ckpt_args[key])


def load_adapter_state_if_present(
    model: DepthPredictionAdapter,
    payload: Dict[str, Any],
    checkpoint_path: Path | None,
) -> None:
    adapter_state = payload.get("adapter_state") if isinstance(payload, dict) else None
    if adapter_state is None:
        return
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    print(f"[INFO] loaded adapter_state from {checkpoint_path}")
    print(f"[INFO] adapter_state missing_keys={len(missing)} unexpected_keys={len(unexpected)}")


def build_optimizer_for_stage(
    model: torch.nn.Module,
    args: argparse.Namespace,
    stage: Dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    lr = float(stage.get("lr", args.lr)) if stage is not None else args.lr
    weight_decay = float(stage.get("weight_decay", args.weight_decay)) if stage is not None else args.weight_decay
    optimizer_type = str(stage.get("optimizer_type", args.optimizer_type)) if stage is not None else args.optimizer_type
    pred_depth_scale_lr = (
        float(stage["pred_depth_scale_lr"])
        if stage is not None and stage.get("pred_depth_scale_lr") is not None
        else args.pred_depth_scale_lr
    )
    return build_optimizer(
        model,
        args,
        lr=lr,
        weight_decay=weight_decay,
        pred_depth_scale_lr=pred_depth_scale_lr,
        optimizer_type=optimizer_type,
    )


def transition_optimizer_for_stage(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    args: argparse.Namespace,
    stage: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, bool]:
    """Keep optimizer moments when a stage only changes scalar hyperparameters."""
    requested_type = str(stage.get("optimizer_type", args.optimizer_type))
    current_type = "adafactor" if isinstance(optimizer, torch.optim.Adafactor) else "adamw"
    current_params = {id(param) for group in optimizer.param_groups for param in group["params"]}
    requested_params = {id(param) for param in model.parameters() if param.requires_grad}
    if requested_type != current_type or current_params != requested_params:
        return build_optimizer_for_stage(model, args, stage), False

    regular_lr = float(stage.get("lr", args.lr))
    scale_lr_value = stage.get("pred_depth_scale_lr", regular_lr)
    scale_lr = regular_lr if scale_lr_value is None else float(scale_lr_value)
    weight_decay = float(stage.get("weight_decay", args.weight_decay))
    for group in optimizer.param_groups:
        if group.get("group_name") == "scale":
            group["lr"] = scale_lr
            group["weight_decay"] = 0.0
        else:
            group["lr"] = regular_lr
            group["weight_decay"] = weight_decay
    return optimizer, True


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    lr: float | None = None,
    weight_decay: float | None = None,
    pred_depth_scale_lr: float | None = None,
    optimizer_type: str | None = None,
) -> torch.optim.Optimizer:
    lr = args.lr if lr is None else float(lr)
    weight_decay = args.weight_decay if weight_decay is None else float(weight_decay)
    optimizer_type = args.optimizer_type if optimizer_type is None else str(optimizer_type)
    scale_params = []
    regular_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("pred_depth_log_scale"):
            scale_params.append(param)
        else:
            regular_params.append(param)
    param_groups = []
    if regular_params:
        param_groups.append(
            {"params": regular_params, "lr": lr, "weight_decay": weight_decay, "group_name": "regular"}
        )
    if scale_params:
        scale_lr = pred_depth_scale_lr if pred_depth_scale_lr is not None else lr
        param_groups.append(
            {"params": scale_params, "lr": scale_lr, "weight_decay": 0.0, "group_name": "scale"}
        )
    if not param_groups:
        raise ValueError("No trainable parameters for optimizer.")
    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups)
    if optimizer_type == "adafactor":
        return torch.optim.Adafactor(param_groups, foreach=False)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def capture_default_sampler_args(args: argparse.Namespace) -> None:
    args.default_window_size = int(args.window_size)
    args.default_patch_size = int(args.patch_size)
    args.default_num_yaw = int(args.num_yaw)
    args.default_pitch_degrees = str(args.pitch_degrees)
    args.default_fov_degrees = float(args.fov_degrees)
    args.default_dense_head_frames_chunk_size = int(args.dense_head_frames_chunk_size)


LOSS_STAGE_OVERRIDE_KEYS = (
    "camera_loss_weight",
    "camera_translation_weight",
    "camera_rotation_weight",
    "camera_fov_weight",
    "pano_translation_consistency_weight",
    "camera_translation_normalization",
    "camera_translation_normalization_eps",
    "global_point_loss_weight",
)


def capture_default_loss_args(args: argparse.Namespace) -> None:
    args.default_loss_overrides = {key: getattr(args, key) for key in LOSS_STAGE_OVERRIDE_KEYS}


def apply_stage_loss_overrides(args: argparse.Namespace, stage: Dict[str, Any] | None) -> None:
    defaults = getattr(args, "default_loss_overrides", {})
    for key in LOSS_STAGE_OVERRIDE_KEYS:
        if key in defaults:
            setattr(args, key, stage.get(key, defaults[key]) if stage else defaults[key])
    if args.camera_supervision_mode == "none":
        args.camera_loss_weight = 0.0
        args.camera_translation_weight = 0.0
        args.camera_rotation_weight = 0.0
        args.camera_fov_weight = 0.0


def apply_stage_sampler_overrides(
    model: torch.nn.Module,
    args: argparse.Namespace,
    stage: Dict[str, Any] | None,
) -> Dict[str, int | float | str]:
    window_size = int(stage.get("window_size", args.default_window_size)) if stage else int(args.default_window_size)
    patch_size = int(stage.get("patch_size", args.default_patch_size)) if stage else int(args.default_patch_size)
    num_yaw = int(stage.get("num_yaw", args.default_num_yaw)) if stage else int(args.default_num_yaw)
    pitch_degrees = str(stage.get("pitch_degrees", args.default_pitch_degrees)) if stage else str(args.default_pitch_degrees)
    fov_degrees = float(stage.get("fov_degrees", args.default_fov_degrees)) if stage else float(args.default_fov_degrees)

    if window_size <= 0:
        raise ValueError(f"stage window_size must be positive, got {window_size}")
    if patch_size <= 0:
        raise ValueError(f"stage patch_size must be positive, got {patch_size}")
    if window_size % patch_size != 0:
        raise ValueError(f"stage window_size={window_size} must be divisible by patch_size={patch_size}")

    base_model = unwrap_model(model)
    if isinstance(base_model, DepthPredictionAdapter):
        base_model = base_model.model
    dense_chunk = (
        stage.get("dense_head_frames_chunk_size", args.default_dense_head_frames_chunk_size)
        if stage
        else args.default_dense_head_frames_chunk_size
    )
    set_dense_head_frames_chunk_size(base_model, args, dense_chunk)
    aggregator = getattr(base_model, "aggregator", None)
    model_patch_size = int(getattr(aggregator, "patch_size", patch_size))
    if patch_size != model_patch_size:
        raise ValueError(
            f"Changing patch_size inside training is not supported: "
            f"stage patch_size={patch_size}, model patch_size={model_patch_size}."
        )

    sampler = getattr(base_model, "pano_sampler", None)
    if sampler is None:
        return current_sampler_status(model, args)

    sampler.window_size = window_size
    sampler.patch_size = patch_size
    sampler.fov_radians = math.radians(fov_degrees)
    yaw, pitch = make_default_view_grid(num_yaw=num_yaw, pitch_degrees=parse_pitch_degrees(pitch_degrees))
    device = sampler.default_yaw.device if torch.is_tensor(getattr(sampler, "default_yaw", None)) else None
    sampler.default_yaw = yaw.to(device=device) if device is not None else yaw
    sampler.default_pitch = pitch.to(device=device) if device is not None else pitch

    args.window_size = window_size
    args.patch_size = patch_size
    args.num_yaw = num_yaw
    args.pitch_degrees = pitch_degrees
    args.fov_degrees = fov_degrees
    return current_sampler_status(model, args)


def normalize_dense_head_frames_chunk_size(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def set_dense_head_frames_chunk_size(
    model: torch.nn.Module,
    args: argparse.Namespace,
    value: int | None,
) -> int | None:
    dense_chunk = normalize_dense_head_frames_chunk_size(value)
    base_model = unwrap_model(model)
    if isinstance(base_model, DepthPredictionAdapter):
        base_model = base_model.model
    if hasattr(base_model, "dense_head_frames_chunk_size"):
        base_model.dense_head_frames_chunk_size = dense_chunk
    args.dense_head_frames_chunk_size = 0 if dense_chunk is None else int(dense_chunk)
    return dense_chunk


def current_sampler_status(model: torch.nn.Module, args: argparse.Namespace) -> Dict[str, int | float | str]:
    base_model = unwrap_model(model)
    if isinstance(base_model, DepthPredictionAdapter):
        base_model = base_model.model
    sampler = getattr(base_model, "pano_sampler", None)
    if sampler is None:
        return {
            "window_size": int(args.window_size),
            "patch_size": int(args.patch_size),
            "num_yaw": int(args.num_yaw),
            "pitch_degrees": str(args.pitch_degrees),
            "fov_degrees": float(args.fov_degrees),
        }
    return {
        "window_size": int(getattr(sampler, "window_size", args.window_size)),
        "patch_size": int(getattr(sampler, "patch_size", args.patch_size)),
        "num_yaw": int(getattr(sampler, "default_yaw", torch.empty(0)).numel()),
        "pitch_degrees": str(args.pitch_degrees),
        "fov_degrees": float(math.degrees(float(getattr(sampler, "fov_radians", math.radians(args.fov_degrees))))),
    }


def normalize_pano_sampling_args(args: argparse.Namespace) -> None:
    legacy_count = getattr(args, "panos_per_sample", None)
    if legacy_count is not None:
        if legacy_count < 1:
            raise ValueError(f"panos_per_sample must be >= 1, got {legacy_count}")
        args.pano_sample_mode = "single" if legacy_count == 1 else "fixed_neighborhood"
        args.pano_min_count = legacy_count
        args.pano_max_count = legacy_count
        print(
            "[WARN] --panos-per-sample is deprecated; use "
            "--pano-sample-mode/--pano-min-count/--pano-max-count instead."
        )
    if args.pano_sample_mode == "single":
        args.pano_min_count = 1
        args.pano_max_count = 1
    if args.pano_min_count < 1 or args.pano_max_count < args.pano_min_count:
        raise ValueError(f"Invalid pano count range: min={args.pano_min_count}, max={args.pano_max_count}")
    if args.pano_sample_mode == "variable_neighborhood" and args.pano_min_count < 2:
        raise ValueError("variable_neighborhood requires --pano-min-count >= 2.")


def normalize_camera_supervision_args(args: argparse.Namespace) -> None:
    if args.camera_supervision_mode == "auto":
        args.camera_supervision_mode = "none" if args.pano_sample_mode == "single" else "pano_relative"
    if args.camera_supervision_mode == "none":
        args.camera_loss_weight = 0.0
        args.camera_translation_weight = 0.0
        args.camera_rotation_weight = 0.0
        args.camera_fov_weight = 0.0
    if args.camera_supervision_mode == "pano_relative" and args.pano_sample_mode == "single":
        raise ValueError("pano_relative camera supervision requires at least two panos per sample.")


def normalize_pred_depth_scale_args(args: argparse.Namespace) -> None:
    if args.dataset_format == "panocity_paired" and float(args.pred_depth_scale) == 1.0:
        args.pred_depth_scale = DEFAULT_PANOCITY_PRED_DEPTH_SCALE
        print(f"[INFO] using PanoCity pred_depth_scale = {args.pred_depth_scale}")


def normalize_training_stages(raw_stages) -> list[Dict[str, Any]]:
    if raw_stages in (None, "", False):
        return []
    if isinstance(raw_stages, str):
        parsed = yaml.safe_load(raw_stages)
    else:
        parsed = raw_stages
    if parsed in (None, "", False):
        return []
    if not isinstance(parsed, list):
        raise ValueError("training_stages must be a list of stage mappings.")

    stages: list[Dict[str, Any]] = []
    previous_until_minutes = -math.inf
    previous_until_steps = -math.inf
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"training_stages[{idx}] must be a mapping, got {type(item).__name__}.")
        stage = dict(item)
        stage.setdefault("name", f"stage{idx + 1}")
        if "until_minutes" in stage and stage["until_minutes"] is not None:
            stage["until_minutes"] = float(stage["until_minutes"])
            if stage["until_minutes"] <= previous_until_minutes:
                raise ValueError("training_stages until_minutes values must be strictly increasing.")
            previous_until_minutes = stage["until_minutes"]
        if "until_steps" in stage and stage["until_steps"] is not None:
            stage["until_steps"] = int(stage["until_steps"])
            if stage["until_steps"] <= previous_until_steps:
                raise ValueError("training_stages until_steps values must be strictly increasing.")
            previous_until_steps = stage["until_steps"]
        stages.append(stage)
    return stages


def select_training_stage(
    stages: Sequence[Dict[str, Any]],
    next_step: int,
    elapsed_minutes: float,
) -> tuple[int, Dict[str, Any]]:
    if not stages:
        raise ValueError("select_training_stage requires at least one stage.")
    for idx, stage in enumerate(stages):
        until_minutes = stage.get("until_minutes")
        until_steps = stage.get("until_steps")
        within_minutes = until_minutes is None or elapsed_minutes < float(until_minutes)
        within_steps = until_steps is None or next_step <= int(until_steps)
        if within_minutes and within_steps:
            return idx, stage
    return len(stages) - 1, stages[-1]


def format_training_stages(stages: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for idx, stage in enumerate(stages):
        boundary = []
        if stage.get("until_minutes") is not None:
            boundary.append(f"until={float(stage['until_minutes']):.1f}m")
        if stage.get("until_steps") is not None:
            boundary.append(f"until_step={int(stage['until_steps'])}")
        if stage.get("window_size") is not None:
            boundary.append(f"window={int(stage['window_size'])}")
        if stage.get("dense_head_frames_chunk_size") is not None:
            boundary.append(f"dense_chunk={int(stage['dense_head_frames_chunk_size'])}")
        if stage.get("pano_max_count") is not None:
            boundary.append(f"pano_max={int(stage['pano_max_count'])}")
        parts.append(f"{idx + 1}:{stage.get('name', f'stage{idx + 1}')}({','.join(boundary) or 'final'})")
    return "; ".join(parts)


def format_stage_status(
    stage_index: int | None,
    stage: Dict[str, Any] | None,
    trainable_count: int,
    frozen_count: int,
) -> str:
    if stage is None:
        return f"default trainable={trainable_count:,} frozen={frozen_count:,}"
    return (
        f"{stage_index + 1}:{stage.get('name', f'stage{stage_index + 1}')} "
        f"trainable_mode={stage.get('trainable', 'default')} "
        f"lr={stage.get('lr', 'default')} "
        f"window={stage.get('window_size', 'default')} "
        f"dense_chunk={stage.get('dense_head_frames_chunk_size', 'default')} "
        f"pano_max={stage.get('pano_max_count', 'default')} "
        f"optimizer={stage.get('optimizer_type', 'default')} "
        f"camera_weight={stage.get('camera_loss_weight', 'default')} "
        f"luna_forward={stage.get('enable_luna_forward', 'default')} "
        f"depth_residual={stage.get('enable_depth_residual', 'default')} "
        f"trainable={trainable_count:,} frozen={frozen_count:,}"
    )


def limit_batch_panos(batch: Dict[str, Any], stage: Dict[str, Any] | None) -> Dict[str, Any]:
    """Apply a stage pano curriculum after collation so worker copies cannot go stale."""
    if stage is None or stage.get("pano_max_count") is None:
        return batch
    pano_image = batch.get("pano_image")
    if not torch.is_tensor(pano_image) or pano_image.ndim != 5:
        return batch
    current_count = int(pano_image.shape[1])
    max_count = max(1, int(stage["pano_max_count"]))
    if current_count <= max_count:
        return batch
    limited: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[1]) == current_count:
            limited[key] = value[:, :max_count]
        else:
            limited[key] = value
    return limited


def configure_trainable_for_stage(
    model: torch.nn.Module,
    args: argparse.Namespace,
    stage: Dict[str, Any],
) -> Tuple[int, int]:
    unwrapped = unwrap_model(model)
    adapter = unwrapped if isinstance(unwrapped, DepthPredictionAdapter) else None
    base_model = adapter.model if adapter is not None else unwrapped

    trainable_mode = str(stage.get("trainable", args.trainable))
    configure_trainable(base_model, trainable_mode)
    if args.camera_supervision_mode == "pano_relative":
        window_camera_head = getattr(base_model, "camera_head", None)
        if window_camera_head is not None:
            for param in window_camera_head.parameters():
                param.requires_grad_(False)

    enable_luna_forward = bool(stage.get("enable_luna_forward", True))
    set_luna_forward_enabled(base_model, enable_luna_forward)
    if not enable_luna_forward or bool(stage.get("freeze_luna_parameters", False)):
        set_luna_parameters_trainable(base_model, False)

    if adapter is not None:
        enable_depth_residual = bool(stage.get("enable_depth_residual", adapter.depth_residual_head is not None))
        adapter.depth_residual_enabled = enable_depth_residual
        train_depth_residual = bool(stage.get("train_depth_residual", enable_depth_residual))
        train_scale = bool(stage.get("learn_pred_depth_scale", args.learn_pred_depth_scale))

        if isinstance(adapter.pred_depth_log_scale, torch.nn.Parameter):
            adapter.pred_depth_log_scale.requires_grad_(train_scale)
        if adapter.depth_residual_head is not None:
            for param in adapter.depth_residual_head.parameters():
                param.requires_grad_(enable_depth_residual and train_depth_residual)

    trainable = sum(param.numel() for param in unwrapped.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in unwrapped.parameters() if not param.requires_grad)
    if trainable == 0:
        raise ValueError(f"No trainable parameters selected for stage {stage.get('name')!r}.")
    return trainable, frozen


def set_luna_forward_enabled(model: torch.nn.Module, enabled: bool) -> None:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None and hasattr(aggregator, "enable_luna"):
        aggregator.enable_luna = bool(enabled)


def set_luna_parameters_trainable(model: torch.nn.Module, enabled: bool) -> None:
    for name, param in model.named_parameters():
        if is_luna_parameter_name(name):
            param.requires_grad_(enabled)


def is_luna_parameter_name(name: str) -> bool:
    return "luna_" in name or "pano_global" in name or "pano_geometry" in name


def build_model(args: argparse.Namespace) -> VGGTOmega_LUNA:
    pitch_degrees = parse_pitch_degrees(args.pitch_degrees)
    sampler = {
        "window_size": args.window_size,
        "patch_size": args.patch_size,
        "fov_degrees": args.fov_degrees,
        "num_yaw": args.num_yaw,
        "pitch_degrees": pitch_degrees,
    }

    if args.smoke:
        embed_dim = 64
        model = VGGTOmega_LUNA(
            patch_size=args.patch_size,
            embed_dim=embed_dim,
            enable_camera=True,
            enable_depth=True,
            enable_alignment=False,
            enable_pano_global_token=args.enable_pano_global_token,
            enable_pano_geometry_residual=args.enable_pano_geometry_residual,
            enable_luna=True,
            luna_patch_layers=args.luna_patch_layers,
            luna_camera_layers=args.luna_camera_layers,
            aggregator_use_checkpoint=args.aggregator_use_checkpoint,
            sampler=sampler,
            dense_head_frames_chunk_size=normalize_dense_head_frames_chunk_size(args.dense_head_frames_chunk_size),
            dense_head_use_checkpoint=args.dense_head_use_checkpoint,
            dense_head_return_confidence=args.dense_head_return_confidence,
            aggregator_kwargs={
                "depth": 24,
                "num_heads": 4,
                "num_register_tokens": 1,
                "register_attention_block_indices": (),
                "cached_layer_indices": (4, 11, 17, 23),
            },
            checkpoint_path=None,
        )
        model.aggregator.patch_embed = PatchEmbed(
            img_size=args.window_size,
            patch_size=args.patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        model.dense_head = DenseHead(
            dim_in=2 * embed_dim,
            patch_size=args.patch_size,
            features=16,
            out_channels=[16, 32, 64, 64],
        )
        return model

    return VGGTOmega_LUNA(
        patch_size=args.patch_size,
        embed_dim=1024,
        enable_camera=args.enable_camera_head,
        enable_depth=True,
        enable_alignment=False,
        enable_pano_global_token=args.enable_pano_global_token,
        enable_pano_geometry_residual=args.enable_pano_geometry_residual,
        enable_luna=True,
        luna_patch_layers=args.luna_patch_layers,
        luna_camera_layers=args.luna_camera_layers,
        aggregator_use_checkpoint=args.aggregator_use_checkpoint,
        sampler=sampler,
        dense_head_frames_chunk_size=normalize_dense_head_frames_chunk_size(args.dense_head_frames_chunk_size),
        dense_head_use_checkpoint=args.dense_head_use_checkpoint,
        dense_head_return_confidence=args.dense_head_return_confidence,
        checkpoint_path=None,
    )


def train_step(
    model: VGGTOmega_LUNA,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    pano_images = batch["pano_image"]
    pano_depths = batch["pano_depth"]
    sampler_model = unwrap_model(model)

    target_depth, target_valid = sample_depth_targets(
        sampler_model,
        pano_depths,
        source_depth_semantics=args.gt_depth_semantics,
        max_range_depth=args.depth_max_m,
    )
    target_valid_float = target_valid.to(dtype=torch.float32)
    depth_valid_ratio = target_valid_float.mean()
    depth_window_keep_ratio = (
        target_valid_float.mean(dim=tuple(range(2, target_valid_float.ndim)))
        >= max(float(args.min_window_valid_ratio), 0.0)
    ).to(dtype=torch.float32).mean()
    amp_enabled = pano_images.device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float32
    with torch.autocast(device_type=pano_images.device.type, dtype=amp_dtype, enabled=amp_enabled):
        predictions = model(
            pano_images=pano_images,
            return_sampler_output=True,
            return_window_pose=args.camera_supervision_mode != "pano_relative",
        )
        pred_depth_scale = predictions.get(
            "_pred_depth_scale",
            predictions["depth"].new_tensor(float(args.pred_depth_scale)),
        )
        base_depth_scale = pred_depth_scale.detach() if args.depth_scale_alignment != "none" else pred_depth_scale
        pred_depth_base = predictions["depth"] * base_depth_scale
        sample_depth_scale = estimate_sample_depth_alignment_scale(
            pred_depth_base,
            target_depth,
            target_valid,
            mode=args.depth_scale_alignment,
            min_scale=args.depth_scale_alignment_min,
            max_scale=args.depth_scale_alignment_max,
            eps=args.depth_scale_alignment_eps,
        )
        pred_depth = pred_depth_base * expand_sample_scale_like(sample_depth_scale, pred_depth_base)
        depth_loss_valid, depth_loss_window_keep_ratio, pred_depth_finite_ratio = depth_loss_validity_stats(
            pred_depth,
            target_depth,
            target_valid,
            min_window_valid_ratio=args.min_window_valid_ratio,
        )
        loss_depth = masked_depth_loss(
            pred_depth,
            target_depth,
            target_valid,
            mode=args.depth_loss_mode,
            huber_delta=args.depth_log_huber_delta,
            error_clip=args.depth_log_error_clip,
            sample_weight=batch.get("sample_weight") if args.loss_sample_weighting else None,
            min_window_valid_ratio=args.min_window_valid_ratio,
            valid_ratio_power=args.valid_ratio_loss_power,
            sample_weight_min=args.sample_weight_min,
            sample_weight_max=args.sample_weight_max,
        )
        loss_depth_unfiltered = masked_depth_loss(
            pred_depth,
            target_depth,
            target_valid,
            mode=args.depth_loss_mode,
            huber_delta=args.depth_log_huber_delta,
            error_clip=args.depth_log_error_clip,
            sample_weight=batch.get("sample_weight") if args.loss_sample_weighting else None,
            min_window_valid_ratio=0.0,
            valid_ratio_power=0.0,
            sample_weight_min=args.sample_weight_min,
            sample_weight_max=args.sample_weight_max,
        )
        loss_overlap = adjacent_edge_overlap_loss(
            pred_depth,
            target_valid,
            sample_weight=batch.get("sample_weight") if args.loss_sample_weighting else None,
            band_fraction=args.overlap_band_fraction,
        )
        if args.debug_depth_dump and _is_rank0_process():
            dump_debug_depth_predictions(
                args=args,
                pred_depth=pred_depth,
                target_depth=target_depth,
                valid_mask=target_valid,
                global_step=global_step,
                batch=batch,
            )
        loss_camera_dict = camera_alignment_loss(
            predictions=predictions,
            batch=batch,
            translation_weight=args.camera_translation_weight,
            rotation_weight=args.camera_rotation_weight,
            fov_weight=args.camera_fov_weight,
            position_mode=args.camera_position_mode,
            supervision_mode=args.camera_supervision_mode,
            pano_consistency_weight=args.pano_translation_consistency_weight,
            translation_normalization=args.camera_translation_normalization,
            translation_normalization_eps=args.camera_translation_normalization_eps,
            pred_translation_scale=(
                (base_depth_scale.detach() * sample_depth_scale)
                if args.camera_depth_scale_alignment and args.depth_scale_alignment != "none"
                else None
            ),
        )
        loss_camera = loss_camera_dict["loss_camera"]
        if float(args.global_point_loss_weight) > 0:
            global_point_dict = shared_frame_point_loss(
                pred_depth=pred_depth,
                target_depth=target_depth,
                valid_mask=target_valid,
                predictions=predictions,
                batch=batch,
                pred_translation_scale=(
                    (base_depth_scale.detach() * sample_depth_scale)
                    if args.camera_depth_scale_alignment and args.depth_scale_alignment != "none"
                    else None
                ),
                stride=args.global_point_stride,
            )
        else:
            zero_global = pred_depth.sum() * 0.0
            global_point_dict = {
                "loss_global_point": zero_global,
                "global_point_valid_ratio": zero_global.detach(),
                "global_point_finite_ratio": zero_global.detach(),
                "global_point_geometry_finite_ratio": zero_global.detach(),
            }
        loss_global_point = global_point_dict["loss_global_point"]
        loss = (
            loss_depth
            + float(args.overlap_consistency_weight) * loss_overlap
            + float(args.global_point_loss_weight) * loss_global_point
            + args.camera_loss_weight * loss_camera
        )
        loss_comparable = (
            loss_depth
            + float(args.overlap_consistency_weight) * loss_overlap
            + float(args.global_point_loss_weight) * loss_global_point
            + float(args.camera_comparable_weight) * loss_camera
        )
    loss.backward()

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=args.grad_clip,
        )
    optimizer.step()
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, DepthPredictionAdapter):
        unwrapped_model.sanitize_parameters()
        logged_pred_depth_scale = unwrapped_model.pred_depth_scale().detach()
    else:
        logged_pred_depth_scale = pred_depth_scale.detach()
    return {
        "loss": loss.detach(),
        "loss_comparable": loss_comparable.detach(),
        "loss_depth": loss_depth.detach(),
        "loss_overlap": loss_overlap.detach(),
        "loss_global_point": loss_global_point.detach(),
        "loss_camera": loss_camera.detach(),
        "loss_overlap_weighted": (float(args.overlap_consistency_weight) * loss_overlap).detach(),
        "loss_global_point_weighted": (
            float(args.global_point_loss_weight) * loss_global_point
        ).detach(),
        "loss_camera_weighted": (float(args.camera_loss_weight) * loss_camera).detach(),
        "pano_count": loss_depth.new_tensor(float(pano_images.shape[1] if pano_images.ndim == 5 else 1)),
        "depth_valid_ratio": depth_valid_ratio.detach(),
        "depth_window_keep_ratio": depth_window_keep_ratio.detach(),
        "depth_loss_valid_ratio": depth_loss_valid.detach(),
        "depth_loss_window_keep_ratio": depth_loss_window_keep_ratio.detach(),
        "pred_depth_finite_ratio": pred_depth_finite_ratio.detach(),
        "loss_depth_unfiltered": loss_depth_unfiltered.detach(),
        "depth_sample_scale_median": sample_depth_scale.detach().float().median(),
        "depth_sample_scale_min": sample_depth_scale.detach().float().min(),
        "depth_sample_scale_max": sample_depth_scale.detach().float().max(),
        "pred_depth_scale": logged_pred_depth_scale,
        **{key: value.detach() for key, value in global_point_dict.items() if key != "loss_global_point"},
        **{key: value.detach() for key, value in loss_camera_dict.items() if key != "loss_camera"},
    }


def _is_rank0_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def print_cuda_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    print(
        f"{prefix} cuda_memory allocated={allocated:.2f}GB reserved={reserved:.2f}GB "
        f"peak_allocated={peak_allocated:.2f}GB peak_reserved={peak_reserved:.2f}GB",
        flush=True,
    )


@torch.no_grad()
def dump_debug_depth_predictions(
    args: argparse.Namespace,
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    global_step: int,
    batch: Dict,
) -> None:
    per_window_loss = per_window_log_l1_depth(pred_depth, target_depth, valid_mask)
    bad_windows = torch.nonzero(per_window_loss > float(args.debug_depth_threshold), as_tuple=False)
    if bad_windows.numel() == 0:
        return

    max_dumps = max(0, int(args.debug_depth_max_dumps_per_step))
    if max_dumps == 0:
        return
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    scene_names = batch.get("scene_name", None)
    dumped = 0
    for batch_idx, view_idx in bad_windows.detach().cpu().tolist():
        if dumped >= max_dumps:
            break
        loss_value = float(per_window_loss[batch_idx, view_idx].detach().cpu())
        scene = _debug_scene_name(scene_names, batch_idx)
        prefix = debug_dir / f"step_{global_step:06d}_b{batch_idx:02d}_v{view_idx:02d}_{scene}_loss_{loss_value:.4f}"
        valid = valid_mask[batch_idx, view_idx, ..., 0].detach().cpu().numpy().astype(bool)
        pred = pred_depth[batch_idx, view_idx, ..., 0].detach().float().cpu().numpy()
        target = target_depth[batch_idx, view_idx, ..., 0].detach().float().cpu().numpy()
        save_depth_debug_pngs(prefix, pred, target, valid, max_depth_m=float(args.depth_max_m))
        dumped += 1


def _debug_scene_name(scene_names, batch_idx: int) -> str:
    if scene_names is None:
        return "sample"
    if isinstance(scene_names, (list, tuple)) and len(scene_names) > batch_idx:
        value = scene_names[batch_idx]
    else:
        value = scene_names
    if isinstance(value, (list, tuple)):
        value = "_".join(str(part) for part in value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))[:80]


def save_depth_debug_pngs(prefix: Path, pred: np.ndarray, target: np.ndarray, valid: np.ndarray, max_depth_m: float) -> None:
    valid = valid & np.isfinite(pred) & np.isfinite(target) & (target > 0)
    pred_to_save = np.where(valid, pred, 0.0)
    target_to_save = np.where(valid, target, 0.0)
    pred_u16 = np.clip(pred_to_save * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    target_u16 = np.clip(target_to_save * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(pred_u16).save(prefix.with_name(prefix.name + "_pred_depth_m.png"))
    Image.fromarray(target_u16).save(prefix.with_name(prefix.name + "_gt_depth_m.png"))
    Image.fromarray((valid.astype(np.uint8) * 255)).save(prefix.with_name(prefix.name + "_valid_mask.png"))

    denom = max(max_depth_m, 1e-6)
    pred_vis = (np.clip(pred_to_save / denom, 0.0, 1.0) * 255.0).astype(np.uint8)
    target_vis = (np.clip(target_to_save / denom, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pred_vis).save(prefix.with_name(prefix.name + "_pred_vis.png"))
    Image.fromarray(target_vis).save(prefix.with_name(prefix.name + "_gt_vis.png"))

    err = np.zeros_like(target_to_save, dtype=np.float32)
    err[valid] = np.abs(np.log(np.clip(pred_to_save[valid], 1e-4, None)) - np.log(np.clip(target_to_save[valid], 1e-4, None)))
    err_vis = (np.clip(err / 1.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(err_vis).save(prefix.with_name(prefix.name + "_abs_log_err.png"))


def per_window_log_l1_depth(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    pred_depth = pred_depth.float()
    target_depth = target_depth.to(device=pred_depth.device, dtype=torch.float32)
    valid = torch.isfinite(pred_depth) & torch.isfinite(target_depth) & (target_depth > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=target_depth.device, dtype=torch.bool)
    diff = torch.zeros_like(pred_depth, dtype=torch.float32)
    diff[valid] = (
        torch.log(pred_depth[valid].clamp_min(1e-4))
        - torch.log(target_depth[valid].clamp_min(1e-4))
    ).abs()
    numerator = diff.sum(dim=(2, 3, 4))
    denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1)
    return numerator / denominator


@torch.no_grad()
def sample_depth_windows(
    model: VGGTOmega_LUNA,
    pano_depths: torch.Tensor,
    source_depth_semantics: str = "range",
    max_range_depth: float | None = None,
) -> torch.Tensor:
    target_depth, _ = sample_depth_targets(model, pano_depths, source_depth_semantics, max_range_depth)
    return target_depth


@torch.no_grad()
def sample_depth_targets(
    model: VGGTOmega_LUNA,
    pano_depths: torch.Tensor,
    source_depth_semantics: str = "range",
    max_range_depth: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode source ERP depth to radial range, sample it, and convert to window Z-depth."""
    source_valid_bool = torch.isfinite(pano_depths) & (pano_depths > 0)
    source_valid = source_valid_bool.to(dtype=pano_depths.dtype)
    range_depth_erp = erp_depth_to_range_depth(pano_depths, source_depth_semantics)
    source_valid_bool &= torch.isfinite(range_depth_erp) & (range_depth_erp > 0)
    if max_range_depth is not None and max_range_depth > 0:
        source_valid_bool &= range_depth_erp <= max_range_depth
    source_valid = source_valid_bool.to(dtype=pano_depths.dtype)
    source_depth = torch.where(source_valid_bool, range_depth_erp, torch.zeros_like(range_depth_erp))
    channel_dim = 1 if source_depth.ndim == 4 else 2
    packed_depth_valid = torch.cat([source_depth, source_valid, source_valid.new_zeros(source_valid.shape)], dim=channel_dim)
    sampled = sample_pano_tensor(model, packed_depth_valid, interpolation_mode="bilinear")
    sampled_depth = sampled.windows[:, :, :1]
    sampled_weight = sampled.windows[:, :, 1:2]
    range_depth = (sampled_depth / sampled_weight.clamp_min(1e-6)).permute(0, 1, 3, 4, 2).contiguous()

    valid_as_rgb = source_valid.repeat(1, 3, 1, 1) if source_valid.ndim == 4 else source_valid.repeat(1, 1, 3, 1, 1)
    sampled_valid = sample_pano_tensor(model, valid_as_rgb, interpolation_mode="nearest").windows[:, :, :1]
    valid = sampled_valid.permute(0, 1, 3, 4, 2).contiguous() > 0.5
    valid = valid & (sampled_weight.permute(0, 1, 3, 4, 2).contiguous() > 1e-6)

    z_factor = build_window_z_factor(sampled.camera_meta, range_depth.shape[2], range_depth.shape[3])
    target_z = range_depth * z_factor[..., None]
    valid = valid & torch.isfinite(target_z) & (target_z > 0)
    target_z = torch.where(valid, target_z, torch.zeros_like(target_z))
    return target_z, valid


def sample_pano_tensor(
    model: VGGTOmega_LUNA,
    tensor: torch.Tensor,
    interpolation_mode: str,
) -> SimpleNamespace:
    if hasattr(model, "sample_pano_windows"):
        return model.sample_pano_windows(tensor, interpolation_mode=interpolation_mode)
    return model.pano_sampler(tensor, interpolation_mode=interpolation_mode)


def erp_depth_to_range_depth(pano_depths: torch.Tensor, semantics: str) -> torch.Tensor:
    if semantics == "range":
        return pano_depths
    if semantics not in {"cubemap_z", "double_cubemap_z"}:
        raise ValueError(f"Unknown ERP depth semantics: {semantics}")
    height, width = pano_depths.shape[-2:]
    rays = merged_cubemap_source_rays_torch(height, width, pano_depths.device, pano_depths.dtype)
    face_z_factor = cube_projection_factor_torch(rays)
    if semantics == "double_cubemap_z":
        back_rays = rotate_source_rays_about_vertical_torch(rays, -math.pi / 4.0)
        back_factor = cube_projection_factor_torch(back_rays)
        front_weight = cube_edge_weight_torch(rays)
        back_weight = cube_edge_weight_torch(back_rays)
        face_z_factor = (front_weight * face_z_factor + back_weight * back_factor) / (
            front_weight + back_weight
        ).clamp_min(torch.finfo(pano_depths.dtype).eps)
    return pano_depths / face_z_factor[None, None]


def merged_cubemap_source_rays_torch(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    vv, uu = torch.meshgrid(y, x, indexing="ij")
    longitude = (uu * 2.0 - 1.0) * math.pi
    latitude = (0.5 - vv) * math.pi
    cos_lat = torch.cos(latitude)
    return torch.stack(
        [cos_lat * torch.cos(longitude), cos_lat * torch.sin(longitude), torch.sin(latitude)],
        dim=-1,
    )


def rotate_source_rays_about_vertical_torch(rays: torch.Tensor, angle_rad: float) -> torch.Tensor:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return torch.stack(
        [
            cosine * rays[..., 0] - sine * rays[..., 1],
            sine * rays[..., 0] + cosine * rays[..., 1],
            rays[..., 2],
        ],
        dim=-1,
    )


def cube_projection_factor_torch(rays: torch.Tensor) -> torch.Tensor:
    return rays.abs().amax(dim=-1).clamp_min(torch.finfo(rays.dtype).eps)


def cube_edge_weight_torch(rays: torch.Tensor) -> torch.Tensor:
    sorted_abs = rays.abs().sort(dim=-1).values
    factor = sorted_abs[..., 2].clamp_min(torch.finfo(rays.dtype).eps)
    return (1.0 - sorted_abs[..., 1] / factor + 1e-6).clamp_min(0.0)


def build_window_z_factor(camera_meta: Dict[str, torch.Tensor], height: int, width: int) -> torch.Tensor:
    """Return cos(angle-to-optical-axis) for each virtual pinhole pixel."""
    yaw = camera_meta["yaw"].reshape(-1)
    pitch = camera_meta["pitch"].reshape(-1)
    fov_x = camera_meta["fov_x"].reshape(-1)
    fov_y = camera_meta["fov_y"].reshape(-1)
    rays = pinhole_rays(
        yaw,
        pitch,
        fov_x,
        fov_y,
        height,
        width,
        device=yaw.device,
        dtype=yaw.dtype,
    )
    forward = camera_meta["rotations"][..., :, 2].reshape(-1, 3)
    z_factor = (rays * forward[:, None, None, :]).sum(dim=-1).clamp_min(0.0)
    return z_factor.reshape(*camera_meta["yaw"].shape, height, width)


def create_tensorboard_writer(path: Path, dist_state: Dict[str, Any], enabled: bool = True):
    if not enabled or not is_main_process(dist_state):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[WARN] tensorboard is not installed; skip TensorBoard scalar logging.")
        return None
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(path))
    print(f"[INFO] tensorboard_dir = {path}")
    return writer


def write_tensorboard_metrics(writer, metrics: Dict[str, Any]) -> None:
    if writer is None:
        return
    step = int(metrics["step"])
    scalar_map = {
        "loss/total": "loss",
        "loss/comparable": "loss_comparable",
        "loss/depth": "loss_depth",
        "loss/overlap": "loss_overlap",
        "loss/overlap_weighted": "loss_overlap_weighted",
        "loss/global_point": "loss_global_point",
        "loss/global_point_weighted": "loss_global_point_weighted",
        "loss/camera": "loss_camera",
        "loss/camera_weighted": "loss_camera_weighted",
        "loss/camera_t": "loss_camera_t",
        "loss/camera_r": "loss_camera_r",
        "loss/camera_fov": "loss_camera_fov",
        "loss/camera_consistency": "loss_camera_consistency",
        "loss/depth_unfiltered": "loss_depth_unfiltered",
        "data/depth_valid_ratio": "depth_valid_ratio",
        "data/depth_window_keep_ratio": "depth_window_keep_ratio",
        "data/depth_loss_valid_ratio": "depth_loss_valid_ratio",
        "data/depth_loss_window_keep_ratio": "depth_loss_window_keep_ratio",
        "data/pred_depth_finite_ratio": "pred_depth_finite_ratio",
        "data/depth_sample_scale_median": "depth_sample_scale_median",
        "data/depth_sample_scale_min": "depth_sample_scale_min",
        "data/depth_sample_scale_max": "depth_sample_scale_max",
        "camera/rotation_deg": "camera_rotation_deg",
        "camera/translation_deg": "camera_translation_deg",
        "camera/translation_valid_count": "camera_translation_valid_count",
        "camera/rotation_valid_count": "camera_rotation_valid_count",
        "data/global_point_valid_ratio": "global_point_valid_ratio",
        "data/global_point_finite_ratio": "global_point_finite_ratio",
        "data/global_point_geometry_finite_ratio": "global_point_geometry_finite_ratio",
        "data/pano_count": "pano_count",
        "train/lr": "lr",
        "train/pred_depth_scale": "pred_depth_scale",
        "train/elapsed_seconds": "elapsed_seconds",
        "train/stage_index": "stage_index",
        "train/window_size": "window_size",
    }
    for tag, key in scalar_map.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(tag, float(value), step)


def append_loss_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def parse_float_list(value: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        parsed = [float(item) for item in items]
    else:
        parsed = [float(item) for item in value]
    parsed = [item for item in parsed if math.isfinite(item)]
    return tuple(sorted(set(parsed))) or (5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0)


def strip_camera_diag_tensors(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in loss_dict.items() if not key.startswith("_camera_diag_")}


def gather_camera_diag_records(
    records: list[Dict[str, Any]],
    dist_state: Dict[str, int | bool],
) -> list[Dict[str, Any]]:
    if not dist_state["distributed"]:
        return records
    gathered: list[list[Dict[str, Any]] | None] = [None for _ in range(int(dist_state["world_size"]))]
    dist.all_gather_object(gathered, records)
    merged: list[Dict[str, Any]] = []
    for item in gathered:
        if item:
            merged.extend(item)
    return merged


def camera_diag_records_from_batch(
    loss_dict: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    global_step: int,
    stage_name: str,
    dist_state: Dict[str, int | bool],
) -> list[Dict[str, Any]]:
    rotation_deg = _camera_diag_2d(loss_dict.get("_camera_diag_rotation_deg"))
    translation_loss = _camera_diag_2d(loss_dict.get("_camera_diag_translation_loss"))
    translation_l2_m = _camera_diag_2d(loss_dict.get("_camera_diag_translation_l2_m"))
    translation_norm_l2 = _camera_diag_2d(loss_dict.get("_camera_diag_translation_norm_l2"))
    rotation_valid = _camera_diag_bool_2d(loss_dict.get("_camera_diag_rotation_valid"))
    translation_valid = _camera_diag_bool_2d(loss_dict.get("_camera_diag_translation_valid"))

    reference = next(
        (
            item
            for item in (rotation_deg, translation_loss, translation_l2_m, translation_norm_l2)
            if item is not None
        ),
        None,
    )
    if reference is None:
        return []
    batch_size = len(reference)
    pano_count = len(reference[0]) if batch_size > 0 else 0
    if batch_size <= 0 or pano_count <= 0:
        return []
    rotation_deg = _ensure_diag_shape(rotation_deg, batch_size, pano_count, fill=0.0)
    translation_loss = _ensure_diag_shape(translation_loss, batch_size, pano_count, fill=0.0)
    translation_l2_m = _ensure_diag_shape(translation_l2_m, batch_size, pano_count, fill=0.0)
    translation_norm_l2 = _ensure_diag_shape(translation_norm_l2, batch_size, pano_count, fill=0.0)
    rotation_valid = _ensure_bool_diag_shape(rotation_valid, batch_size, pano_count, fill=False)
    translation_valid = _ensure_bool_diag_shape(translation_valid, batch_size, pano_count, fill=False)

    sequence_names = normalize_collated_string_matrix(batch.get("sequence_name"), batch_size, pano_count)
    scene_names = normalize_collated_string_matrix(batch.get("scene_name"), batch_size, pano_count, split_pipe=True)
    rgb_paths = normalize_collated_string_matrix(batch.get("rgb_path"), batch_size, pano_count)

    sample_scale = _loss_scalar(loss_dict, "depth_sample_scale_median")
    records: list[Dict[str, Any]] = []
    for batch_index in range(batch_size):
        for pano_index in range(pano_count):
            t_valid = bool(translation_valid[batch_index][pano_index])
            r_valid = bool(rotation_valid[batch_index][pano_index])
            if not (t_valid or r_valid):
                continue
            sequence_name = sequence_names[batch_index][pano_index]
            scene_name = scene_names[batch_index][pano_index]
            dataset = canonical_dataset_name(sequence_name)
            if dataset == "unknown":
                dataset = canonical_dataset_name(scene_name)
            records.append(
                {
                    "step": int(global_step),
                    "stage": str(stage_name),
                    "rank": int(dist_state.get("rank", 0)),
                    "batch_index": int(batch_index),
                    "pano_index": int(pano_index),
                    "pano_count": int(pano_count),
                    "dataset": dataset,
                    "sequence_name": sequence_name,
                    "scene_name": scene_name,
                    "rgb_path": rgb_paths[batch_index][pano_index],
                    "rotation_valid": r_valid,
                    "translation_valid": t_valid,
                    "rotation_deg": _finite_or_none(rotation_deg[batch_index][pano_index]) if r_valid else None,
                    "translation_loss": _finite_or_none(translation_loss[batch_index][pano_index]) if t_valid else None,
                    "translation_l2_m": _finite_or_none(translation_l2_m[batch_index][pano_index]) if t_valid else None,
                    "translation_norm_l2": _finite_or_none(translation_norm_l2[batch_index][pano_index]) if t_valid else None,
                    "sample_scale": sample_scale,
                    "loss_camera": _loss_scalar(loss_dict, "loss_camera"),
                    "loss_camera_t": _loss_scalar(loss_dict, "loss_camera_t"),
                    "loss_camera_r": _loss_scalar(loss_dict, "loss_camera_r"),
                }
            )
    return records


def _camera_diag_2d(value: torch.Tensor | None) -> list[list[float]] | None:
    if value is None or not torch.is_tensor(value):
        return None
    item = value.detach().float().cpu()
    if item.ndim == 0:
        item = item.reshape(1, 1)
    elif item.ndim == 1:
        item = item.reshape(1, -1)
    elif item.ndim > 2:
        item = item.reshape(item.shape[0], item.shape[1], -1).mean(dim=-1)
    return item.tolist()


def _camera_diag_bool_2d(value: torch.Tensor | None) -> list[list[bool]] | None:
    if value is None or not torch.is_tensor(value):
        return None
    item = value.detach().bool().cpu()
    if item.ndim == 0:
        item = item.reshape(1, 1)
    elif item.ndim == 1:
        item = item.reshape(1, -1)
    elif item.ndim > 2:
        item = item.reshape(item.shape[0], item.shape[1], -1).any(dim=-1)
    return [[bool(cell) for cell in row] for row in item.tolist()]


def _ensure_diag_shape(
    value: list[list[float]] | None,
    batch_size: int,
    pano_count: int,
    fill: float,
) -> list[list[float]]:
    result = [[float(fill) for _ in range(pano_count)] for _ in range(batch_size)]
    if value is None:
        return result
    for batch_index in range(min(batch_size, len(value))):
        row = value[batch_index]
        for pano_index in range(min(pano_count, len(row))):
            result[batch_index][pano_index] = float(row[pano_index])
    return result


def _ensure_bool_diag_shape(
    value: list[list[bool]] | None,
    batch_size: int,
    pano_count: int,
    fill: bool,
) -> list[list[bool]]:
    result = [[bool(fill) for _ in range(pano_count)] for _ in range(batch_size)]
    if value is None:
        return result
    for batch_index in range(min(batch_size, len(value))):
        row = value[batch_index]
        for pano_index in range(min(pano_count, len(row))):
            result[batch_index][pano_index] = bool(row[pano_index])
    return result


def normalize_collated_string_matrix(
    raw: Any,
    batch_size: int,
    pano_count: int,
    split_pipe: bool = False,
) -> list[list[str]]:
    result = [["unknown" for _ in range(pano_count)] for _ in range(batch_size)]
    if raw is None:
        return result
    if isinstance(raw, str):
        values = raw.split("|") if split_pipe and "|" in raw else [raw]
        for batch_index in range(batch_size):
            for pano_index in range(pano_count):
                result[batch_index][pano_index] = stringify_nested(values[min(pano_index, len(values) - 1)])
        return result
    if not isinstance(raw, (list, tuple)):
        text = stringify_nested(raw)
        for batch_index in range(batch_size):
            for pano_index in range(pano_count):
                result[batch_index][pano_index] = text
        return result

    if len(raw) >= pano_count and all(isinstance(item, (list, tuple)) for item in raw[:pano_count]):
        for pano_index, item in enumerate(raw[:pano_count]):
            for batch_index in range(min(batch_size, len(item))):
                result[batch_index][pano_index] = stringify_nested(item[batch_index])
    elif len(raw) >= batch_size and all(isinstance(item, (list, tuple)) for item in raw[:batch_size]):
        for batch_index, item in enumerate(raw[:batch_size]):
            for pano_index in range(min(pano_count, len(item))):
                result[batch_index][pano_index] = stringify_nested(item[pano_index])
    elif batch_size == 1 and len(raw) >= pano_count:
        for pano_index, item in enumerate(raw[:pano_count]):
            result[0][pano_index] = stringify_nested(item)
    elif len(raw) >= batch_size:
        for batch_index, item in enumerate(raw[:batch_size]):
            text = stringify_nested(item)
            for pano_index in range(pano_count):
                result[batch_index][pano_index] = text

    if split_pipe:
        for batch_index in range(batch_size):
            row = result[batch_index]
            if row and all(value == row[0] for value in row) and "|" in row[0]:
                parts = [part for part in row[0].split("|") if part]
                if parts:
                    for pano_index in range(pano_count):
                        result[batch_index][pano_index] = stringify_nested(parts[min(pano_index, len(parts) - 1)])
    return result


def stringify_nested(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return stringify_nested(value[0])
    return str(value)


def canonical_dataset_name(value: Any) -> str:
    text = stringify_nested(value).strip().lower()
    compact = text.replace("_", "").replace("-", "").replace(" ", "")
    if not compact or compact in {"none", "unknown"}:
        return "unknown"
    if "matterport" in compact or "mp3d" in compact:
        return "matterport3d"
    if "stanford" in compact or "2d3ds" in compact:
        return "stanford2d3ds"
    if "structured3d" in compact or compact.startswith("s3d"):
        return "structured3d"
    if "panocity" in compact or "pano-city" in compact:
        return "panocity"
    return compact


def _loss_scalar(loss_dict: Dict[str, torch.Tensor], key: str) -> float | None:
    value = loss_dict.get(key)
    if value is None or not torch.is_tensor(value) or value.numel() != 1:
        return None
    return _finite_or_none(float(value.detach().cpu().item()))


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class CameraErrorStats:
    def __init__(self, thresholds_deg: Sequence[float], example_limit: int) -> None:
        self.thresholds_deg = tuple(sorted(float(item) for item in thresholds_deg))
        self.example_limit = max(0, int(example_limit))
        self.total_records = 0
        self.overall = self._new_group()
        self.by_dataset: Dict[str, Dict[str, Any]] = {}
        self.by_stage: Dict[str, Dict[str, Any]] = {}
        self.by_stage_dataset: Dict[str, Dict[str, Any]] = {}
        self.high_rotation_examples: list[Dict[str, Any]] = []

    def update(self, records: Sequence[Dict[str, Any]]) -> None:
        for record in records:
            self.total_records += 1
            dataset = str(record.get("dataset") or "unknown")
            stage = str(record.get("stage") or "unknown")
            self._update_group(self.overall, record)
            self._update_group(self.by_dataset.setdefault(dataset, self._new_group()), record)
            self._update_group(self.by_stage.setdefault(stage, self._new_group()), record)
            stage_dataset = f"{stage}/{dataset}"
            self._update_group(self.by_stage_dataset.setdefault(stage_dataset, self._new_group()), record)
            self._maybe_add_high_rotation_example(record)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def to_json(self) -> Dict[str, Any]:
        return {
            "updated_at_unix": time.time(),
            "thresholds_deg": list(self.thresholds_deg),
            "total_records": int(self.total_records),
            "overall": self._finalize_group(self.overall),
            "by_dataset": {
                key: self._finalize_group(value)
                for key, value in sorted(self.by_dataset.items())
            },
            "by_stage": {
                key: self._finalize_group(value)
                for key, value in sorted(self.by_stage.items())
            },
            "by_stage_dataset": {
                key: self._finalize_group(value)
                for key, value in sorted(self.by_stage_dataset.items())
            },
            "high_rotation_examples": list(self.high_rotation_examples),
        }

    def _new_group(self) -> Dict[str, Any]:
        return {
            "records": 0,
            "rotation": self._new_metric_group(),
            "translation_loss": self._new_metric_group(),
            "translation_l2_m": self._new_metric_group(),
            "translation_norm_l2": self._new_metric_group(),
            "rotation_over_threshold_counts": {
                self._threshold_key(threshold): 0 for threshold in self.thresholds_deg
            },
        }

    @staticmethod
    def _new_metric_group() -> Dict[str, Any]:
        return {
            "count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "min": None,
            "max": None,
        }

    def _update_group(self, group: Dict[str, Any], record: Dict[str, Any]) -> None:
        group["records"] += 1
        rotation_deg = record.get("rotation_deg")
        if record.get("rotation_valid") and rotation_deg is not None:
            self._update_metric_group(group["rotation"], float(rotation_deg))
            for threshold in self.thresholds_deg:
                if float(rotation_deg) >= threshold:
                    group["rotation_over_threshold_counts"][self._threshold_key(threshold)] += 1
        for key in ("translation_loss", "translation_l2_m", "translation_norm_l2"):
            value = record.get(key)
            if record.get("translation_valid") and value is not None:
                self._update_metric_group(group[key], float(value))

    @staticmethod
    def _update_metric_group(group: Dict[str, Any], value: float) -> None:
        if not math.isfinite(value):
            return
        group["count"] += 1
        group["sum"] += value
        group["sumsq"] += value * value
        group["min"] = value if group["min"] is None else min(float(group["min"]), value)
        group["max"] = value if group["max"] is None else max(float(group["max"]), value)

    def _finalize_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        rotation = self._finalize_metric_group(group["rotation"])
        rotation_count = max(1, int(group["rotation"]["count"]))
        threshold_counts = dict(group["rotation_over_threshold_counts"])
        threshold_ratios = {
            key: float(value) / float(rotation_count)
            for key, value in threshold_counts.items()
        }
        rotation["over_threshold_counts"] = threshold_counts
        rotation["over_threshold_ratios"] = threshold_ratios
        return {
            "records": int(group["records"]),
            "rotation_deg": rotation,
            "translation_loss": self._finalize_metric_group(group["translation_loss"]),
            "translation_l2_m": self._finalize_metric_group(group["translation_l2_m"]),
            "translation_norm_l2": self._finalize_metric_group(group["translation_norm_l2"]),
        }

    @staticmethod
    def _finalize_metric_group(group: Dict[str, Any]) -> Dict[str, Any]:
        count = int(group["count"])
        if count <= 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = float(group["sum"]) / float(count)
        variance = max(0.0, float(group["sumsq"]) / float(count) - mean * mean)
        return {
            "count": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": group["min"],
            "max": group["max"],
        }

    def _maybe_add_high_rotation_example(self, record: Dict[str, Any]) -> None:
        rotation_deg = record.get("rotation_deg")
        if self.example_limit <= 0 or not record.get("rotation_valid") or rotation_deg is None:
            return
        example = {
            key: record.get(key)
            for key in (
                "step",
                "stage",
                "rank",
                "dataset",
                "sequence_name",
                "scene_name",
                "rgb_path",
                "pano_index",
                "pano_count",
                "rotation_deg",
                "translation_l2_m",
                "translation_norm_l2",
                "translation_loss",
                "sample_scale",
            )
        }
        self.high_rotation_examples.append(example)
        self.high_rotation_examples.sort(
            key=lambda item: float(item.get("rotation_deg") or float("-inf")),
            reverse=True,
        )
        del self.high_rotation_examples[self.example_limit :]

    @staticmethod
    def _threshold_key(threshold: float) -> str:
        return f"ge_{threshold:g}"


def save_loss_plot(path: Path, metrics_history: list[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skip loss plot.")
        return

    steps = [item["step"] for item in metrics_history]
    plt.figure(figsize=(8, 5), dpi=140)
    for key, label in [
        ("loss", "total"),
        ("loss_depth", "depth"),
        ("loss_camera", "camera"),
    ]:
        values = [item[key] for item in metrics_history]
        plt.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=label)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[INFO] saved loss plot = {path}")


def save_checkpoint(path: Path, model: torch.nn.Module, args: argparse.Namespace, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_depth_scale = current_pred_depth_scale(model, args)
    args_payload = vars(args).copy()
    args_payload["pred_depth_scale"] = pred_depth_scale
    args_payload.update(current_sampler_status(model, args))
    adapter_state = None
    checkpoint_model = model
    if isinstance(model, DepthPredictionAdapter):
        checkpoint_model = model.model
        adapter_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("model.")
        }
    if args.checkpoint_format == "trainable_delta":
        delta = {
            name: param.detach().cpu()
            for name, param in checkpoint_model.named_parameters()
            if param.requires_grad
        }
        torch.save(
            {
                "checkpoint_format": "trainable_delta",
                "model_delta": delta,
                "foundation_checkpoint": (
                    str(args.base_checkpoint) if args.base_checkpoint is not None else None
                ),
                "base_checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                "trainable": args.trainable,
                "pred_depth_scale": pred_depth_scale,
                "learn_pred_depth_scale": bool(args.learn_pred_depth_scale),
                "depth_residual_mode": args.depth_residual_mode,
                "adapter_state": adapter_state,
                "args": args_payload,
                "step": step,
            },
            path,
        )
        return

    cpu_state = {key: value.detach().cpu() for key, value in checkpoint_model.state_dict().items()}
    torch.save(
        {
            "checkpoint_format": "full",
            "model": cpu_state,
            "foundation_checkpoint": (
                str(args.base_checkpoint) if args.base_checkpoint is not None else None
            ),
            "base_checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "pred_depth_scale": pred_depth_scale,
            "learn_pred_depth_scale": bool(args.learn_pred_depth_scale),
            "depth_residual_mode": args.depth_residual_mode,
            "adapter_state": adapter_state,
            "args": args_payload,
            "step": step,
        },
        path,
    )


def current_pred_depth_scale(model: torch.nn.Module, args: argparse.Namespace) -> float:
    if isinstance(model, DepthPredictionAdapter):
        return float(model.pred_depth_scale().detach().cpu())
    return float(args.pred_depth_scale)


def masked_log_l1_depth(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return masked_depth_loss(pred_depth, target_depth, valid_mask, mode="log_l1")


def estimate_sample_depth_alignment_scale(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    mode: str = "none",
    min_scale: float = 0.05,
    max_scale: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = pred_depth.shape[0]
    if mode == "none":
        return pred_depth.new_ones(batch_size)
    if mode not in {"sample_lstsq", "sample_log_median"}:
        raise ValueError(f"Unknown depth-scale-alignment mode: {mode}")

    pred = pred_depth.detach().float()
    target = target_depth.detach().to(device=pred.device, dtype=torch.float32)
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0) & (target > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=pred.device, dtype=torch.bool)

    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    valid_flat = valid.reshape(batch_size, -1)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    eps = max(float(eps), 1e-12)

    if mode == "sample_lstsq":
        weights = valid_flat.to(dtype=torch.float32)
        numerator = (pred_flat * target_flat * weights).sum(dim=1)
        denominator = (pred_flat.square() * weights).sum(dim=1).clamp_min(eps)
        scale = numerator / denominator
        has_valid = valid_flat.any(dim=1)
        scale = torch.where(has_valid, scale, torch.ones_like(scale))
    else:
        scales = []
        for item_pred, item_target, item_valid in zip(pred_flat, target_flat, valid_flat):
            if bool(item_valid.any()):
                ratio = (item_target[item_valid] / item_pred[item_valid]).clamp_min(eps)
                scales.append(torch.exp(torch.log(ratio).median()))
            else:
                scales.append(item_pred.new_tensor(1.0))
        scale = torch.stack(scales, dim=0)

    scale = torch.nan_to_num(scale, nan=1.0, posinf=max_scale, neginf=min_scale)
    return scale.clamp(min=min_scale, max=max_scale).to(device=pred_depth.device, dtype=pred_depth.dtype)


def expand_sample_scale_like(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=target.device, dtype=target.dtype)
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def depth_loss_validity_stats(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    min_window_valid_ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_depth = pred_depth.float()
    target_depth = target_depth.to(device=pred_depth.device, dtype=torch.float32)
    pred_finite = torch.isfinite(pred_depth)
    valid = pred_finite & torch.isfinite(target_depth) & (target_depth > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=target_depth.device, dtype=torch.bool)

    valid_float = valid.to(dtype=torch.float32)
    depth_loss_valid_ratio = valid_float.mean()
    pred_depth_finite_ratio = pred_finite.to(dtype=torch.float32).mean()

    min_ratio = max(float(min_window_valid_ratio), 0.0)
    if valid_float.ndim >= 3:
        reduce_dims = tuple(range(2, valid_float.ndim))
        window_valid_ratio = valid_float.mean(dim=reduce_dims)
        depth_loss_window_keep_ratio = (window_valid_ratio >= min_ratio).to(dtype=torch.float32).mean()
    else:
        depth_loss_window_keep_ratio = (depth_loss_valid_ratio >= min_ratio).to(dtype=torch.float32)

    return depth_loss_valid_ratio, depth_loss_window_keep_ratio, pred_depth_finite_ratio


def masked_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    mode: str = "log_l1",
    huber_delta: float = 0.2,
    error_clip: float = 0.5,
    sample_weight: torch.Tensor | None = None,
    min_window_valid_ratio: float = 0.0,
    valid_ratio_power: float = 0.0,
    sample_weight_min: float = 0.0,
    sample_weight_max: float = 10.0,
) -> torch.Tensor:
    pred_depth = pred_depth.float()
    target_depth = target_depth.to(device=pred_depth.device, dtype=torch.float32)
    valid = torch.isfinite(pred_depth) & torch.isfinite(target_depth) & (target_depth > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=target_depth.device, dtype=torch.bool)
    if not bool(valid.any()):
        return torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    log_abs_error = torch.zeros_like(pred_depth, dtype=torch.float32)
    log_abs_error[valid] = (
        torch.log(pred_depth[valid].clamp_min(1e-4))
        - torch.log(target_depth[valid].clamp_min(1e-4))
    ).abs()
    if mode == "log_l1":
        per_pixel_loss = log_abs_error
    if mode == "log_huber":
        delta = max(float(huber_delta), 1e-6)
        per_pixel_loss = torch.where(
            log_abs_error < delta,
            0.5 * log_abs_error.square() / delta,
            log_abs_error - 0.5 * delta,
        )
    elif mode == "clipped_log_l1":
        clip_value = max(float(error_clip), 1e-6)
        per_pixel_loss = log_abs_error.clamp_max(clip_value)
    elif mode != "log_l1":
        raise ValueError(f"Unknown depth loss mode: {mode}")

    valid_float = valid.to(dtype=torch.float32)
    reduce_dims = tuple(range(2, per_pixel_loss.ndim))
    per_window_loss = per_pixel_loss.sum(dim=reduce_dims) / valid_float.sum(dim=reduce_dims).clamp_min(1.0)
    window_valid_ratio = valid_float.mean(dim=reduce_dims)
    window_weight = torch.ones_like(per_window_loss)
    min_ratio = max(float(min_window_valid_ratio), 0.0)
    if min_ratio > 0:
        window_weight = window_weight * (window_valid_ratio >= min_ratio).to(dtype=window_weight.dtype)
    power = max(float(valid_ratio_power), 0.0)
    if power > 0:
        window_weight = window_weight * window_valid_ratio.clamp_min(1e-6).pow(power)
    if sample_weight is not None:
        window_weight = window_weight * _expand_sample_weight(
            sample_weight,
            per_window_loss,
            sample_weight_min=sample_weight_min,
            sample_weight_max=sample_weight_max,
        )
    if not bool((window_weight > 0).any()):
        return (0.0 * pred_depth).sum()
    return (per_window_loss * window_weight).sum() / window_weight.sum().clamp_min(1e-6)


def _expand_sample_weight(
    sample_weight: torch.Tensor,
    target: torch.Tensor,
    sample_weight_min: float = 0.0,
    sample_weight_max: float = 10.0,
) -> torch.Tensor:
    weight = sample_weight.to(device=target.device, dtype=torch.float32)

    if weight.ndim == 0:
        weight = weight.reshape(*([1] * target.ndim))
    elif weight.ndim == 1 and target.ndim >= 2 and weight.shape[0] == target.shape[0]:
        weight = weight.reshape(target.shape[0], *([1] * (target.ndim - 1)))

    while weight.ndim > target.ndim:
        weight = weight.squeeze(-1)
    while weight.ndim < target.ndim:
        weight = weight.unsqueeze(-1)

    # Multi-pano samples carry one dataset/sample weight per pano [B, N],
    # while depth/camera windows are flattened as [B, N * views_per_pano].
    # Repeat each pano weight across its yaw/pitch windows before broadcasting.
    if (
        weight.ndim >= 2
        and target.ndim >= 2
        and weight.shape[0] in (1, target.shape[0])
        and weight.shape[1] not in (1, target.shape[1])
    ):
        if target.shape[1] % weight.shape[1] != 0:
            raise ValueError(
                f"Cannot expand sample_weight shape {tuple(sample_weight.shape)} "
                f"to target shape {tuple(target.shape)}."
            )
        repeat = target.shape[1] // weight.shape[1]
        weight = weight.repeat_interleave(repeat, dim=1)

    return weight.expand_as(target).clamp(min=float(sample_weight_min), max=float(sample_weight_max))


def shared_frame_point_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    predictions: Dict,
    batch: Dict,
    pred_translation_scale: torch.Tensor | None,
    stride: int = 16,
) -> Dict[str, torch.Tensor]:
    """Couple depth and pano pose in one GT-anchored 3D coordinate frame.

    This is a memory-bounded analogue of PanoVGGT's global-point supervision:
    existing window depth predictions are sparsely lifted to 3D and transformed
    by the predicted pano poses. It requires valid translation and rotation GT,
    so Structured3D is excluded through ``pano_translation_valid``.
    """
    zero = pred_depth.sum() * 0.0
    pred_center = predictions.get("pano_camera_center")
    pred_quat = predictions.get("pano_rotation_quat_w2c")
    camera_meta = predictions.get("pano_camera_meta")
    if pred_center is None or pred_quat is None or camera_meta is None:
        return {"loss_global_point": zero, "global_point_valid_ratio": zero.detach()}

    targets = build_relative_pano_pose_targets(
        batch=batch,
        position_mode="relative_anchor",
        device=pred_depth.device,
        dtype=pred_depth.dtype,
    )
    if targets is None:
        return {"loss_global_point": zero, "global_point_valid_ratio": zero.detach()}
    target_center, target_quat, position_valid, rotation_valid = targets
    if pred_center.shape != target_center.shape or pred_quat.shape != target_quat.shape:
        return {"loss_global_point": zero, "global_point_valid_ratio": zero.detach()}

    pred_center = torch.nan_to_num(pred_center.float(), nan=0.0, posinf=0.0, neginf=0.0)
    pred_quat = F.normalize(
        torch.nan_to_num(pred_quat.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    pred_center, pred_quat = omega_y_up_pose_to_official_y_down(pred_center, pred_quat)
    if pred_translation_scale is not None:
        pred_center = pred_center * expand_sample_scale_like(pred_translation_scale, pred_center)

    batch_size, view_count, height, width, _ = pred_depth.shape
    num_panos = pred_center.shape[1]
    if num_panos < 1 or view_count % num_panos != 0:
        return {"loss_global_point": zero, "global_point_valid_ratio": zero.detach()}
    views_per_pano = view_count // num_panos
    low_h = max(2, int(math.ceil(height / max(int(stride), 1))))
    low_w = max(2, int(math.ceil(width / max(int(stride), 1))))

    def resize_depth(values: torch.Tensor, mode: str) -> torch.Tensor:
        flat = values[..., 0].reshape(batch_size * view_count, 1, height, width)
        resized = F.interpolate(
            flat.float(),
            size=(low_h, low_w),
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )
        return resized.reshape(batch_size, view_count, low_h, low_w)

    pred_z = resize_depth(pred_depth, "bilinear")
    target_z = resize_depth(target_depth, "nearest")
    point_valid = resize_depth(valid_mask.to(dtype=pred_depth.dtype), "nearest") > 0.5

    yaw = camera_meta["yaw"].reshape(-1)
    pitch = camera_meta["pitch"].reshape(-1)
    fov_x = camera_meta["fov_x"].reshape(-1)
    fov_y = camera_meta["fov_y"].reshape(-1)
    rays_omega = pinhole_rays(
        yaw,
        pitch,
        fov_x,
        fov_y,
        low_h,
        low_w,
        device=pred_depth.device,
        dtype=torch.float32,
    ).reshape(batch_size, view_count, low_h, low_w, 3)
    z_factor = build_window_z_factor(camera_meta, low_h, low_w).float().clamp_min(1e-4)
    pred_range = pred_z.float() / z_factor
    target_range = target_z.float() / z_factor
    point_valid = (
        point_valid
        & torch.isfinite(pred_range)
        & torch.isfinite(target_range)
        & (pred_range > 0)
        & (target_range > 0)
    )

    pred_local = omega_y_up_vectors_to_official_y_down(rays_omega * pred_range[..., None])
    target_local = omega_y_up_vectors_to_official_y_down(rays_omega * target_range[..., None])

    pred_w2c = quat_to_mat(pred_quat)
    target_w2c = quat_to_mat(F.normalize(target_quat.float(), dim=-1, eps=1e-6))
    pred_c2w = pred_w2c.transpose(-1, -2).repeat_interleave(views_per_pano, dim=1)
    target_c2w = target_w2c.transpose(-1, -2).repeat_interleave(views_per_pano, dim=1)
    pred_centers = pred_center.repeat_interleave(views_per_pano, dim=1)
    target_centers = target_center.float().repeat_interleave(views_per_pano, dim=1)
    pred_world = torch.matmul(pred_c2w[:, :, None, None], pred_local[..., None])[..., 0]
    pred_world = pred_world + pred_centers[:, :, None, None]
    target_world = torch.matmul(target_c2w[:, :, None, None], target_local[..., None])[..., 0]
    target_world = target_world + target_centers[:, :, None, None]

    camera_valid = (position_valid & rotation_valid).repeat_interleave(views_per_pano, dim=1)
    point_valid = point_valid & camera_valid[:, :, None, None]
    geometry_finite = torch.isfinite(pred_world).all(dim=-1) & torch.isfinite(target_world).all(dim=-1)
    depth_pose_valid_count = point_valid.sum().clamp_min(1)
    geometry_finite_ratio = (point_valid & geometry_finite).sum().to(dtype=torch.float32) / depth_pose_valid_count
    point_valid = point_valid & geometry_finite
    valid_float = point_valid.to(dtype=torch.float32)
    valid_count = valid_float.sum(dim=(1, 2, 3)).clamp_min(1.0)
    safe_target_range = torch.where(point_valid, target_range, torch.zeros_like(target_range))
    scene_scale = safe_target_range.sum(dim=(1, 2, 3)) / valid_count
    scene_scale = scene_scale.clamp_min(1.0)
    normalized_delta = (pred_world - target_world) / scene_scale[:, None, None, None, None]
    point_error = F.smooth_l1_loss(
        normalized_delta,
        torch.zeros_like(normalized_delta),
        beta=0.05,
        reduction="none",
    ).mean(dim=-1)
    loss = distributed_masked_mean(point_error, point_valid)
    finite_point_valid = point_valid & torch.isfinite(point_error)
    finite_ratio = finite_point_valid.sum().to(dtype=torch.float32) / point_valid.sum().clamp_min(1)
    return {
        "loss_global_point": loss,
        "global_point_valid_ratio": valid_float.mean().detach(),
        "global_point_finite_ratio": finite_ratio.detach(),
        "global_point_geometry_finite_ratio": geometry_finite_ratio.detach(),
    }


def adjacent_edge_overlap_loss(
    pred_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    band_fraction: float = 0.20,
) -> torch.Tensor:
    if pred_depth.ndim != 5 or pred_depth.shape[1] < 2:
        return torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    finite_depth = torch.isfinite(pred_depth)
    pred = torch.nan_to_num(pred_depth.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-4)
    batch_size, view_count, height, width, channels = pred.shape
    band = max(1, min(width // 2, int(round(width * float(band_fraction)))))
    left = torch.log(pred[:, :, :, :band, :])
    right_next = torch.log(torch.roll(pred, shifts=-1, dims=1)[:, :, :, -band:, :])
    finite_left = finite_depth[:, :, :, :band, :]
    finite_right = torch.roll(finite_depth, shifts=-1, dims=1)[:, :, :, -band:, :]
    valid = torch.isfinite(left) & torch.isfinite(right_next) & finite_left & finite_right
    if valid_mask is not None:
        valid_bool = valid_mask.to(device=pred.device, dtype=torch.bool)
        valid_left = valid_bool[:, :, :, :band, :]
        valid_right = torch.roll(valid_bool, shifts=-1, dims=1)[:, :, :, -band:, :]
        valid = valid & valid_left & valid_right
    if not bool(valid.any()):
        return torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    diff = torch.zeros_like(left)
    diff[valid] = (left[valid] - right_next[valid]).abs().clamp_max(0.5)
    per_pair = diff.sum(dim=(2, 3, 4)) / valid.to(dtype=torch.float32).sum(dim=(2, 3, 4)).clamp_min(1.0)
    per_pair = torch.nan_to_num(per_pair, nan=0.0, posinf=0.0, neginf=0.0)
    if sample_weight is None:
        return torch.nan_to_num(per_pair.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    weight = _expand_sample_weight(sample_weight, per_pair, sample_weight_min=0.0, sample_weight_max=10.0)
    if not bool((weight > 0).any()):
        return torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    result = (per_pair * weight).sum() / weight.sum().clamp_min(1e-6)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def camera_alignment_loss(
    predictions: Dict,
    batch: Dict,
    translation_weight: float,
    rotation_weight: float,
    fov_weight: float,
    position_mode: str,
    supervision_mode: str = "window_pose",
    pano_consistency_weight: float = 0.1,
    translation_normalization: str = "none",
    translation_normalization_eps: float = 1.0,
    pred_translation_scale: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    pred_pose = predictions.get("pose_enc")
    camera_meta = predictions.get("pano_camera_meta")
    if supervision_mode == "none":
        zero = predictions["depth"].new_zeros(())
        return {
            "loss_camera": zero,
            "loss_camera_t": zero,
            "loss_camera_r": zero,
            "loss_camera_fov": zero,
            "loss_camera_consistency": zero,
            "camera_rotation_deg": zero,
            "camera_translation_deg": zero,
            "camera_translation_valid_count": zero,
            "camera_rotation_valid_count": zero,
        }

    if supervision_mode == "pano_relative":
        pred_center = predictions.get("pano_camera_center")
        pred_quat = predictions.get("pano_rotation_quat_w2c")
        if pred_center is None or pred_quat is None:
            raise ValueError(
                "pano_relative supervision requires pano_camera_center and "
                "pano_rotation_quat_w2c from PanoCameraHead"
            )
        pred_center = torch.nan_to_num(pred_center.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_quat = F.normalize(
            torch.nan_to_num(pred_quat.float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
        # The pano sampler/head operates in Omega's right/up/forward basis,
        # while all loader GT poses are canonicalized to official OpenCV
        # right/down/forward coordinates. Convert at the supervision boundary;
        # the model and dense-depth path remain in their pretrained basis.
        pred_center, pred_quat = omega_y_up_pose_to_official_y_down(pred_center, pred_quat)
        if pred_translation_scale is not None:
            pred_center = pred_center * expand_sample_scale_like(pred_translation_scale, pred_center)
        pano_losses = pano_relative_pose_loss(
            pred_center=pred_center,
            pred_quat=pred_quat,
            batch=batch,
            position_mode=position_mode,
            translation_normalization=translation_normalization,
            translation_normalization_eps=translation_normalization_eps,
        )
        zero = pred_center.new_zeros(())
        return {
            "loss_camera": (
                translation_weight * pano_losses["loss_camera_t"]
                + rotation_weight * pano_losses["loss_camera_r"]
            ),
            "loss_camera_fov": zero,
            "loss_camera_consistency": zero,
            **pano_losses,
        }
    if supervision_mode != "window_pose":
        raise ValueError(f"Unknown camera-supervision-mode: {supervision_mode}")
    if pred_pose is None or camera_meta is None:
        zero = predictions["depth"].new_zeros(())
        return {
            "loss_camera": zero,
            "loss_camera_t": zero,
            "loss_camera_r": zero,
            "loss_camera_fov": zero,
            "loss_camera_consistency": zero,
            "camera_rotation_deg": zero,
            "camera_translation_deg": zero,
            "camera_translation_valid_count": zero,
            "camera_rotation_valid_count": zero,
        }

    pred_pose = torch.nan_to_num(pred_pose.float(), nan=0.0, posinf=0.0, neginf=0.0)
    rotations_c2w = camera_meta["rotations"].to(device=pred_pose.device, dtype=pred_pose.dtype)
    rotations_w2c = rotations_c2w.transpose(-1, -2).contiguous()
    pred_translation = pred_pose[..., :3]
    if pred_translation_scale is not None:
        pred_translation = pred_translation * expand_sample_scale_like(pred_translation_scale, pred_translation)

    target_quat = F.normalize(mat_to_quat(rotations_w2c), dim=-1)
    target_translation = build_target_translation(
        rotations_w2c=rotations_w2c,
        batch=batch,
        position_mode=position_mode,
    ).to(device=pred_pose.device, dtype=pred_pose.dtype)

    pred_quat = F.normalize(pred_pose[..., 3:7], dim=-1)
    pred_fov = pred_pose[..., 7:9]
    target_fov = torch.stack(
        [
            camera_meta["fov_y"].to(device=pred_pose.device, dtype=pred_pose.dtype),
            camera_meta["fov_x"].to(device=pred_pose.device, dtype=pred_pose.dtype),
        ],
        dim=-1,
    )

    if position_mode == "none" or translation_weight == 0:
        loss_t = pred_translation.new_zeros(())
    else:
        scale = camera_translation_normalization_scale(
            target_translation,
            mode=translation_normalization,
            eps=translation_normalization_eps,
        )
        loss_t = ((pred_translation - target_translation).abs() / scale[:, None, None]).mean()
    loss_r = torch.minimum(
        (pred_quat - target_quat).abs().sum(dim=-1),
        (pred_quat + target_quat).abs().sum(dim=-1),
    ).mean()
    loss_fov = (pred_fov - target_fov).abs().mean()
    loss_camera = translation_weight * loss_t + rotation_weight * loss_r + fov_weight * loss_fov
    return {
        "loss_camera": loss_camera,
        "loss_camera_t": loss_t,
        "loss_camera_r": loss_r,
        "loss_camera_fov": loss_fov,
        "loss_camera_consistency": pred_translation.new_zeros(()),
        "camera_rotation_deg": pred_translation.new_zeros(()),
        "camera_translation_deg": pred_translation.new_zeros(()),
        "camera_translation_valid_count": pred_translation.new_tensor(float(pred_translation.shape[1])),
        "camera_rotation_valid_count": pred_translation.new_tensor(float(pred_translation.shape[1])),
    }


def pano_relative_pose_loss(
    pred_center: torch.Tensor,
    pred_quat: torch.Tensor,
    batch: Dict,
    position_mode: str,
    translation_normalization: str,
    translation_normalization_eps: float,
) -> Dict[str, torch.Tensor]:
    """Supervise one predicted camera center and rotation per panorama."""
    targets = build_relative_pano_pose_targets(
        batch=batch,
        position_mode=position_mode,
        device=pred_center.device,
        dtype=pred_center.dtype,
    )
    if targets is None:
        valid = torch.zeros(pred_center.shape[:2], device=pred_center.device, dtype=torch.bool)
        loss_t = distributed_masked_mean(pred_center[..., 0] * 0.0, valid)
        loss_r = distributed_masked_mean(pred_quat[..., 0] * 0.0, valid)
        rotation_deg = distributed_masked_mean(pred_quat[..., 0].detach() * 0.0, valid)
        diag_zero = pred_center[..., 0].detach() * 0.0
        return {
            "loss_camera_t": loss_t,
            "loss_camera_r": loss_r,
            "camera_rotation_deg": rotation_deg,
            "camera_translation_deg": rotation_deg,
            "camera_translation_valid_count": pred_center.new_zeros(()),
            "camera_rotation_valid_count": pred_center.new_zeros(()),
            "_camera_diag_translation_loss": diag_zero,
            "_camera_diag_translation_l2_m": diag_zero,
            "_camera_diag_translation_norm_l2": diag_zero,
            "_camera_diag_translation_valid": valid.detach(),
            "_camera_diag_rotation_deg": diag_zero,
            "_camera_diag_rotation_valid": valid.detach(),
        }
    target_center, target_quat, position_valid, rotation_valid = targets
    if pred_center.shape != target_center.shape or pred_quat.shape != target_quat.shape:
        raise ValueError(
            "Pano camera prediction/target shapes disagree: "
            f"center {tuple(pred_center.shape)} vs {tuple(target_center.shape)}, "
            f"rotation {tuple(pred_quat.shape)} vs {tuple(target_quat.shape)}"
        )
    # Retain anchor-relative diagnostics for per-pano logging, but optimize and
    # evaluate every valid ordered camera pair, matching PanoVGGT's pose loss.
    translation_loss_map, translation_l2_m, translation_norm_l2, translation_valid = pano_center_error_maps(
        pred_center=pred_center,
        target_center=target_center,
        position_valid=position_valid,
        translation_normalization=translation_normalization,
        translation_normalization_eps=translation_normalization_eps,
    )
    _anchor_loss_r, _anchor_rotation_deg, rotation_deg_map, rotation_valid_mask = pano_rotation_geodesic_loss(
        pred_quat=pred_quat,
        target_quat=target_quat,
        rotation_valid=rotation_valid,
    )
    pair_errors = pano_pairwise_pose_error_maps(
        pred_center=pred_center,
        pred_quat_w2c=pred_quat,
        target_center=target_center,
        target_quat_w2c=target_quat,
        position_valid=position_valid,
        rotation_valid=rotation_valid,
        translation_normalization=translation_normalization,
        translation_normalization_eps=translation_normalization_eps,
    )
    loss_t = distributed_masked_mean(
        pair_errors["translation_loss"],
        pair_errors["translation_valid"],
    )
    loss_r = distributed_masked_mean(
        pair_errors["rotation_rad"],
        pair_errors["rotation_valid"],
    )
    translation_deg = distributed_masked_mean(
        pair_errors["translation_deg"].detach(),
        pair_errors["translation_angle_valid"],
    )
    rotation_deg = distributed_masked_mean(
        pair_errors["rotation_rad"].detach(),
        pair_errors["rotation_valid"],
    ) * (180.0 / math.pi)
    return {
        "loss_camera_t": loss_t,
        "loss_camera_r": loss_r,
        "camera_rotation_deg": rotation_deg.detach(),
        "camera_translation_deg": translation_deg.detach(),
        "camera_translation_valid_count": pair_errors["translation_angle_valid"].sum().to(dtype=pred_center.dtype),
        "camera_rotation_valid_count": pair_errors["rotation_valid"].sum().to(dtype=pred_center.dtype),
        "_camera_diag_translation_loss": translation_loss_map.detach(),
        "_camera_diag_translation_l2_m": translation_l2_m.detach(),
        "_camera_diag_translation_norm_l2": translation_norm_l2.detach(),
        "_camera_diag_translation_valid": translation_valid.detach(),
        "_camera_diag_rotation_deg": rotation_deg_map.detach(),
        "_camera_diag_rotation_valid": rotation_valid_mask.detach(),
    }


def pano_pairwise_pose_error_maps(
    pred_center: torch.Tensor,
    pred_quat_w2c: torch.Tensor,
    target_center: torch.Tensor,
    target_quat_w2c: torch.Tensor,
    position_valid: torch.Tensor,
    rotation_valid: torch.Tensor,
    translation_normalization: str,
    translation_normalization_eps: float,
) -> Dict[str, torch.Tensor]:
    """Return PanoVGGT-style errors for every ordered relative camera pair."""
    if pred_center.ndim != 3 or pred_center.shape[1] < 2:
        shape = (*pred_center.shape[:2], pred_center.shape[1])
        zero = pred_center.new_zeros(shape)
        valid = torch.zeros(shape, device=pred_center.device, dtype=torch.bool)
        return {
            "translation_loss": zero,
            "translation_deg": zero,
            "translation_valid": valid,
            "translation_angle_valid": valid,
            "rotation_rad": zero,
            "rotation_valid": valid,
        }

    pred_w2c = quat_to_mat(F.normalize(pred_quat_w2c, dim=-1, eps=1e-6))
    target_w2c = quat_to_mat(F.normalize(target_quat_w2c, dim=-1, eps=1e-6))
    pred_delta = pred_center[:, None, :, :] - pred_center[:, :, None, :]
    target_delta = target_center[:, None, :, :] - target_center[:, :, None, :]
    pred_translation = torch.matmul(pred_w2c[:, :, None], pred_delta[..., None])[..., 0]
    target_translation = torch.matmul(target_w2c[:, :, None], target_delta[..., None])[..., 0]

    pred_relative_rotation = pred_w2c[:, :, None] @ pred_w2c.transpose(-1, -2)[:, None, :]
    target_relative_rotation = target_w2c[:, :, None] @ target_w2c.transpose(-1, -2)[:, None, :]

    pair_count = pred_center.shape[1]
    off_diagonal = ~torch.eye(pair_count, device=pred_center.device, dtype=torch.bool)[None]
    translation_valid = (
        position_valid.to(device=pred_center.device).bool()[:, :, None]
        & position_valid.to(device=pred_center.device).bool()[:, None, :]
        & off_diagonal
    )
    rotation_pair_valid = (
        rotation_valid.to(device=pred_center.device).bool()[:, :, None]
        & rotation_valid.to(device=pred_center.device).bool()[:, None, :]
        & off_diagonal
    )

    scale = camera_translation_normalization_scale(
        target_translation,
        mode=translation_normalization,
        eps=translation_normalization_eps,
        valid_mask=translation_valid,
    )
    normalized_delta = (pred_translation - target_translation) / scale[:, None, None, None]
    translation_loss = F.smooth_l1_loss(
        normalized_delta,
        torch.zeros_like(normalized_delta),
        beta=0.1,
        reduction="none",
    ).mean(dim=-1)

    target_norm = torch.linalg.vector_norm(target_translation, dim=-1)
    pred_norm = torch.linalg.vector_norm(pred_translation, dim=-1)
    cosine = (pred_translation * target_translation).sum(dim=-1) / (
        pred_norm.clamp_min(1e-8) * target_norm.clamp_min(1e-8)
    )
    translation_deg = torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi)
    translation_angle_valid = translation_valid & (target_norm > 1e-8)

    rotation_delta = pred_relative_rotation.transpose(-1, -2) @ target_relative_rotation
    trace = torch.diagonal(rotation_delta, dim1=-2, dim2=-1).sum(dim=-1)
    rotation_cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    rotation_floor = torch.acos(rotation_cosine.new_tensor(1.0 - 1e-7))
    rotation_rad = (torch.acos(rotation_cosine) - rotation_floor).clamp_min(0.0)
    rotation_rad = torch.where(rotation_pair_valid, rotation_rad, torch.zeros_like(rotation_rad))
    return {
        "translation_loss": translation_loss,
        "translation_deg": translation_deg,
        "translation_valid": translation_valid,
        "translation_angle_valid": translation_angle_valid,
        "rotation_rad": rotation_rad,
        "rotation_valid": rotation_pair_valid,
    }


def pano_center_error_maps(
    pred_center: torch.Tensor,
    target_center: torch.Tensor,
    position_valid: torch.Tensor,
    translation_normalization: str,
    translation_normalization_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_map = pred_center[..., 0] * 0.0
    false_valid = torch.zeros(pred_center.shape[:2], device=pred_center.device, dtype=torch.bool)
    if target_center.ndim != 3 or target_center.shape[1] < 2:
        return zero_map, zero_map.detach(), zero_map.detach(), false_valid
    valid = position_valid.to(device=pred_center.device).bool()
    if valid.shape != pred_center.shape[:2]:
        return zero_map, zero_map.detach(), zero_map.detach(), false_valid
    valid = valid.clone()
    valid[:, 0] = False
    scale = camera_translation_normalization_scale(
        target_center,
        mode=translation_normalization,
        eps=translation_normalization_eps,
        valid_mask=valid,
    )
    delta = pred_center - target_center
    normalized_error = delta / scale[:, None, None]
    per_pano = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=0.1,
        reduction="none",
    ).mean(dim=-1)
    translation_l2_m = torch.linalg.vector_norm(delta.detach(), dim=-1)
    translation_norm_l2 = torch.linalg.vector_norm(normalized_error.detach(), dim=-1)
    return per_pano, translation_l2_m, translation_norm_l2, valid


def pano_center_loss(
    pred_center: torch.Tensor,
    target_center: torch.Tensor,
    position_valid: torch.Tensor,
    translation_normalization: str,
    translation_normalization_eps: float,
) -> torch.Tensor:
    per_pano, _, _, valid = pano_center_error_maps(
        pred_center=pred_center,
        target_center=target_center,
        position_valid=position_valid,
        translation_normalization=translation_normalization,
        translation_normalization_eps=translation_normalization_eps,
    )
    return distributed_masked_mean(per_pano, valid)


def camera_translation_normalization_scale(
    target_translation: torch.Tensor,
    mode: str,
    eps: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_shape = target_translation.shape[0]
    if mode == "none":
        return target_translation.new_ones(batch_shape)
    norms = torch.linalg.vector_norm(target_translation.float(), dim=-1)
    if valid_mask is None:
        valid = torch.ones_like(norms, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=norms.device).bool()
        if valid.shape != norms.shape:
            raise ValueError(f"Translation validity shape {tuple(valid.shape)} != {tuple(norms.shape)}")
    valid = valid & torch.isfinite(norms)
    safe_norms = torch.where(valid, norms, torch.zeros_like(norms))
    valid_float = valid.to(dtype=norms.dtype)
    reduce_dims = tuple(range(1, norms.ndim))
    count = valid_float.sum(dim=reduce_dims).clamp_min(1.0)
    if mode == "target_rms":
        scale = ((safe_norms.square() * valid_float).sum(dim=reduce_dims) / count).sqrt()
    elif mode == "target_mean_norm":
        scale = (safe_norms * valid_float).sum(dim=reduce_dims) / count
    elif mode == "target_max_norm":
        scale = safe_norms.masked_fill(~valid, float("-inf")).flatten(start_dim=1).max(dim=-1).values
        scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    else:
        raise ValueError(f"Unknown camera translation normalization: {mode}")
    return scale.to(device=target_translation.device, dtype=target_translation.dtype).clamp_min(float(eps))


def pano_rotation_geodesic_loss(
    pred_quat: torch.Tensor,
    target_quat: torch.Tensor,
    rotation_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_quat.ndim != 3 or rotation_valid.ndim != 2 or target_quat.shape[1] < 2:
        zero = pred_quat.new_zeros(())
        zero_map = pred_quat[..., 0].detach() * 0.0
        valid = torch.zeros(pred_quat.shape[:2], device=pred_quat.device, dtype=torch.bool)
        return zero, zero, zero_map, valid
    valid = rotation_valid.to(device=pred_quat.device).bool()
    if valid.shape != pred_quat.shape[:2]:
        zero = pred_quat.new_zeros(())
        zero_map = pred_quat[..., 0].detach() * 0.0
        valid = torch.zeros(pred_quat.shape[:2], device=pred_quat.device, dtype=torch.bool)
        return zero, zero, zero_map, valid
    valid = valid.clone()
    valid = valid & valid[:, :1]
    valid[:, 0] = False

    pred = F.normalize(pred_quat, dim=-1)
    target = F.normalize(target_quat, dim=-1)
    cosine = (pred * target).sum(dim=-1).abs().clamp(0.0, 1.0)
    eps = 1e-7
    floor = 2.0 * torch.acos(cosine.new_tensor(1.0 - eps))
    angle = (2.0 * torch.acos(cosine.clamp_max(1.0 - eps)) - floor).clamp_min(0.0)
    loss = distributed_masked_mean(angle, valid)
    degrees = distributed_masked_mean(angle.detach(), valid) * (180.0 / math.pi)
    return loss, degrees, angle.detach() * (180.0 / math.pi), valid.detach()


def distributed_masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid labels with correct normalization under variable-label DDP."""
    finite_valid = valid.to(device=values.device, dtype=torch.bool) & torch.isfinite(values)
    valid_float = finite_valid.to(dtype=values.dtype)
    safe_values = torch.where(finite_valid, values, torch.zeros_like(values))
    numerator = safe_values.sum()
    count = valid_float.sum()
    if not (dist.is_available() and dist.is_initialized()):
        return numerator / count.clamp_min(1.0)
    global_count = count.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if not bool(global_count > 0):
        return numerator * 0.0
    # DDP averages gradients over ranks. Multiplying the local numerator by
    # world_size/global_count yields a true global valid-label mean.
    return numerator * (float(dist.get_world_size()) / global_count)


def build_relative_pano_pose_targets(
    batch: Dict,
    position_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    pano_position = batch.get("pano_position_m", None)
    if pano_position is None:
        return None
    centers = pano_position.to(device=device, dtype=dtype)
    if centers.ndim == 1:
        centers = centers[None, None, :]
    elif centers.ndim == 2:
        centers = centers[:, None, :]
    if centers.ndim != 3 or centers.shape[1] < 2:
        return None
    # Position availability and translation-supervision eligibility are
    # separate. Structured3D keeps metric centers for reprojection/debugging,
    # but explicitly disables camera translation learning and evaluation.
    position_valid_raw = batch.get("pano_translation_valid", batch.get("pano_position_valid", None))
    if position_valid_raw is None:
        position_valid = torch.ones(centers.shape[:2], device=device, dtype=torch.bool)
    else:
        position_valid = position_valid_raw.to(device=device).bool()
        if position_valid.ndim == 1:
            position_valid = position_valid[None, :]
        if position_valid.shape != centers.shape[:2]:
            return None
    position_finite = torch.isfinite(centers).all(dim=-1)
    position_valid = position_valid & position_finite
    centers = torch.where(position_finite[..., None], centers, torch.zeros_like(centers))

    pano_rotation = batch.get("pano_rotation_c2w", None)
    if pano_rotation is None:
        rotations_c2w = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3).expand(
            centers.shape[0],
            centers.shape[1],
            3,
            3,
        )
        rotation_valid = torch.zeros(centers.shape[:2], device=device, dtype=torch.bool)
    else:
        rotations_c2w = pano_rotation.to(device=device, dtype=dtype)
        if rotations_c2w.ndim == 2:
            rotations_c2w = rotations_c2w[None, None, :, :]
        elif rotations_c2w.ndim == 3:
            rotations_c2w = rotations_c2w[:, None, :, :]
        if rotations_c2w.shape[:2] != centers.shape[:2] or rotations_c2w.shape[-2:] != (3, 3):
            return None
        rotation_valid_raw = batch.get("pano_rotation_valid", None)
        if rotation_valid_raw is None:
            rotation_valid = torch.ones(centers.shape[:2], device=device, dtype=torch.bool)
        else:
            rotation_valid = rotation_valid_raw.to(device=device).bool()
            if rotation_valid.ndim == 1:
                rotation_valid = rotation_valid[None, :]
            elif rotation_valid.ndim == 0:
                rotation_valid = rotation_valid.reshape(1, 1).expand(*centers.shape[:2])
        if rotation_valid.shape != centers.shape[:2]:
            return None
    rotation_finite = torch.isfinite(rotations_c2w).all(dim=(-1, -2))
    rotation_valid = rotation_valid & rotation_finite
    identity = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3)
    rotations_c2w = torch.where(rotation_finite[..., None, None], rotations_c2w, identity)

    if position_mode in {"none", "local_zero"}:
        target_centers = torch.zeros_like(centers)
        position_valid = torch.zeros_like(position_valid)
        target_rotations_c2w = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3).expand_as(rotations_c2w)
        rotation_valid = torch.zeros_like(rotation_valid)
    elif position_mode == "relative_anchor":
        anchor_w2c = rotations_c2w[:, :1].transpose(-1, -2)
        target_centers = (anchor_w2c @ (centers - centers[:, :1])[..., None])[..., 0]
        position_valid = position_valid & position_valid[:, :1]
        target_rotations_c2w = anchor_w2c @ rotations_c2w
        rotation_valid = rotation_valid & rotation_valid[:, :1]
    elif position_mode == "relative_mean":
        target_centers = centers - centers.mean(dim=1, keepdim=True)
        target_rotations_c2w = rotations_c2w
    elif position_mode == "world":
        target_centers = centers
        target_rotations_c2w = rotations_c2w
    else:
        raise ValueError(f"Unknown camera-position-mode: {position_mode}")

    target_rotations_w2c = target_rotations_c2w.transpose(-1, -2).contiguous()
    target_quat = F.normalize(mat_to_quat(target_rotations_w2c), dim=-1)
    return target_centers.contiguous(), target_quat, position_valid, rotation_valid


def build_relative_pano_centers(
    pano_position: torch.Tensor,
    reference: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    centers = pano_position.to(device=device, dtype=dtype)
    if centers.ndim == 1:
        centers = centers[None, None, :]
    elif centers.ndim == 2:
        centers = centers[:, None, :]
    if reference in {"none", "local_zero"}:
        return torch.zeros_like(centers)
    if reference == "relative_anchor":
        return centers - centers[:, :1, :]
    if reference == "relative_mean":
        return centers - centers.mean(dim=1, keepdim=True)
    if reference == "world":
        return centers
    raise ValueError(f"Unknown camera-position-mode: {reference}")


def build_target_translation(rotations_w2c: torch.Tensor, batch: Dict, position_mode: str) -> torch.Tensor:
    if position_mode == "none":
        return rotations_w2c.new_zeros(*rotations_w2c.shape[:2], 3)
    if position_mode == "local_zero":
        return rotations_w2c.new_zeros(*rotations_w2c.shape[:2], 3)
    if position_mode not in {"world", "relative_anchor", "relative_mean"}:
        raise ValueError(f"Unknown camera-position-mode: {position_mode}")

    pano_position = batch.get("pano_position_m", None)
    if pano_position is None:
        return rotations_w2c.new_zeros(*rotations_w2c.shape[:2], 3)
    center_world = pano_position.to(device=rotations_w2c.device, dtype=rotations_w2c.dtype)
    if center_world.ndim == 1:
        center_world = center_world[None]
    if center_world.ndim == 2:
        center_world = center_world[:, None, :]
    if position_mode == "relative_anchor":
        center_world = center_world - center_world[:, :1, :]
    elif position_mode == "relative_mean":
        center_world = center_world - center_world.mean(dim=1, keepdim=True)

    batch_size, total_views = rotations_w2c.shape[:2]
    num_panos = center_world.shape[1]
    if total_views % num_panos != 0:
        raise ValueError(f"Cannot map {total_views} views to {num_panos} panos.")
    views_per_pano = total_views // num_panos
    center_world = center_world[:, :, None, :].expand(batch_size, num_panos, views_per_pano, 3)
    center_world = center_world.reshape(batch_size, total_views, 3)[..., None]
    return (-(rotations_w2c @ center_world)[..., 0]).contiguous()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, strict: bool) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Place vggt_omega_1b_512.pt under project/ckpt or pass --checkpoint."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_delta" in checkpoint:
        foundation_checkpoint = checkpoint.get("foundation_checkpoint")
        if foundation_checkpoint is None:
            foundation_checkpoint = checkpoint.get("args", {}).get("base_checkpoint")
        if foundation_checkpoint:
            foundation_checkpoint_path = Path(foundation_checkpoint)
            if foundation_checkpoint_path.exists() and foundation_checkpoint_path.resolve() != checkpoint_path.resolve():
                load_checkpoint(model, foundation_checkpoint_path, strict=False)
            else:
                print(f"[WARN] foundation checkpoint is unavailable for delta load: {foundation_checkpoint}")
        base_checkpoint = checkpoint.get("base_checkpoint")
        if base_checkpoint is None:
            base_checkpoint = checkpoint.get("args", {}).get("checkpoint")
        if base_checkpoint:
            base_checkpoint_path = Path(base_checkpoint)
            if base_checkpoint_path.exists() and base_checkpoint_path.resolve() != checkpoint_path.resolve():
                load_checkpoint(model, base_checkpoint_path, strict=False)
            else:
                print(f"[WARN] base checkpoint is unavailable for delta load: {base_checkpoint}")
        else:
            print("[WARN] delta checkpoint has no base_checkpoint; applying delta to current model.")
        state_dict = strip_state_dict_prefix(checkpoint["model_delta"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] loaded trainable-delta checkpoint = {checkpoint_path}")
        print(f"[INFO] missing_keys = {len(missing)}; unexpected_keys = {len(unexpected)}")
        print_checkpoint_key_analysis(missing, unexpected)
        return

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    state_dict = strip_state_dict_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"[INFO] loaded checkpoint = {checkpoint_path}")
    print(f"[INFO] missing_keys = {len(missing)}; unexpected_keys = {len(unexpected)}")
    print_checkpoint_key_analysis(missing, unexpected)


def print_checkpoint_key_analysis(missing: Sequence[str], unexpected: Sequence[str], max_items: int = 20) -> None:
    if missing:
        print(f"[INFO] missing_key_prefixes = {_format_key_prefix_counts(missing)}")
        print(f"[INFO] missing_keys_sample = {list(missing)[:max_items]}")
    if unexpected:
        print(f"[INFO] unexpected_key_prefixes = {_format_key_prefix_counts(unexpected)}")
        print(f"[INFO] unexpected_keys_sample = {list(unexpected)[:max_items]}")


def _format_key_prefix_counts(keys: Sequence[str], depth: int = 2, max_groups: int = 12) -> str:
    counts: Dict[str, int] = {}
    for key in keys:
        parts = str(key).split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else str(key)
        counts[prefix] = counts.get(prefix, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_groups]
    return ", ".join(f"{prefix}:{count}" for prefix, count in ranked)


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        stripped[key] = value
    return stripped


def configure_trainable(model: torch.nn.Module, mode: str) -> Tuple[int, int]:
    if mode == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)

        train_luna = mode in {
            "luna",
            "luna_dense",
            "luna_heads",
            "luna_dense_tail",
            "luna_residual",
            "luna_residual_dense",
            "luna_residual_dense_tail",
            "luna_residual_tail_heads",
            "luna_residual_heads",
        }
        train_dense = mode in {"dense", "heads", "luna_dense", "luna_heads", "luna_residual_dense", "luna_residual_heads"}
        train_dense_tail = mode in {"luna_dense_tail", "luna_residual_dense_tail", "luna_residual_tail_heads"}
        train_camera = mode in {"camera", "heads", "luna_heads", "luna_residual_heads", "luna_residual_tail_heads"}
        for name, param in model.named_parameters():
            if train_luna and (
                "luna_" in name or "pano_global" in name or "pano_geometry" in name
            ):
                param.requires_grad_(True)
            if train_dense and "dense_head" in name:
                param.requires_grad_(True)
            if train_dense_tail and is_dense_tail_parameter(name):
                param.requires_grad_(True)
            if train_camera and "camera_head" in name:
                param.requires_grad_(True)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    if trainable == 0:
        raise ValueError(f"No trainable parameters selected for mode {mode!r}")
    return trainable, frozen


def is_dense_tail_parameter(name: str) -> bool:
    if "dense_head" not in name:
        return False
    tail_markers = (
        "dense_head.scratch.refinenet1",
        "dense_head.scratch.refinenet2",
        "dense_head.proj.",
        "dense_head.proj_conf.",
    )
    return any(marker in name for marker in tail_markers)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def setup_distributed(args: argparse.Namespace) -> Dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = args.distributed == "ddp" or (args.distributed == "auto" and world_size > 1)
    if args.distributed == "ddp" and world_size <= 1:
        raise RuntimeError("distributed=ddp requires torchrun or WORLD_SIZE > 1.")
    if use_ddp:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")
        if args.dist_backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA. Use --dist-backend gloo for CPU debugging.")
        if not dist.is_initialized():
            dist.init_process_group(backend=args.dist_backend, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return {
        "distributed": use_ddp,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def cleanup_distributed(dist_state: Dict[str, int | bool]) -> None:
    if dist_state["distributed"] and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(dist_state: Dict[str, int | bool]) -> bool:
    return int(dist_state["rank"]) == 0


def rank0_print(message: str, dist_state: Dict[str, int | bool]) -> None:
    if is_main_process(dist_state):
        print(message)


def barrier(dist_state: Dict[str, int | bool]) -> None:
    if dist_state["distributed"] and dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_loss_dict(
    loss_dict: Dict[str, torch.Tensor],
    dist_state: Dict[str, int | bool],
) -> Dict[str, torch.Tensor]:
    if not dist_state["distributed"]:
        return loss_dict
    reduced = {}
    for key, value in loss_dict.items():
        item = value.detach().clone()
        dist.all_reduce(item, op=dist.ReduceOp.SUM)
        item /= int(dist_state["world_size"])
        reduced[key] = item
    return reduced


def should_stop_for_duration(
    started_at: float,
    max_duration_seconds: float | None,
    device: torch.device,
    dist_state: Dict[str, int | bool],
) -> bool:
    if max_duration_seconds is None:
        return False
    local_stop = time.time() - started_at >= max_duration_seconds
    if not dist_state["distributed"]:
        return local_stop
    flag = torch.tensor(1 if local_stop else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def resolve_device(requested: str, dist_state: Dict[str, int | bool]) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if requested == "cuda" and dist_state["distributed"]:
        return torch.device("cuda", int(dist_state["local_rank"]))
    return torch.device(requested)


def parse_pitch_degrees(value: str) -> Tuple[float, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(float(item) for item in items) or (0.0,)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_smoke_dataset(root: Path) -> None:
    height, width = 32, 64
    sequence_lines = []
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    for idx in range(3):
        scene_name = f"pano_smoke_{idx:02d}"
        rgb_dir = root / scene_name / "clone" / "frames" / "rgb" / "Camera_0"
        depth_dir = root / scene_name / "clone" / "frames" / "depth" / "Camera_0"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        rgb = np.stack(
            [
                np.broadcast_to((x + idx * 35).astype(np.uint8), (height, width)),
                np.broadcast_to(y, (height, width)),
                np.full((height, width), 128 + idx * 20, dtype=np.uint8),
            ],
            axis=-1,
        )
        Image.fromarray(rgb, mode="RGB").save(rgb_dir / "rgb_00000.jpg")

        depth_m = np.full((height, width), 2.0 + idx * 0.25, dtype=np.float32)
        cv2.imwrite(str(depth_dir / "depth_00000.png"), np.round(depth_m * 100.0).astype(np.uint16))

        meta = {
            "scene_name": scene_name,
            "pano_shape_hw": [height, width],
            "camera_alignment": {
                "alignment_policy": "smoke sample uses pano-local or relative-anchor translation",
                "panorama_position_m_xyz": [float(idx), 0.0, 0.0],
            },
            "depth_policy": {
                "unit": "meters",
                "invalid_depth_value": 0,
                "output_depth_scale": 100.0,
                "saved_depth_decode": "depth_m = uint16_png / 100.0",
            },
        }
        (root / scene_name / "clone" / "pano_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        sequence_lines.append(f"{scene_name}/clone/frames/rgb/Camera_0")
    (root / "sequence_list.txt").write_text("\n".join(sequence_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
