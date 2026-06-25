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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import PanoCityPairedOmegaDataset, PanoVKittiOmegaDataset  # noqa: E402
from vggt_omega.data.pano_sampler import make_default_view_grid  # noqa: E402
from vggt_omega.models.heads.dense_head import DenseHead  # noqa: E402
from vggt_omega.models.layers import PatchEmbed  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402
from vggt_omega.utils.rotation import mat_to_quat  # noqa: E402


DEFAULT_DATASET_ROOT = Path("whitehole/AOKI/datasets/PANO_LUNA_omega")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ckpt" / "vggt_omega_1b_512.pt"
CONFIG_PATH_KEYS = {
    "dataset_root",
    "checkpoint",
    "output_dir",
    "log_csv",
    "loss_plot",
    "tensorboard_dir",
    "debug_dir",
    "metadata_path",
}
DEFAULT_PANOCITY_PRED_DEPTH_SCALE = 5.491308212280273


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VGGT-Omega LUNA on converted pano VKitti-style data.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config; explicit CLI values override it.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-format", choices=["vkitti", "panocity_paired"], default="vkitti")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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
            "all",
        ],
        default="luna_heads",
    )
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--enable-camera-head", dest="enable_camera_head", action="store_true", default=True)
    parser.add_argument("--disable-camera-head", dest="enable_camera_head", action="store_false")
    parser.add_argument("--camera-loss-weight", type=float, default=1.0)
    parser.add_argument("--camera-translation-weight", type=float, default=1.0)
    parser.add_argument("--camera-rotation-weight", type=float, default=1.0)
    parser.add_argument("--camera-fov-weight", type=float, default=0.1)
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
            ).to(device)

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
        rank0_print(
            "[INFO] camera_supervision = "
            f"{args.camera_supervision_mode} position={args.camera_position_mode} "
            f"weight={args.camera_loss_weight}",
            dist_state,
        )
        rank0_print(f"[INFO] pred_depth_scale = {args.pred_depth_scale}", dist_state)
        rank0_print(f"[INFO] learn_pred_depth_scale = {args.learn_pred_depth_scale}", dist_state)
        rank0_print(
            "[INFO] depth_residual = "
            f"{args.depth_residual_mode} hidden={args.depth_residual_hidden} "
            f"max_log={args.depth_residual_max_log}",
            dist_state,
        )
        rank0_print(
            "[INFO] depth_loss = "
            f"{args.depth_loss_mode} huber_delta={args.depth_log_huber_delta} "
            f"clip={args.depth_log_error_clip}",
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
        loss_plot = args.loss_plot or (args.output_dir / "loss_curve.png")
        tensorboard_dir = args.tensorboard_dir or (args.output_dir / "tensorboard")
        tensorboard_writer = create_tensorboard_writer(tensorboard_dir, dist_state, enabled=args.tensorboard)
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
                        trainable_count, frozen_count = configure_trainable_for_stage(
                            unwrap_model(model),
                            args,
                            active_stage,
                        )
                        optimizer = build_optimizer_for_stage(model, args, active_stage)
                        rank0_print(
                            f"[INFO] active_stage = {format_stage_status(active_stage_index, active_stage, trainable_count, frozen_count)}",
                            dist_state,
                        )

                global_step += 1
                batch = move_batch_to_device(batch, device)
                loss_dict = train_step(model, batch, optimizer, args, global_step=global_step)
                loss_dict = reduce_loss_dict(loss_dict, dist_state)
                elapsed_seconds = time.time() - started_at
                sampler_status = current_sampler_status(unwrap_model(model), args)
                metrics = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "elapsed_seconds": elapsed_seconds,
                    "loss": float(loss_dict["loss"].item()),
                    "loss_depth": float(loss_dict["loss_depth"].item()),
                    "loss_overlap": float(loss_dict.get("loss_overlap", torch.tensor(0.0)).item()),
                    "loss_camera": float(loss_dict["loss_camera"].item()),
                    "loss_camera_t": float(loss_dict.get("loss_camera_t", torch.tensor(0.0)).item()),
                    "loss_camera_r": float(loss_dict.get("loss_camera_r", torch.tensor(0.0)).item()),
                    "loss_camera_fov": float(loss_dict.get("loss_camera_fov", torch.tensor(0.0)).item()),
                    "loss_camera_consistency": float(
                        loss_dict.get("loss_camera_consistency", torch.tensor(0.0)).item()
                    ),
                    "pred_depth_scale": float(loss_dict["pred_depth_scale"].item()),
                    "stage": active_stage_name,
                    "stage_index": int(active_stage_index + 1) if active_stage_index is not None else 0,
                    "window_size": sampler_status["window_size"],
                    "patch_size": sampler_status["patch_size"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if is_main_process(dist_state):
                    metrics_history.append(metrics)
                    append_loss_csv(log_csv, metrics)
                    write_tensorboard_metrics(tensorboard_writer, metrics)
                    message = (
                        f"[TRAIN] epoch={epoch + 1} step={global_step} "
                        f"stage={active_stage_name} elapsed={elapsed_seconds / 60.0:.2f}m "
                        f"loss={metrics['loss']:.6f} depth={metrics['loss_depth']:.6f} "
                        f"overlap={metrics['loss_overlap']:.6f} camera={metrics['loss_camera']:.6f} "
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
            curriculum_bins=args.curriculum_bins,
            use_metadata_weights=args.use_metadata_weights,
        )
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
    ) -> None:
        super().__init__()
        if initial_scale <= 0:
            raise ValueError(f"initial pred_depth_scale must be positive, got {initial_scale}")
        self.model = model
        self.learn_scale = bool(learn_scale)
        self.residual_mode = residual_mode
        self.residual_max_log = float(residual_max_log)
        self.depth_residual_enabled = residual_mode != "none"
        log_scale = torch.tensor(math.log(float(initial_scale)), dtype=torch.float32)
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
        if self.depth_residual_head is None:
            return predictions

        raw_depth = predictions["depth"].float().clamp_min(1e-4)
        if not self.depth_residual_enabled:
            return predictions
        windows = predictions.get("pano_windows")
        if windows is None:
            return predictions
        batch_size, num_views, _, height, width = windows.shape
        rgb = windows.reshape(batch_size * num_views, 3, height, width).float()
        log_depth = torch.log(raw_depth).permute(0, 1, 4, 2, 3).reshape(batch_size * num_views, 1, height, width)
        residual_input = torch.cat([rgb, log_depth], dim=1)
        delta = self.depth_residual_head(residual_input)
        if self.residual_max_log > 0:
            delta = torch.tanh(delta) * self.residual_max_log
        delta = delta.reshape(batch_size, num_views, 1, height, width).permute(0, 1, 3, 4, 2)
        predictions["raw_depth"] = predictions["depth"]
        predictions["depth_log_residual"] = delta
        predictions["depth"] = raw_depth * torch.exp(delta.float())
        return predictions

    def pred_depth_scale(self) -> torch.Tensor:
        return self.pred_depth_log_scale.float().exp()

    def sample_pano_windows(self, *args, **kwargs):
        return self.model.sample_pano_windows(*args, **kwargs)

    @property
    def pano_sampler(self):
        return self.model.pano_sampler


LearnablePredDepthScale = DepthPredictionAdapter


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
        param_groups.append({"params": regular_params, "lr": lr, "weight_decay": weight_decay})
    if scale_params:
        scale_lr = pred_depth_scale_lr if pred_depth_scale_lr is not None else lr
        param_groups.append({"params": scale_params, "lr": scale_lr, "weight_decay": 0.0})
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
        f"optimizer={stage.get('optimizer_type', 'default')} "
        f"luna_forward={stage.get('enable_luna_forward', 'default')} "
        f"depth_residual={stage.get('enable_depth_residual', 'default')} "
        f"trainable={trainable_count:,} frozen={frozen_count:,}"
    )


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
            enable_luna=True,
            luna_patch_layers=args.luna_patch_layers,
            luna_camera_layers=args.luna_camera_layers,
            sampler=sampler,
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
        enable_luna=True,
        luna_patch_layers=args.luna_patch_layers,
        luna_camera_layers=args.luna_camera_layers,
        sampler=sampler,
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
    amp_enabled = pano_images.device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float32
    with torch.autocast(device_type=pano_images.device.type, dtype=amp_dtype, enabled=amp_enabled):
        predictions = model(pano_images=pano_images, return_sampler_output=True)
        pred_depth_scale = predictions.get(
            "_pred_depth_scale",
            predictions["depth"].new_tensor(float(args.pred_depth_scale)),
        )
        pred_depth = predictions["depth"] * pred_depth_scale
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
        )
        loss_camera = loss_camera_dict["loss_camera"]
        loss = loss_depth + float(args.overlap_consistency_weight) * loss_overlap + args.camera_loss_weight * loss_camera
    loss.backward()

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=args.grad_clip,
        )
    optimizer.step()
    logged_pred_depth_scale = pred_depth_scale.detach()
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, DepthPredictionAdapter):
        logged_pred_depth_scale = unwrapped_model.pred_depth_scale().detach()
    return {
        "loss": loss.detach(),
        "loss_depth": loss_depth.detach(),
        "loss_overlap": loss_overlap.detach(),
        "loss_camera": loss_camera.detach(),
        "pred_depth_scale": logged_pred_depth_scale,
        **{key: value.detach() for key, value in loss_camera_dict.items() if key != "loss_camera"},
    }


def _is_rank0_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


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
        "loss/depth": "loss_depth",
        "loss/overlap": "loss_overlap",
        "loss/camera": "loss_camera",
        "loss/camera_t": "loss_camera_t",
        "loss/camera_r": "loss_camera_r",
        "loss/camera_fov": "loss_camera_fov",
        "loss/camera_consistency": "loss_camera_consistency",
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
    while weight.ndim > target.ndim:
        weight = weight.squeeze(-1)
    while weight.ndim < target.ndim:
        weight = weight.unsqueeze(-1)
    return weight.expand_as(target).clamp(min=float(sample_weight_min), max=float(sample_weight_max))


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
) -> Dict[str, torch.Tensor]:
    pred_pose = predictions.get("pose_enc")
    camera_meta = predictions.get("pano_camera_meta")
    if supervision_mode == "none" or pred_pose is None or camera_meta is None:
        zero = predictions["depth"].new_zeros(())
        return {
            "loss_camera": zero,
            "loss_camera_t": zero,
            "loss_camera_r": zero,
            "loss_camera_fov": zero,
            "loss_camera_consistency": zero,
        }

    pred_pose = torch.nan_to_num(pred_pose.float(), nan=0.0, posinf=0.0, neginf=0.0)
    rotations_c2w = camera_meta["rotations"].to(device=pred_pose.device, dtype=pred_pose.dtype)
    rotations_w2c = rotations_c2w.transpose(-1, -2).contiguous()
    pred_translation = pred_pose[..., :3]
    if supervision_mode == "pano_relative":
        loss_t, loss_consistency = pano_relative_translation_loss(
            pred_translation=pred_translation,
            rotations_c2w=rotations_c2w,
            batch=batch,
            position_mode=position_mode,
            consistency_weight=pano_consistency_weight,
        )
        zero = pred_translation.new_zeros(())
        return {
            "loss_camera": translation_weight * loss_t,
            "loss_camera_t": loss_t,
            "loss_camera_r": zero,
            "loss_camera_fov": zero,
            "loss_camera_consistency": loss_consistency,
        }
    if supervision_mode != "window_pose":
        raise ValueError(f"Unknown camera-supervision-mode: {supervision_mode}")

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
        loss_t = (pred_translation - target_translation).abs().mean()
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
    }


def pano_relative_translation_loss(
    pred_translation: torch.Tensor,
    rotations_c2w: torch.Tensor,
    batch: Dict,
    position_mode: str,
    consistency_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pano_position = batch.get("pano_position_m", None)
    if pano_position is None:
        zero = pred_translation.new_zeros(())
        return zero, zero

    target_centers = build_relative_pano_centers(
        pano_position=pano_position,
        reference=position_mode,
        device=pred_translation.device,
        dtype=pred_translation.dtype,
    )
    if target_centers.ndim != 3 or target_centers.shape[1] < 2:
        zero = pred_translation.new_zeros(())
        return zero, zero

    batch_size, total_views = pred_translation.shape[:2]
    num_panos = target_centers.shape[1]
    if total_views % num_panos != 0:
        raise ValueError(f"Cannot map {total_views} views to {num_panos} panos.")

    views_per_pano = total_views // num_panos
    pred_centers = (-(rotations_c2w @ pred_translation[..., None])[..., 0]).reshape(
        batch_size,
        num_panos,
        views_per_pano,
        3,
    )
    pred_pano_centers = pred_centers.mean(dim=2)
    loss_center = (pred_pano_centers - target_centers).abs().mean()
    loss_consistency = (pred_centers - pred_pano_centers[:, :, None, :]).abs().mean()
    return loss_center + consistency_weight * loss_consistency, loss_consistency


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
        }
        train_dense = mode in {"dense", "heads", "luna_dense", "luna_heads", "luna_residual_dense"}
        train_dense_tail = mode in {"luna_dense_tail", "luna_residual_dense_tail"}
        train_camera = mode in {"camera", "heads", "luna_heads"}
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
