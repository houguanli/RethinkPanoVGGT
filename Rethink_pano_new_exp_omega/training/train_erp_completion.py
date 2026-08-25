#!/usr/bin/env python3
"""Train the lightweight full-ERP remaining-band head with frozen Omega teacher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_pano_omega import (
    build_dataset,
    build_model,
    load_checkpoint,
    parse_args as parse_omega_args,
    set_seed,
)
from vggt_omega.models.erp_completion import (
    ERPRemainingBandHead,
    soft_coverage_distance,
    spherical_pixel_weights,
    splat_omega_window_depth_to_erp,
)


CSV_FIELDS = [
    "step", "elapsed_minutes", "loss", "loss_remaining", "loss_boundary",
    "loss_distill", "loss_smooth", "remaining_valid_ratio", "coverage_ratio",
    "pred_finite_ratio", "lr", "stage",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Canonical Omega/data config.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--omega-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--duration-minutes", type=float, required=True)
    parser.add_argument("--stage", choices=["main", "refine"], default="main")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--max-log-residual", type=float, default=2.5)
    parser.add_argument("--blend-width-pixels", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "none"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--progress-bar", dest="progress_bar", action="store_true", default=True)
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if distributed:
            raise ValueError("Distributed ERP completion currently requires CUDA/NCCL")
        device = torch.device("cpu")
    is_main = rank == 0
    set_seed(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    omega_args = parse_omega_args(["--config", str(args.config)])
    if args.dataset_root is not None:
        omega_args.dataset_root = str(args.dataset_root)
    # Completion always uses the distribution-aligned canonical Omega windows.
    omega_args.pitch_degrees = "-15"
    omega_args.fov_degrees = 75.0
    omega_args.fov_x_degrees = 75.0
    omega_args.fov_y_degrees = 75.0
    omega_args.num_yaw = 4
    omega_args.window_size = 384
    omega_args.pano_height = 512
    omega_args.pano_width = 1024
    omega_args.pano_sample_mode = "fixed_neighborhood"
    omega_args.randomize_pano_order = False
    omega_args.batch_size = 1
    omega_args.num_workers = args.num_workers
    dataset = build_dataset(omega_args, (512, 1024))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    omega = build_model(omega_args).to(device)
    if args.base_checkpoint is not None and args.base_checkpoint.resolve() != args.omega_checkpoint.resolve():
        load_checkpoint(omega, args.base_checkpoint, strict=False)
    load_checkpoint(omega, args.omega_checkpoint, strict=False)
    omega.eval()
    for parameter in omega.parameters():
        parameter.requires_grad_(False)

    head = ERPRemainingBandHead(args.head_width, args.max_log_residual).to(device)
    global_step = 0
    total_elapsed_before = 0.0
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        head.load_state_dict(resume_payload["completion_head"])
        global_step = int(resume_payload.get("step", 0))
        total_elapsed_before = float(resume_payload.get("total_elapsed_minutes", 0.0))
        if is_main:
            print(f"[INFO] resumed completion head from {args.resume} at step {global_step}", flush=True)
    if distributed:
        head = DistributedDataParallel(head, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_payload is not None and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])

    log_path = args.output_dir / "loss.csv"
    append = log_path.exists() and log_path.stat().st_size > 0
    log_file = log_path.open("a", newline="", encoding="utf-8") if is_main else None
    writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS) if log_file is not None else None
    if writer is not None and not append:
        writer.writeheader()
    start = time.monotonic()
    stop_reason = "duration"
    amp_enabled = device.type == "cuda" and args.amp_dtype == "bfloat16"
    iterator = iter(loader)
    loader_epoch = 0
    progress = None
    last_progress_elapsed = 0.0
    if is_main and args.progress_bar:
        progress = tqdm(
            total=max(1, int(args.duration_minutes * 60.0)),
            desc=f"ERP completion {args.stage}",
            unit="s",
            dynamic_ncols=True,
            leave=True,
        )
    head.train()
    try:
        while True:
            elapsed = (time.monotonic() - start) / 60.0
            reached_duration = elapsed >= args.duration_minutes
            reached_max_steps = args.max_steps > 0 and global_step >= args.max_steps
            should_stop = reached_duration or reached_max_steps
            if distributed:
                stop_tensor = torch.tensor(int(should_stop), device=device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                if reached_max_steps:
                    stop_reason = "max_steps"
                elif not reached_duration:
                    stop_reason = "peer_duration"
                break
            try:
                batch = next(iterator)
            except StopIteration:
                loader_epoch += 1
                if sampler is not None:
                    sampler.set_epoch(loader_epoch)
                iterator = iter(loader)
                batch = next(iterator)
            pano_image = batch["pano_image"].to(device, non_blocking=True)
            pano_depth = batch["pano_depth"].to(device, non_blocking=True)
            common_mask = batch["pano_rgb_depth_common_mask"].to(device, non_blocking=True)
            batch_size, num_panos = pano_image.shape[:2]
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                prediction = omega(pano_images=pano_image, return_sampler_output=True)
                splat = splat_omega_window_depth_to_erp(
                    prediction["depth"], prediction["pano_camera_meta"], num_panos=num_panos,
                    erp_height=512, erp_width=1024,
                )
            rgb = pano_image.reshape(batch_size * num_panos, 3, 512, 1024)
            gt = pano_depth.reshape(batch_size * num_panos, 1, 512, 1024)
            gt_valid = common_mask.reshape(batch_size * num_panos, 1, 512, 1024).bool()
            rgb = F.interpolate(rgb, size=(args.height, args.width), mode="bilinear", align_corners=False)
            gt = F.interpolate(torch.nan_to_num(gt, nan=0.0, posinf=0.0), size=(args.height, args.width), mode="nearest")
            gt_valid = F.interpolate(gt_valid.float(), size=(args.height, args.width), mode="nearest") > 0.5
            coverage_float = splat.valid_mask.float()
            omega_sum = F.interpolate(splat.depth * coverage_float, size=(args.height, args.width), mode="area")
            omega_weight = F.interpolate(coverage_float, size=(args.height, args.width), mode="area")
            omega_depth = omega_sum / omega_weight.clamp_min(1e-6)
            coverage = omega_weight > 0.05

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = head(rgb, omega_depth, coverage, blend_width_pixels=args.blend_width_pixels)
                losses = completion_losses(output, gt, gt_valid, rgb, args.stage, args.blend_width_pixels)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            row = {
                "step": global_step,
                "elapsed_minutes": total_elapsed_before + elapsed,
                "lr": optimizer.param_groups[0]["lr"],
                "stage": args.stage,
                **{key: float(value.detach().float().cpu()) for key, value in losses.items()},
            }
            if writer is not None:
                writer.writerow(row)
                log_file.flush()
            if progress is not None:
                elapsed_seconds = time.monotonic() - start
                update = max(0, int(elapsed_seconds) - int(last_progress_elapsed))
                if update:
                    progress.update(update)
                    last_progress_elapsed = elapsed_seconds
                progress.set_postfix(
                    step=global_step,
                    loss=f"{row['loss']:.4f}",
                    remaining=f"{row['loss_remaining']:.4f}",
                    boundary=f"{row['loss_boundary']:.4f}",
                    coverage=f"{row['coverage_ratio']:.3f}",
                    finite=f"{row['pred_finite_ratio']:.3f}",
                )
            if is_main and global_step % args.log_every == 0:
                message = (
                    f"[TRAIN completion:{args.stage}] step={global_step} elapsed={elapsed:.1f}m "
                    f"loss={row['loss']:.5f} remaining={row['loss_remaining']:.5f} "
                    f"boundary={row['loss_boundary']:.5f} coverage={row['coverage_ratio']:.3f}"
                )
                if progress is not None:
                    progress.write(message)
                else:
                    print(message, flush=True)
            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(args.output_dir / f"step_{global_step:06d}.pt", head, optimizer, args, global_step, total_elapsed_before + elapsed)
    finally:
        elapsed = (time.monotonic() - start) / 60.0
        if is_main:
            if progress is not None:
                progress.close()
            save_checkpoint(args.output_dir / "last.pt", head, optimizer, args, global_step, total_elapsed_before + elapsed)
            if log_file is not None:
                log_file.close()
            print(f"[INFO] completion stop_reason={stop_reason} step={global_step} checkpoint={args.output_dir / 'last.pt'}", flush=True)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def completion_losses(output, target, target_valid, rgb, stage: str, blend_width: int):
    pred = output["depth"].float().clamp_min(1e-6)
    completion = output["completion_depth"].float().clamp_min(1e-6)
    coverage = output["coverage"] > 0.5
    valid = target_valid & torch.isfinite(target) & (target > 0)
    remaining = valid & ~coverage
    boundary = remaining & (soft_coverage_distance(coverage.float(), blend_width) > 0)
    # Scale-only alignment preserves Omega-relative geometry without leaking an absolute scale at inference.
    aligned = []
    for index in range(pred.shape[0]):
        values = valid[index]
        if values.any():
            shift = (torch.log(target[index][values]) - torch.log(pred[index][values])).median().detach()
        else:
            shift = pred.new_tensor(0.0)
        aligned.append(pred[index] * torch.exp(shift))
    aligned_pred = torch.stack(aligned)
    error = F.smooth_l1_loss(torch.log(aligned_pred), torch.log(target.clamp_min(1e-6)), reduction="none", beta=0.2)
    sphere = spherical_pixel_weights(pred.shape[-2], pred.shape[-1], pred.device, pred.dtype)
    loss_remaining = masked_weighted_mean(error, remaining, sphere)
    loss_boundary = masked_weighted_mean(error, boundary, sphere)
    teacher_valid = coverage & torch.isfinite(output["base_depth"]) & (output["base_depth"] > 0)
    distill_error = (torch.log(completion) - torch.log(output["base_depth"].float().clamp_min(1e-6))).abs()
    loss_distill = masked_weighted_mean(distill_error, teacher_valid, sphere)
    log_depth = torch.log(completion)
    dx = torch.roll(log_depth, shifts=-1, dims=-1) - log_depth
    dy = log_depth[..., 1:, :] - log_depth[..., :-1, :]
    rgb_dx = (torch.roll(rgb.float(), shifts=-1, dims=-1) - rgb.float()).abs().mean(1, keepdim=True)
    rgb_dy = (rgb.float()[..., 1:, :] - rgb.float()[..., :-1, :]).abs().mean(1, keepdim=True)
    smooth_x = (dx.abs() * torch.exp(-4.0 * rgb_dx) * (~coverage).float()).mean()
    smooth_y = (dy.abs() * torch.exp(-4.0 * rgb_dy) * (~coverage[..., 1:, :]).float()).mean()
    loss_smooth = smooth_x + smooth_y
    if stage == "refine":
        total = loss_remaining + 1.0 * loss_boundary + 0.03 * loss_distill + 0.03 * loss_smooth
    else:
        total = loss_remaining + 0.35 * loss_boundary + 0.08 * loss_distill + 0.02 * loss_smooth
    return {
        "loss": total,
        "loss_remaining": loss_remaining,
        "loss_boundary": loss_boundary,
        "loss_distill": loss_distill,
        "loss_smooth": loss_smooth,
        "remaining_valid_ratio": remaining.float().mean(),
        "coverage_ratio": coverage.float().mean(),
        "pred_finite_ratio": torch.isfinite(pred).float().mean(),
    }


def masked_weighted_mean(value, mask, weight):
    effective = mask.float() * weight
    return (value * effective).sum() / effective.sum().clamp_min(1.0)


def save_checkpoint(path, head, optimizer, args, step, elapsed):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "erp_remaining_band_head_v1",
        "completion_head": unwrap_completion_head(head).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "total_elapsed_minutes": float(elapsed),
        "omega_checkpoint": str(args.omega_checkpoint),
        "base_checkpoint": str(args.base_checkpoint) if args.base_checkpoint else None,
        "config": str(args.config),
        "head_args": {
            "width": int(args.head_width),
            "max_log_residual": float(args.max_log_residual),
            "height": int(args.height),
            "width_erp": int(args.width),
            "blend_width_pixels": int(args.blend_width_pixels),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def unwrap_completion_head(head):
    return head.module if isinstance(head, DistributedDataParallel) else head


if __name__ == "__main__":
    main()
