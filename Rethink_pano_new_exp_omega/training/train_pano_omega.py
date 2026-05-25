"""Train VGGT-Omega LUNA on converted panorama data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import PanoVKittiOmegaDataset  # noqa: E402
from vggt_omega.models.heads.dense_head import DenseHead  # noqa: E402
from vggt_omega.models.layers import PatchEmbed  # noqa: E402
from vggt_omega.models.layers.pano_position import pinhole_rays  # noqa: E402
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402
from vggt_omega.utils.rotation import mat_to_quat  # noqa: E402


DEFAULT_DATASET_ROOT = Path("whitehole/AOKI/datasets/PANO_LUNA_omega")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ckpt" / "vggt_omega_1b_512.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VGGT-Omega LUNA on converted pano VKitti-style data.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pano_omega_luna")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--num-yaw", type=int, default=8)
    parser.add_argument("--pitch-degrees", type=str, default="0")
    parser.add_argument("--fov-degrees", type=float, default=75.0)
    parser.add_argument("--pano-height", type=int, default=0, help="Optional resize height before sampling.")
    parser.add_argument("--pano-width", type=int, default=0, help="Optional resize width before sampling.")

    parser.add_argument("--trainable", choices=["luna", "heads", "luna_heads", "all"], default="luna_heads")
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--enable-camera-head", dest="enable_camera_head", action="store_true", default=True)
    parser.add_argument("--disable-camera-head", dest="enable_camera_head", action="store_false")
    parser.add_argument("--camera-loss-weight", type=float, default=1.0)
    parser.add_argument("--camera-translation-weight", type=float, default=1.0)
    parser.add_argument("--camera-rotation-weight", type=float, default=1.0)
    parser.add_argument("--camera-fov-weight", type=float, default=0.1)
    parser.add_argument("--camera-position-mode", choices=["local_zero", "world"], default="local_zero")
    parser.add_argument(
        "--pred-depth-scale",
        type=float,
        default=1.0,
        help="Fixed metric calibration applied to predicted Z-depth before loss/export.",
    )
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--max-duration-minutes", type=float, default=0.0)
    parser.add_argument("--log-csv", type=Path, default=None)
    parser.add_argument("--loss-plot", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny generated-data training step.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="pano_omega_smoke_") as tmp:
            smoke_root = Path(tmp) / "converted_pano4vggt_omega"
            write_smoke_dataset(smoke_root)
            args.dataset_root = smoke_root
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


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    pano_size = (args.pano_height, args.pano_width) if args.pano_height > 0 and args.pano_width > 0 else None
    dataset = PanoVKittiOmegaDataset(root=args.dataset_root, pano_size=pano_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args).to(device)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, strict=args.strict_checkpoint)

    trainable_count, frozen_count = configure_trainable(model, args.trainable)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[INFO] dataset_root = {dataset.root}")
    print(f"[INFO] samples = {len(dataset)}")
    print(f"[INFO] device = {device}")
    print(f"[INFO] trainable_params = {trainable_count:,}; frozen_params = {frozen_count:,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = args.log_csv or (args.output_dir / "loss.csv")
    loss_plot = args.loss_plot or (args.output_dir / "loss_curve.png")
    max_duration_seconds = args.max_duration_minutes * 60.0 if args.max_duration_minutes > 0 else None
    started_at = time.time()
    metrics_history = []

    model.train()
    global_step = 0
    stop_reason = "max_steps"
    try:
        for epoch in range(args.epochs):
            for batch in loader:
                if max_duration_seconds is not None and time.time() - started_at >= max_duration_seconds:
                    stop_reason = "max_duration"
                    break

                global_step += 1
                batch = move_batch_to_device(batch, device)
                loss_dict = train_step(model, batch, optimizer, args)
                elapsed_seconds = time.time() - started_at
                metrics = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "elapsed_seconds": elapsed_seconds,
                    "loss": float(loss_dict["loss"].item()),
                    "loss_depth": float(loss_dict["loss_depth"].item()),
                    "loss_camera": float(loss_dict["loss_camera"].item()),
                    "loss_camera_t": float(loss_dict.get("loss_camera_t", torch.tensor(0.0)).item()),
                    "loss_camera_r": float(loss_dict.get("loss_camera_r", torch.tensor(0.0)).item()),
                    "loss_camera_fov": float(loss_dict.get("loss_camera_fov", torch.tensor(0.0)).item()),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                metrics_history.append(metrics)
                append_loss_csv(log_csv, metrics)
                print(
                    f"[TRAIN] epoch={epoch + 1} step={global_step} "
                    f"elapsed={elapsed_seconds / 60.0:.2f}m "
                    f"loss={metrics['loss']:.6f} depth={metrics['loss_depth']:.6f} "
                    f"camera={metrics['loss_camera']:.6f}",
                    flush=True,
                )

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0 and not args.smoke:
                    save_checkpoint(args.output_dir / f"step_{global_step:06d}.pt", model, args, global_step)

                if global_step >= args.max_steps:
                    stop_reason = "max_steps"
                    break

            if global_step >= args.max_steps or stop_reason == "max_duration":
                break
        else:
            stop_reason = "epochs_complete"
    finally:
        if metrics_history:
            save_loss_plot(loss_plot, metrics_history)
        if args.save_last and not args.smoke:
            ckpt_path = args.output_dir / "last.pt"
            save_checkpoint(ckpt_path, model, args, global_step)
            print(f"[INFO] saved checkpoint = {ckpt_path}")
        print(f"[INFO] stop_reason = {stop_reason}; steps = {global_step}")


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
            enable_pano_global_token=True,
            enable_luna=True,
            luna_patch_layers=[23],
            luna_camera_layers=[23],
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
        enable_pano_global_token=True,
        enable_luna=True,
        luna_patch_layers="second_half",
        luna_camera_layers=[23],
        sampler=sampler,
        checkpoint_path=None,
    )


def train_step(
    model: VGGTOmega_LUNA,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    pano_images = batch["pano_image"]
    pano_depths = batch["pano_depth"]

    target_depth, target_valid = sample_depth_targets(model, pano_depths)
    predictions = model(pano_images=pano_images, return_sampler_output=True)
    pred_depth = predictions["depth"] * args.pred_depth_scale

    loss_depth = masked_log_l1_depth(pred_depth, target_depth, target_valid)
    loss_camera_dict = camera_alignment_loss(
        predictions=predictions,
        batch=batch,
        translation_weight=args.camera_translation_weight,
        rotation_weight=args.camera_rotation_weight,
        fov_weight=args.camera_fov_weight,
        position_mode=args.camera_position_mode,
    )
    loss_camera = loss_camera_dict["loss_camera"]
    loss = loss_depth + args.camera_loss_weight * loss_camera
    loss.backward()

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=args.grad_clip,
        )
    optimizer.step()
    return {
        "loss": loss.detach(),
        "loss_depth": loss_depth.detach(),
        "loss_camera": loss_camera.detach(),
        **{key: value.detach() for key, value in loss_camera_dict.items() if key != "loss_camera"},
    }


@torch.no_grad()
def sample_depth_windows(model: VGGTOmega_LUNA, pano_depths: torch.Tensor) -> torch.Tensor:
    target_depth, _ = sample_depth_targets(model, pano_depths)
    return target_depth


@torch.no_grad()
def sample_depth_targets(model: VGGTOmega_LUNA, pano_depths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ERP range-depth, convert it to pinhole Z-depth, and keep a hard validity mask."""
    source_valid = (pano_depths > 0).to(dtype=pano_depths.dtype)
    packed_depth_valid = torch.cat([pano_depths * source_valid, source_valid, source_valid.new_zeros(source_valid.shape)], dim=1)
    sampled = model.pano_sampler(packed_depth_valid, interpolation_mode="bilinear")
    sampled_depth = sampled.windows[:, :, :1]
    sampled_weight = sampled.windows[:, :, 1:2]
    range_depth = (sampled_depth / sampled_weight.clamp_min(1e-6)).permute(0, 1, 3, 4, 2).contiguous()

    valid_as_rgb = source_valid.repeat(1, 3, 1, 1)
    sampled_valid = model.pano_sampler(valid_as_rgb, interpolation_mode="nearest").windows[:, :, :1]
    valid = sampled_valid.permute(0, 1, 3, 4, 2).contiguous() > 0.5
    valid = valid & (sampled_weight.permute(0, 1, 3, 4, 2).contiguous() > 1e-6)

    z_factor = build_window_z_factor(sampled.camera_meta, range_depth.shape[2], range_depth.shape[3])
    target_z = range_depth * z_factor[..., None]
    valid = valid & torch.isfinite(target_z) & (target_z > 0)
    target_z = torch.where(valid, target_z, torch.zeros_like(target_z))
    return target_z, valid


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
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, path)


def masked_log_l1_depth(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target_depth = target_depth.to(device=pred_depth.device, dtype=pred_depth.dtype)
    pred_depth = torch.nan_to_num(pred_depth, nan=1e-4, posinf=1e4, neginf=1e-4)
    target_depth = torch.nan_to_num(target_depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.isfinite(target_depth) & (target_depth > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=target_depth.device, dtype=torch.bool)
    if not bool(valid.any()):
        return pred_depth.new_zeros(())
    pred = pred_depth.clamp_min(1e-4)
    target = target_depth.clamp_min(1e-4)
    return (torch.log(pred[valid]) - torch.log(target[valid])).abs().mean()


def camera_alignment_loss(
    predictions: Dict,
    batch: Dict,
    translation_weight: float,
    rotation_weight: float,
    fov_weight: float,
    position_mode: str,
) -> Dict[str, torch.Tensor]:
    pred_pose = predictions.get("pose_enc")
    camera_meta = predictions.get("pano_camera_meta")
    if pred_pose is None or camera_meta is None:
        zero = predictions["depth"].new_zeros(())
        return {
            "loss_camera": zero,
            "loss_camera_t": zero,
            "loss_camera_r": zero,
            "loss_camera_fov": zero,
        }

    pred_pose = torch.nan_to_num(pred_pose.float(), nan=0.0, posinf=0.0, neginf=0.0)
    rotations_c2w = camera_meta["rotations"].to(device=pred_pose.device, dtype=pred_pose.dtype)
    rotations_w2c = rotations_c2w.transpose(-1, -2).contiguous()
    target_quat = F.normalize(mat_to_quat(rotations_w2c), dim=-1)

    pred_translation = pred_pose[..., :3]
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
    }


def build_target_translation(rotations_w2c: torch.Tensor, batch: Dict, position_mode: str) -> torch.Tensor:
    if position_mode == "local_zero":
        return rotations_w2c.new_zeros(*rotations_w2c.shape[:2], 3)
    if position_mode != "world":
        raise ValueError(f"Unknown camera-position-mode: {position_mode}")

    pano_position = batch.get("pano_position_m", None)
    if pano_position is None:
        return rotations_w2c.new_zeros(*rotations_w2c.shape[:2], 3)
    center_world = pano_position.to(device=rotations_w2c.device, dtype=rotations_w2c.dtype)
    if center_world.ndim == 1:
        center_world = center_world[None]
    center_world = center_world[:, None, :, None]
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

        train_luna = mode in {"luna", "luna_heads"}
        train_heads = mode in {"heads", "luna_heads"}
        for name, param in model.named_parameters():
            if train_luna and (
                "luna_" in name or "pano_global" in name or "pano_geometry" in name
            ):
                param.requires_grad_(True)
            if train_heads and ("dense_head" in name or "camera_head" in name):
                param.requires_grad_(True)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    if trainable == 0:
        raise ValueError(f"No trainable parameters selected for mode {mode!r}")
    return trainable, frozen


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
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
    rgb_dir = root / "pano_smoke" / "clone" / "frames" / "rgb" / "Camera_0"
    depth_dir = root / "pano_smoke" / "clone" / "frames" / "depth" / "Camera_0"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    height, width = 32, 64
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    rgb = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 128, dtype=np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(rgb, mode="RGB").save(rgb_dir / "rgb_00000.jpg")

    depth_m = np.full((height, width), 2.0, dtype=np.float32)
    cv2.imwrite(str(depth_dir / "depth_00000.png"), np.round(depth_m * 100.0).astype(np.uint16))

    meta = {
        "scene_name": "pano_smoke",
        "pano_shape_hw": [height, width],
        "camera_alignment": {
            "alignment_policy": "smoke sample uses pano-local zero translation",
            "panorama_position_m_xyz": [0.0, 0.0, 0.0],
        },
        "depth_policy": {
            "unit": "meters",
            "invalid_depth_value": 0,
            "output_depth_scale": 100.0,
            "saved_depth_decode": "depth_m = uint16_png / 100.0",
        },
    }
    (root / "pano_smoke" / "clone" / "pano_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    (root / "sequence_list.txt").write_text(
        "pano_smoke/clone/frames/rgb/Camera_0\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
