"""Train VGGT-Omega LUNA on converted panorama data."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
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
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA  # noqa: E402


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
    parser.add_argument("--enable-camera-head", action="store_true")
    parser.add_argument("--save-last", action="store_true")
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

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            loss_dict = train_step(model, batch, optimizer, grad_clip=args.grad_clip)
            print(
                f"[TRAIN] epoch={epoch + 1} step={global_step} "
                f"loss={loss_dict['loss'].item():.6f} depth={loss_dict['loss_depth'].item():.6f}"
            )
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    if args.save_last and not args.smoke:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.output_dir / "last.pt"
        torch.save({"model": model.state_dict(), "args": vars(args), "step": global_step}, ckpt_path)
        print(f"[INFO] saved checkpoint = {ckpt_path}")


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
            enable_camera=False,
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
    )


def train_step(
    model: VGGTOmega_LUNA,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> Dict[str, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    pano_images = batch["pano_image"]
    pano_depths = batch["pano_depth"]

    target_depth = sample_depth_windows(model, pano_depths)
    predictions = model(pano_images=pano_images)
    pred_depth = predictions["depth"]

    loss_depth = masked_log_l1_depth(pred_depth, target_depth)
    loss = loss_depth
    loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=grad_clip,
        )
    optimizer.step()
    return {"loss": loss.detach(), "loss_depth": loss_depth.detach()}


@torch.no_grad()
def sample_depth_windows(model: VGGTOmega_LUNA, pano_depths: torch.Tensor) -> torch.Tensor:
    depth_as_rgb = pano_depths.repeat(1, 3, 1, 1)
    sampled = model.pano_sampler(depth_as_rgb).windows[:, :, :1]
    return sampled.permute(0, 1, 3, 4, 2).contiguous()


def masked_log_l1_depth(pred_depth: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
    target_depth = target_depth.to(device=pred_depth.device, dtype=pred_depth.dtype)
    pred_depth = torch.nan_to_num(pred_depth, nan=1e-4, posinf=1e4, neginf=1e-4)
    target_depth = torch.nan_to_num(target_depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.isfinite(target_depth) & (target_depth > 0)
    if not bool(valid.any()):
        return pred_depth.new_zeros(())
    pred = pred_depth.clamp_min(1e-4)
    target = target_depth.clamp_min(1e-4)
    return (torch.log(pred[valid]) - torch.log(target[valid])).abs().mean()


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
