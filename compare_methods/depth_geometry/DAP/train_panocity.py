from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import networks.dap  # noqa: F401 - registers the DAP model.
from datasets.panocity import PanoCity
from networks.models import make


def setup_distributed():
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def build_dataset(config: Dict, split: str, smoke: bool = False):
    ds_cfg = config[f"{split}_dataset"]
    args = ds_cfg["args"]
    dataset = PanoCity(
        root_dir=ds_cfg.get("root_path") or os.environ.get("PANOCITY_ROOT"),
        list_file=ds_cfg.get("list_path"),
        height=args.get("height", 512),
        width=args.get("width", 1024),
        color_augmentation=args.get("augment_color", split == "train"),
        LR_filp_augmentation=args.get("augment_flip", split == "train"),
        yaw_rotation_augmentation=args.get("augment_rotation", split == "train"),
        repeat=args.get("repeat", 1),
        is_training=(split == "train"),
        split="smoke" if smoke else ("train" if split == "train" else "test"),
        max_samples=8 if smoke else args.get("max_samples"),
        depth_scale=args.get("depth_scale", 100.0),
        max_depth_meters=args.get("max_depth_meters", 100.0),
        target_mode=args.get("target_mode", "normalized"),
    )
    return dataset


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    mask_dx = mask[..., :, 1:] & mask[..., :, :-1]

    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_dy = mask[..., 1:, :] & mask[..., :-1, :]

    loss_x = torch.abs(pred_dx - target_dx)[mask_dx].mean() if mask_dx.any() else pred.sum() * 0.0
    loss_y = torch.abs(pred_dy - target_dy)[mask_dy].mean() if mask_dy.any() else pred.sum() * 0.0
    return loss_x + loss_y


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_cfg: Dict) -> Dict[str, torch.Tensor]:
    pred = outputs["pred_depth"]
    target = batch["gt_depth"]
    mask = batch["val_mask"].bool()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, target.shape[-2:], mode="bilinear", align_corners=True)
    if "pred_mask" in outputs:
        pred_valid = (1.0 - outputs["pred_mask"]) > 0.5
        if pred_valid.shape[-2:] != target.shape[-2:]:
            pred_valid = F.interpolate(pred_valid.float(), target.shape[-2:], mode="nearest").bool()
        mask = mask & pred_valid

    l1 = torch.abs(pred - target)[mask].mean() if mask.any() else pred.sum() * 0.0
    grad = gradient_loss(pred, target, mask)
    total = loss_cfg.get("l1_weight", 1.0) * l1 + loss_cfg.get("gradient_weight", 1.0) * grad
    return {"loss": total, "loss_l1": l1.detach(), "loss_grad": grad.detach()}


def maybe_load_weights(model, config: Dict, device):
    load_dir = config.get("load_weights_dir")
    if not load_dir:
        return
    path = Path(load_dir)
    if path.is_dir():
        path = path / "model.pth"
    if not path.exists():
        raise FileNotFoundError(f"load_weights_dir/model.pth not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported DAP checkpoint payload in {path}: {type(state)}")
    target = model.module if hasattr(model, "module") else model
    target_state = target.state_dict()

    candidates = state
    if not any(k in target_state for k in candidates):
        candidates = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}

    matched = {
        k: v
        for k, v in candidates.items()
        if k in target_state and hasattr(v, "shape") and v.shape == target_state[k].shape
    }
    if not matched:
        sample_ckpt = list(state.keys())[:5]
        sample_model = list(target_state.keys())[:5]
        raise RuntimeError(
            f"DAP checkpoint {path} did not match any model parameters. "
            f"checkpoint keys sample={sample_ckpt}; model keys sample={sample_model}"
        )
    missing = target.load_state_dict(matched, strict=False)
    print(f"Loaded DAP weights from {path}: matched={len(matched)}/{len(target_state)} missing={missing}")


def save_checkpoint(model, output_dir: Path, epoch: int, step: int, rank: int):
    if not is_main(rank):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    target = model.module if hasattr(model, "module") else model
    torch.save(target.state_dict(), output_dir / "model.pth")
    torch.save({"epoch": epoch, "step": step, "model": target.state_dict()}, output_dir / "checkpoint_last.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_panocity_4rtx5000.yaml")
    parser.add_argument("--output-dir", default="outputs/panocity_4rtx5000")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_dataset = build_dataset(config, "train", smoke=args.smoke)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if args.smoke else config["train_dataset"].get("batch_size", 1),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0 if args.smoke else config["train_dataset"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    model = make(config["model"]).to(device)
    maybe_load_weights(model, config, device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimizer"]["lr"]), weight_decay=float(config["optimizer"].get("weight_decay", 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("amp", True)) and torch.cuda.is_available())
    max_epochs = 1 if args.smoke else int(config.get("epoch_max", 20))
    max_steps = args.max_steps if args.max_steps is not None else (1 if args.smoke else None)
    output_dir = Path(args.output_dir)

    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"DAP PanoCity train samples: {len(train_dataset)}, world_size={world_size}, output={output_dir}")

    global_step = 0
    start = time.time()
    for epoch in range(max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, disable=not is_main(rank), desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(config.get("amp", True)) and torch.cuda.is_available()):
                outputs = model(batch["rgb"])
                losses = compute_loss(outputs, batch, config.get("loss", {}))
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if is_main(rank):
                pbar.set_postfix(loss=float(losses["loss"].detach().cpu()), l1=float(losses["loss_l1"].cpu()))
            if max_steps is not None and global_step >= max_steps:
                break
        save_checkpoint(model, output_dir, epoch, global_step, rank)
        if max_steps is not None and global_step >= max_steps:
            break

    if is_main(rank):
        print(f"DAP PanoCity finetune finished in {time.time() - start:.1f}s, steps={global_step}")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
