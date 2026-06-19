from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

COMPARE_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityDepthTorchDataset  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def depth_loss(pred, target, mask):
    if pred.ndim == 5:
        pred = pred[:, 0]
    if pred.shape[-1] == 1:
        pred = pred.permute(0, 3, 1, 2)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, target.shape[-2:], mode="bilinear", align_corners=True)
    pred = torch.clamp(pred, min=1e-6, max=100.0)
    return torch.abs(pred - target)[mask.bool()].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/vggt_omega_camera_panocity_4rtx5000.yaml")
    parser.add_argument("--checkpoint", default="../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt")
    parser.add_argument("--output-dir", default="outputs/panocity_4rtx5000")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    root = os.environ.get("PANOCITY_ROOT") or cfg.get("panocity", {}).get("root")
    image_size = int(train_cfg.get("image_size", 512 if not args.smoke else 256))

    dataset = PanoCityDepthTorchDataset(
        root_dir=root,
        height=image_size,
        width=image_size * 2,
        split="smoke" if args.smoke else "train",
        is_training=True,
        max_depth_meters=float(train_cfg.get("max_depth_meters", 100.0)),
    )
    loader = DataLoader(dataset, batch_size=1 if args.smoke else int(train_cfg.get("batch_size", 1)), shuffle=True, num_workers=0 if args.smoke else int(train_cfg.get("num_workers", 4)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGTOmega().to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-5)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = 1 if args.smoke else int(cfg.get("budget", {}).get("max_epochs", 20))
    max_steps = args.max_steps if args.max_steps is not None else (1 if args.smoke else None)
    step = 0
    for epoch in range(max_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            rgb = batch["rgb"].to(device).unsqueeze(1)
            target = batch["gt_depth"].to(device)
            mask = batch["val_mask"].to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(rgb)["depth"]
            loss = depth_loss(pred, target, mask)
            loss.backward()
            optim.step()
            step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"VGGT-Omega PanoCity finetune finished, steps={step}, output={output_dir}")


if __name__ == "__main__":
    main()
