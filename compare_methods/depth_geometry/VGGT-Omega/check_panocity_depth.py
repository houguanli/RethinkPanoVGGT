from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

COMPARE_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.depth_validation import depth_metrics, load_partial_state, tensor_stats  # noqa: E402
from common.panocity_paired import PanoCityDepthSequenceTorchDataset, PanoCityDepthTorchDataset  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config: Dict, split: str, max_samples: int | None) -> PanoCityDepthTorchDataset:
    train_cfg = config.get("train", {})
    root = os.environ.get("PANOCITY_ROOT") or config.get("panocity", {}).get("root")
    image_size = int(train_cfg.get("image_size", 512))
    num_views = int(train_cfg.get("num_views", 1))
    if split == "smoke":
        image_size = int(train_cfg.get("smoke_image_size", min(image_size, 256)))
        num_views = int(train_cfg.get("smoke_num_views", num_views))
    dataset_cls = PanoCityDepthSequenceTorchDataset if num_views > 1 else PanoCityDepthTorchDataset
    kwargs = dict(
        root_dir=root,
        height=image_size,
        width=image_size * 2,
        split="smoke" if split == "smoke" else ("train" if split == "train" else "test"),
        is_training=False,
        max_samples=max_samples,
        depth_scale=train_cfg.get("depth_scale"),
        max_depth_meters=float(train_cfg.get("max_depth_meters", 100.0)),
        target_mode=train_cfg.get("target_mode", "metric"),
        normalize_rgb=bool(train_cfg.get("normalize_rgb", False)),
    )
    if num_views > 1:
        kwargs["num_views"] = num_views
    return dataset_cls(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/vggt_omega_depth_panocity_4rtx5000.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "smoke"])
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = build_dataset(config, args.split, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    summary = {
        "method": "vggt_omega",
        "split": args.split,
        "samples": len(dataset),
        "target_mode": getattr(dataset, "target_mode", "metric"),
        "max_depth_meters": float(getattr(dataset, "max_depth_meters", 100.0)),
        "normalize_rgb": bool(getattr(dataset, "normalize_rgb", False)),
        "raw_depth_stats_m": tensor_stats(batch["raw_depth"], batch["val_mask"]),
        "target_depth_stats": tensor_stats(batch["gt_depth"], batch["val_mask"]),
    }

    if not args.no_forward:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGGTOmega().to(device).eval()
        matched = load_partial_state(model, str(checkpoint), device)
        summary["checkpoint"] = str(checkpoint.resolve())
        summary["checkpoint_matched_tensors"] = matched
        with torch.inference_mode():
            rgb = batch["rgb"].to(device)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            pred = model(rgb)["depth"].detach().cpu()
        summary["pred_metric_stats_m"] = tensor_stats(pred, batch["val_mask"])
        summary["metric_metrics_m"] = depth_metrics(pred, batch["raw_depth"], batch["val_mask"])

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
