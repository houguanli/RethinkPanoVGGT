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

import networks.dap  # noqa: F401,E402 - registers the DAP model.
from common.depth_validation import depth_metrics, load_partial_state, tensor_stats  # noqa: E402
from datasets.panocity import PanoCity  # noqa: E402
from networks.models import make  # noqa: E402


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config: Dict, split: str, max_samples: int | None) -> PanoCity:
    ds_key = "train_dataset" if split == "train" else "val_dataset"
    ds_cfg = config[ds_key]
    args = ds_cfg["args"]
    return PanoCity(
        root_dir=ds_cfg.get("root_path") or os.environ.get("PANOCITY_ROOT"),
        list_file=ds_cfg.get("list_path"),
        height=args.get("height", 512),
        width=args.get("width", 1024),
        color_augmentation=False,
        LR_filp_augmentation=False,
        yaw_rotation_augmentation=False,
        repeat=1,
        is_training=False,
        split="smoke" if split == "smoke" else ("train" if split == "train" else "test"),
        max_samples=max_samples if max_samples is not None else args.get("max_samples"),
        depth_scale=args.get("depth_scale", 100.0),
        max_depth_meters=args.get("max_depth_meters", 100.0),
        target_mode=args.get("target_mode", "normalized"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_panocity_4rtx5000.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test", "smoke"])
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = build_dataset(config, args.split, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    target_mode = getattr(dataset, "target_mode", "normalized")
    max_depth = float(getattr(dataset, "max_depth_meters", 100.0))
    summary = {
        "method": "dap",
        "split": args.split,
        "samples": len(dataset),
        "target_mode": target_mode,
        "max_depth_meters": max_depth,
        "raw_depth_stats_m": tensor_stats(batch["raw_depth"], batch["val_mask"]),
        "target_depth_stats": tensor_stats(batch["gt_depth"], batch["val_mask"]),
    }

    if not args.no_forward:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make(config["model"]).to(device).eval()
        checkpoint = args.checkpoint or config.get("load_weights_dir")
        if checkpoint:
            matched = load_partial_state(model, checkpoint, device)
            summary["checkpoint"] = str(Path(checkpoint).resolve())
            summary["checkpoint_matched_tensors"] = matched
        with torch.inference_mode():
            outputs = model(batch["rgb"].to(device))
        pred_native = outputs["pred_depth"].detach().cpu()
        summary["pred_native_stats"] = tensor_stats(pred_native, batch["val_mask"])
        summary["native_metrics"] = depth_metrics(pred_native, batch["gt_depth"], batch["val_mask"])
        pred_metric = pred_native * max_depth if target_mode == "normalized" else pred_native
        summary["metric_metrics_m"] = depth_metrics(pred_metric, batch["raw_depth"], batch["val_mask"])

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
