from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

COMPARE_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex  # noqa: E402
from panovggt.models.panovggt_model import PanoVGGTModel  # noqa: E402


def _default_checkpoint() -> Path:
    smoke_ckpt = Path("outputs/panocity_deep_smoke/ckpts/checkpoint.pt")
    if smoke_ckpt.exists():
        return smoke_ckpt
    return Path("../../../ckpt/PanoVGGT/model.pt")


def _load_rgb(path: Path, height: int, width: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


def _build_model(cfg) -> PanoVGGTModel:
    model_cfg = cfg.model
    aggregator_cfg = OmegaConf.to_container(model_cfg.aggregator, resolve=True)
    aggregator_cfg["load_dinov2_pretrained"] = False
    aggregator_cfg["allow_dinov2_download"] = False
    return PanoVGGTModel(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        enable_camera=model_cfg.enable_camera,
        enable_depth=model_cfg.enable_depth,
        enable_point=model_cfg.enable_point,
        aggregator=aggregator_cfg,
    )


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"PanoVGGT checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
            break
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in payload.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] checkpoint={checkpoint} missing={len(missing)} unexpected={len(unexpected)}")


def _summarize_outputs(outputs) -> dict:
    summary = {}
    for key, value in outputs.items():
        if torch.is_tensor(value):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict):
            summary[key] = _summarize_outputs(value)
        else:
            summary[key] = type(value).__name__
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="smoke")
    parser.add_argument("--config", default="panocity_smoke")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/panocity_eval_smoke")
    parser.add_argument("--max-images", type=int, default=2)
    args = parser.parse_args()

    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name=args.config)
    OmegaConf.resolve(cfg)

    height = int(cfg.img_size)
    width = height * 2
    index = PanoCityPairedIndex(split="smoke", max_samples=max(2, args.max_images))
    images = torch.stack([_load_rgb(index[i].rgb_path, height, width) for i in range(args.max_images)], dim=0)
    images = images.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device).eval()
    _load_checkpoint(model, Path(args.checkpoint) if args.checkpoint else _default_checkpoint(), device)

    with torch.no_grad():
        outputs = model(images.to(device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": args.stage,
        "records": [str(index[i].rgb_path) for i in range(args.max_images)],
        "input_shape": list(images.shape),
        "outputs": _summarize_outputs(outputs),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
