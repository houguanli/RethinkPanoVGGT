#!/usr/bin/env python3
"""Run the shared full mixed4 ablation evaluation from two path arguments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "ablation_mixed4_full_eval.yaml"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_multipano_mixed4_eval_4gpu.sh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full mixed4 evaluation. Only CHECKPOINT and OUTPUT_DIR are "
            "required; evaluation hyperparameters come from the YAML config."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Final training checkpoint, normally last.pt.")
    parser.add_argument("output_dir", type=Path, help="Directory for merged evaluation outputs.")
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG,
        help="Evaluation hyperparameter YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate all settings without starting evaluation.",
    )
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config must contain a mapping: {path}")
    return payload


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    return payload


def checkpoint_args(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args", {})
    return args if isinstance(args, dict) else {}


def resolve_existing_path(raw: Any, candidates: list[Path], label: str) -> Path:
    if raw not in (None, "", "checkpoint"):
        raw_path = Path(str(raw)).expanduser()
        candidates = [raw_path, *candidates]
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate
    rendered = "\n  ".join(str(path) for path in checked)
    raise FileNotFoundError(f"Unable to resolve {label}; checked:\n  {rendered}")


def resolve_training_config(eval_config: dict[str, Any], payload: dict[str, Any]) -> Path:
    configured = eval_config.get("model", {}).get("training_config", "checkpoint")
    saved = checkpoint_args(payload).get("config")
    saved_path = Path(str(saved)) if saved not in (None, "") else None
    candidates: list[Path] = []
    if saved_path is not None:
        candidates.extend(
            [
                saved_path,
                PROJECT_ROOT / saved_path,
                PROJECT_ROOT / "configs" / saved_path.name,
            ]
        )
    return resolve_existing_path(configured, candidates, "training config")


def resolve_dataset_root(eval_config: dict[str, Any], payload: dict[str, Any]) -> Path:
    configured = eval_config.get("data", {}).get("dataset_root", "checkpoint")
    saved = checkpoint_args(payload).get("dataset_root")
    candidates = [Path(str(saved))] if saved not in (None, "") else []
    path = resolve_existing_path(configured, candidates, "dataset root")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {path}")
    return path


def validate_delta_dependencies(checkpoint: Path, payload: dict[str, Any]) -> None:
    if "model_delta" not in payload:
        return
    args = checkpoint_args(payload)
    dependencies = {
        "foundation checkpoint": payload.get("foundation_checkpoint") or args.get("base_checkpoint"),
        "warm-up/base checkpoint": payload.get("base_checkpoint") or args.get("checkpoint"),
    }
    missing = []
    for label, raw in dependencies.items():
        if raw in (None, ""):
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_file() or path.resolve() == checkpoint.resolve():
            missing.append(f"{label}: {path}")
    if missing:
        detail = "\n  ".join(missing)
        raise FileNotFoundError(
            "The trainable-delta checkpoint has unavailable dependencies:\n"
            f"  {detail}\n"
            "Keep the original foundation and warm-up checkpoints at the paths "
            "saved during training before starting evaluation."
        )


def detect_gpus(value: Any) -> str:
    requested = str(value or "auto").strip()
    if requested.lower() != "auto":
        return requested
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < 1:
        raise RuntimeError("No CUDA GPU is visible.")
    return ",".join(str(index) for index in range(count))


def as_env_bool(value: Any) -> str:
    return "1" if bool(value) else "0"


def build_environment(
    eval_config: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
    training_config: Path,
    dataset_root: Path,
) -> dict[str, str]:
    data = eval_config.get("data", {})
    geometry = eval_config.get("geometry", {})
    runtime = eval_config.get("runtime", {})
    train_loss_csv = checkpoint.parent / "loss.csv"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "GPUS": detect_gpus(runtime.get("gpus", "auto")),
            "CONFIG": str(training_config),
            "DATASET_ROOT": str(dataset_root),
            "LUNA_OUT": str(checkpoint.parent),
            "CHECKPOINT": str(checkpoint),
            "TRAIN_LOSS_CSV": str(train_loss_csv),
            "EVAL_OUT": str(output_dir),
            "LIMIT_PER_DATASET": str(int(data.get("limit_per_dataset", 0))),
            "EVAL_DATASETS": str(data.get("datasets", "all")),
            "NUM_WORKERS_PER_GPU": str(int(runtime.get("num_workers_per_gpu", 2))),
            "AMP_DTYPE": str(runtime.get("amp_dtype", "bfloat16")),
            "SEED": str(int(runtime.get("seed", 123))),
            "PROGRESS_EVERY": str(int(runtime.get("progress_every", 10))),
            "PROGRESS_INTERVAL_SECONDS": str(int(runtime.get("progress_interval_seconds", 30))),
            "SHOW_PROGRESS": as_env_bool(runtime.get("show_progress", True)),
            "PROGRESS_STYLE": str(runtime.get("progress_style", "bar")),
            "PROGRESS_BAR_WIDTH": str(int(runtime.get("progress_bar_width", 32))),
            "SHOW_GPU_PROC": as_env_bool(runtime.get("show_gpu_processes", False)),
            "SAMPLE_POLICY": str(data.get("sample_policy", "anchor")),
            "PANO_COUNT_POLICY": str(geometry.get("pano_count_policy", "panovggt")),
            "DATASET_PANO_COUNTS": str(geometry.get("dataset_pano_counts", "")),
            "CAMERA_EVAL_MAX_PANOS": str(int(geometry.get("camera_eval_max_panos", 3))),
            "NUM_YAW": str(int(geometry.get("num_yaw", 8))),
            "ERP_LATITUDE_LIMIT_DEG": str(float(geometry.get("erp_latitude_limit_deg", 75))),
        }
    )
    return env


def public_settings(env: dict[str, str], eval_config: Path) -> dict[str, Any]:
    keys = [
        "GPUS",
        "CONFIG",
        "DATASET_ROOT",
        "CHECKPOINT",
        "TRAIN_LOSS_CSV",
        "EVAL_OUT",
        "LIMIT_PER_DATASET",
        "EVAL_DATASETS",
        "NUM_WORKERS_PER_GPU",
        "AMP_DTYPE",
        "SAMPLE_POLICY",
        "PANO_COUNT_POLICY",
        "DATASET_PANO_COUNTS",
        "CAMERA_EVAL_MAX_PANOS",
        "NUM_YAW",
        "ERP_LATITUDE_LIMIT_DEG",
    ]
    return {"eval_config": str(eval_config), **{key: env[key] for key in keys}}


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    eval_config_path = args.eval_config.expanduser().resolve()
    eval_config = load_yaml(eval_config_path)
    payload = load_checkpoint_payload(checkpoint)
    validate_delta_dependencies(checkpoint, payload)
    training_config = resolve_training_config(eval_config, payload)
    dataset_root = resolve_dataset_root(eval_config, payload)
    env = build_environment(
        eval_config,
        checkpoint,
        output_dir,
        training_config,
        dataset_root,
    )
    print(json.dumps(public_settings(env, eval_config_path), indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", str(EVAL_SCRIPT)], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

