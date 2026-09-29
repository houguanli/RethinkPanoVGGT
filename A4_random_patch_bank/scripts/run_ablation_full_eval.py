#!/usr/bin/env python3
"""Independent evaluation: python scripts/run_ablation_full_eval.py CKPT OUT."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_pano_omega import resolve_checkpoint_reference


def run_eval(checkpoint, output, settings=None):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if settings is None:
        settings = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completion = "completion_head" in payload
    teacher = checkpoint
    if completion:
        teacher = resolve_checkpoint_reference(payload["omega_checkpoint"], checkpoint)
        if not teacher.is_file():
            raise FileNotFoundError(f"Omega teacher missing: {teacher}. Keep warmup and completion folders together.")
    env = os.environ.copy()
    env.update(
        PYTHON=sys.executable,
        GPUS=str(settings["gpus"]),
        CONFIG=str(ROOT / settings["train_config"]),
        DATASET_ROOT=str(settings["dataset_root"]),
        CHECKPOINT=str(teacher),
        ERP_COMPLETION_CHECKPOINT=str(checkpoint) if completion else "",
        EVAL_OUT=str(output),
        TRAIN_LOSS_CSV=str(checkpoint.parent / "loss.csv"),
    )
    env.update({("EVAL_DATASETS" if key == "datasets" else key.upper()): str(value)
                for key, value in settings["eval"].items()})
    output.mkdir(parents=True, exist_ok=True)
    (output / "eval_inputs.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "omega_checkpoint": str(teacher),
        "completion": completion, "settings": settings,
        "geora_ablation": payload.get("geora_ablation", payload.get("args", {}).get("geora_ablation")),
    }, indent=2))
    subprocess.run(["bash", str(ROOT / "scripts/run_multipano_mixed4_eval_4gpu.sh")],
                   cwd=ROOT, env=env, check=True)
    summary = output / "validation_mixed4_by_dataset_valtestfull_summary.json"
    if not summary.is_file() or not summary.stat().st_size:
        raise RuntimeError(f"Evaluation returned without a summary: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        print(run_eval(args.checkpoint, args.output_dir))
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode or 1)


if __name__ == "__main__":
    main()
