#!/usr/bin/env python3
"""One independent experiment: 2h Omega + 8h completion + 2h refine + full eval."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ablation_full_eval import run_eval


def build_plan(args, settings):
    run_dir = args.run_dir.resolve()
    config = ROOT / settings["train_config"]
    teacher = args.warmup_checkpoint.resolve() if args.warmup_checkpoint else run_dir / "warmup/last.pt"
    foundation = args.base_checkpoint or Path(settings["base_checkpoint"])
    dataset = args.dataset_root or Path(settings["dataset_root"])
    nproc = args.nproc_per_node or int(settings["nproc_per_node"])
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"]
    common = ["--config", str(config), "--dataset-root", str(dataset),
              "--base-checkpoint", str(foundation),
              "--num-workers", str(settings["num_workers"]), "--progress-bar"]
    plan = []
    if not args.warmup_checkpoint:
        plan.append(("warmup", prefix + ["training/train_pano_omega.py"] + common + [
            "--checkpoint", str(foundation), "--output-dir", str(run_dir / "warmup"),
            "--tensorboard-dir", str(run_dir / "warmup/tensorboard"),
            "--debug-dir", str(run_dir / "warmup/debug"),
            "--max-duration-minutes", str(settings["warmup_minutes"]),
            "--save-every-steps", str(settings["save_every_steps"]),
            "--no-inherit-checkpoint-training-defaults",
        ], teacher))
    for stage, folder, duration in (
        ("main", "completion_main", settings["completion_main_minutes"]),
        ("refine", "completion_refine", settings["completion_refine_minutes"]),
    ):
        command = prefix + ["training/train_erp_completion.py"] + common + [
            "--omega-checkpoint", str(teacher), "--output-dir", str(run_dir / folder),
            "--duration-minutes", str(duration), "--stage", stage,
            "--save-every", str(settings["save_every_steps"]),
        ]
        if stage == "refine":
            command += ["--resume", str(run_dir / "completion_main/last.pt"), "--lr", "5e-5"]
        plan.append((folder, command, run_dir / folder / "last.pt"))
    return plan


def run_pipeline(args, settings):
    plan = build_plan(args, settings)
    for stage, command, _ in plan:
        print(f"[{stage}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Output is not empty: {run_dir}. Use a new run directory; old results are never overwritten.")
    for value in (args.base_checkpoint or settings["base_checkpoint"],):
        if not Path(value).is_file():
            raise FileNotFoundError(f"Foundation checkpoint missing: {value}")
    if not Path(args.dataset_root or settings["dataset_root"]).is_dir():
        raise FileNotFoundError("Dataset root missing; edit configs/pipeline.yaml or pass --dataset-root")
    if args.warmup_checkpoint and not args.warmup_checkpoint.is_file():
        raise FileNotFoundError(args.warmup_checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "settings": settings,
             "warmup_checkpoint": str(args.warmup_checkpoint) if args.warmup_checkpoint else None,
             "stages": []}
    status_file = run_dir / "pipeline_status.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(settings["gpus"])
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    try:
        for stage, command, checkpoint in plan:
            state["active_stage"] = stage
            status_file.write_text(json.dumps(state, indent=2))
            stage_dir = checkpoint.parent
            stage_dir.mkdir(parents=True, exist_ok=True)
            with (stage_dir / "command.txt").open("w") as log:
                # Inherited console keeps the training progress visible; the
                # canonical trainer also records loss.csv/status in stage_dir.
                log.write(shlex.join(command) + "\n")
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            if not checkpoint.is_file() or not checkpoint.stat().st_size:
                raise RuntimeError(f"{stage} returned without checkpoint: {checkpoint}")
            state["stages"].append({"stage": stage, "checkpoint": str(checkpoint)})
        if settings["auto_eval"] and not args.no_auto_eval:
            state["active_stage"] = "eval"
            status_file.write_text(json.dumps(state, indent=2))
            eval_settings = dict(settings, dataset_root=str(args.dataset_root or settings["dataset_root"]))
            state["summary"] = str(run_eval(plan[-1][2], run_dir / "eval_full", eval_settings))
        state["state"] = "completed"
    except BaseException as error:
        state["state"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        status_file.write_text(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "logs/run")
    parser.add_argument("--warmup-checkpoint", type=Path, help="Skip the 2h warmup and reuse this teacher.")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--no-auto-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        settings["gpus"] = os.environ["CUDA_VISIBLE_DEVICES"]
    if args.nproc_per_node and args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be positive")
    if len(str(settings["gpus"]).split(",")) != (args.nproc_per_node or settings["nproc_per_node"]):
        parser.error("GPU list and nproc_per_node differ")
    try:
        run_pipeline(args, settings)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode or 1)


if __name__ == "__main__":
    main()
