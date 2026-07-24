#!/usr/bin/env python3
"""Run ablation training and, on success, immediately start full evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_LAUNCHER = PROJECT_ROOT / "scripts" / "run_ablation_full_eval.py"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "ablation_mixed4_full_eval.yaml"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (1/0, true/false, yes/no, on/off), got: {value!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a training command and launch full mixed4 evaluation only after "
            "training exits successfully and writes a checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument(
        "--auto-eval",
        type=parse_bool,
        default=True,
        help="Explicitly enable or disable evaluation after successful training.",
    )
    parser.add_argument("--eval-launcher", type=Path, default=DEFAULT_EVAL_LAUNCHER)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "train_command",
        nargs=argparse.REMAINDER,
        help="Training command, separated from lifecycle options by --.",
    )
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now(), **values}
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized.pop(0)
    if not normalized:
        raise ValueError("A training command is required after --.")
    return normalized


def run_and_tee(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = normalize_command(args.train_command)
    checkpoint = args.checkpoint.expanduser().resolve()
    eval_output = args.eval_output.expanduser().resolve()
    train_log = args.train_log.expanduser().resolve()
    status_json = args.status_json.expanduser().resolve()
    eval_launcher = args.eval_launcher.expanduser().resolve()
    eval_config = args.eval_config.expanduser().resolve()

    write_status(
        status_json,
        state="training",
        auto_eval=args.auto_eval,
        checkpoint=str(checkpoint),
        eval_output=str(eval_output),
        train_command=command,
    )
    train_returncode = run_and_tee(command, train_log)
    if train_returncode != 0:
        write_status(
            status_json,
            state="training_failed",
            auto_eval=args.auto_eval,
            train_returncode=train_returncode,
            checkpoint=str(checkpoint),
            eval_started=False,
        )
        print(
            f"[train-then-eval] training failed with exit code {train_returncode}; "
            "full eval was not started.",
            file=sys.stderr,
        )
        return train_returncode

    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        write_status(
            status_json,
            state="checkpoint_missing",
            auto_eval=args.auto_eval,
            train_returncode=0,
            checkpoint=str(checkpoint),
            eval_started=False,
        )
        print(
            f"[train-then-eval] training returned success but checkpoint is missing or empty: "
            f"{checkpoint}",
            file=sys.stderr,
        )
        return 3

    if not args.auto_eval:
        write_status(
            status_json,
            state="eval_disabled",
            auto_eval=False,
            train_returncode=0,
            checkpoint=str(checkpoint),
            eval_started=False,
        )
        print("[train-then-eval] training succeeded; automatic full eval is disabled.")
        return 0

    eval_command = [
        args.python,
        str(eval_launcher),
        str(checkpoint),
        str(eval_output),
        "--eval-config",
        str(eval_config),
    ]
    write_status(
        status_json,
        state="evaluating",
        auto_eval=True,
        train_returncode=0,
        checkpoint=str(checkpoint),
        eval_output=str(eval_output),
        eval_started=True,
        eval_command=eval_command,
    )
    print(
        f"[train-then-eval] training succeeded; starting full eval with checkpoint: "
        f"{checkpoint}"
    )
    eval_completed = subprocess.run(
        eval_command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    if eval_completed.returncode != 0:
        write_status(
            status_json,
            state="eval_failed",
            auto_eval=True,
            train_returncode=0,
            eval_returncode=eval_completed.returncode,
            checkpoint=str(checkpoint),
            eval_output=str(eval_output),
            eval_started=True,
        )
        print(
            f"[train-then-eval] full eval failed with exit code "
            f"{eval_completed.returncode}.",
            file=sys.stderr,
        )
        return eval_completed.returncode

    summary = eval_output / "validation_mixed4_by_dataset_valtestfull_summary.json"
    if not summary.is_file() or summary.stat().st_size == 0:
        write_status(
            status_json,
            state="eval_output_missing",
            auto_eval=True,
            train_returncode=0,
            eval_returncode=0,
            checkpoint=str(checkpoint),
            eval_output=str(eval_output),
            eval_started=True,
        )
        print(
            f"[train-then-eval] eval returned success but summary is missing or empty: "
            f"{summary}",
            file=sys.stderr,
        )
        return 4

    write_status(
        status_json,
        state="completed",
        auto_eval=True,
        train_returncode=0,
        eval_returncode=0,
        checkpoint=str(checkpoint),
        eval_output=str(eval_output),
        summary=str(summary),
        eval_started=True,
    )
    print(f"[train-then-eval] full eval completed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
