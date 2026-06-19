"""Unified finetune/evaluate launcher for compare methods."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .method_registry import get_group, get_method, is_group, runnable_names
from .panocity_paired import smoke_summary


def _run(cmd, cwd: Path, dry_run: bool, timeout_seconds: int | None = None) -> None:
    print(f"[cmd] cwd={cwd} {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), start_new_session=True)
        return_code = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        hours = timeout_seconds / 3600 if timeout_seconds else 0
        raise SystemExit(
            f"Command exceeded finetune timeout ({hours:g} hours): {' '.join(cmd)}"
        ) from exc
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _run_finetune(spec, dry_run: bool, allow_unsupported: bool, timeout_seconds: int | None) -> None:
    if not spec.supports_native_finetune and not allow_unsupported:
        raise SystemExit(
            f"{spec.name} has no native public finetune implementation in this checkout. "
            "Run with --allow-unsupported-finetune to execute its adapter stub."
        )
    if not spec.finetune_command:
        raise SystemExit(f"No finetune command configured for {spec.name}")
    print(f"[finetune] method={spec.name}", flush=True)
    _run(spec.finetune_command, spec.path, dry_run, timeout_seconds)


def _run_evaluate(spec, dry_run: bool) -> None:
    if not spec.evaluate_command:
        raise SystemExit(f"No evaluate command configured for {spec.name}")
    print(f"[evaluate] method={spec.name}", flush=True)
    _run(spec.evaluate_command, spec.path, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=runnable_names())
    parser.add_argument("--stage", choices=["finetune", "evaluate", "both", "smoke"], default="both")
    parser.add_argument("--panocity-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unsupported-finetune", action="store_true")
    parser.add_argument(
        "--finetune-timeout-seconds",
        type=int,
        default=None,
        help="Optional wall-clock timeout applied only to the finetune command.",
    )
    args = parser.parse_args()

    if args.panocity_root:
        os.environ["PANOCITY_ROOT"] = args.panocity_root

    summary = smoke_summary(args.panocity_root, split="train")
    if is_group(args.method):
        group = get_group(args.method)
        print(
            json.dumps(
                {
                    "method_group": group.name,
                    "finetune_method": group.finetune_method,
                    "evaluate_methods": group.evaluate_methods,
                    "panocity": summary,
                },
                indent=2,
            ),
            flush=True,
        )
        if args.stage == "smoke":
            return 0

        if args.stage in {"finetune", "both"}:
            _run_finetune(
                get_method(group.finetune_method),
                args.dry_run,
                args.allow_unsupported_finetune,
                args.finetune_timeout_seconds,
            )
        if args.stage in {"evaluate", "both"}:
            for method_name in group.evaluate_methods:
                _run_evaluate(get_method(method_name), args.dry_run)
        return 0

    spec = get_method(args.method)
    print(json.dumps({"method": spec.name, "panocity": summary}, indent=2), flush=True)

    if args.stage == "smoke":
        return 0

    if args.stage in {"finetune", "both"}:
        _run_finetune(spec, args.dry_run, args.allow_unsupported_finetune, args.finetune_timeout_seconds)
    if args.stage in {"evaluate", "both"}:
        _run_evaluate(spec, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
