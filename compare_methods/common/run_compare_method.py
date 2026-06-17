"""Unified finetune/evaluate launcher for compare methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .method_registry import get_method, method_names
from .panocity_paired import smoke_summary


def _run(cmd, cwd: Path, dry_run: bool) -> None:
    print(f"[cmd] cwd={cwd} {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=method_names())
    parser.add_argument("--stage", choices=["finetune", "evaluate", "both", "smoke"], default="both")
    parser.add_argument("--panocity-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unsupported-finetune", action="store_true")
    args = parser.parse_args()

    spec = get_method(args.method)
    if args.panocity_root:
        os.environ["PANOCITY_ROOT"] = args.panocity_root

    summary = smoke_summary(args.panocity_root, split="train")
    print(json.dumps({"method": spec.name, "panocity": summary}, indent=2), flush=True)

    if args.stage == "smoke":
        return 0

    if args.stage in {"finetune", "both"}:
        if not spec.supports_native_finetune and not args.allow_unsupported_finetune:
            raise SystemExit(
                f"{spec.name} has no native public finetune implementation in this checkout. "
                "Run with --allow-unsupported-finetune to execute its adapter stub."
            )
        _run(spec.finetune_command or [], spec.path, args.dry_run)
    if args.stage in {"evaluate", "both"}:
        _run(spec.evaluate_command or [], spec.path, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
