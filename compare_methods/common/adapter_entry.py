"""Per-method finetune/evaluate adapter entrypoint helpers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .panocity_paired import smoke_summary


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run(command, cwd: Path, dry_run: bool) -> None:
    print(f"[cmd] cwd={cwd} {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(cwd), check=True)


def run_adapter(default_config: Path, default_stage: str = "finetune") -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--stage", choices=["finetune", "evaluate", "smoke"], default=default_stage)
    parser.add_argument("--panocity-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unsupported-finetune", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    if args.panocity_root:
        os.environ["PANOCITY_ROOT"] = args.panocity_root

    raw_method_root = Path(cfg["method"]["path"]).expanduser()
    method_root = raw_method_root if raw_method_root.is_absolute() else (config_path.parents[1] / raw_method_root)
    method_root = method_root.resolve()
    print(json.dumps({"config": str(config_path), "summary": smoke_summary(args.panocity_root)}, indent=2), flush=True)

    if args.stage == "smoke":
        return 0

    stage_cfg = cfg[args.stage]
    if args.stage == "finetune" and not bool(stage_cfg.get("supported", True)) and not args.allow_unsupported_finetune:
        raise SystemExit(stage_cfg.get("unsupported_reason", f"{cfg['method']['name']} finetune is not supported."))

    command = stage_cfg.get("command")
    if not command:
        raise SystemExit(f"No command configured for stage={args.stage} in {config_path}")
    _run([str(part) for part in command], method_root, args.dry_run)
    return 0


def main(default_config: str, default_stage: str = "finetune") -> int:
    return run_adapter(Path(default_config), default_stage=default_stage)


if __name__ == "__main__":
    sys.exit(run_adapter(Path("panocity_4rtx5000.yaml")))
