"""Per-method finetune/evaluate adapter entrypoint helpers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .dinov2_assets import prepare_panovggt_dinov2_asset
from .panocity_paired import smoke_summary


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


PATH_FLAGS = {
    "--checkpoint",
    "--config",
    "--load",
    "--load-weights",
    "--load_weights",
    "--load_weights_dir",
    "--pretrained",
    "--resume",
}
CONFIG_ASSET_KEYS = {"resume_checkpoint_path", "load_weights_dir"}


def _iter_config_assets(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in CONFIG_ASSET_KEYS and child:
                yield key, str(child)
            yield from _iter_config_assets(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_config_assets(child)


def _looks_generated_path(path: Path) -> bool:
    return bool({"logs", "outputs", "tmp"}.intersection(set(path.parts)))


def _check_command_path(label: str, value: str, cwd: Path, *, allow_missing_generated: bool) -> None:
    if "${" in value:
        print(f"[asset] {label}: {value} (deferred interpolation)", flush=True)
        return
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    resolved = path if path.is_absolute() else (cwd / path).resolve()
    print(f"[asset] {label}: {value} -> {resolved}", flush=True)
    if resolved.exists():
        return
    if allow_missing_generated and _looks_generated_path(resolved):
        print(f"[asset] {label}: missing now; expected to be produced by a prior finetune stage", flush=True)
        return
    raise SystemExit(f"Missing asset for {label}: {value} -> {resolved}")


def _validate_config_asset_paths(config_path: Path, cwd: Path, *, allow_missing_generated: bool) -> None:
    if config_path.suffix not in {".yaml", ".yml"} or not config_path.exists():
        return
    try:
        cfg = _load_yaml(config_path)
    except Exception:
        return
    for key, value in _iter_config_assets(cfg):
        _check_command_path(f"{config_path.name}:{key}", value, cwd, allow_missing_generated=allow_missing_generated)


def _resolve_command_paths(command, cwd: Path, *, allow_missing_generated: bool = False) -> list[str]:
    resolved_command = [str(part) for part in command]
    for index, part in enumerate(resolved_command[:-1]):
        if part not in PATH_FLAGS:
            continue
        value = resolved_command[index + 1]
        if not value or value.startswith("-"):
            continue
        if part == "--config" and "/" not in value and "\\" not in value and not Path(value).suffix:
            print(f"[asset] command {part}: {value} (config name)", flush=True)
            _validate_config_asset_paths(cwd / "training" / "config" / f"{value}.yaml", cwd, allow_missing_generated=allow_missing_generated)
            continue
        _check_command_path(f"command {part}", value, cwd, allow_missing_generated=allow_missing_generated)
        if part == "--config":
            config_path = Path(os.path.expandvars(os.path.expanduser(value)))
            config_path = config_path if config_path.is_absolute() else (cwd / config_path).resolve()
            _validate_config_asset_paths(config_path, cwd, allow_missing_generated=allow_missing_generated)
    return resolved_command


def _run(command, cwd: Path, dry_run: bool) -> None:
    command = _resolve_command_paths(command, cwd, allow_missing_generated=dry_run)
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
    repo_root = config_path.parents[2]
    print(json.dumps({"config": str(config_path), "summary": smoke_summary(args.panocity_root)}, indent=2), flush=True)

    if args.stage == "smoke":
        return 0

    stage_cfg = cfg[args.stage]
    if args.stage == "finetune" and not bool(stage_cfg.get("supported", True)) and not args.allow_unsupported_finetune:
        raise SystemExit(stage_cfg.get("unsupported_reason", f"{cfg['method']['name']} finetune is not supported."))

    command = stage_cfg.get("command")
    if not command:
        raise SystemExit(f"No command configured for stage={args.stage} in {config_path}")
    if args.stage == "finetune" and str(cfg["method"]["name"]).startswith("panovggt"):
        prepare_panovggt_dinov2_asset(repo_root, dry_run=args.dry_run)
    _run([str(part) for part in command], method_root, args.dry_run)
    return 0


def main(default_config: str, default_stage: str = "finetune") -> int:
    return run_adapter(Path(default_config), default_stage=default_stage)


if __name__ == "__main__":
    sys.exit(run_adapter(Path("panocity_4rtx5000.yaml")))
