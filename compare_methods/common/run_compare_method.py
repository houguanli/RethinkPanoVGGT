"""Unified finetune/evaluate launcher for compare methods."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .dinov2_assets import prepare_panovggt_dinov2_asset
from .method_registry import REPO_ROOT, get_group, get_method, is_group, normalize_name, runnable_names
from .panocity_paired import smoke_summary


FORBIDDEN_RUNTIME_PATHS = ("/public/home/", "/home/tione/", "/hpc2hdd/home/")
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


def _runtime_config_paths(spec) -> list[Path]:
    paths = [spec.config]
    if spec.name.startswith("panovggt_"):
        paths.extend(
            [
                spec.path / "training" / "config" / "default.yaml",
                spec.path / "training" / "config" / "panocity_4rtx5000.yaml",
            ]
        )
    elif spec.name == "dap":
        paths.append(spec.path / "config" / "train_panocity_4rtx5000.yaml")
    elif spec.name == "panda":
        paths.append(spec.path / "config" / "metric_depth" / "train_panocity_4rtx5000.yaml")
    return list(dict.fromkeys(path for path in paths if path and path.is_file()))


def _iter_config_assets(value: Any, parent_key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in CONFIG_ASSET_KEYS and child:
                yield key, str(child)
            yield from _iter_config_assets(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_config_assets(child, parent_key)


def _iter_config_commands(value: Any) -> Iterable[list[Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "command" and isinstance(child, list):
                yield child
            yield from _iter_config_commands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_config_commands(child)


def _looks_generated_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool({"logs", "outputs", "tmp"}.intersection(parts))


def _check_asset_path(label: str, raw_value: str, cwd: Path, *, allow_missing_generated: bool = False) -> None:
    if "${" in raw_value:
        print(f"[asset] {label}: {raw_value} (deferred interpolation)", flush=True)
        return
    path = Path(os.path.expandvars(os.path.expanduser(raw_value)))
    resolved = path if path.is_absolute() else (cwd / path).resolve()
    print(f"[asset] {label}: {raw_value} -> {resolved}", flush=True)
    if resolved.exists():
        return
    if allow_missing_generated and _looks_generated_path(resolved):
        print(f"[asset] {label}: missing now; expected to be produced by a prior finetune stage", flush=True)
        return
    raise SystemExit(f"Missing asset for {label}: {raw_value} -> {resolved}")


def _validate_config_assets(spec) -> None:
    config_paths = _runtime_config_paths(spec)
    if not config_paths:
        return
    try:
        import yaml
    except Exception as exc:
        raise SystemExit(f"PyYAML is required to validate config assets: {exc}") from exc

    for path in config_paths:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for key, value in _iter_config_assets(payload):
            _check_asset_path(f"{path.relative_to(spec.path) if path.is_relative_to(spec.path) else path}:{key}", value, spec.path)
        for command in _iter_config_commands(payload):
            _resolve_command_paths(command, spec.path, allow_missing_generated=True)


def _validate_runtime_paths(spec) -> None:
    offenders = []
    for path in _runtime_config_paths(spec):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in FORBIDDEN_RUNTIME_PATHS:
            if forbidden in text:
                offenders.append(f"{path}: contains {forbidden}")
    if offenders:
        raise SystemExit(
            "Refusing to run with hard-coded external runtime paths:\n"
            + "\n".join(offenders)
            + "\nUse project-relative paths or environment variables in the PanoCity configs."
        )
    _validate_config_assets(spec)


def _resolve_command_paths(cmd, cwd: Path, *, allow_missing_generated: bool = False) -> list[str]:
    resolved_cmd = [str(part) for part in cmd]
    for index, part in enumerate(resolved_cmd[:-1]):
        if part not in PATH_FLAGS:
            continue
        value = resolved_cmd[index + 1]
        if not value or value.startswith("-"):
            continue
        if part == "--config" and "/" not in value and "\\" not in value and not Path(value).suffix:
            print(f"[asset] command {part}: {value} (config name)", flush=True)
            continue
        _check_asset_path(f"command {part}", value, cwd, allow_missing_generated=allow_missing_generated)
    return resolved_cmd


def _run(cmd, cwd: Path, dry_run: bool, timeout_seconds: int | None = None) -> None:
    cmd = _resolve_command_paths(cmd, cwd, allow_missing_generated=dry_run)
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


def _run_finetune(spec, dry_run: bool, allow_unsupported: bool, timeout_seconds: int | None, smoke: bool = False) -> None:
    _validate_runtime_paths(spec)
    if spec.name.startswith("panovggt_"):
        prepare_panovggt_dinov2_asset(REPO_ROOT, dry_run=dry_run)
    if not spec.supports_native_finetune and not allow_unsupported:
        raise SystemExit(
            f"{spec.name} has no native public finetune implementation in this checkout. "
            "Run with --allow-unsupported-finetune to execute its adapter stub."
        )
    command = spec.smoke_finetune_command if smoke and spec.smoke_finetune_command else spec.finetune_command
    if not command:
        raise SystemExit(f"No finetune command configured for {spec.name}")
    label = "deep_smoke_finetune" if smoke else "finetune"
    print(f"[{label}] method={spec.name}", flush=True)
    _run(command, spec.path, dry_run, timeout_seconds)


def _run_evaluate(spec, dry_run: bool, smoke: bool = False) -> None:
    _validate_runtime_paths(spec)
    command = spec.smoke_evaluate_command if smoke and spec.smoke_evaluate_command else spec.evaluate_command
    if not command:
        raise SystemExit(f"No evaluate command configured for {spec.name}")
    label = "deep_smoke_evaluate" if smoke else "evaluate"
    print(f"[{label}] method={spec.name}", flush=True)
    _run(command, spec.path, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=runnable_names())
    parser.add_argument("--stage", choices=["finetune", "evaluate", "both", "smoke", "deep_smoke"], default="both")
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
    args.method = normalize_name(args.method)

    summary = smoke_summary(args.panocity_root, split="train")
    if is_group(args.method):
        group = get_group(args.method)
        for method_name in {group.finetune_method, *group.evaluate_methods}:
            _validate_runtime_paths(get_method(method_name))
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

        if args.stage == "deep_smoke":
            _run_finetune(
                get_method(group.finetune_method),
                args.dry_run,
                args.allow_unsupported_finetune,
                args.finetune_timeout_seconds,
                smoke=True,
            )
            for method_name in group.evaluate_methods:
                _run_evaluate(get_method(method_name), args.dry_run, smoke=True)
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
    _validate_runtime_paths(spec)
    print(json.dumps({"method": spec.name, "panocity": summary}, indent=2), flush=True)

    if args.stage == "smoke":
        return 0

    if args.stage == "deep_smoke":
        _run_finetune(spec, args.dry_run, args.allow_unsupported_finetune, args.finetune_timeout_seconds, smoke=True)
        _run_evaluate(spec, args.dry_run, smoke=True)
        return 0

    if args.stage in {"finetune", "both"}:
        _run_finetune(spec, args.dry_run, args.allow_unsupported_finetune, args.finetune_timeout_seconds)
    if args.stage in {"evaluate", "both"}:
        _run_evaluate(spec, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
