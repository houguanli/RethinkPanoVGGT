from pathlib import Path
from typing import Sequence, Optional, Union

import torch
import torch.nn as nn


DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "ckpt" / "vggt_omega_1b_512.pt"


def resolve_checkpoint_path(checkpoint_path: Optional[Union[str, Path]] = None) -> Path:
    path = DEFAULT_CHECKPOINT_PATH if checkpoint_path is None else Path(checkpoint_path)
    return path.expanduser().resolve()


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Optional[Union[str, Path]] = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    path = resolve_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f"[INFO] loaded checkpoint = {path}")
    print(f"[INFO] missing_keys = {len(missing)}; unexpected_keys = {len(unexpected)}")
    print_checkpoint_key_analysis(missing, unexpected)
    return missing, unexpected


def print_checkpoint_key_analysis(missing: Sequence[str], unexpected: Sequence[str], max_items: int = 20) -> None:
    if missing:
        print(f"[INFO] missing_key_prefixes = {_format_key_prefix_counts(missing)}")
        print(f"[INFO] missing_keys_sample = {list(missing)[:max_items]}")
    if unexpected:
        print(f"[INFO] unexpected_key_prefixes = {_format_key_prefix_counts(unexpected)}")
        print(f"[INFO] unexpected_keys_sample = {list(unexpected)[:max_items]}")


def _format_key_prefix_counts(keys: Sequence[str], depth: int = 2, max_groups: int = 12) -> str:
    counts: dict[str, int] = {}
    for key in keys:
        parts = str(key).split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else str(key)
        counts[prefix] = counts.get(prefix, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_groups]
    return ", ".join(f"{prefix}:{count}" for prefix, count in ranked)
