from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "ckpt" / "vggt_omega_1b_512.pt"


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

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
