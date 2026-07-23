"""LUNA configuration dataclass shared by the omega aggregator and top model."""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class LunaConfig:
    enable_luna: bool = True
    patch_layers: Optional[Sequence[int]] = None
    camera_layers: Optional[Sequence[int]] = None
    sphere_dim: int = 7
    camera_meta_dim: int = 16
    hidden_dim: Optional[int] = None
    patch_bank_mode: str = "aligned"
    patch_bank_shuffle_seed: Optional[int] = None


def default_luna_config() -> LunaConfig:
    """Default MVP config: insert lightweight residual adapters in the second half."""
    return LunaConfig()
