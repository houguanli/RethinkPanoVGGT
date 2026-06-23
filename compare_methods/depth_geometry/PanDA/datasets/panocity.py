"""PanoCity paired depth dataset adapter for PanDA."""

from __future__ import annotations

import sys
from pathlib import Path

COMPARE_ROOT = Path(__file__).resolve().parents[3]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityDepthTorchDataset  # noqa: E402


class PanoCity(PanoCityDepthTorchDataset):
    """PanDA-compatible PanoCity depth dataset."""

    def __init__(self, *args, **kwargs):
        if "split" not in kwargs:
            kwargs["split"] = "train" if kwargs.get("is_training", False) else "test"
        kwargs.setdefault("target_mode", "metric")
        kwargs.setdefault("max_depth_meters", 10.0)
        super().__init__(*args, **kwargs)
