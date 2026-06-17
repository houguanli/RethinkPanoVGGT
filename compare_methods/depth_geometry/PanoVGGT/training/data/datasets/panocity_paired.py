"""PanoCity paired CSV adapter for PanoVGGT depth training configs."""

from __future__ import annotations

import sys
from pathlib import Path

COMPARE_ROOT = Path(__file__).resolve().parents[5]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex, read_depth, read_rgb, smoke_summary  # noqa: E402,F401


class PanoCityPairedDataset(PanoCityPairedIndex):
    """Lightweight paired index; use existing panocity.py for full PanoVGGT training."""

    pass
