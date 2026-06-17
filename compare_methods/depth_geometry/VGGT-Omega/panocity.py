"""PanoCity paired reader for VGGT-Omega depth experiments."""

from __future__ import annotations

import sys
from pathlib import Path

COMPARE_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex, read_depth, read_rgb, smoke_summary  # noqa: E402,F401


class PanoCityOmegaDataset(PanoCityPairedIndex):
    """CSV-backed PanoCity index for VGGT-Omega adapters."""

    pass
