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

    pass
