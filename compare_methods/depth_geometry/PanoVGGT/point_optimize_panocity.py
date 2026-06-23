from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parents[2] / "camera_pose" / "PanoVGGT" / "point_optimize_panocity.py"
    if str(target.parent) not in sys.path:
        sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
