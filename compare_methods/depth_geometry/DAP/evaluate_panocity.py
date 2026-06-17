from pathlib import Path
import sys

COMPARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMPARE_ROOT))

from common.adapter_entry import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(str(COMPARE_ROOT / "configs" / "dap_panocity_4rtx5000.yaml"), "evaluate"))
