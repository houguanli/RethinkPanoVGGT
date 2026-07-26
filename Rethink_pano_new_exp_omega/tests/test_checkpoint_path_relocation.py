import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.train_pano_omega import resolve_checkpoint_reference


class CheckpointPathRelocationTest(unittest.TestCase):
    def test_relocates_home_aoki_to_storage_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            storage_root = temp_path / "whitehole" / "AOKI"
            relocated = storage_root / "RethinkPanoVGGT_omega" / "ckpt" / "model.pt"
            relocated.parent.mkdir(parents=True)
            relocated.write_bytes(b"checkpoint")

            with patch.dict(os.environ, {"AOKI_STORAGE_ROOT": str(storage_root)}):
                result = resolve_checkpoint_reference(
                    "/home/aoki/RethinkPanoVGGT_omega/ckpt/model.pt",
                    temp_path / "run" / "last.pt",
                )

            self.assertEqual(result, relocated)

    def test_resolves_project_relative_checkpoint(self):
        project_root = Path(__file__).resolve().parents[1]
        relative = Path("tests") / "checkpoint_reference_fixture.pt"
        fixture = project_root / relative
        fixture.write_bytes(b"checkpoint")
        try:
            with patch.dict(os.environ, {}, clear=True):
                result = resolve_checkpoint_reference(relative, Path("/tmp/run/last.pt"))
            self.assertEqual(result.resolve(), fixture.resolve())
        finally:
            fixture.unlink()


if __name__ == "__main__":
    unittest.main()
