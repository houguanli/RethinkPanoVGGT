import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.train_pano_omega import resolve_checkpoint_reference


class CheckpointPathRelocationTest(unittest.TestCase):
    def test_uses_explicit_checkpoint_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            relocated = temp_path / "checkpoints" / "model.pt"
            relocated.parent.mkdir(parents=True)
            relocated.write_bytes(b"checkpoint")

            with patch.dict(os.environ, {"P2P_VGGT_CHECKPOINT_OVERRIDE": str(relocated)}):
                result = resolve_checkpoint_reference(
                    "/previous/machine/checkpoints/model.pt",
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
