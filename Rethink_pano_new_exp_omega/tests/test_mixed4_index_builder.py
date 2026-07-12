import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_mixed4_official_indexes import build_matterport3d


class MatterportIndexBuilderTest(unittest.TestCase):
    def test_groups_panoramas_by_parsed_room(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            scan = "scan1"
            scan_dir = root / scan
            for folder in ("pano_depth", "pano_skybox_color", "pano_poses"):
                (scan_dir / folder).mkdir(parents=True, exist_ok=True)
            rooms = {"0": {"room_name": "room_a", "panoramas": ["a", "b"]},
                     "1": {"room_name": "room_b", "panoramas": ["c", "d"]}}
            (root / "parsed_json").mkdir()
            (root / "parsed_json" / f"{scan}.json").write_text(json.dumps(rooms), encoding="utf-8")
            for pano_id in ("a", "b", "c", "d"):
                (scan_dir / "pano_depth" / f"{pano_id}.png").touch()
                (scan_dir / "pano_skybox_color" / f"{pano_id}.jpg").touch()
                (scan_dir / "pano_poses" / f"{pano_id}.txt").touch()

            result = build_matterport3d(root)

            rows = json.loads((root / "cache" / "matterport3d_train_index.json").read_text())
            self.assertEqual([(row[1], row[2], row[3]) for row in rows], [
                ("0", "room_a", ["a", "b"]),
                ("1", "room_b", ["c", "d"]),
            ])
            self.assertEqual(result["train"]["rows"], 2)
            self.assertEqual(result["train"]["expanded"], 4)


if __name__ == "__main__":
    unittest.main()
