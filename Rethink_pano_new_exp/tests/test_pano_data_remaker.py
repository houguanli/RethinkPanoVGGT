import tempfile
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pano_data_remaker import find_rgb_depth_normal_items, remake_one_item


def test_pano_data_remaker_numbered_output():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        out = root / "out"
        src.mkdir()
        out.mkdir()

        rgb = np.zeros((8, 16, 3), dtype=np.uint8)
        rgb[..., 1] = 128
        depth = np.full((8, 16), 123, dtype=np.uint16)
        normal = np.zeros((8, 16, 3), dtype=np.uint8)
        normal[..., 2] = 255

        cv2.imwrite(str(src / "x_100_y_-200_rgb.png"), rgb)
        cv2.imwrite(str(src / "x_100_y_-200_depth.png"), depth)
        cv2.imwrite(str(src / "x_100_y_-200_normal.png"), normal)

        items = find_rgb_depth_normal_items(src, depth_suffix="_depth.png")
        meta = remake_one_item(
            item=items[0],
            out_dir=out,
            item_id=0,
            input_depth_scale=0.01,
            invalid_depth_raw_min=0.0,
            invalid_depth_raw_max=65000.0,
            max_depth_m=80.0,
            output_depth_scale=100.0,
            xy_scale=0.01,
            z_m=0.5,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            depth_suffix_channel=None,
            image_ext=".png",
        )

        sample = out / "00000"
        assert meta["id"] == "00000"
        assert (sample / "rgb.png").exists()
        assert (sample / "depth.png").exists()
        assert (sample / "normal.png").exists()
        assert (sample / "camera_6dof.txt").exists()
        assert np.loadtxt(sample / "pose_c2w.txt").shape == (4, 4)


if __name__ == "__main__":
    test_pano_data_remaker_numbered_output()
    print("pano data remaker smoke ok")
