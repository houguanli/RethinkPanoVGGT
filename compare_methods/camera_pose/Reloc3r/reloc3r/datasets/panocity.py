"""PanoCity paired dataset adapter for Reloc3r training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset

COMPARE_ROOT = Path(__file__).resolve().parents[4]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex  # noqa: E402


class PanoCityReloc3r(BaseStereoViewDataset):
    """Adjacent-pair PanoCity loader matching Reloc3r's two-view contract.

    PanoCity_paired currently provides RGB/depth paths, not calibrated camera
    poses. For finetune plumbing and weak pose adaptation, this dataset supplies
    identity cam2world placeholders and centered pinhole intrinsics. Replace
    `camera_pose` with calibrated labels when available.
    """

    def __init__(self, root=None, split="train", ROOT=None, resolution=512, max_samples=None, **kwargs):
        self.index = PanoCityPairedIndex(root=root or ROOT, split=split, max_samples=max_samples)
        super().__init__(split=split, resolution=resolution, **kwargs)

    def __len__(self):
        return max(0, len(self.index) - 1)

    def _get_views(self, idx, resolution, rng):
        records = [self.index[idx], self.index[idx + 1]]
        views = []
        for rec in records:
            image = Image.open(rec.rgb_path).convert("RGB")
            width, height = image.size
            focal = 0.5 * min(width, height)
            intrinsics = np.array(
                [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            image, intrinsics = self._crop_resize_if_necessary(
                image, intrinsics, resolution, rng=rng, info=(rec.block, rec.num_id)
            )
            views.append(
                dict(
                    img=image,
                    camera_pose=np.eye(4, dtype=np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="PanoCity",
                    label=rec.block,
                    instance=rec.num_id,
                )
            )
        return views


def smoke(root=None):
    ds = PanoCityReloc3r(root=root, split="smoke", max_samples=3)
    sample = ds[0]
    return {
        "length": len(ds),
        "view_count": len(sample),
        "img_shape": tuple(sample[0]["img"].shape),
        "camera_pose_shape": tuple(sample[0]["camera_pose"].shape),
    }
