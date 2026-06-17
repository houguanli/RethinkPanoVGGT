"""PanoCity paired dataset adapter for Reloc3r."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

COMPARE_ROOT = Path(__file__).resolve().parents[4]
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from common.panocity_paired import PanoCityPairedIndex  # noqa: E402


class PanoCityReloc3r(data.Dataset):
    """Adjacent-pair RGB adapter for Reloc3r smoke/finetune scheduling.

    PanoCity paired data has no camera pose annotations in pairs.csv, so this
    adapter returns identity relative pose placeholders and explicit metadata.
    Real pose-supervised Reloc3r finetuning should replace `relpose` with
    calibrated labels if they become available.
    """

    def __init__(self, root=None, split="train", size=512, max_samples=None):
        self.index = PanoCityPairedIndex(root=root, split=split, max_samples=max_samples)
        self.size = int(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return max(0, len(self.index) - 1)

    def __getitem__(self, idx):
        rec1 = self.index[idx]
        rec2 = self.index[idx + 1]
        img1 = Image.open(rec1.rgb_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        img2 = Image.open(rec2.rgb_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return {
            "img1": self.normalize(self.to_tensor(img1)),
            "img2": self.normalize(self.to_tensor(img2)),
            "relpose": torch.eye(4, dtype=torch.float32),
            "valid_pose": torch.tensor(False),
            "label": f"{rec1.num_id}_{rec2.num_id}",
            "instance": rec1.block,
            "metadata": {
                "rgb1": str(rec1.rgb_path),
                "rgb2": str(rec2.rgb_path),
                "note": "PanoCity pairs.csv does not provide relative camera pose labels.",
            },
        }


def smoke(root=None):
    ds = PanoCityReloc3r(root=root, split="smoke", max_samples=3)
    sample = ds[0]
    return {
        "length": len(ds),
        "img1_shape": tuple(sample["img1"].shape),
        "img2_shape": tuple(sample["img2"].shape),
        "valid_pose": bool(sample["valid_pose"]),
    }
