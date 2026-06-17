"""PanoCity paired RGB/depth reader shared by compare methods.

The paired PanoCity layout is expected to look like:

    PanoCity_paired/
      pairs.csv
      rgb/*.png
      depth/*.png

The default resolver first honors PANOCITY_ROOT, then searches for a
PanoCity_paired directory next to the project, and finally checks the common
Windows mount path used by the local workstation.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PanoCityRecord:
    pair_idx: int
    block: str
    num_id: str
    rgb_path: Path
    depth_path: Path
    rgb_source: str = ""
    depth_source: str = ""


def resolve_panocity_root(root: Optional[str] = None, start: Optional[Path] = None) -> Path:
    """Resolve the paired PanoCity root with workstation-friendly defaults."""
    candidates: List[Path] = []
    if root:
        candidates.append(Path(root).expanduser())
    env_root = os.environ.get("PANOCITY_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        candidates.append(parent / "PanoCity_paired")
        candidates.append(parent.parent / "PanoCity_paired")

    candidates.extend(
        [
            Path("/mnt/f/panovggt/PanoCity_paired"),
            Path("/mnt/win_f/panovggt/PanoCity_paired"),
            Path("F:/panovggt/PanoCity_paired"),
        ]
    )

    for candidate in candidates:
        if (candidate / "pairs.csv").exists() and (candidate / "rgb").is_dir() and (candidate / "depth").is_dir():
            return candidate.resolve()

    tried = "\n".join(str(c) for c in candidates[:12])
    raise FileNotFoundError(
        "Cannot locate PanoCity_paired. Set PANOCITY_ROOT or place it next to the project. "
        f"First candidates tried:\n{tried}"
    )


class PanoCityPairedIndex:
    """Lightweight CSV-backed index for PanoCity paired samples."""

    def __init__(
        self,
        root: Optional[str] = None,
        pairs_file: str = "pairs.csv",
        split: str = "train",
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        split_seed: int = 42,
        max_samples: Optional[int] = None,
        start: Optional[Path] = None,
    ) -> None:
        self.root = resolve_panocity_root(root, start=start)
        self.pairs_path = self.root / pairs_file
        self.split = split
        quick_limit = int(max_samples) if max_samples is not None and split in {"smoke", "all"} else None
        self.records = self._load_records(limit=quick_limit)
        if quick_limit is None:
            self.records = self._select_split(self.records, split, train_ratio, val_ratio, split_seed)
            if max_samples is not None:
                self.records = self.records[: int(max_samples)]
        if not self.records:
            raise RuntimeError(f"No PanoCity records selected for split={split} from {self.pairs_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PanoCityRecord:
        return self.records[index % len(self.records)]

    def _load_records(self, limit: Optional[int] = None) -> List[PanoCityRecord]:
        records: List[PanoCityRecord] = []
        with self.pairs_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"pair_idx", "block", "num_id", "rgb_path", "depth_path"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.pairs_path} is missing required columns: {sorted(missing)}")
            for row in reader:
                rgb_rel = Path(row["rgb_path"].replace("\\", "/"))
                depth_rel = Path(row["depth_path"].replace("\\", "/"))
                rgb_path = self.root / rgb_rel
                depth_path = self.root / depth_rel
                records.append(
                    PanoCityRecord(
                        pair_idx=int(row["pair_idx"]),
                        block=row["block"],
                        num_id=row["num_id"],
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        rgb_source=row.get("rgb_source", ""),
                        depth_source=row.get("depth_source", ""),
                    )
                )
                if limit is not None and len(records) >= limit:
                    break
        return records

    @staticmethod
    def _select_split(
        records: Sequence[PanoCityRecord],
        split: str,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> List[PanoCityRecord]:
        indices = list(range(len(records)))
        random.Random(seed).shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        if split == "train":
            selected = indices[:n_train]
        elif split in {"val", "valid", "validation", "test"}:
            selected = indices[n_train : n_train + n_val]
        elif split in {"test_final", "holdout"}:
            selected = indices[n_train + n_val :]
        elif split in {"all", "smoke"}:
            selected = indices
        else:
            raise ValueError(f"Unsupported PanoCity split: {split}")
        return [records[i] for i in selected]

    def write_list_file(self, path: str | Path) -> Path:
        """Write `rgb_rel depth_rel` lines for methods that expect list files."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for record in self.records:
            lines.append(
                f"{record.rgb_path.relative_to(self.root).as_posix()} "
                f"{record.depth_path.relative_to(self.root).as_posix()}"
            )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out


def read_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        width, height = size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return image


def read_depth(path: Path, depth_scale: float = 1000.0, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    import cv2
    import numpy as np

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    if size is not None:
        width, height = size
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    if depth_scale:
        depth = depth / float(depth_scale)
    return depth


class PanoCityDepthTorchDataset:
    """Depth-estimation dataset adapter returning rgb/gt_depth/val_mask tensors."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        list_file: Optional[str] = None,
        height: int = 512,
        width: int = 1024,
        color_augmentation: bool = True,
        LR_filp_augmentation: bool = True,
        yaw_rotation_augmentation: bool = True,
        repeat: int = 1,
        is_training: bool = False,
        split: str = "train",
        depth_scale: float = 1000.0,
        max_depth_meters: float = 100.0,
    ) -> None:
        import torch
        from torchvision import transforms

        self.index = PanoCityPairedIndex(root=root_dir, split=split if list_file is None else "all")
        if list_file:
            self.records = self._records_from_list(root_dir, list_file)
        else:
            self.records = self.index.records
        self.records = self.records * max(1, int(repeat))
        self.height = int(height)
        self.width = int(width)
        self.depth_scale = float(depth_scale)
        self.max_depth_meters = float(max_depth_meters)
        self.is_training = bool(is_training)
        self.color_augmentation = bool(color_augmentation)
        self.flip_augmentation = bool(LR_filp_augmentation)
        self.yaw_rotation_augmentation = bool(yaw_rotation_augmentation)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_aug = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self._torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import cv2
        import numpy as np

        record = self.records[index % len(self.records)]
        rgb = read_rgb(record.rgb_path, size=(self.width, self.height))
        depth = read_depth(record.depth_path, self.depth_scale, size=(self.width, self.height))

        if self.is_training and self.yaw_rotation_augmentation:
            roll = random.randint(0, max(0, self.width - 1))
            rgb = np.roll(rgb, roll, axis=1)
            depth = np.roll(depth, roll, axis=1)
        if self.is_training and self.flip_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
        if self.is_training and self.color_augmentation and random.random() > 0.5:
            from torchvision import transforms

            rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))

        rgb_tensor = self.normalize(self.to_tensor(rgb.copy()))
        depth_tensor = self._torch.from_numpy(depth[None].copy()).to(self._torch.float32)
        val_mask = (depth_tensor > 0) & (depth_tensor <= self.max_depth_meters) & ~self._torch.isnan(depth_tensor)
        depth_norm = self._torch.clamp(depth_tensor / self.max_depth_meters, 0.001, 1.0)
        return {
            "rgb": rgb_tensor,
            "gt_depth": depth_norm,
            "val_mask": val_mask,
            "mask_100": (depth_tensor > 0) & (depth_tensor <= 100.0),
            "path": str(record.rgb_path),
            "num_id": record.num_id,
        }

    def _records_from_list(self, root_dir: Optional[str], list_file: str) -> List[PanoCityRecord]:
        root = resolve_panocity_root(root_dir)
        records: List[PanoCityRecord] = []
        for idx, line in enumerate(Path(list_file).read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            rgb_rel, depth_rel = line.split()[:2]
            records.append(
                PanoCityRecord(
                    pair_idx=idx,
                    block="list",
                    num_id=str(idx),
                    rgb_path=root / rgb_rel,
                    depth_path=root / depth_rel,
                )
            )
        return records


def smoke_summary(root: Optional[str] = None, split: str = "train") -> dict:
    index = PanoCityPairedIndex(root=root, split="smoke", max_samples=2)
    first = index[0]
    summary = {
        "root": str(index.root),
        "requested_split": split,
        "split": "smoke",
        "records": len(index),
        "first_rgb": str(first.rgb_path),
        "first_depth": str(first.depth_path),
    }
    try:
        import numpy as np

        rgb = read_rgb(first.rgb_path)
        depth = read_depth(first.depth_path)
        summary.update(
            {
                "rgb_shape": tuple(rgb.shape),
                "depth_shape": tuple(depth.shape),
                "depth_minmax": (float(np.nanmin(depth)), float(np.nanmax(depth))),
            }
        )
    except Exception as exc:
        summary["array_read_skipped"] = f"{type(exc).__name__}: {exc}"
    return summary
