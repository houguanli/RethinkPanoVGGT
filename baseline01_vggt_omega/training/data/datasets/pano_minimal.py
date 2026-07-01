import json
from pathlib import Path
from typing import Iterable, Optional

from data.datasets.panocity_paired import (
    PanoCityPairedPinholeDataset,
    _apply_metadata,
    _is_bad_sample,
)


class PanoMinimalPinholeDataset(PanoCityPairedPinholeDataset):
    """Pinhole-window reader for the official PanoVGGT minimal dataset bundle."""

    def __init__(
        self,
        common_conf,
        split: str = "train",
        root: str = "../../panovggt",
        len_train: int = 100000,
        len_test: int = 10000,
        datasets: Optional[str | Iterable[str]] = "all",
        output_depth_scale: float = 1000.0,
        invalid_depth_value: Optional[float] = 65535.0,
        depth_max_m: float = 80.0,
        num_yaw: int = 8,
        pitch_degrees: float = 0.0,
        fov_degrees: float = 75.0,
        max_samples: Optional[int] = None,
        train_split_fraction: float = 0.95,
        split_seed: int = 42,
        metadata_path: Optional[str] = None,
        bad_sample_list: Optional[str] = None,
        curriculum_bins: Optional[str | Iterable[str]] = None,
        use_metadata_weights: bool = True,
    ):
        self.dataset_names = _parse_dataset_names(datasets)
        super().__init__(
            common_conf=common_conf,
            split=split,
            root=root,
            len_train=len_train,
            len_test=len_test,
            output_depth_scale=output_depth_scale,
            invalid_depth_value=invalid_depth_value,
            depth_max_m=depth_max_m,
            num_yaw=num_yaw,
            pitch_degrees=pitch_degrees,
            fov_degrees=fov_degrees,
            max_samples=max_samples,
            train_split_fraction=train_split_fraction,
            split_seed=split_seed,
            metadata_path=metadata_path,
            bad_sample_list=bad_sample_list,
            curriculum_bins=curriculum_bins,
            use_metadata_weights=use_metadata_weights,
        )

    def _build_index(self, max_samples: Optional[int] = None):
        root = Path(self.root)
        items = []
        if "matterport3d" in self.dataset_names:
            items.extend(_index_matterport3d(root / "Matterport3D", self.split))
        if "stanford2d3ds" in self.dataset_names:
            items.extend(_index_stanford2d3ds(root / "Stanford2D3DS", self.split))
        if "structured3d" in self.dataset_names:
            items.extend(_index_structured3d(root / "Structured3D", self.split))
        if "panocity" in self.dataset_names:
            items.extend(_index_panocity_official(root / "Panocity", self.split))

        if self.bad_samples:
            items = [
                item
                for item in items
                if not _is_bad_sample(str(item["rgb_path"]), str(item["depth_path"]), self.bad_samples)
            ]
        items = _apply_metadata(
            items,
            metadata_by_name=self.metadata_by_name,
            curriculum_bins=self.curriculum_bins,
            use_metadata_weights=self.use_metadata_weights,
        )
        if max_samples is not None:
            items = items[: int(max_samples)]
        return items


def _parse_dataset_names(raw: Optional[str | Iterable[str]]) -> set[str]:
    if raw in (None, "", "all"):
        return {"matterport3d", "stanford2d3ds", "structured3d", "panocity"}
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = [str(value) for value in raw]
    aliases = {
        "2d3ds": "stanford2d3ds",
        "stanford": "stanford2d3ds",
        "mp3d": "matterport3d",
        "s3d": "structured3d",
        "pano_city": "panocity",
        "panocityofficial": "panocity",
    }
    return {aliases.get(value.strip().lower(), value.strip().lower()) for value in values if value.strip()}


def _index_matterport3d(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"matterport3d_{split}_index.json")
    items = []
    for row in rows:
        scan, room_id, _room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = root / str(scan) / "pano_skybox_color" / f"{pano_id}.jpg"
            depth_path = root / str(scan) / "pano_depth" / f"{pano_id}.png"
            if rgb_path.exists() and depth_path.exists():
                items.append(_item("Matterport3D", f"{scan}_{room_id}_{pano_id}", rgb_path, depth_path))
    return items


def _index_stanford2d3ds(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"2d3ds_{split}_index.json")
    items = []
    for row in rows:
        area, _room_id, room_name, pano_ids, _size = row
        for pano_id in pano_ids:
            rgb_path = _first_match(root / str(area) / "pano" / "rgb", f"camera_{pano_id}_*_rgb.png")
            depth_path = _first_match(root / str(area) / "pano" / "depth", f"camera_{pano_id}_*_depth.png")
            if rgb_path is not None and depth_path is not None:
                items.append(_item("Stanford2D3DS", f"{area}_{room_name}_{pano_id}", rgb_path, depth_path))
    return items


def _index_structured3d(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"structured3d_{split}_index.json")
    if not rows:
        for fallback_split in ("val", "test"):
            rows = _read_json_list(root / "cache" / f"structured3d_{fallback_split}_index.json")
            if rows:
                print(
                    f"[WARN] Structured3D {split} index not found under {root}; "
                    f"using structured3d_{fallback_split}_index.json instead."
                )
                break
    items = []
    for row in rows:
        scene, pano_ids, _size = row
        for pano_id in pano_ids:
            pano_dir = root / str(scene) / "2D_rendering" / str(pano_id) / "panorama" / "full"
            rgb_path = pano_dir / "rgb_rawlight.png"
            depth_path = pano_dir / "depth.png"
            if rgb_path.exists() and depth_path.exists():
                items.append(_item("Structured3D", f"{scene}_{pano_id}", rgb_path, depth_path))
    return items


def _index_panocity_official(root: Path, split: str) -> list[dict]:
    rows = _read_json_list(root / "cache" / f"panocity_{split}_index.json")
    if not rows and split != "all":
        rows = _read_json_list(root / "cache" / "panocity_all_index.json")
        if rows:
            print(f"[WARN] Panocity {split} index not found under {root}; using panocity_all_index.json instead.")
    if not rows:
        rows = _build_panocity_official_rows(root)
    items = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rgb_path = _resolve_cached_path(root, row.get("rgb_path"))
        depth_path = _resolve_cached_path(root, row.get("depth_path"))
        if rgb_path is not None and depth_path is not None and rgb_path.exists() and depth_path.exists():
            items.append(_item("Panocity", str(row.get("scene_name") or rgb_path.stem), rgb_path, depth_path))
    return items


def _build_panocity_official_rows(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for pose_path in sorted(root.glob("*/*/*_poses.json")):
        block_dir = pose_path.parent
        city = block_dir.parent.name
        block = block_dir.name
        try:
            payload = json.loads(pose_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARN] skipping malformed Panocity pose file {pose_path}: {exc}")
            continue
        frames = payload.get("frames", []) if isinstance(payload, dict) else []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            rgb_name = frame.get("name")
            depth_name = frame.get("depth")
            if not rgb_name or not depth_name:
                continue
            rgb_path = block_dir / "pano_images" / str(rgb_name)
            depth_path = block_dir / "panodepth_images" / str(depth_name)
            if rgb_path.exists() and depth_path.exists():
                rows.append(
                    {
                        "dataset": "Panocity",
                        "city": city,
                        "block": block,
                        "scene_name": f"{city}_{block}_{Path(str(rgb_name)).stem}",
                        "rgb_path": str(rgb_path.relative_to(root)),
                        "depth_path": str(depth_path.relative_to(root)),
                    }
                )
    return rows


def _item(dataset: str, name: str, rgb_path: Path, depth_path: Path) -> dict:
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "name": f"{dataset}_{name}",
    }


def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _resolve_cached_path(root: Path, value: object) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _first_match(folder: Path, pattern: str) -> Optional[Path]:
    if not folder.exists():
        return None
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None
