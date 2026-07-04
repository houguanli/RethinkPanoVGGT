#!/usr/bin/env python3
"""Build cache indexes for the official mixed PanoVGGT minimal datasets.

Expected root:

  <root>/Panocity
  <root>/Matterport3D
  <root>/Stanford2D3DS
  <root>/Structured3D

The generated cache files are the files consumed by the mixed4 readers:

  Panocity/cache/panocity_{all,train,val}_index.json
  Matterport3D/cache/matterport3d_{train,val,test}_index.json
  Stanford2D3DS/cache/2d3ds_{train,val,test}_index.json
  Structured3D/cache/structured3d_{train,val,test}_index.json

Structured3D minimal bundles may ship only a subset of scenes. When no
train.txt exists, auto mode first checks whether the folder matches the
official scene ids and uses the official Structured3D split. It falls back to a
deterministic generated split for subset bundles.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANOCITY_SPLIT_DIR = REPO_ROOT / "training" / "data" / "splits" / "panocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Parent folder containing the four datasets.")
    parser.add_argument("--datasets", default="all", help="Comma list: panocity,matterport3d,stanford2d3ds,structured3d or all.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.90,
        help="Panocity generated fallback train fraction. Official splits are used by default when present.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--panocity-split-source",
        choices=("auto", "official", "generated"),
        default="auto",
        help="Use PanoVGGT official Panocity split JSONs or generate a fallback split.",
    )
    parser.add_argument(
        "--panocity-split-dir",
        type=Path,
        default=DEFAULT_PANOCITY_SPLIT_DIR,
        help="Directory containing panocity_{train,val,test}_index.json official split files.",
    )
    parser.add_argument(
        "--structured3d-train-source",
        choices=("auto", "official", "generated", "val", "all", "not-val-test"),
        default="auto",
        help="How to build Structured3D train index when train.txt is absent.",
    )
    parser.add_argument("--structured3d-train-fraction", type=float, default=0.90)
    parser.add_argument("--structured3d-val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--bad-scene-list",
        type=Path,
        default=None,
        help="Optional newline-delimited scene ids or paths to exclude from Structured3D indexes.",
    )
    parser.add_argument("--summary-name", default="mixed4_index_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Mixed4 root does not exist: {root}")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError(f"--train-fraction must be in (0, 1), got {args.train_fraction}")

    datasets = parse_dataset_names(args.datasets)
    summary: dict[str, Any] = {
        "root": str(root),
        "seed": int(args.seed),
        "panocity_train_fraction": float(args.train_fraction),
        "panocity_split_source": args.panocity_split_source,
        "panocity_split_dir": str(args.panocity_split_dir),
        "structured3d_train_source": args.structured3d_train_source,
        "datasets": {},
    }

    if "panocity" in datasets:
        summary["datasets"]["panocity"] = build_panocity(
            root / "Panocity",
            args.train_fraction,
            args.seed,
            split_source=args.panocity_split_source,
            split_dir=args.panocity_split_dir,
        )
    if "matterport3d" in datasets:
        summary["datasets"]["matterport3d"] = build_matterport3d(root / "Matterport3D")
    if "stanford2d3ds" in datasets:
        summary["datasets"]["stanford2d3ds"] = build_stanford2d3ds(root / "Stanford2D3DS")
    if "structured3d" in datasets:
        summary["datasets"]["structured3d"] = build_structured3d(
            root / "Structured3D",
            train_source=args.structured3d_train_source,
            seed=args.seed,
            train_fraction=args.structured3d_train_fraction,
            val_fraction=args.structured3d_val_fraction,
            bad_scene_list=args.bad_scene_list,
        )

    summary_path = root / args.summary_name
    write_json(summary_path, summary)
    print(f"[OK] wrote mixed4 summary: {summary_path}")
    for name, dataset_summary in summary["datasets"].items():
        parts = []
        for split in ("all", "train", "val", "test"):
            if split in dataset_summary:
                info = dataset_summary[split]
                parts.append(f"{split}=rows:{info['rows']} expanded:{info['expanded']}")
        print(f"[OK] {name}: " + " ".join(parts))


def parse_dataset_names(raw: str) -> set[str]:
    if raw in ("", "all", None):
        return {"panocity", "matterport3d", "stanford2d3ds", "structured3d"}
    aliases = {
        "pano_city": "panocity",
        "panocityofficial": "panocity",
        "mp3d": "matterport3d",
        "matterport": "matterport3d",
        "2d3ds": "stanford2d3ds",
        "stanford": "stanford2d3ds",
        "s3d": "structured3d",
    }
    values = {aliases.get(value.strip().lower(), value.strip().lower()) for value in raw.split(",") if value.strip()}
    supported = {"panocity", "matterport3d", "stanford2d3ds", "structured3d"}
    invalid = values - supported
    if invalid:
        raise ValueError(f"Unknown datasets: {sorted(invalid)}. Supported: {sorted(supported)}")
    return values


def build_panocity(
    root: Path,
    train_fraction: float,
    seed: int,
    split_source: str = "auto",
    split_dir: Path = DEFAULT_PANOCITY_SPLIT_DIR,
) -> dict[str, Any]:
    official_paths = {
        split: split_dir / f"panocity_{split}_index.json"
        for split in ("train", "val", "test")
    }
    official_available = all(path.exists() for path in official_paths.values())
    if split_source in {"auto", "official"} and official_available:
        return build_panocity_from_official_splits(root, official_paths)
    if split_source == "official":
        missing = [str(path) for path in official_paths.values() if not path.exists()]
        raise FileNotFoundError(f"Missing official Panocity split files: {missing}")
    if split_source == "auto":
        print(f"[WARN] official Panocity split files not found under {split_dir}; generating fallback split.")
    return build_panocity_generated(root, train_fraction, seed)


def build_panocity_from_official_splits(root: Path, split_paths: dict[str, Path]) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, path in split_paths.items():
        split_rows = read_json(path)
        if not isinstance(split_rows, list):
            raise ValueError(f"Official Panocity split is not a list: {path}")
        rows_by_split[split] = expand_panocity_official_split_rows(root, split_rows)

    all_rows_by_path: dict[str, dict[str, Any]] = {}
    for rows in rows_by_split.values():
        for row in rows:
            all_rows_by_path.setdefault(str(row.get("rgb_path")), row)
    all_rows = sorted(
        all_rows_by_path.values(),
        key=lambda row: (row.get("city", ""), row.get("block", ""), row.get("scene_name", "")),
    )

    cache = root / "cache"
    write_json(cache / "panocity_all_index.json", all_rows)
    for split, rows in rows_by_split.items():
        write_json(cache / f"panocity_{split}_index.json", rows)
    write_json(
        cache / "panocity_index_summary.json",
        {
            "root": str(root),
            "split_source": "official",
            "split_dir": str(next(iter(split_paths.values())).parent),
            "total": len(all_rows),
            "train": len(rows_by_split["train"]),
            "val": len(rows_by_split["val"]),
            "test": len(rows_by_split["test"]),
            "official_trajectory_rows": {
                split: len(read_json(path)) for split, path in split_paths.items()
            },
        },
    )
    return {
        "root": str(root),
        "split_source": "official",
        "split_dir": str(next(iter(split_paths.values())).parent),
        "all": split_summary(all_rows, "panocity"),
        "train": split_summary(rows_by_split["train"], "panocity"),
        "val": split_summary(rows_by_split["val"], "panocity"),
        "test": split_summary(rows_by_split["test"], "panocity"),
    }


def expand_panocity_official_split_rows(root: Path, split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_row in split_rows:
        if not isinstance(split_row, dict):
            continue
        scene = str(split_row.get("scene") or "")
        block = str(split_row.get("block") or "")
        rgb_paths = split_row.get("pano_images") or []
        depth_paths = split_row.get("panodepth_images") or []
        if not scene or not block:
            continue
        if len(rgb_paths) != len(depth_paths):
            print(f"[WARN] Panocity split row has mismatched RGB/depth counts: {scene}/{block}")
            continue
        for rgb_value, depth_value in zip(rgb_paths, depth_paths):
            rgb_rel = Path(str(rgb_value))
            depth_rel = Path(str(depth_value))
            rows.append(
                {
                    "dataset": "Panocity",
                    "city": scene,
                    "block": block,
                    "scene_name": f"{scene}_{block}_{rgb_rel.stem}",
                    "rgb_path": str(rgb_rel),
                    "depth_path": str(depth_rel),
                    "pano_position_m": [0.0, 0.0, 0.0],
                }
            )
    return rows


def build_panocity_generated(root: Path, train_fraction: float, seed: int) -> dict[str, Any]:
    rows = build_panocity_rows(root)
    rows.sort(key=lambda row: (row.get("city", ""), row.get("block", ""), row.get("scene_name", "")))
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    train_count = int(round(len(order) * train_fraction))
    val_count = int(round(len(order) * 0.05))
    train_indices = set(order[:train_count])
    val_indices = set(order[train_count : train_count + val_count])
    train_rows = [row for index, row in enumerate(rows) if index in train_indices]
    val_rows = [row for index, row in enumerate(rows) if index in val_indices]
    test_rows = [row for index, row in enumerate(rows) if index not in train_indices and index not in val_indices]

    cache = root / "cache"
    write_json(cache / "panocity_all_index.json", rows)
    write_json(cache / "panocity_train_index.json", train_rows)
    write_json(cache / "panocity_val_index.json", val_rows)
    write_json(cache / "panocity_test_index.json", test_rows)
    write_json(
        cache / "panocity_index_summary.json",
        {
            "root": str(root),
            "split_source": "generated",
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "train_fraction": train_fraction,
            "val_fraction": 0.05,
            "seed": seed,
        },
    )
    return {
        "root": str(root),
        "split_source": "generated",
        "all": split_summary(rows, "panocity"),
        "train": split_summary(train_rows, "panocity"),
        "val": split_summary(val_rows, "panocity"),
        "test": split_summary(test_rows, "panocity"),
    }


def build_panocity_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for pose_path in sorted(root.glob("*/*/*_poses.json")):
        block_dir = pose_path.parent
        city = block_dir.parent.name
        block = block_dir.name
        rgb_dir = block_dir / "pano_images"
        depth_dir = block_dir / "panodepth_images"
        if not rgb_dir.exists() or not depth_dir.exists():
            continue
        rgb_names = {path.name for path in rgb_dir.glob("*.png")}
        depth_names = {path.name for path in depth_dir.glob("*.png")}
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
            rgb_name = str(rgb_name)
            depth_name = str(depth_name)
            if rgb_name not in rgb_names or depth_name not in depth_names:
                continue
            rows.append(
                {
                    "dataset": "Panocity",
                    "city": city,
                    "block": block,
                    "scene_name": f"{city}_{block}_{Path(rgb_name).stem}",
                    "rgb_path": str(Path(city) / block / "pano_images" / rgb_name),
                    "depth_path": str(Path(city) / block / "panodepth_images" / depth_name),
                    "pano_position_m": translation_from_matrix(frame.get("transformation_matrix") or []),
                }
            )
    return rows


def build_matterport3d(root: Path) -> dict[str, Any]:
    rows_by_split: dict[str, list[list[Any]]] = {"train": [], "val": [], "test": []}
    if not root.exists():
        return {"root": str(root), "train": split_summary([], "matterport3d"), "val": split_summary([], "matterport3d"), "test": split_summary([], "matterport3d")}

    val_scans = read_name_set(root / "val.txt")
    test_scans = read_name_set(root / "test.txt")
    for scan_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "cache"):
        scan = scan_dir.name
        depth_dir = scan_dir / "pano_depth"
        rgb_dir = scan_dir / "pano_skybox_color"
        if not depth_dir.exists() or not rgb_dir.exists():
            continue
        pano_ids = [
            depth_path.stem
            for depth_path in sorted(depth_dir.glob("*.png"))
            if (rgb_dir / f"{depth_path.stem}.jpg").exists()
        ]
        if not pano_ids:
            continue
        split = "val" if scan in val_scans else "test" if scan in test_scans else "train"
        rows_by_split[split].append([scan, "all", "all", pano_ids, [1024, 2048]])

    cache = root / "cache"
    for split, rows in rows_by_split.items():
        write_json(cache / f"matterport3d_{split}_index.json", rows)
    return {
        "root": str(root),
        "train": split_summary(rows_by_split["train"], "matterport3d"),
        "val": split_summary(rows_by_split["val"], "matterport3d"),
        "test": split_summary(rows_by_split["test"], "matterport3d"),
    }


def build_stanford2d3ds(root: Path) -> dict[str, Any]:
    rows_by_split: dict[str, list[list[Any]]] = {"train": [], "val": [], "test": []}
    if root.exists():
        for area_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("area_")):
            area = area_dir.name
            split = "val" if area == "area_5a" else "test" if area == "area_5b" else "train"
            depth_dir = area_dir / "pano" / "depth"
            rgb_dir = area_dir / "pano" / "rgb"
            if not depth_dir.exists() or not rgb_dir.exists():
                continue
            grouped: dict[str, list[str]] = defaultdict(list)
            for depth_path in sorted(depth_dir.glob("camera_*_depth.png")):
                parsed = parse_stanford_depth_name(depth_path.name)
                if parsed is None:
                    continue
                pano_id, room_name = parsed
                if first_match(rgb_dir, f"camera_{pano_id}_*_rgb.png") is None:
                    continue
                grouped[room_name].append(pano_id)
            for room_index, (room_name, pano_ids) in enumerate(sorted(grouped.items())):
                rows_by_split[split].append([area, str(room_index), room_name, sorted(pano_ids), [1024, 2048]])

    cache = root / "cache"
    for split, rows in rows_by_split.items():
        write_json(cache / f"2d3ds_{split}_index.json", rows)
    return {
        "root": str(root),
        "train": split_summary(rows_by_split["train"], "stanford2d3ds"),
        "val": split_summary(rows_by_split["val"], "stanford2d3ds"),
        "test": split_summary(rows_by_split["test"], "stanford2d3ds"),
    }


def build_structured3d(
    root: Path,
    train_source: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    bad_scene_list: Path | None = None,
) -> dict[str, Any]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"--structured3d-train-fraction must be in (0, 1), got {train_fraction}")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"--structured3d-val-fraction must be in [0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "--structured3d-train-fraction + --structured3d-val-fraction must be < 1.0 "
            f"got {train_fraction + val_fraction}"
        )
    all_scenes = sorted(path.name for path in root.glob("scene_*") if path.is_dir()) if root.exists() else []
    val_scenes = sorted(read_name_set(root / "val.txt"))
    test_scenes = sorted(read_name_set(root / "test.txt"))
    train_txt = sorted(read_name_set(root / "train.txt"))
    bad_scenes = read_bad_scene_set(bad_scene_list, root)
    if bad_scenes:
        all_scenes = [scene for scene in all_scenes if scene not in bad_scenes]
        train_txt = [scene for scene in train_txt if scene not in bad_scenes]
        val_scenes = [scene for scene in val_scenes if scene not in bad_scenes]
        test_scenes = [scene for scene in test_scenes if scene not in bad_scenes]

    if train_source == "auto":
        if train_txt:
            train_scenes = train_txt
            val_source_scenes = val_scenes
            test_source_scenes = test_scenes
            resolved_train_source = "train.txt"
        elif has_official_structured3d_scene_ids(all_scenes):
            train_scenes, val_source_scenes, test_source_scenes = make_official_structured3d_split(all_scenes)
            resolved_train_source = "official"
        else:
            train_scenes, val_source_scenes, test_source_scenes = make_generated_split(
                all_scenes,
                seed=seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
            )
            resolved_train_source = "generated"
    elif train_source == "official":
        train_scenes, val_source_scenes, test_source_scenes = make_official_structured3d_split(all_scenes)
        resolved_train_source = "official"
    elif train_source == "generated":
        train_scenes, val_source_scenes, test_source_scenes = make_generated_split(
            all_scenes,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        resolved_train_source = "generated"
    elif train_source == "val":
        train_scenes = val_scenes
        val_source_scenes = val_scenes
        test_source_scenes = test_scenes
        resolved_train_source = "val"
    elif train_source == "not-val-test":
        excluded = set(val_scenes) | set(test_scenes)
        train_scenes = [scene for scene in all_scenes if scene not in excluded]
        val_source_scenes = val_scenes
        test_source_scenes = test_scenes
        resolved_train_source = "not-val-test"
    else:
        train_scenes = all_scenes
        val_source_scenes = val_scenes
        test_source_scenes = test_scenes
        resolved_train_source = "all"

    if resolved_train_source in {"generated", "official"}:
        cache = root / "cache"
        write_lines(cache / "structured3d_train_scenes.txt", train_scenes)
        write_lines(cache / "structured3d_val_scenes.txt", val_source_scenes)
        write_lines(cache / "structured3d_test_scenes.txt", test_source_scenes)
        write_json(
            cache / f"structured3d_{resolved_train_source}_split_summary.json",
            {
                "source": resolved_train_source,
                "seed": int(seed),
                "scene_count": len(all_scenes),
                "train_fraction": float(train_fraction) if resolved_train_source == "generated" else None,
                "val_fraction": float(val_fraction) if resolved_train_source == "generated" else None,
                "test_fraction": float(1.0 - train_fraction - val_fraction) if resolved_train_source == "generated" else None,
                "train_scenes": len(train_scenes),
                "val_scenes": len(val_source_scenes),
                "test_scenes": len(test_source_scenes),
                "bad_scenes_excluded": len(bad_scenes),
            },
        )

    rows_by_split = {
        "train": build_structured3d_rows(root, train_scenes),
        "val": build_structured3d_rows(root, val_source_scenes),
        "test": build_structured3d_rows(root, test_source_scenes),
    }
    cache = root / "cache"
    for split, rows in rows_by_split.items():
        write_json(cache / f"structured3d_{split}_index.json", rows)
    return {
        "root": str(root),
        "resolved_train_source": resolved_train_source,
        "bad_scenes_excluded": len(bad_scenes),
        "train": split_summary(rows_by_split["train"], "structured3d"),
        "val": split_summary(rows_by_split["val"], "structured3d"),
        "test": split_summary(rows_by_split["test"], "structured3d"),
    }


def has_official_structured3d_scene_ids(scenes: list[str]) -> bool:
    scene_ids = {parse_structured3d_scene_id(scene) for scene in scenes}
    scene_ids.discard(None)
    # Full official Structured3D has scene_00000 through scene_03499. Accept a
    # nearly complete folder so a few bad/missing scenes do not force a random
    # split.
    return len(scene_ids) >= 3000 and any(scene_id < 3000 for scene_id in scene_ids) and any(scene_id >= 3250 for scene_id in scene_ids)


def make_official_structured3d_split(scenes: list[str]) -> tuple[list[str], list[str], list[str]]:
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    unknown: list[str] = []
    for scene in sorted(scenes):
        scene_id = parse_structured3d_scene_id(scene)
        if scene_id is None:
            unknown.append(scene)
        elif 0 <= scene_id <= 2999:
            train.append(scene)
        elif 3000 <= scene_id <= 3249:
            val.append(scene)
        elif 3250 <= scene_id <= 3499:
            test.append(scene)
        else:
            unknown.append(scene)
    if unknown:
        print(f"[WARN] Structured3D official split ignored {len(unknown)} non-standard scene ids.")
    return train, val, test


def parse_structured3d_scene_id(scene: str) -> int | None:
    prefix = "scene_"
    if not scene.startswith(prefix):
        return None
    try:
        return int(scene[len(prefix) :])
    except ValueError:
        return None


def make_generated_split(
    scenes: list[str],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    order = list(sorted(scenes))
    rng = random.Random(seed)
    rng.shuffle(order)
    train_count = int(round(len(order) * train_fraction))
    val_count = int(round(len(order) * val_fraction))
    train_scenes = sorted(order[:train_count])
    val_scenes = sorted(order[train_count : train_count + val_count])
    test_scenes = sorted(order[train_count + val_count :])
    return train_scenes, val_scenes, test_scenes


def build_structured3d_rows(root: Path, scenes: Iterable[str]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scene in sorted(set(scenes)):
        render_dir = root / scene / "2D_rendering"
        if not render_dir.exists():
            continue
        pano_ids = []
        for camera_dir in sorted(path for path in render_dir.iterdir() if path.is_dir()):
            pano_dir = camera_dir / "panorama" / "full"
            if (pano_dir / "rgb_rawlight.png").exists() and (pano_dir / "depth.png").exists():
                pano_ids.append(camera_dir.name)
        if pano_ids:
            rows.append([scene, pano_ids, [1024, 2048]])
    return rows


def parse_stanford_depth_name(name: str) -> tuple[str, str] | None:
    prefix = "camera_"
    suffix = "_depth.png"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    body = name[len(prefix) : -len(suffix)]
    if "_" not in body:
        return body, "unknown"
    pano_id, room_name = body.split("_", 1)
    room_name = room_name.replace("_frame_equirectangular_domain", "")
    return pano_id, room_name


def split_summary(rows: list[Any], dataset: str) -> dict[str, int]:
    return {"rows": len(rows), "expanded": expanded_count(rows, dataset)}


def expanded_count(rows: list[Any], dataset: str) -> int:
    if dataset == "panocity":
        return len(rows)
    if dataset in {"matterport3d", "stanford2d3ds"}:
        return sum(len(row[3]) for row in rows if isinstance(row, list) and len(row) > 3 and isinstance(row[3], list))
    if dataset == "structured3d":
        return sum(len(row[1]) for row in rows if isinstance(row, list) and len(row) > 1 and isinstance(row[1], list))
    return len(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def read_name_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_bad_scene_set(path: Path | None, root: Path) -> set[str]:
    if path in (None, ""):
        return set()
    requested = Path(path)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(root / requested)
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        print(f"[WARN] Structured3D bad scene list not found: {path}")
        return set()
    scenes: set[str] = set()
    for line in resolved.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        parts = Path(value).parts
        scene = next((part for part in parts if part.startswith("scene_")), Path(value).name)
        scenes.add(scene)
    print(f"[INFO] loaded Structured3D bad scenes = {len(scenes)} from {resolved}")
    return scenes


def first_match(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern)) if folder.exists() else []
    return matches[0] if matches else None


def translation_from_matrix(matrix: object) -> list[float]:
    try:
        if len(matrix) >= 3:
            return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
    except (TypeError, ValueError, IndexError):
        pass
    return [0.0, 0.0, 0.0]


if __name__ == "__main__":
    main()
