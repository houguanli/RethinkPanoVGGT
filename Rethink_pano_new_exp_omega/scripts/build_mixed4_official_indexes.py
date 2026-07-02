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

Structured3D minimal bundles often ship only val/test scenes. By default this
script mirrors the current reader behavior by using val as train when no
train.txt exists.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Parent folder containing the four datasets.")
    parser.add_argument("--datasets", default="all", help="Comma list: panocity,matterport3d,stanford2d3ds,structured3d or all.")
    parser.add_argument("--train-fraction", type=float, default=0.95, help="Panocity random train split fraction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--structured3d-train-source",
        choices=("auto", "generated", "val", "all", "not-val-test"),
        default="auto",
        help="How to build Structured3D train index when train.txt is absent.",
    )
    parser.add_argument("--structured3d-train-fraction", type=float, default=0.90)
    parser.add_argument("--structured3d-val-fraction", type=float, default=0.05)
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
        "structured3d_train_source": args.structured3d_train_source,
        "datasets": {},
    }

    if "panocity" in datasets:
        summary["datasets"]["panocity"] = build_panocity(root / "Panocity", args.train_fraction, args.seed)
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


def build_panocity(root: Path, train_fraction: float, seed: int) -> dict[str, Any]:
    rows = build_panocity_rows(root)
    rows.sort(key=lambda row: (row.get("city", ""), row.get("block", ""), row.get("scene_name", "")))
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    train_count = int(round(len(order) * train_fraction))
    train_indices = set(order[:train_count])
    train_rows = [row for index, row in enumerate(rows) if index in train_indices]
    val_rows = [row for index, row in enumerate(rows) if index not in train_indices]

    cache = root / "cache"
    write_json(cache / "panocity_all_index.json", rows)
    write_json(cache / "panocity_train_index.json", train_rows)
    write_json(cache / "panocity_val_index.json", val_rows)
    write_json(
        cache / "panocity_index_summary.json",
        {
            "root": str(root),
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "train_fraction": train_fraction,
            "seed": seed,
        },
    )
    return {
        "root": str(root),
        "all": split_summary(rows, "panocity"),
        "train": split_summary(train_rows, "panocity"),
        "val": split_summary(val_rows, "panocity"),
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


def build_structured3d(root: Path, train_source: str, seed: int, train_fraction: float, val_fraction: float) -> dict[str, Any]:
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

    if train_source == "auto":
        if train_txt:
            train_scenes = train_txt
            val_source_scenes = val_scenes
            test_source_scenes = test_scenes
            resolved_train_source = "train.txt"
        else:
            train_scenes, val_source_scenes, test_source_scenes = make_generated_split(
                all_scenes,
                seed=seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
            )
            resolved_train_source = "generated"
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

    if resolved_train_source == "generated":
        cache = root / "cache"
        write_lines(cache / "structured3d_train_scenes.txt", train_scenes)
        write_lines(cache / "structured3d_val_scenes.txt", val_source_scenes)
        write_lines(cache / "structured3d_test_scenes.txt", test_source_scenes)
        write_json(
            cache / "structured3d_generated_split_summary.json",
            {
                "source": "generated",
                "seed": int(seed),
                "scene_count": len(all_scenes),
                "train_fraction": float(train_fraction),
                "val_fraction": float(val_fraction),
                "test_fraction": float(1.0 - train_fraction - val_fraction),
                "train_scenes": len(train_scenes),
                "val_scenes": len(val_source_scenes),
                "test_scenes": len(test_source_scenes),
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
        "train": split_summary(rows_by_split["train"], "structured3d"),
        "val": split_summary(rows_by_split["val"], "structured3d"),
        "test": split_summary(rows_by_split["test"], "structured3d"),
    }


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


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def read_name_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


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
