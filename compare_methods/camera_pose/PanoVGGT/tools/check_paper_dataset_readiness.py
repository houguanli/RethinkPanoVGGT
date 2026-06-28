#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def exists_any(root: Path, patterns: list[str]) -> bool:
    return any(next(root.glob(pattern), None) is not None for pattern in patterns)


def check_matterport(root: Path) -> list[str]:
    issues = []
    if not (root / "parsed_json").is_dir():
        issues.append("missing parsed_json/*.json")
    if not exists_any(root, ["*/pano_skybox_color/*", "*/pano_color/*"]):
        issues.append("missing <scene>/pano_skybox_color or pano_color")
    if not exists_any(root, ["*/pano_depth/*"]):
        issues.append("missing <scene>/pano_depth")
    if not exists_any(root, ["*/pano_poses/*"]):
        issues.append("missing <scene>/pano_poses")
    cache = root / "cache"
    for split in ["val", "test"]:
        index_path = cache / f"matterport3d_{split}_index.json"
        if not index_path.is_file():
            issues.append(f"missing cache/{index_path.name}")
            continue
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"invalid cache/{index_path.name}: {exc}")
            continue
        missing_files = 0
        for scene, _room_id, _room_name, panos, _res in data:
            if not (root / "parsed_json" / f"{scene}.json").is_file():
                missing_files += 1
            for pano in panos:
                color_png = root / scene / "pano_skybox_color" / f"{pano}.png"
                color_jpg = root / scene / "pano_skybox_color" / f"{pano}.jpg"
                depth = root / scene / "pano_depth" / f"{pano}.png"
                pose = root / scene / "pano_poses" / f"{pano}.txt"
                if not (color_png.is_file() or color_jpg.is_file()) or not depth.is_file() or not pose.is_file():
                    missing_files += 1
                    if missing_files >= 10:
                        break
            if missing_files >= 10:
                break
        if missing_files:
            issues.append(f"incomplete Matterport3D processed {split} files; first missing count capped at {missing_files}")
    return issues


def check_stanford(root: Path) -> list[str]:
    issues = []
    cache = root / "cache"
    for name in ["2d3ds_train_index.json", "2d3ds_val_index.json", "2d3ds_test_index.json"]:
        if not (cache / name).is_file():
            issues.append(f"missing cache/{name}")
    for area in ["area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"]:
        area_root = root / area
        if not area_root.is_dir():
            issues.append(f"missing {area}/")
            continue
        for sub in ["rgb", "depth", "pose"]:
            if not (area_root / "pano" / sub).is_dir():
                issues.append(f"missing {area}/pano/{sub}")
    return issues


def check_structured3d(root: Path) -> list[str]:
    issues = []
    nested = root / "Structured3D"
    if nested.is_dir():
        root = nested
    if not exists_any(root, ["scene_*/2D_rendering/*/panorama/*/rgb_*.png"]):
        issues.append("missing scene_*/2D_rendering/*/panorama/*/rgb_*.png")
    if not exists_any(root, ["scene_*/2D_rendering/*/panorama/*/depth.png"]):
        issues.append("missing scene_*/2D_rendering/*/panorama/*/depth.png")
    if not exists_any(root, ["scene_*/2D_rendering/*/panorama/camera_xyz.txt"]):
        issues.append("missing scene_*/2D_rendering/*/panorama/camera_xyz.txt")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matterport-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Matterport3D")
    parser.add_argument("--stanford-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Stanford2D3DS")
    parser.add_argument("--structured3d-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Structured3D")
    args = parser.parse_args()

    checks = {
        "matterport": (Path(args.matterport_root), check_matterport),
        "stanford": (Path(args.stanford_root), check_stanford),
        "structured3d": (Path(args.structured3d_root), check_structured3d),
    }

    failed = False
    for name, (root, fn) in checks.items():
        issues = fn(root)
        print(f"\n[{name}] root={root}")
        if issues:
            failed = True
            for issue in issues:
                print(f"  MISSING: {issue}")
        else:
            print("  OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
