#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


FACE_ORDER = {
    # Matterport skybox files are named skybox0..5. This follows the common
    # cubemap order used by the public release and keeps RGB/depth aligned.
    0: "front",
    1: "right",
    2: "back",
    3: "left",
    4: "up",
    5: "down",
}


def _read_zip_image(zf: zipfile.ZipFile, member: str, flags: int) -> np.ndarray:
    data = zf.read(member)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise RuntimeError(f"Could not decode {member}")
    return img


def _cube_maps(size: int, out_h: int, out_w: int):
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32), np.arange(out_w, dtype=np.float32), indexing="ij")
    lon = (xs / out_w - 0.5) * (2.0 * math.pi)
    lat = (0.5 - ys / out_h) * math.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    face = np.empty((out_h, out_w), dtype=np.uint8)
    u = np.empty((out_h, out_w), dtype=np.float32)
    v = np.empty((out_h, out_w), dtype=np.float32)

    m = (az >= ax) & (az >= ay) & (z > 0)
    face[m] = 0
    u[m] = x[m] / az[m]
    v[m] = -y[m] / az[m]

    m = (ax >= ay) & (ax >= az) & (x > 0)
    face[m] = 1
    u[m] = -z[m] / ax[m]
    v[m] = -y[m] / ax[m]

    m = (az >= ax) & (az >= ay) & (z <= 0)
    face[m] = 2
    u[m] = -x[m] / az[m]
    v[m] = -y[m] / az[m]

    m = (ax >= ay) & (ax >= az) & (x <= 0)
    face[m] = 3
    u[m] = z[m] / ax[m]
    v[m] = -y[m] / ax[m]

    m = (ay >= ax) & (ay >= az) & (y > 0)
    face[m] = 4
    u[m] = x[m] / ay[m]
    v[m] = z[m] / ay[m]

    m = (ay >= ax) & (ay >= az) & (y <= 0)
    face[m] = 5
    u[m] = x[m] / ay[m]
    v[m] = -z[m] / ay[m]

    map_x = ((u + 1.0) * 0.5 * (size - 1)).astype(np.float32)
    map_y = ((v + 1.0) * 0.5 * (size - 1)).astype(np.float32)
    return face, map_x, map_y


def cube_to_equirect(faces: dict[int, np.ndarray], out_h: int = 1024, out_w: int = 2048, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    size = next(iter(faces.values())).shape[0]
    face_map, map_x, map_y = _cube_maps(size, out_h, out_w)
    sample = next(iter(faces.values()))
    if sample.ndim == 2:
        out = np.zeros((out_h, out_w), dtype=sample.dtype)
    else:
        out = np.zeros((out_h, out_w, sample.shape[2]), dtype=sample.dtype)
    for idx, img in faces.items():
        remapped = cv2.remap(img, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = face_map == idx
        out[mask] = remapped[mask]
    return out


def load_split_indexes(split_dir: Path, splits: list[str]):
    needed: dict[str, set[str]] = defaultdict(set)
    cache_payloads = {}
    for split in splits:
        src = split_dir / f"matterport3d_{split}_index.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        cache_payloads[split] = data
        for scene, _room_id, _room_name, panos, _res in data:
            needed[scene].update(panos)
    return needed, cache_payloads


def read_panorama_to_region(zf: zipfile.ZipFile, scene: str) -> dict[str, str]:
    member = f"{scene}/house_segmentations/panorama_to_region.txt"
    out = {}
    with zf.open(member) as fh:
        for raw in fh:
            parts = raw.decode("utf-8", errors="ignore").strip().split()
            if len(parts) >= 4:
                _idx, pano_id, room_id, room_name = parts[:4]
                out[pano_id] = room_id if room_id != "-1" else "unassigned"
    return out


def create_scene_json(cache_payloads: dict[str, list], scene: str) -> dict:
    rooms = {}
    for data in cache_payloads.values():
        for item in data:
            item_scene, room_id, room_name, panos, _res = item
            if item_scene != scene:
                continue
            rooms[str(room_id)] = {"room_name": room_name, "panoramas": panos}
    return rooms or {"0": {"room_name": "room_0", "panoramas": []}}


def zip_member_lookup(zf: zipfile.ZipFile) -> dict[str, str]:
    return {Path(name).name: name for name in zf.namelist() if not name.endswith("/")}


def convert_scene(raw_root: Path, out_root: Path, scene: str, panos: set[str], cache_payloads: dict[str, list], force: bool) -> tuple[int, int]:
    scene_raw = raw_root / scene
    scene_out = out_root / scene
    color_dir = scene_out / "pano_skybox_color"
    depth_dir = scene_out / "pano_depth"
    pose_dir = scene_out / "pano_poses"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    parsed_dir = out_root / "parsed_json"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{scene}.json").write_text(json.dumps(create_scene_json(cache_payloads, scene), ensure_ascii=False, indent=2), encoding="utf-8")

    with zipfile.ZipFile(scene_raw / "matterport_skybox_images.zip") as z_color, \
            zipfile.ZipFile(scene_raw / "matterport_depth_images.zip") as z_depth, \
            zipfile.ZipFile(scene_raw / "matterport_camera_poses.zip") as z_pose:
        color_members = zip_member_lookup(z_color)
        depth_members = zip_member_lookup(z_depth)
        pose_members = zip_member_lookup(z_pose)
        done = 0
        skipped = 0
        for pano in sorted(panos):
            color_path = color_dir / f"{pano}.jpg"
            depth_path = depth_dir / f"{pano}.png"
            pose_path = pose_dir / f"{pano}.txt"
            if not force and color_path.exists() and depth_path.exists() and pose_path.exists():
                skipped += 1
                continue

            color_faces = {}
            depth_faces = {}
            for face in range(6):
                color_name = f"{pano}_skybox{face}_sami.jpg"
                depth_name = f"{pano}_d0_{face}.png"
                if color_name not in color_members or depth_name not in depth_members:
                    raise FileNotFoundError(f"Missing face {face} for {scene}/{pano}")
                color_faces[face] = _read_zip_image(z_color, color_members[color_name], cv2.IMREAD_COLOR)
                depth_faces[face] = _read_zip_image(z_depth, depth_members[depth_name], cv2.IMREAD_UNCHANGED)

            color = cube_to_equirect(color_faces, interpolation=cv2.INTER_LINEAR)
            depth = cube_to_equirect(depth_faces, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(color_path), color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(str(depth_path), depth)

            pose_name = f"{pano}_pose_0_0.txt"
            if pose_name not in pose_members:
                alternatives = [name for key, name in pose_members.items() if key.startswith(f"{pano}_pose_")]
                if not alternatives:
                    raise FileNotFoundError(f"Missing pose for {scene}/{pano}")
                pose_member = alternatives[0]
            else:
                pose_member = pose_members[pose_name]
            pose_path.write_bytes(z_pose.read(pose_member))
            done += 1
    return done, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Matterport3D_raw/v1/scans")
    parser.add_argument("--out-root", default="/mnt/e/PanoVGGT_minimal_datasets/datasets/Matterport3D")
    parser.add_argument("--split-dir", default="/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/compare_methods/camera_pose/PanoVGGT/training/data/splits/matterport3d")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-scenes", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    split_dir = Path(args.split_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    needed, cache_payloads = load_split_indexes(split_dir, args.splits)

    cache_dir = out_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, data in cache_payloads.items():
        (cache_dir / f"matterport3d_{split}_index.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        scenes = sorted({row[0] for row in data})
        mode = "val" if split == "val" else "test"
        (out_root / f"{mode}.txt").write_text("\n".join(scenes) + "\n", encoding="utf-8")

    scenes = sorted(needed)
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]
    total_done = total_skipped = 0
    for idx, scene in enumerate(scenes, 1):
        done, skipped = convert_scene(raw_root, out_root, scene, needed[scene], cache_payloads, args.force)
        total_done += done
        total_skipped += skipped
        print(f"[{idx}/{len(scenes)}] {scene}: converted={done} skipped={skipped}")
    print(f"done converted={total_done} skipped={total_skipped} scenes={len(scenes)} out={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
