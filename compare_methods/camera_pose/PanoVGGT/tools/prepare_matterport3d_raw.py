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


MATTERPORT_FACE_ORDER = "1,2,3,4,0,5"
MATTERPORT_FACE_ROTATIONS = "0,0,0,0,0,0"


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


def parse_int_list(text: str, expected: int, name: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated integers, got {text!r}")
    return values


def rotate_face(img: np.ndarray, turns: int) -> np.ndarray:
    turns = turns % 4
    if turns == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=turns))


def remap_faces(
    raw_faces: dict[int, np.ndarray],
    face_order: list[int],
    face_rotations: list[int],
) -> dict[int, np.ndarray]:
    return {
        out_face: rotate_face(raw_faces[src_face], face_rotations[out_face])
        for out_face, src_face in enumerate(face_order)
    }


def apply_yaw_offset(equi: np.ndarray, yaw_deg: float) -> np.ndarray:
    if yaw_deg == 0:
        return equi
    shift = int(round((yaw_deg / 360.0) * equi.shape[1]))
    return np.roll(equi, shift=shift, axis=1)


def cube_to_equirect(
    faces: dict[int, np.ndarray],
    out_h: int = 1024,
    out_w: int = 2048,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
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


def convert_scene(
    raw_root: Path,
    out_root: Path,
    scene: str,
    panos: set[str],
    cache_payloads: dict[str, list],
    force: bool,
    face_order: list[int],
    face_rotations: list[int],
    yaw_deg: float,
    out_h: int,
    out_w: int,
    depth_camera: int,
    pose_camera: int,
) -> tuple[int, int]:
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

            raw_color_faces = {}
            raw_depth_faces = {}
            for face in range(6):
                color_name = f"{pano}_skybox{face}_sami.jpg"
                depth_name = f"{pano}_d{depth_camera}_{face}.png"
                if color_name not in color_members or depth_name not in depth_members:
                    raise FileNotFoundError(f"Missing face {face} for {scene}/{pano}")
                raw_color_faces[face] = _read_zip_image(z_color, color_members[color_name], cv2.IMREAD_COLOR)
                raw_depth_faces[face] = _read_zip_image(z_depth, depth_members[depth_name], cv2.IMREAD_UNCHANGED)

            color_faces = remap_faces(raw_color_faces, face_order, face_rotations)
            depth_faces = remap_faces(raw_depth_faces, face_order, face_rotations)
            color = apply_yaw_offset(cube_to_equirect(color_faces, out_h=out_h, out_w=out_w, interpolation=cv2.INTER_LINEAR), yaw_deg)
            depth = apply_yaw_offset(cube_to_equirect(depth_faces, out_h=out_h, out_w=out_w, interpolation=cv2.INTER_NEAREST), yaw_deg)
            cv2.imwrite(str(color_path), color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(str(depth_path), depth)

            pose_name = f"{pano}_pose_{pose_camera}_0.txt"
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
    parser.add_argument("--only-scene", default=None, help="Convert only this Matterport scene id.")
    parser.add_argument("--only-pano", default=None, help="Convert only this panorama id.")
    parser.add_argument(
        "--face-order",
        default=MATTERPORT_FACE_ORDER,
        help="Raw skybox face index for output cube faces front,right,back,left,up,down. "
        "Matterport skybox0 is up and skybox5 is down, so the default is 1,2,3,4,0,5.",
    )
    parser.add_argument(
        "--face-rotations",
        default=MATTERPORT_FACE_ROTATIONS,
        help="Per output face rotation in 90-degree CCW turns.",
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Horizontal roll/yaw offset applied after equirect conversion.")
    parser.add_argument("--out-h", type=int, default=1024)
    parser.add_argument("--out-w", type=int, default=2048)
    parser.add_argument("--depth-camera", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--pose-camera", type=int, default=0, choices=[0, 1, 2])
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

    if args.only_scene:
        if args.only_scene not in needed:
            needed[args.only_scene] = set()
        scenes = [args.only_scene]
        if args.only_pano:
            needed[args.only_scene] = {args.only_pano}
    else:
        scenes = sorted(needed)
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]
    face_order = parse_int_list(args.face_order, 6, "--face-order")
    face_rotations = parse_int_list(args.face_rotations, 6, "--face-rotations")
    total_done = total_skipped = 0
    for idx, scene in enumerate(scenes, 1):
        done, skipped = convert_scene(
            raw_root,
            out_root,
            scene,
            needed[scene],
            cache_payloads,
            args.force,
            face_order,
            face_rotations,
            args.yaw_deg,
            args.out_h,
            args.out_w,
            args.depth_camera,
            args.pose_camera,
        )
        total_done += done
        total_skipped += skipped
        print(f"[{idx}/{len(scenes)}] {scene}: converted={done} skipped={skipped}")
    print(f"done converted={total_done} skipped={total_skipped} scenes={len(scenes)} out={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
