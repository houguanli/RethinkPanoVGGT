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
MATTERPORT_SKYBOX_TRANSFORMS = (
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    np.eye(3, dtype=np.float64),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
)


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


def parse_undistorted_camera_parameters(text: str):
    intrinsics: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    extrinsics: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    current_k = None

    for line in text.splitlines():
        if line.startswith("intrinsics_matrix"):
            vals = [float(x) for x in line.split()[1:]]
            current_k = np.array(vals, dtype=np.float64).reshape(3, 3)
            continue
        if not line.startswith("scan "):
            continue
        if current_k is None:
            raise RuntimeError(f"Camera scan entry before intrinsics: {line}")

        parts = line.split()
        depth_name = parts[1]
        color_name = parts[2]
        pano = depth_name.rsplit("_d", 1)[0]
        camera_key = color_name.rsplit(".", 1)[0].rsplit("_i", 1)[1]
        camera_group = camera_key.split("_", 1)[0]
        pose_c2w = np.array([float(x) for x in parts[3:19]], dtype=np.float64).reshape(4, 4)
        intrinsics[pano][camera_group] = current_k.copy()
        extrinsics[pano][camera_key] = (pose_c2w, np.linalg.inv(pose_c2w))

    return intrinsics, extrinsics


def _skybox_intrinsic(width: int, height: int) -> np.ndarray:
    k = np.zeros((3, 3), dtype=np.float64)
    k[0, 0] = width / 2.0
    k[1, 1] = height / 2.0
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[2, 2] = 1.0
    return k


def _z_depth_to_euclidean(k_inv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.indices((h, w))
    pix = np.vstack((xx.reshape(-1), yy.reshape(-1), np.ones(xx.size)))
    rays = k_inv.dot(pix)
    cos_theta = np.array([0.0, 0.0, 1.0], dtype=np.float64).dot(rays) / np.linalg.norm(rays, axis=0)
    out = depth.astype(np.float64) / cos_theta.reshape(h, w)
    out[~np.isfinite(out)] = 0
    out[out < 0] = 0
    out[out > 65535] = 65535
    return out.astype(np.uint16)


def fill_small_depth_holes(depth: np.ndarray, max_area: int) -> np.ndarray:
    valid = depth > 0
    invalid = (~valid).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, connectivity=8)
    fill_mask = np.zeros_like(invalid, dtype=np.uint8)
    for label in range(1, num):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or (x + w) >= depth.shape[1] or (y + h) >= depth.shape[0]
        if not touches_border and area <= max_area:
            fill_mask[labels == label] = 255
    if not np.any(fill_mask):
        return depth

    values = depth[valid].astype(np.float32)
    if values.size == 0:
        return depth
    lo, hi = np.percentile(values, [1, 99])
    scale = max(float(hi - lo), 1.0)
    depth_8u = np.clip((depth.astype(np.float32) - lo) / scale * 255.0, 0, 255).astype(np.uint8)
    filled_8u = cv2.inpaint(depth_8u, fill_mask, 5, cv2.INPAINT_TELEA)
    filled = depth.copy()
    locs = fill_mask > 0
    filled[locs] = np.clip(filled_8u[locs].astype(np.float32) / 255.0 * scale + lo, 0, 65535).astype(np.uint16)
    return filled


def project_undistorted_depth_to_equirect(
    z_depth: zipfile.ZipFile,
    depth_members: dict[str, str],
    pano: str,
    pano_intrinsics: dict[str, np.ndarray],
    pano_extrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
    out_h: int,
    out_w: int,
    hole_fill: str,
    small_hole_area: int,
    depth_max_m: float,
) -> np.ndarray:
    if len(pano_intrinsics) < 3 or len(pano_extrinsics) < 18:
        raise RuntimeError(f"Missing undistorted camera parameters for {pano}")

    depth_euclidean = {}
    for camera in range(3):
        k_inv = np.linalg.inv(pano_intrinsics[str(camera)])
        for angle in range(6):
            key = f"{camera}_{angle}"
            depth_name = f"{pano}_d{key}.png"
            if depth_name not in depth_members:
                raise FileNotFoundError(depth_name)
            depth_z = _read_zip_image(z_depth, depth_members[depth_name], cv2.IMREAD_ANYDEPTH)
            depth_euclidean[key] = _z_depth_to_euclidean(k_inv, depth_z)

    face_size = 1024
    k_skybox = _skybox_intrinsic(face_size, face_size)
    center_key = "1_5"
    if center_key not in pano_extrinsics:
        raise RuntimeError(f"Missing central skybox reference pose {pano}_i{center_key}")
    center_c2w, _ = pano_extrinsics[center_key]
    skybox_faces = {}
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for skybox_ix, transform in enumerate(MATTERPORT_SKYBOX_TRANSFORMS):
        skybox_c2w_rot = center_c2w[:3, :3].dot(transform)
        skybox_wtc_rot = np.linalg.inv(skybox_c2w_rot)
        face_depth = np.zeros((face_size, face_size), dtype=np.uint16)

        for camera in range(3):
            k_im = pano_intrinsics[str(camera)]
            for angle in range(6):
                key = f"{camera}_{angle}"
                if key not in pano_extrinsics:
                    raise RuntimeError(f"Missing undistorted pose {pano}_i{key}")
                pose_c2w, _ = pano_extrinsics[key]
                image_c2w_rot = pose_c2w[:3, :3]
                if image_c2w_rot.dot(z_axis).dot(skybox_c2w_rot.dot(z_axis)) < 0:
                    continue
                homography = k_skybox.dot(skybox_wtc_rot.dot(image_c2w_rot.dot(np.linalg.inv(k_im))))
                src_depth = cv2.flip(depth_euclidean[key], 1)
                warped = cv2.warpPerspective(src_depth, homography, (face_size, face_size), flags=cv2.INTER_NEAREST)
                mask = cv2.warpPerspective(
                    np.ones_like(src_depth, dtype=np.uint8),
                    homography,
                    (face_size, face_size),
                    flags=cv2.INTER_LINEAR,
                )
                mask[warped == 0] = 0
                mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
                locs = np.where(mask == 1)
                face_depth[locs] = warped[locs]

        skybox_faces[skybox_ix] = cv2.flip(face_depth, 1)

    equirect_faces = {
        0: skybox_faces[1],
        1: skybox_faces[2],
        2: skybox_faces[3],
        3: skybox_faces[4],
        4: skybox_faces[0],
        5: skybox_faces[5],
    }
    depth = cube_to_equirect(equirect_faces, out_h=out_h, out_w=out_w, interpolation=cv2.INTER_NEAREST)
    if hole_fill == "small":
        scaled_area = max(1, int(round(small_hole_area * (out_h * out_w) / float(1024 * 2048))))
        depth = fill_small_depth_holes(depth, scaled_area)
    if depth_max_m > 0:
        depth = depth.copy()
        depth[depth.astype(np.float32) / 4000.0 > depth_max_m] = 0
    return depth


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
    depth_source: str,
    hole_fill: str,
    small_hole_area: int,
    depth_max_m: float,
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
            zipfile.ZipFile(scene_raw / "matterport_camera_poses.zip") as z_pose:
        color_members = zip_member_lookup(z_color)
        pose_members = zip_member_lookup(z_pose)
        if depth_source == "undistorted":
            z_depth = zipfile.ZipFile(scene_raw / "undistorted_depth_images.zip")
            z_params = zipfile.ZipFile(scene_raw / "undistorted_camera_parameters.zip")
            conf_text = z_params.read(f"{scene}/undistorted_camera_parameters/{scene}.conf").decode("utf-8")
            scene_intrinsics, scene_extrinsics = parse_undistorted_camera_parameters(conf_text)
        else:
            z_depth = zipfile.ZipFile(scene_raw / "matterport_depth_images.zip")
            z_params = None
            scene_intrinsics = scene_extrinsics = None
        depth_members = zip_member_lookup(z_depth)
        done = 0
        skipped = 0
        try:
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
                    if color_name not in color_members:
                        raise FileNotFoundError(f"Missing color face {face} for {scene}/{pano}")
                    raw_color_faces[face] = _read_zip_image(z_color, color_members[color_name], cv2.IMREAD_COLOR)
                    if depth_source == "legacy":
                        depth_name = f"{pano}_d{depth_camera}_{face}.png"
                        if depth_name not in depth_members:
                            raise FileNotFoundError(f"Missing depth face {face} for {scene}/{pano}")
                        raw_depth_faces[face] = _read_zip_image(z_depth, depth_members[depth_name], cv2.IMREAD_UNCHANGED)

                color_faces = remap_faces(raw_color_faces, face_order, face_rotations)
                color = cube_to_equirect(color_faces, out_h=out_h, out_w=out_w, interpolation=cv2.INTER_LINEAR)
                if depth_source == "undistorted":
                    depth = project_undistorted_depth_to_equirect(
                        z_depth,
                        depth_members,
                        pano,
                        scene_intrinsics.get(pano, {}),
                        scene_extrinsics.get(pano, {}),
                        out_h,
                        out_w,
                        hole_fill,
                        small_hole_area,
                        depth_max_m,
                    )
                else:
                    depth_faces = remap_faces(raw_depth_faces, face_order, face_rotations)
                    depth = cube_to_equirect(depth_faces, out_h=out_h, out_w=out_w, interpolation=cv2.INTER_NEAREST)
                    if depth_max_m > 0:
                        depth = depth.copy()
                        depth[depth.astype(np.float32) / 4000.0 > depth_max_m] = 0
                color = apply_yaw_offset(color, yaw_deg)
                depth = apply_yaw_offset(depth, yaw_deg)
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
        finally:
            z_depth.close()
            if z_params is not None:
                z_params.close()
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
    parser.add_argument("--depth-source", default="undistorted", choices=["undistorted", "legacy"])
    parser.add_argument("--hole-fill", default="small", choices=["none", "small"])
    parser.add_argument("--small-hole-area", type=int, default=18000)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
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
            args.depth_source,
            args.hole_fill,
            args.small_hole_area,
            args.depth_max_m,
        )
        total_done += done
        total_skipped += skipped
        print(f"[{idx}/{len(scenes)}] {scene}: converted={done} skipped={skipped}")
    print(f"done converted={total_done} skipped={total_skipped} scenes={len(scenes)} out={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
