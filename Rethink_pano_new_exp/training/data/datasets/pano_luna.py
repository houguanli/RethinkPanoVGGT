import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points, read_image_cv2, threshold_depth_map


class PanoLunaDataset(BaseDataset):
    """Loader for numbered whole-pano datasets produced by pano_data_remaker.py.

    Each sample keeps the whole panorama in `pano_image` and also generates the
    virtual pinhole windows needed by VGGT losses. The model path can consume
    `pano_images`, while the losses still see standard VGGT-style
    `images/depths/extrinsics/intrinsics/world_points`.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        Pano_DIR: str = "/YOUR/PATH/TO/PANO_LUNA",
        len_train: int = 100000,
        len_test: int = 10000,
        depth_scale: float = 100.0,
        depth_is_range: bool = True,
        depth_max: float = 80.0,
        fov_degrees: float = 75.0,
        num_yaw: int = 8,
        pitch_degrees: Sequence[float] = (0.0,),
        view_yaw_degrees: Optional[Sequence[float]] = None,
        pano_resize_width: int = 0,
        allow_missing_normal: bool = True,
        seam_width: float = 0.02,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.Pano_DIR = Pano_DIR
        self.depth_scale = depth_scale
        self.depth_is_range = depth_is_range
        self.depth_max = depth_max
        self.fov_degrees = fov_degrees
        self.fov_radians = math.radians(fov_degrees)
        self.num_yaw = num_yaw
        self.pitch_degrees = list(pitch_degrees)
        self.view_yaw_degrees = list(view_yaw_degrees) if view_yaw_degrees is not None else None
        self.pano_resize_width = pano_resize_width
        self.allow_missing_normal = allow_missing_normal
        self.seam_width = seam_width

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.sequence_list = self._load_sequence_list(split)
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise RuntimeError(f"No pano sequences found under {self.Pano_DIR}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Pano LUNA data size: {self.sequence_list_len}")
        logging.info(f"{status}: Pano LUNA dataset length: {len(self)}")

    def _load_sequence_list(self, split: str):
        root = Path(self.Pano_DIR)
        split_list = root / f"sequence_list_{split}.txt"
        default_list = root / "sequence_list.txt"
        list_path = split_list if split_list.exists() else default_list
        if list_path.exists():
            return [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index % self.sequence_list_len]

        sample_root = Path(self.Pano_DIR) / seq_name
        meta = self._read_meta(sample_root)

        pano_image = read_image_cv2(str(sample_root / meta.get("rgb", "rgb.png")))
        if pano_image is None:
            raise FileNotFoundError(f"Cannot read pano RGB under {sample_root}")

        depth_path = sample_root / meta.get("depth", "depth.png")
        pano_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if pano_depth is None:
            raise FileNotFoundError(f"Cannot read pano depth: {depth_path}")
        if pano_depth.ndim == 3:
            pano_depth = pano_depth[..., 0]
        pano_depth = pano_depth.astype(np.float32) / float(self.depth_scale)
        pano_depth = threshold_depth_map(pano_depth, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

        pano_normal = self._read_optional_normal(sample_root, meta)
        pano_image, pano_depth, pano_normal = self._maybe_resize_pano(pano_image, pano_depth, pano_normal)

        pose_c2w = self._read_pose_c2w(sample_root, meta)
        yaw, pitch = self._build_view_grid()
        fov_x = np.full_like(yaw, self.fov_radians, dtype=np.float32)
        fov_y = np.full_like(yaw, self.fov_radians, dtype=np.float32)

        windows, depths, normals, local_rays, token_meta = self._sample_virtual_windows(
            pano_image=pano_image,
            pano_depth=pano_depth,
            pano_normal=pano_normal,
            yaw=yaw,
            pitch=pitch,
            fov_x=fov_x,
            fov_y=fov_y,
        )

        intrinsics = []
        extrinsics = []
        cam_points = []
        world_points = []
        point_masks = []
        rotations_w2c = []

        for view_idx in range(len(yaw)):
            K = make_intrinsics(self.img_size, self.img_size, fov_x[view_idx], fov_y[view_idx])
            R_c2w_local = camera_c2w_opencv(yaw[view_idx], pitch[view_idx])
            R_c2w = pose_c2w[:3, :3] @ R_c2w_local
            C_world = pose_c2w[:3, 3]
            extri = c2w_to_w2c_3x4(R_c2w, C_world)

            depth_z = depths[view_idx]
            if self.depth_is_range:
                depth_z = depth_z * local_rays[view_idx][..., 2]
                valid = np.isfinite(depth_z) & (depth_z > 0.0) & (depth_z <= self.depth_max)
                clean = np.zeros_like(depth_z, dtype=np.float32)
                clean[valid] = depth_z[valid]
                depth_z = clean

            world_coords, cam_coords, point_mask = depth_to_world_coords_points(depth_z, extri, K)

            intrinsics.append(K)
            extrinsics.append(extri)
            rotations_w2c.append(extri[:3, :3])
            depths[view_idx] = depth_z
            cam_points.append(cam_coords)
            world_points.append(world_coords)
            point_masks.append(point_mask)

        ids = np.arange(len(yaw), dtype=np.int64)
        return {
            "seq_name": "pano_luna_" + seq_name,
            "ids": ids,
            "frame_num": len(yaw),
            "images": windows,
            "depths": depths,
            "normals": normals,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": [np.array(window.shape[:2]) for window in windows],
            "pano_image": pano_image,
            "pano_depth": pano_depth,
            "pano_normal": pano_normal,
            "pano_view_params": np.stack([yaw, pitch, fov_y, fov_x], axis=-1).astype(np.float32),
            "pano_rotations": np.stack(rotations_w2c).astype(np.float32),
            "pano_valid_mask": np.ones(len(yaw), dtype=bool),
            "is_pano": np.array(True),
            "pano_camera_6dof": self._read_camera_6dof(sample_root),
            "pano_token_meta": token_meta,
        }

    def _read_meta(self, sample_root: Path) -> Dict:
        meta_path = sample_root / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {"rgb": "rgb.png", "depth": "depth.png", "normal": "normal.png"}

    def _read_optional_normal(self, sample_root: Path, meta: Dict):
        normal_name = meta.get("normal", "normal.png")
        if normal_name is None:
            return None
        normal_path = sample_root / normal_name
        if not normal_path.exists():
            if self.allow_missing_normal:
                return None
            raise FileNotFoundError(f"Missing normal: {normal_path}")
        normal = read_image_cv2(str(normal_path))
        if normal is None and not self.allow_missing_normal:
            raise FileNotFoundError(f"Cannot read normal: {normal_path}")
        return normal

    def _maybe_resize_pano(self, image, depth, normal):
        if self.pano_resize_width is None or self.pano_resize_width <= 0:
            return image, depth, normal
        h, w = image.shape[:2]
        if w == self.pano_resize_width:
            return image, depth, normal
        scale = self.pano_resize_width / float(w)
        new_size = (int(round(w * scale)), int(round(h * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
        if normal is not None:
            normal = cv2.resize(normal, new_size, interpolation=cv2.INTER_LINEAR)
        return image, depth, normal

    def _read_pose_c2w(self, sample_root: Path, meta: Dict) -> np.ndarray:
        pose_path = sample_root / meta.get("pose_c2w", "pose_c2w.txt")
        if pose_path.exists():
            pose = np.loadtxt(pose_path).astype(np.float32)
            if pose.shape == (4, 4):
                return pose
        six = self._read_camera_6dof(sample_root)
        return pose_from_6dof(six)

    def _read_camera_6dof(self, sample_root: Path) -> np.ndarray:
        pose_path = sample_root / "camera_6dof.txt"
        if not pose_path.exists():
            return np.zeros(6, dtype=np.float32)
        rows = []
        for line in pose_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()[:6]])
        if not rows:
            return np.zeros(6, dtype=np.float32)
        return np.array(rows[0], dtype=np.float32)

    def _build_view_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.view_yaw_degrees is None:
            yaw = np.arange(self.num_yaw, dtype=np.float32) * (2.0 * math.pi / self.num_yaw) - math.pi
        else:
            yaw = np.array([math.radians(v) for v in self.view_yaw_degrees], dtype=np.float32)
        pitch = np.array([math.radians(v) for v in self.pitch_degrees], dtype=np.float32)
        yaw_grid, pitch_grid = np.meshgrid(yaw, pitch, indexing="ij")
        return yaw_grid.reshape(-1).astype(np.float32), pitch_grid.reshape(-1).astype(np.float32)

    def _sample_virtual_windows(self, pano_image, pano_depth, pano_normal, yaw, pitch, fov_x, fov_y):
        H, W = pano_depth.shape
        rays, local_rays = pinhole_rays_np(yaw, pitch, fov_x, fov_y, self.img_size, self.img_size)
        theta, phi, u, v = rays_to_equirectangular_np(rays)
        map_x = (u * (W - 1)).astype(np.float32)
        map_y = (v * (H - 1)).astype(np.float32)

        images = []
        depths = []
        normals = []
        for idx in range(len(yaw)):
            images.append(
                cv2.remap(pano_image, map_x[idx], map_y[idx], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            )
            depths.append(
                cv2.remap(pano_depth.astype(np.float32), map_x[idx], map_y[idx], interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
            )
            if pano_normal is not None:
                normals.append(
                    cv2.remap(pano_normal, map_x[idx], map_y[idx], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                )

        token_meta = self._build_token_meta(yaw, pitch, fov_x, fov_y, pano_hw=(H, W))
        return images, depths, normals if normals else None, local_rays, token_meta

    def _build_token_meta(self, yaw, pitch, fov_x, fov_y, pano_hw):
        pano_h, pano_w = pano_hw
        patch_h = self.img_size // self.patch_size
        patch_w = self.img_size // self.patch_size
        rays, _ = pinhole_rays_np(yaw, pitch, fov_x, fov_y, patch_h, patch_w)
        theta, phi, u, v = rays_to_equirectangular_np(rays)
        sphere_dir = rays.reshape(len(yaw), patch_h * patch_w, 3).astype(np.float32)
        theta = theta.reshape(len(yaw), patch_h * patch_w).astype(np.float32)
        phi = phi.reshape(len(yaw), patch_h * patch_w).astype(np.float32)
        u = u.reshape(len(yaw), patch_h * patch_w).astype(np.float32)
        v = v.reshape(len(yaw), patch_h * patch_w).astype(np.float32)

        global_w = max(1, int(math.ceil(pano_w / self.patch_size)))
        global_h = max(1, int(math.ceil(pano_h / self.patch_size)))
        global_x = np.clip(np.floor(u * global_w), 0, global_w - 1).astype(np.int64)
        global_y = np.clip(np.floor(v * global_h), 0, global_h - 1).astype(np.int64)

        local_y, local_x = np.meshgrid(np.arange(patch_h), np.arange(patch_w), indexing="ij")
        sphere_encoding = np.concatenate(
            [
                np.sin(theta)[..., None],
                np.cos(theta)[..., None],
                np.sin(phi)[..., None],
                np.cos(phi)[..., None],
                sphere_dir,
            ],
            axis=-1,
        ).astype(np.float32)
        return {
            "window_id": np.repeat(np.arange(len(yaw))[:, None], patch_h * patch_w, axis=1).astype(np.int64),
            "local_patch_x": np.repeat(local_x.reshape(1, -1), len(yaw), axis=0).astype(np.int64),
            "local_patch_y": np.repeat(local_y.reshape(1, -1), len(yaw), axis=0).astype(np.int64),
            "pano_u": (u * (pano_w - 1)).astype(np.float32),
            "pano_v": (v * (pano_h - 1)).astype(np.float32),
            "theta": theta,
            "phi": phi,
            "sphere_dir": sphere_dir,
            "sphere_encoding": sphere_encoding,
            "global_patch_id": (global_y * global_w + global_x).astype(np.int64),
            "is_seam_region": ((u <= self.seam_width) | (u >= 1.0 - self.seam_width)),
        }


def make_intrinsics(width: int, height: int, fov_x: float, fov_y: float) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = 0.5 * width / math.tan(0.5 * float(fov_x))
    K[1, 1] = 0.5 * height / math.tan(0.5 * float(fov_y))
    K[0, 2] = (width - 1) * 0.5
    K[1, 2] = (height - 1) * 0.5
    return K


def yaw_pitch_axes_np(yaw, pitch):
    yaw = np.asarray(yaw, dtype=np.float32)
    pitch = np.asarray(pitch, dtype=np.float32)
    cos_pitch = np.cos(pitch)
    forward = np.stack([cos_pitch * np.sin(yaw), np.sin(pitch), cos_pitch * np.cos(yaw)], axis=-1)
    right = np.stack([np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)], axis=-1)
    up = np.cross(forward, right)
    return normalize_np(forward), normalize_np(right), normalize_np(up)


def pinhole_rays_np(yaw, pitch, fov_x, fov_y, height: int, width: int):
    yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
    pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
    fov_x = np.asarray(fov_x, dtype=np.float32).reshape(-1)
    fov_y = np.asarray(fov_y, dtype=np.float32).reshape(-1)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    x_plane = xx[None] * np.tan(fov_x[:, None, None] * 0.5)
    y_plane_up = -yy[None] * np.tan(fov_y[:, None, None] * 0.5)
    y_plane_down = yy[None] * np.tan(fov_y[:, None, None] * 0.5)
    local_rays = normalize_np(np.stack([x_plane, y_plane_down, np.ones_like(x_plane)], axis=-1))
    forward, right, up = yaw_pitch_axes_np(yaw, pitch)
    rays = forward[:, None, None, :] + x_plane[..., None] * right[:, None, None, :] + y_plane_up[..., None] * up[:, None, None, :]
    return normalize_np(rays).astype(np.float32), local_rays.astype(np.float32)


def rays_to_equirectangular_np(rays):
    rays = normalize_np(rays)
    theta = np.arctan2(rays[..., 0], rays[..., 2])
    phi = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    u = np.mod(theta / (2.0 * math.pi) + 0.5, 1.0)
    v = np.clip(0.5 - phi / math.pi, 0.0, 1.0)
    return theta, phi, u, v


def camera_c2w_opencv(yaw: float, pitch: float) -> np.ndarray:
    forward, right, up = yaw_pitch_axes_np(np.array([yaw]), np.array([pitch]))
    return np.stack([right[0], -up[0], forward[0]], axis=1).astype(np.float32)


def c2w_to_w2c_3x4(R_c2w: np.ndarray, C_world: np.ndarray) -> np.ndarray:
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ C_world.reshape(3)
    return np.concatenate([R_w2c, t_w2c[:, None]], axis=1).astype(np.float32)


def pose_from_6dof(camera_6dof: np.ndarray) -> np.ndarray:
    tx, ty, tz, roll, pitch, yaw = camera_6dof.astype(np.float32).tolist()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = euler_xyz_to_matrix(roll, pitch, yaw)
    pose[:3, 3] = [tx, ty, tz]
    return pose


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def normalize_np(values, eps=1e-8):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)
