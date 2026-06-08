import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation as R
# from pdb import set_trace as bb


class Reloc3rVisloc:
    def __init__(self):
        super(Reloc3rVisloc, self).__init__()

    def rotation_averaging(self, matrices): 
        """
        Args:
            matrices (np.ndarray): shape of (N, 3, 3), absolute rotation matrices 
        Returns:
            avg_rotation_matrix (np.ndarray): shape of (3, 3), absolute rotation matrix
        """
        quaternions = [R.from_matrix(mat).as_quat() for mat in matrices]
        quaternions = np.array(quaternions)
        # ret_quaternion = np.mean(quaternions, axis=0)
        ret_quaternion = np.median(quaternions, axis=0)  # slightly better than mean
        norm = np.linalg.norm(ret_quaternion)
        ret_quaternion /= norm
        avg_rotation_matrix = R.from_quat(ret_quaternion).as_matrix()
        return avg_rotation_matrix

    def camera_center_triangulation(self, points):  
        """
        Args:
            points (np.ndarray): shape of (N, 2, 3), 3D coordinates of the 2 endpoints of N lines (directions)
        Returns:
            x (np.ndarray): shape of (3), least squares intersection point
        """
        n = points.shape[0]
        p = points[:, 0, :]
        q = points[:, 1, :]
        d = q - p
        d_norm_sq = np.sum(d ** 2, axis=1, keepdims=True)
        eye_3 = np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(n, axis=0)
        d_dT = np.expand_dims(d, axis=2) * np.expand_dims(d, axis=1)
        A_blocks = eye_3 - d_dT / d_norm_sq.reshape(n, 1, 1)
        A = A_blocks.reshape(-1, 3)
        b_blocks = np.matmul(eye_3 - d_dT / d_norm_sq.reshape(n, 1, 1), np.expand_dims(p, axis=2))
        b = b_blocks.reshape(-1)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1 / S)
        x = np.dot(Vt.T, np.dot(S_inv, np.dot(U.T, b)))
        return x

    def camera_center_triangulation_torch(self, points):  
        """
        Args:
            points (torch.Tensor): shape of (N, 2, 3), 3D coordinates of the 2 endpoints of N lines (directions)
        Returns:
            x (torch.Tensor): shape of (3), least squares intersection point
        """
        n = points.shape[0]
        p = points[:, 0, :]
        q = points[:, 1, :]
        d = q - p
        d_norm_sq = (d ** 2).sum(dim=1, keepdim=True)
        eye_3 = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        d_dT = d.unsqueeze(2) * d.unsqueeze(1)
        A_blocks = eye_3 - d_dT / d_norm_sq.unsqueeze(2)
        A = A_blocks.reshape(-1, 3)
        b_blocks = (eye_3 - d_dT / d_norm_sq.unsqueeze(2)) @ p.unsqueeze(2)
        b = b_blocks.reshape(-1)
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        S_inv = torch.diag(1 / S)
        x = Vt.T @ (S_inv @ (U.T @ b))
        return x

    def motion_averaging(self, poses_db, poses_q2d): 
        """
        Args:
            poses_db (List): list of (4, 4) absolute poses that transform points from camera to world
            poses_q2d (List): list of (4, 4) relative poses that transform query to dbs
        Returns:
            Rt (np.ndarray): shape of (4, 4), absolute pose of query 
        """
        assert len(poses_db) == len(poses_q2d)
        qR = []
        lines = []
        for pid in range(len(poses_db)):
            pose_q = poses_db[pid] @ poses_q2d[pid]
            qR.append(pose_q[0:3,0:3])
            p_beg = poses_db[pid][0:3,3]
            p_end = (poses_db[pid] @ poses_q2d[pid][0:4,3])[0:3]
            endpoints = np.concatenate((p_beg[None,...], p_end[None,...]), axis=0)
            lines.append(endpoints)
        
        # rotation averaging
        avg_qR = self.rotation_averaging(np.array(qR))

        # camera center triangulation
        cam_cen = self.camera_center_triangulation(np.array(lines))
        # cam_cen = self.camera_center_triangulation_torch(torch.tensor(np.array(lines)))
        
        Rt = np.identity(4)
        Rt[0:3,0:3] = avg_qR
        Rt[0:3,3] = cam_cen
        return Rt

