import os.path as osp
import numpy as np
import json
import itertools
from collections import deque

import pickle

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2

from pdb import set_trace as bb


DATA_ROOT = "./data/dl3dv_processed"


class DL3DV(BaseStereoViewDataset):
    def __init__(self, split = 'train', *args, **kwargs):
        self.ROOT = DATA_ROOT
        super().__init__(*args, **kwargs)
        self.split = split

        # load all scenes
        with open(osp.join(self.ROOT, f'metadata_{self.split}.pkl'), 'rb') as f:
            self.scenes = pickle.load(f)

    def __len__(self):
        return len(self.scenes)

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        pair_info = self.scenes[idx]
        sence_name, label0, label1, pose0, pose1, K = pair_info[0], pair_info[3], pair_info[4], pair_info[5], pair_info[6], pair_info[7]

        views = []
        groups = [(sence_name, label0, pose0), (sence_name, label1, pose1)]
        for group in groups:
            scene, label, pose = group

            impath = osp.join(self.ROOT, '1K', scene, label)  

            # load image
            input_rgb_image = imread_cv2(impath)
            intrinsics = K.astype(np.float32)
            camera_pose = pose.astype(np.float32)
            
            # convert the camera pose from opengl to opencv
            camera_pose[2, :] *= -1
            camera_pose = camera_pose[np.array([1, 0, 2, 3]), :]
            camera_pose[0:3, 1:3] *= -1
            
            rgb_image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
            img=rgb_image,
            camera_pose=camera_pose,  # cam2world
            camera_intrinsics=intrinsics,
            dataset='DL3DV',
            label=self.ROOT,
            instance=osp.join(scene, label)))

        return views

