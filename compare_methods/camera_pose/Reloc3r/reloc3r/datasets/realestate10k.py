import os.path as osp
import numpy as np
import json
import itertools
from collections import deque

import pickle

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
from pdb import set_trace as bb


DATA_ROOT = './data/realestate10k_processed' 


class RealEstate(BaseStereoViewDataset):
    def __init__(self, split='train', *args, **kwargs):
        self.ROOT = DATA_ROOT
        super().__init__(*args, **kwargs)
        self.split = split

        # load all scenes
        scenes = np.load(osp.join(self.ROOT, f'{split}_metadata_pairs.npz'), allow_pickle=True)
        self.img_pairs = scenes['img_pairs']
        self.camera_params = scenes['camera_params']
        self.w2c = scenes['w2c']

    def __len__(self):
        return len(self.img_pairs)

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        img0, img1 = self.img_pairs[idx]
        K0, K1 = self.camera_params[idx]
        pose0, pose1 = self.w2c[idx]

        views = []
        groups = [(img0, K0, pose0), (img1, K1, pose1)]
        for group in groups:
            scene, K, pose = group
            impath = osp.join(self.ROOT, 'images', scene)
            # load image
            input_rgb_image = imread_cv2(impath)
            intrinsics = K.astype(np.float32)
            camera_pose = pose.astype(np.float32)  # w2c

            rgb_image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
            img=rgb_image,
            camera_pose=np.linalg.inv(camera_pose),  # cam2world
            camera_intrinsics=intrinsics,
            dataset='realestate',
            label=self.ROOT,
            instance=scene))

        return views

