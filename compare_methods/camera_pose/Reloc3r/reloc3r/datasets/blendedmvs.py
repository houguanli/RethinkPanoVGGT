import os.path as osp
import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2


DATA_ROOT = "./data/blendedmvs_processed" 


class BlendedMVS (BaseStereoViewDataset):
    def __init__(self, *args, ROOT=DATA_ROOT, split=None, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self._load_data(split)

    def _load_data(self, split):
        pairs = np.load(osp.join(self.ROOT, 'blendedmvs_pairs.npy'))
        if split is None:
            selection = slice(None)
        if split == 'train':
            # select 90% of all scenes
            selection = (pairs['seq_low'] % 10) > 0
        if split == 'val':
            # select 10% of all scenes
            selection = (pairs['seq_low'] % 10) == 0
        self.pairs = pairs[selection]

        # list of all scenes
        self.scenes = np.unique(self.pairs['seq_low'])  # low is unique enough

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        seqh, seql, img1, img2, score = self.pairs[pair_idx]

        seq = f"{seqh:08x}{seql:016x}"
        seq_path = osp.join(self.ROOT, seq)

        views = []

        for view_index in [img1, img2]:
            impath = f"{view_index:08n}"
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']

            image, intrinsics = self._crop_resize_if_necessary(
                image, intrinsics, resolution, rng=rng, info=(seq_path, impath))

            views.append(dict(
                img=image,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='BlendedMVS',
                label=osp.relpath(seq_path, self.ROOT),
                instance=impath))

        return views

