import os.path as osp
import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2
import h5py

DATA_ROOT='./data/megadepth1500' 

class MegaDepth_valid(BaseStereoViewDataset):
    def __init__(self, *args, **kwargs):
        self.ROOT = DATA_ROOT
        super().__init__(*args, **kwargs)
        self.metadata = dict(np.load(osp.join(self.ROOT, f'megadepth_meta_test.npz'), allow_pickle=True))
        with open(osp.join(self.ROOT, f'megadepth_test_pairs.txt'), 'r') as f:
            self.scenes = f.readlines()
        self.load_depth = False

    def __len__(self):
        return len(self.scenes)
    
    def _get_views(self, idx, resolution,  rng):
        """
        load data for megadepth_validation views
        """
        # load metadata
        views = []
        image_idx1, image_idx2 = self.scenes[idx].strip().split(' ')
        view_idxs = [image_idx1, image_idx2]
        for view_idx in view_idxs:
            input_image_filename = osp.join(self.ROOT, view_idx)
            # load rgb images
            input_rgb_image = imread_cv2(input_image_filename)
            # load metadata
            intrinsics = np.float32(self.metadata[view_idx].item()['intrinsic'])
            camera_pose = np.linalg.inv(np.float32(self.metadata[view_idx].item()['pose']))

            image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=(self.ROOT, view_idx))
            
            views.append(dict(
                img=image,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=self.ROOT,
                instance=view_idx))
        return views