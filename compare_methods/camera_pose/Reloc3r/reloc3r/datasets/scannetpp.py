import os.path as osp
import numpy as np
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2


DATA_ROOT = './data/scannetpp_processed' 


class ScanNetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = DATA_ROOT
        super().__init__(*args, **kwargs)
        assert self.split == 'train' # only include training data for scannetpp
        self.loaded_data = self._load_data()

    def _load_data(self):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        for view_idx in [image_idx1, image_idx2]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            color_image = imread_cv2(osp.join(scene_dir, 'images', basename + '.jpg'))
            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     intrinsics, 
                                                                     resolution, 
                                                                     rng=rng)

            views.append(dict(
                img=color_image,
                camera_intrinsics=intrinsics.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),  # cam2world
                dataset='ScanNet++',
                label=self.scenes[scene_id] + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views

