import numpy as np
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
# from pdb import set_trace as bb


DATA_ROOT = './data/scannet1500' 


def label_to_str(label):
    return '_'.join(label)


class ScanNet1500(BaseStereoViewDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT
        self.pairs_path = '{}/test.npz'.format(self.data_root)
        self.subfolder_mask = 'scannet_test_1500/scene{:04d}_{:02d}'
        with np.load(self.pairs_path) as data:
            self.pair_names = data['name']

    def __len__(self):
        return len(self.pair_names)

    def _get_views(self, idx, resolution, rng):
        scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]

        views = []

        for name in [name1, name2]: 
            
            color_path = '{}/{}/color/{}.jpg'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            color_image = imread_cv2(color_path)  
            color_image = cv2.resize(color_image, (640, 480))

            intrinsics_path = '{}/{}/intrinsic/intrinsic_depth.txt'.format(self.data_root, self.subfolder_mask).format(scene_name, scene_sub_name)
            intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[0:3,0:3]

            pose_path = '{}/{}/pose/{}.txt'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     intrinsics, 
                                                                     resolution, 
                                                                     rng=rng)

            view_idx_splits = color_path.split('/')

            views.append(dict(
                img = color_image,
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'ScanNet1500',
                label = label_to_str(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                ))
        return views

