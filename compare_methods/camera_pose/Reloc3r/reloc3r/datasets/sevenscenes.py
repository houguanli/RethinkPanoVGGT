from pdb import set_trace as bb
import os.path as osp
import numpy as np
import sys # noqa: E402
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
from pdb import set_trace as bb
import copy
import os
import imagesize


from reloc3r.image_retrieval.topk_retrieval import PREPROCESS_FOLDER, DB_DESCS_FILE_MASK, PAIR_INFO_FILE_MASK


DATA_ROOT = './data/7scenes'


def label_to_str(label):
    return '_'.join(label)


class SevenScenesRelpose(BaseStereoViewDataset):
    
    def __init__(self, 
                 scene, 
                 pairs_info_file_mask=PREPROCESS_FOLDER + '/' + PAIR_INFO_FILE_MASK, 
                 db_step=1,
                 topk=10,
                 pair_id=None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_folder = DATA_ROOT
        self.scene = scene
        self.intrinsics = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.pairs_path = pairs_info_file_mask.format(scene, db_step, topk)
        # load pair info 
        self.loaded_data = {}
        pairs_info = np.load(self.pairs_path, allow_pickle=True)  # list len #images, each len #topk, each element (idx, name)
        for test_fid in range(len(pairs_info)):
            pairs = pairs_info[test_fid]
            img2s = []
            for pair_info in pairs:
                img2s.append(pair_info['db_path']) 
            self.loaded_data[pairs[0]['query_path']] = img2s
        self.queries = list(self.loaded_data)
        self.pair_id = pair_id

    def __len__(self):
        return len(self.queries)

    def _get_views(self, idx, resolution, rng):
        # choose a pair from topk
        image_path1 = self.queries[idx]
        image_paths2 = self.loaded_data[image_path1]
        if self.pair_id is None:
            image_path2 = image_paths2[rng.integers(0,len(image_paths2))]
        else:
            image_path2 = image_paths2[self.pair_id]

        views = []

        for view_path in [image_path2, image_path1]:  # [database, query]
            view_idx_splits = view_path.split('/')

            color_image = imread_cv2(view_path)

            pose = np.loadtxt(view_path.replace('.color.png', '.pose.txt')).astype(np.float32)

            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     copy.deepcopy(self.intrinsics), 
                                                                     resolution, 
                                                                     rng=rng)

            views.append(dict(
                img = color_image,
                camera_intrinsics = intrinsics,
                camera_pose = pose,
                dataset = '7Scenes-' + self.scene,
                label = label_to_str(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                ))
        return views

