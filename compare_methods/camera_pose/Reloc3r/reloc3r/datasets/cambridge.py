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


DATA_ROOT = './data/cambridge'


def label_to_str(label):
    return '_'.join(label)


def rotation_from_quaternion(quad):
    norm = np.linalg.norm(quad)
    if norm < 1e-10:
        raise ValueError("Error! the quaternion is not robust. quad.norm() = {0}".format(norm))
    quad = quad / norm
    qr, qi, qj, qk = quad[0], quad[1], quad[2], quad[3]
    rot_mat = np.zeros((3, 3))
    rot_mat[0,0] = 1 - 2 * (qj ** 2 + qk ** 2)
    rot_mat[0,1] = 2 * (qi * qj - qk * qr)
    rot_mat[0,2] = 2 * (qi * qk + qj * qr)
    rot_mat[1,0] = 2 * (qi * qj + qk * qr)
    rot_mat[1,1] = 1 - 2 * (qi ** 2 + qk ** 2)
    rot_mat[1,2] = 2 * (qj * qk - qi * qr)
    rot_mat[2,0] = 2 * (qi * qk - qj * qr)
    rot_mat[2,1] = 2 * (qj * qk + qi * qr)
    rot_mat[2,2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return rot_mat


def ReadModelVisualSfM(vsfm_path, nvm_file="reconstruction.nvm"):
    input_file = os.path.join(vsfm_path, nvm_file)
    if not os.path.exists(input_file):
        raise ValueError("Error! Input file {0} does not exist.".format(input_file))
    with open(input_file, 'r') as f:
        txt_lines = f.readlines()

    # read camviews
    counter = 2  # start from the third line
    n_images = int(txt_lines[counter].strip())
    counter += 1
    camviews = []

    params_dict = {}
    for img_id in range(n_images):
        line = txt_lines[counter].strip().split()
        counter += 1

        imname = os.path.join(vsfm_path, line[0])
        f = float(line[1])
        qvec = np.array([float(line[k]) for k in np.arange(2, 6).tolist()])
        center_vec = np.array([float(line[k]) for k in np.arange(6, 9).tolist()])
        k1 = -float(line[9])

        # camera
        imname = imname.replace('.jpg', '.png')
        if not os.path.exists(imname):
            raise ValueError("Error! Image not found: {0}".format(imname))
        width, height = imagesize.get(imname)
        img_hw = [height, width]
        # TODO: check cx cy 
        cx = img_hw[1] / 2.0
        cy = img_hw[0] / 2.0
        intrinsics = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)

        # view
        R = rotation_from_quaternion(qvec)
        T = -R @ center_vec
        Rt = np.identity(4)
        Rt[0:3,0:3] = R
        Rt[0:3,3] = T
        pose = np.linalg.inv(Rt)  # cam2world
        
        params_dict[imname] = {'intrinsics': intrinsics, 'pose_c2w':pose}
        
    return params_dict


class CambridgeRelpose(BaseStereoViewDataset):
    
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
        self.params_dict = ReadModelVisualSfM('{}/{}'.format(self.root_folder, self.scene))
        self.intrinsics = np.array([[1671.31, 0.0, 960.0], [0.0, 1671.31, 540.0], [0.0, 0.0, 1.0]], dtype=float)
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

            intrinsics = self.params_dict[view_path]['intrinsics'].astype(np.float32)  
            pose = self.params_dict[view_path]['pose_c2w'].astype(np.float32)  

            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     copy.deepcopy(intrinsics), 
                                                                     resolution, 
                                                                     rng=rng)

            views.append(dict(
                img = color_image,
                camera_intrinsics = intrinsics,
                camera_pose = pose,
                dataset = 'Cambridge-' + self.scene,
                label = label_to_str(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                ))
        return views

