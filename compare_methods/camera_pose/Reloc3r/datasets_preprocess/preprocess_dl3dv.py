import argparse
import random
import json
import os
import os.path as osp
import itertools

import numpy as np
import cv2

from tqdm.auto import tqdm
import pickle
import json
from pdb import set_trace as bb
import trimesh


# Function to create a camera frame using a transformation matrix
def create_camera_pose_frame(translation, rotation_matrix, scale=0.1):
    """
    Create a coordinate frame representing the camera's pose.

    :param translation: (3,) Translation vector for camera position
    :param rotation_matrix: (3, 3) Rotation matrix for camera orientation
    :param scale: Scaling factor for the size of the frame (default: 0.1)
    :return: A Trimesh object representing the camera pose frame
    """
    # Create a transformation matrix combining rotation and translation
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation

    # Create the frame and apply the transformation matrix
    camera_frame = trimesh.creation.axis(origin_size=0.005, axis_length=scale)
    camera_frame.apply_transform(transform)
    return camera_frame


def get_camera_info(seq_path, img_size):
    with open(os.path.join(seq_path, 'transforms.json'), 'r') as file:
        data = json.load(file)
    H = data['h']
    W = data['w']
    scale_h = img_size[0] / H
    scale_w = img_size[1] / W
    cx = data['cx'] * scale_w
    cy = data['cy'] * scale_h
    focal_length_x = data['fl_x'] * scale_w
    focal_length_y = data['fl_y'] * scale_h
    K = np.eye(3)
    K[0,0] = focal_length_x
    K[1,1] = focal_length_y
    K[0,2] = cx
    K[1,2] = cy
    frames = data['frames']
    poses = {}
    for frame in frames:
        file_path = frame['file_path'].replace("images", "images_8")
        transform_matrix = np.array(frame['transform_matrix'])
        poses[file_path] = transform_matrix # c2w, opengl coordinate, need to convert it to opencv/colmap coordinate

    return K, poses

def process_seq(seq_path, img_size):
    scene_id, seq_name = seq_path.split('/')[-2], seq_path.split('/')[-1]
    K, poses = get_camera_info(seq_path, img_size)
    #for each scene, we have len(poses) images ==> around 360 degrees (so int(len(poses) * 30 / 360) frames ~= 45 degrees)
    n_images = len(poses)
    combinations = [(i, j) for i, j in itertools.combinations(range(0, n_images), 2)
                                if 0 < abs(i-j) <= int(n_images * 45 / 360)  and abs(i-j) % 5 == 0]
    bb()
    impath = osp.join(seq_path, "images_8/frame_00001.png")
    input_rgb_image = cv2.imread(impath)
    H, W = input_rgb_image.shape[:2]
    # ramdom sample 500 pairs
    n_samples = min(500, len(combinations))
    sampled_combinations = random.sample(combinations, n_samples)
    # sampled_combinations_sorted = sorted(sampled_combinations, key=lambda x: sampled_combinations.index(x))
    image_names = list(poses.keys())
    pairs = []
    for combination in sampled_combinations:
        idx0, idx1 = combination
        image_name0, image_name1 = image_names[idx0], image_names[idx1]
        if  int(image_name0[15: 20]) != idx0+1 or int(image_name1[15: 20]) != idx1+1 or H != img_size[0] or W != img_size[1]:
            return [] # skip the seq that are not in order or not in the right resolution
        assert int(image_name0[15: 20]) == idx0+1, "idx0 should align with image_name0"
        assert int(image_name1[15: 20]) == idx1+1, "idx1 should align with image_name1"
        pose0, pose1 = poses[image_name0], poses[image_name1]
        pair_info = (scene_id + '/' + seq_name, idx0, idx1, image_name0, image_name1, pose0, pose1, K )
        pairs.append(pair_info)
    return pairs
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl3dv_dir", type=str, default="/data/dataset/DL3DV-10K/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_quality", type=float, default=0.5, help="Minimum viewpoint quality score.")
    parser.add_argument("--output_dir",  type=str, default="/data/dataset/DL3DV-10K_processed/")
    parser.add_argument("--img_size", type=int, default=(270, 480),
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    assert args.img_size[0] == 270 and args.img_size[1] == 480, "input resolution invalid"
    random.seed(args.seed)
    for split in ['train', 'test']:
        with open(os.path.join(args.dl3dv_dir, f"{split}.txt"), 'r') as f:
            lines = f.readlines()
        pairs_all = []
        for line in tqdm(lines):
            seq_path = os.path.join(args.dl3dv_dir, line.strip())
            seq_pairs_info = process_seq(seq_path, img_size= args.img_size)
            pairs_all+=seq_pairs_info
        with open(os.path.join(args.dl3dv_dir, f"metadata_{split}.pkl"), 'wb') as f:
            pickle.dump(pairs_all, f)