import numpy as np
import cv2
import torch
from torch.utils import data
from pdb import set_trace as bb
import os
# from tqdm import tqdm


DATA_ROOT = './data/7scenes'


stat_7Scenes = {
    'scene':[
        'chess', 
        'fire', 
        'heads', 
        'office', 
        'pumpkin', 
        'redkitchen', 
        'stairs'],
    'seqs_train':{
        'chess': [1,2,4,6],
        'fire': [1,2],
        'heads': [2],
        'office': [1,3,4,5,8,10],
        'pumpkin': [2,3,6,8],
        'redkitchen': [1,2,5,7,8,11,13],
        'stairs': [2,3,5,6]
        },
    'seqs_test':{
        'chess': [3,5],
        'fire': [3,4],
        'heads': [1],
        'office': [2,6,7,9],
        'pumpkin': [1,7],
        'redkitchen': [3,4,6,12,14],
        'stairs': [1,4]
        },
    'n_frames': {
        'chess': 1000, 
        'fire': 1000, 
        'heads': 1000, 
        'office': 1000, 
        'pumpkin': 1000, 
        'redkitchen': 1000, 
        'stairs': 500
        }
}
mask_gt_db_cam_7Scenes = 'seq-{:02d}_{:06d}_pose-db.txt'
mask_q2d_cam_7Scenes = 'seq-{:02d}_{:06d}_pose-q2d.txt'
mask_gt_q_cam_7Scenes = 'seq-{:02d}_{:06d}_pose-gt.txt'


class SevenScenesRetrieval:
    def __init__(self, 
                 scene, 
                 split): 
        super(SevenScenesRetrieval, self).__init__()
        self.root_folder = DATA_ROOT
        self.color_file_format = 'frame-{:06d}.color.png'
        self.scene = scene
        self.split = split
        self.names_color = []

        assert self.split in ['train', 'test']
        seqs = stat_7Scenes['seqs_{}'.format(self.split)][self.scene]
        for seq in seqs:
            for fid in range(stat_7Scenes['n_frames'][scene]):
                name = '{}/{}/seq-{:02d}/{}'.format(self.root_folder, self.scene, seq, self.color_file_format).format(fid)
                self.names_color.append(name)

    def load_image(self, name, device):
        color = cv2.imread(name)
        data = {}
        data['image'] = torch.Tensor(color).permute(2,0,1)[None].to(device)
        data['image'] = (data['image'] - data['image'].min()) / (data['image'].max() - data['image'].min())
        return data

