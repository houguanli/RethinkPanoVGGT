import numpy as np
import cv2
import torch
from torch.utils import data
from pdb import set_trace as bb
import os
# from tqdm import tqdm


DATA_ROOT = './data/cambridge'


stat_Cambridge = {
    'GreatCourt':{
        'seq': [1,4],
        'range':{
            '1': (1,612),
            '4': (1,149),
        }
    },
    'KingsCollege':{
        'seq': [2,3,7],
        'range':{
            '2': (1,61),
            '3': (1,179),  # 181 no GT
            '7': (1,103),
        },
    },
    'OldHospital':{
        'seq': [4,8],
        'range':{
            '4': (1,56),
            '8': (1,126),
        },
    },
    'ShopFacade':{
        'seq': [1,3],
        'range':{
            '1': (1,36),
            '3': (1,67),
        },
    },
    'StMarysChurch':{
        'seq': [3,5,13],
        'range':{
            '3': (1,98),
            '5': (1,82),
            '13': (1,350),
        }
    },
}
mask_gt_db_cam_Cambridge = 'seq{}_frame{:05d}_pose-db.txt'
mask_q2d_cam_Cambridge = 'seq{}_frame{:05d}_pose-q2d.txt'
mask_gt_q_cam_Cambridge = 'seq{}_frame{:05d}_pose-gt.txt'


def read_frames_of_the_split(file_path): 
    if not os.path.exists(file_path):
        raise ValueError("Error! Input file {0} does not exist.".format(file_path))
    with open(file_path, 'r') as f:
        txt_lines = f.readlines()
    names = []
    for idx in range(3, len(txt_lines)):
        name = txt_lines[idx].rstrip('\n').split()[0]
        names.append(name)
    names = sorted(names)
    return names


class CambridgeRetrieval:
    def __init__(self, 
                 scene, 
                 split): 
        super(CambridgeRetrieval, self).__init__()
        self.root_folder = DATA_ROOT
        self.scene = scene
        self.split = split
        self.names_color = []
        names = None
        assert self.split in ['train', 'test']
        names = read_frames_of_the_split('{}/{}/dataset_{}.txt'.format(self.root_folder, self.scene, self.split))
        for name in names:
            self.names_color.append('{}/{}/{}'.format(self.root_folder, self.scene, name))

    def load_image(self, name, device):
        color = cv2.imread(name)
        data = {}
        data['image'] = torch.Tensor(color).permute(2,0,1)[None].to(device)
        data['image'] = (data['image'] - data['image'].min()) / (data['image'].max() - data['image'].min())
        return data

