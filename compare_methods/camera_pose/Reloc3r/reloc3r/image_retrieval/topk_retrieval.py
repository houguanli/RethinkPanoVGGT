import numpy as np
import torch
import cv2
import os
from tqdm import tqdm
from .netvlad import NetVLAD
from reloc3r.utils.device import to_numpy


from pdb import set_trace as bb


PREPROCESS_FOLDER = './_db-q_pair_info'
DB_DESCS_FILE_MASK = '{}_db-descs_step={}.txt'
PAIR_INFO_FILE_MASK = '{}_q-retrieval_db-step={}_topk={}.npy'


class TopkRetrieval: 
    def __init__(self, cfg):
        super(TopkRetrieval, self).__init__()
        self.cfg = cfg
        self.extractor_img = NetVLAD(NetVLAD.default_conf).eval().to(self.cfg.device)
        self.db_descs = None
        if not os.path.exists(self.cfg.cache_folder):
            os.mkdir(self.cfg.cache_folder )
        self.db_descs_path_mask = self.cfg.cache_folder + '/' + self.cfg.db_descs_file_mask

    @torch.no_grad()
    def build_database(self, dataset_db): 
        db_descs_path = self.db_descs_path_mask.format(dataset_db.scene, self.cfg.db_step)
        db_descs = []
        if not self.cfg.load_bd_desc:
            for fid in tqdm(range(0, len(dataset_db.names_color), self.cfg.db_step)): 
                f_name = dataset_db.names_color[fid]
                data = dataset_db.load_image(f_name, self.cfg.device)
                desc_full = self.extractor_img(data)['global_descriptor']
                db_descs.append(to_numpy(desc_full.squeeze()))
            db_descs = np.array(db_descs)
            np.savetxt(db_descs_path, db_descs)
            print('Database descriptors saved to {}.'.format(db_descs_path))
        else:
            db_descs = np.loadtxt(db_descs_path)
            print('Database descriptors loaded from {}.'.format(db_descs_path))
        self.db_descs = torch.Tensor(db_descs)
        print('Database: {} images.'.format(len(self.db_descs))) 

    @torch.no_grad()
    def retrieve_topk(self, dataset_db, dataset_q):
        # without filtering by camera poses
        all_retrieved = []
        for fid in tqdm(range(len(dataset_q.names_color))):
            f_name = dataset_q.names_color[fid]
            data = dataset_q.load_image(f_name, self.cfg.device)
            query_desc = self.extractor_img(data)['global_descriptor']
            sim = torch.einsum('id,jd->ij', query_desc.to(self.cfg.device), self.db_descs.to(self.cfg.device)).cpu()
            values, indices = torch.topk(sim, k=self.cfg.topk)
            values, indices = values[0], indices[0]
            retrieved = []
            for i in range(len(indices)):
                idx = indices[i] * self.cfg.db_step
                name = dataset_db.names_color[idx]
                info = {'query_path': f_name, 'db_path':name, 'query_idx':fid, 'db_idx': idx.item()}
                retrieved.append(info)
            all_retrieved.append(retrieved)
        print('Query: {} images.'.format(len(all_retrieved)))
        return all_retrieved

