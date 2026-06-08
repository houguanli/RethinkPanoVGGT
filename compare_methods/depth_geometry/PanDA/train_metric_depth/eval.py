from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import yaml
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import datasets
from metrics import Evaluator
from networks.models import *
from saver import Saver


def main(config):
    model_path = os.path.join(config["load_weights_dir"], 'model.pth')
    model_dict = torch.load(model_path)

    # data
    datasets_dict = { "stanford2d3d": datasets.Stanford2D3D,
                      "matterport3d": datasets.Matterport3D}
    cf_test = config['test_dataset']
    dataset = datasets_dict[cf_test['name']]

    test_dataset = dataset(cf_test['root_path'], 
                            cf_test['list_path'],
                            cf_test['args']['height'],
                            cf_test['args']['width'])
    test_loader = DataLoader(test_dataset, 
                            cf_test['batch_size'], 
                            False,
                            num_workers=cf_test['num_workers'], 
                            pin_memory=True, 
                            drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // cf_test['batch_size']
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    model = make(config['model'])
    if any(key.startswith('module') for key in model_dict.keys()):
        model = nn.DataParallel(model)
    model.cuda()
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    evaluator = Evaluator(config['median_align'])
    evaluator.reset_eval_metrics()
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):
            gt_depth = inputs["gt_depth"].cuda()
            b, c, h, w = inputs["rgb"].shape
            equi_inputs = inputs["rgb"].cuda()
            
            equi_inputs = F.interpolate(equi_inputs, (504, 1008), mode='bilinear', align_corners=False)
            pred_depth = model(equi_inputs)['pred_depth'].detach().cpu()
            gt_depth = gt_depth.detach().cpu()
            val_mask = inputs["val_mask"]
            pred_depth = torch.nn.functional.interpolate(pred_depth, (512, 1024), mode='bilinear', align_corners=False)
        
            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], val_mask[i:i + 1])
                
    print("evaluation results:")
    evaluator.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config)