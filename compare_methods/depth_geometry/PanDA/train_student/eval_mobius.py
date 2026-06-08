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
from metrics_st import Evaluator
from networks.models import *

import matplotlib.pyplot as plot
from einops import rearrange

from mobius_utils import make_coord, warp_mobius_image


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

    evaluator_r1 = Evaluator(config['median_align'])
    evaluator_r1.reset_eval_metrics()
    evaluator_r2 = Evaluator(config['median_align'])
    evaluator_r2.reset_eval_metrics()
    evaluator_r3 = Evaluator(config['median_align'])
    evaluator_r3.reset_eval_metrics()
    evaluator_r4 = Evaluator(config['median_align'])
    evaluator_r4.reset_eval_metrics()
    evaluator_r5 = Evaluator(config['median_align'])
    evaluator_r5.reset_eval_metrics()
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):
            gt_depth = inputs["gt_depth"].cuda()
            b, c, h, w = inputs["rgb"].shape
            equi_inputs = inputs["rgb"].cuda()
            M_groups = []
            for i in [0, 18, 9]:
                if i == 0:
                    theta = 0
                else:
                    theta = np.pi / i
                phi = 0
                scale = 1
                M_scale = np.array([[scale, 0], [0, 1]])
                M_horizon = np.array([[np.cos(theta) + 1j * np.sin(theta), 0], [0, 1]])
                M_vertical = np.array([[np.cos(phi / 2), np.sin(phi / 2)], [-np.sin(phi / 2), np.cos(phi / 2)]])
                M = M_horizon @ M_vertical @ M_scale
                M_groups.append(torch.from_numpy(M).cuda())
            for i in [2.0, 3.0]:
                theta = 0
                phi = 0
                scale = i
                M_scale = np.array([[scale, 0], [0, 1]])
                M_horizon = np.array([[np.cos(theta) + 1j * np.sin(theta), 0], [0, 1]])
                M_vertical = np.array([[np.cos(phi / 2), np.sin(phi / 2)], [-np.sin(phi / 2), np.cos(phi / 2)]])
                M = M_horizon @ M_vertical @ M_scale
                M_groups.append(torch.from_numpy(M).cuda())

            coord_hr = make_coord([h, w], flatten=True).unsqueeze(0)

            equi_r1 = equi_inputs
            gt_r1 = gt_depth.cpu()
            val_mask_r1 = inputs['val_mask']
            pred_r1 = model(equi_r1)['pred_depth'].detach().cpu()

            equi_r2 = warp_mobius_image(equi_inputs, M_groups[1], coord_hr, pole='Equator')
            gt_r2 = warp_mobius_image(gt_depth, M_groups[1], coord_hr, pole='Equator').detach().cpu()
            val_mask_r2 = warp_mobius_image(inputs['val_mask'].float().cuda(), M_groups[1], coord_hr, pole='Equator').detach().cpu()
            val_mask_r2 = ((gt_r2 > 0) & ~torch.isnan(gt_r2)) & (val_mask_r2 > 0)
            pred_r2 = model(equi_r2)['pred_depth'].detach().cpu()

            equi_r3 = warp_mobius_image(equi_inputs, M_groups[2], coord_hr, pole='Equator')
            gt_r3 = warp_mobius_image(gt_depth, M_groups[2], coord_hr, pole='Equator').detach().cpu()
            val_mask_r3 = warp_mobius_image(inputs['val_mask'].float().cuda(), M_groups[2], coord_hr, pole='Equator').detach().cpu()
            val_mask_r3 = ((gt_r3 > 0) & ~torch.isnan(gt_r3)) & (val_mask_r3 > 0)
            pred_r3 = model(equi_r3)['pred_depth'].detach().cpu()

            equi_r4 = warp_mobius_image(equi_inputs, M_groups[3], coord_hr, pole='Equator')
            gt_r4 = warp_mobius_image(gt_depth, M_groups[3], coord_hr, pole='Equator').detach().cpu()
            val_mask_r4 = warp_mobius_image(inputs['val_mask'].float().cuda(), M_groups[3], coord_hr, pole='Equator').detach().cpu()
            val_mask_r4 = ((gt_r4 > 0) & ~torch.isnan(gt_r4)) & (val_mask_r4 > 0)
            pred_r4 = model(equi_r4)['pred_depth'].detach().cpu()

            equi_r5 = warp_mobius_image(equi_inputs, M_groups[4], coord_hr, pole='Equator')
            gt_r5 = warp_mobius_image(gt_depth, M_groups[4], coord_hr, pole='Equator').detach().cpu()
            val_mask_r5 = warp_mobius_image(inputs['val_mask'].float().cuda(), M_groups[4], coord_hr, pole='Equator').detach().cpu()
            val_mask_r5 = ((gt_r5 > 0) & ~torch.isnan(gt_r5)) & (val_mask_r5 > 0)
            pred_r5 = model(equi_r5)['pred_depth'].detach().cpu()
        
            for i in range(gt_depth.shape[0]):
                evaluator_r1.compute_eval_metrics(gt_r1[i:i + 1], pred_r1[i:i + 1], val_mask_r1[i:i + 1])
                evaluator_r2.compute_eval_metrics(gt_r2[i:i + 1], pred_r2[i:i + 1], val_mask_r2[i:i + 1])
                evaluator_r3.compute_eval_metrics(gt_r3[i:i + 1], pred_r3[i:i + 1], val_mask_r3[i:i + 1])
                evaluator_r4.compute_eval_metrics(gt_r4[i:i + 1], pred_r4[i:i + 1], val_mask_r4[i:i + 1])
                evaluator_r5.compute_eval_metrics(gt_r5[i:i + 1], pred_r5[i:i + 1], val_mask_r5[i:i + 1])
                
    print("r1")
    evaluator_r1.print()
    print("r2")
    evaluator_r2.print()
    print("r3")
    evaluator_r3.print()
    print("r4")
    evaluator_r4.print()
    print("r5")
    evaluator_r5.print()

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