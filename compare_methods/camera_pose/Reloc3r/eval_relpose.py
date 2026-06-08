import argparse
import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.datasets import get_data_loader
from reloc3r.utils.metric import *
from reloc3r.utils.device import to_numpy

from tqdm import tqdm
# from pdb import set_trace as bb


def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for relative camera pose estimation')

    # model
    parser.add_argument('--model', type=str, 
        # default='Reloc3rRelpose(img_size=224)')
        default='Reloc3rRelpose(img_size=512)')
    
    # test set
    parser.add_argument('--test_dataset', type=str, 
        # default="ScanNet1500(resolution=(224,224), seed=777)")
        default="ScanNet1500(resolution=(512,384), seed=777)")
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--num_workers', type=int,
        default=10)
    parser.add_argument('--amp', type=int, default=1,
                                choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # parser.add_argument('--output_dir', type=str, 
    #     default='./output', help='path where to save the pose errors')

    return parser


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))
    print('Dataset length: ', len(loader))
    return loader


def test(args):
    
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    reloc3r_relpose = setup_reloc3r_relpose_model(args.model, device)
    
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # start evaluation
    rerrs, terrs = [], []
    for test_name, testset in data_loader_test.items():
        print('Testing {:s}'.format(test_name))
        with torch.no_grad():
            for batch in tqdm(testset):

                pose = inference_relpose(batch, reloc3r_relpose, device, use_amp=bool(args.amp))

                view1, view2 = batch
                gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
                rerrs_prh = []
                terrs_prh = []

                # rotation angular err
                R_prd = pose[:,0:3,0:3]
                for sid in range(len(R_prd)):
                    rerrs_prh.append(get_rot_err(to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid,0:3,0:3])))
                
                # translation direction angular err
                t_prd = pose[:,0:3,3]
                for sid in range(len(t_prd)): 
                    transl = to_numpy(t_prd[sid])
                    gt_transl = to_numpy(gt_pose2to1[sid,0:3,-1])
                    transl_dir = transl / np.linalg.norm(transl)
                    gt_transl_dir = gt_transl / np.linalg.norm(gt_transl)
                    terrs_prh.append(get_transl_ang_err(transl_dir, gt_transl_dir)) 

                rerrs += rerrs_prh
                terrs += terrs_prh

        rerrs = np.array(rerrs)
        terrs = np.array(terrs)
        print('In total {} pairs'.format(len(rerrs)))

        # auc
        print(error_auc(rerrs, terrs, thresholds=[5, 10, 20]))

        # # save err list to file
        # err_list = np.concatenate((rerrs[:,None], terrs[:,None]), axis=-1)
        # output_file = '{}/pose_error_list.txt'.format(args.output_dir)
        # np.savetxt(output_file, err_list)
        # print('Pose errors saved to {}'.format(output_file))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)

