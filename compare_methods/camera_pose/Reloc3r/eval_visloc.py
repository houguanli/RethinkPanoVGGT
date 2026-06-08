import argparse
import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from reloc3r.image_retrieval.topk_retrieval import TopkRetrieval, PREPROCESS_FOLDER, DB_DESCS_FILE_MASK, PAIR_INFO_FILE_MASK
from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.datasets.sevenscenes_retrieval import *  
from reloc3r.datasets.cambridge_retrieval import *  
from eval_relpose import build_dataset
from reloc3r.utils.metric import *
from reloc3r.utils.device import to_numpy

from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for visual localization')

    # model
    parser.add_argument('--model', type=str, 
        # default='Reloc3rRelpose(img_size=224)')
        default='Reloc3rRelpose(img_size=512)')
    parser.add_argument('--resolution', 
        # default=(224,224))  # by default (224,224) for Reloc3r-224
        default=(512,384))  # by default (512,384) for Reloc3r-512

    # test set: process the database
    parser.add_argument('--dataset_db', type=str, 
        default="CambridgeRetrieval(scene='{}', split='train')")
    parser.add_argument('--dataset_q', type=str, 
        default="CambridgeRetrieval(scene='{}', split='test')")
    parser.add_argument('--db_step', type=int, 
        default=1, help='process all database images or skip every db_step images') 
    parser.add_argument('--topk', type=int, 
        default=10, help='topk similar images for motion averaging')
    parser.add_argument('--cache_folder', type=str, default=PREPROCESS_FOLDER)
    parser.add_argument('--db_descs_file_mask', type=str, default=DB_DESCS_FILE_MASK)
    parser.add_argument('--pair_info_file_mask', type=str, default=PAIR_INFO_FILE_MASK)

    # test set: relpose
    parser.add_argument('--dataset_relpose', type=str, 
        default="CambridgeRelpose(scene='{}', pair_id={}, resolution={}, seed=777)")
    parser.add_argument('--batch_size', type=int,
        default=10)
    parser.add_argument('--num_workers', type=int,
        default=10)

    parser.add_argument('--scene', type=str, 
        default='KingsCollege')  
    parser.add_argument('--amp', type=int, 
        default=0,
        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # parser.add_argument('--output_dir', type=str, 
    #     default='./output', help='path where to save the output') 

    return parser


def test(args):
    assert args.scene in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs', 
        'GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch'] 

    if not os.path.exists(args.cache_folder):
        os.mkdir(args.cache_folder)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    args.device = device

    # set up the evaluation 
    args.pair_info_available = False
    if not args.pair_info_available:
        args.load_bd_desc = False 
        args.dataset_db = args.dataset_db.format(args.scene)
        args.dataset_q = args.dataset_q.format(args.scene)
        runner = TopkRetrieval(args)
        dataset_db = eval(args.dataset_db)
        runner.build_database(dataset_db)
        dataset_q = eval(args.dataset_q)
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)
        pair_info_path = '{}/{}'.format(args.cache_folder, args.pair_info_file_mask).format(dataset_q.scene, args.db_step, args.topk)
        np.save(pair_info_path, all_retrieved, allow_pickle=True)
        print('Database-query pairs saved to {}.'.format(pair_info_path)) 

    # infer relative poses
    args.relative_pose_available = False
    if not args.relative_pose_available:
        
        reloc3r_relpose = setup_reloc3r_relpose_model(args.model, device)

        data_loader_test = {'{} pair_id={}'.format(args.dataset_relpose.split('(')[0], pair_id): build_dataset(args.dataset_relpose.format(args.scene, pair_id, args.resolution), args.batch_size, args.num_workers, test=True)
                            for pair_id in range(args.topk)}
        for test_name, testset in data_loader_test.items():
            print('Testing {:s}'.format(test_name))
            pose_folder = '{}/poses_{}_pair-id={}'.format(args.cache_folder, testset.dataset.scene, testset.dataset.pair_id)
            if not os.path.exists(pose_folder):
                os.mkdir(pose_folder)
            with torch.no_grad():
                for batch in tqdm(testset):
                    pose = inference_relpose(batch, reloc3r_relpose, device, use_amp=args.amp)
                    view1, view2 = batch
                    for sid in range(len(pose)):
                        Rt = np.identity(4)
                        Rt[0:3,0:3] = to_numpy(pose[sid][0:3,0:3])
                        Rt[0:3,3] = to_numpy(pose[sid][0:3,3])
                        np.savetxt('{}/{}_{}_pose-q2d.txt'.format(pose_folder, 
                                   view2['label'][sid].split('_')[-1], 
                                   view2['instance'][sid].split('.')[0].split('-')[-1]), 
                                   Rt)
                        np.savetxt('{}/{}_{}_pose-db.txt'.format(pose_folder, 
                                   view2['label'][sid].split('_')[-1], 
                                   view2['instance'][sid].split('.')[0].split('-')[-1]), 
                                   to_numpy(view1['camera_pose'][sid]))
                        np.savetxt('{}/{}_{}_pose-gt.txt'.format(pose_folder, 
                                   view2['label'][sid].split('_')[-1], 
                                   view2['instance'][sid].split('.')[0].split('-')[-1]), 
                                   to_numpy(view2['camera_pose'][sid]))

    # infer absolute poses
    reloc3r_visloc = Reloc3rVisloc()
    rerrs, terrs = [], []
    if 'SevenScenes' in args.dataset_q:
        seqs = stat_7Scenes['seqs_test'][args.scene]
        beg, end = 0, stat_7Scenes['n_frames'][args.scene]-1
        mask_gt_db_cam = mask_gt_db_cam_7Scenes
        mask_q2d_cam = mask_q2d_cam_7Scenes
        mask_gt_q_cam = mask_gt_q_cam_7Scenes
    elif 'Cambridge' in args.dataset_q:
        seqs = stat_Cambridge[args.scene]['seq']
        mask_gt_db_cam = mask_gt_db_cam_Cambridge
        mask_q2d_cam = mask_q2d_cam_Cambridge
        mask_gt_q_cam = mask_gt_q_cam_Cambridge
    for seq in seqs:
        if 'Cambridge' in args.dataset_q:
            beg, end = stat_Cambridge[args.scene]['range']['{}'.format(seq)]
        for fid in tqdm(range(beg, end+1)): 
            if args.scene == 'GreatCourt' and seq == 4 and fid == 73:  # no GT
                continue 
            pose_db, pose_q2d = [], []
            for pid in range(args.topk):
                pose_db.append(np.loadtxt('{}/poses_{}_pair-id={}/{}'.format(args.cache_folder, args.scene, pid, mask_gt_db_cam).format(seq, fid))) 
                pose_q2d.append(np.loadtxt('{}/poses_{}_pair-id={}/{}'.format(args.cache_folder, args.scene, pid, mask_q2d_cam).format(seq, fid)))
            gt_q = np.loadtxt('{}/poses_{}_pair-id={}/{}'.format(args.cache_folder, args.scene, pid, mask_gt_q_cam).format(seq, fid))
            
            Rt = reloc3r_visloc.motion_averaging(pose_db, pose_q2d)

            rerr = get_rot_err(Rt[0:3,0:3], gt_q[0:3,0:3])
            terr = np.linalg.norm(Rt[0:3,3] - gt_q[0:3,3])
            rerrs.append(rerr)
            terrs.append(terr)
    print('Scene {} median pose error: {:.2f} m {:.2f} deg'.format(args.scene, np.median(terrs), np.median(rerrs)))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)

