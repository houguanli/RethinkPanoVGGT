import os
import numpy as np
import torch
from reloc3r.utils.image import parse_video, load_images, check_images_shape_format
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.device import to_numpy
from tqdm import tqdm


def wild_visloc(img_reso, video_path, output_folder=None, max_frames=30, mode='seq', use_amp = False):
    if output_folder is None:
        output_folder = video_path[0:video_path.rfind('/')]
    name = video_path[video_path.rfind('/')+1:video_path.rfind('.')]
    image_folder = output_folder + '/{}_images'.format(name)
    pose_folder = output_folder + '/{}_poses'.format(name)
    for folder in [image_folder, pose_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load Reloc3r
    print('Loading Reloc3r...')
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device)
    reloc3r_visloc = Reloc3rVisloc()

    # load sampled images from video
    print('Loading images from the input video...')
    parse_video(video_path, image_folder, max_frames=max_frames)
    images = load_images(image_folder, size=int(img_reso))
    images = check_images_shape_format(images, device)

    # setup a batabase with the first and the last frames
    print('Building database...')
    batch = [images[0], images[-1]]
    pose2to1 = to_numpy(inference_relpose(batch, reloc3r_relpose, device, use_amp = use_amp)[0])
    pose2to1[0:3,3] = pose2to1[0:3,3] / np.linalg.norm(pose2to1[0:3,3])  # normalize the scale to 1 meter
    pose_beg = np.identity(4)
    pose_end = pose_beg @ pose2to1
    poses_c2w = [pose_beg, pose_end]

    # visloc: w/ or w/o seq info
    print('Running visual localization...')
    for fid in tqdm(range(1, len(images)-1)):
        db1 = images[0]
        db2 = images[-1]
        query = images[fid]

        view1 = {'img': torch.cat((db1['img'], db2['img']), dim=0),
                 'true_shape': torch.cat((db1['true_shape'], db2['true_shape']), dim=0)}
        view2 = {'img': torch.cat((query['img'], query['img']), dim=0),
                 'true_shape': torch.cat((query['true_shape'], query['true_shape']), dim=0)}

        if mode == 'seq' and fid > 1: 
            db3 = images[fid-1]
            view1['img'] = torch.cat((view1['img'], db3['img']), dim=0)
            view1['true_shape'] = torch.cat((view1['true_shape'], db3['true_shape']), dim=0)
            view2['img'] = torch.cat((view2['img'], query['img']), dim=0)
            view2['true_shape'] = torch.cat((view2['true_shape'], query['true_shape']), dim=0)

        batch = [view1, view2]
        poses2to1 = to_numpy(inference_relpose(batch, reloc3r_relpose, device))
        poses_db = [poses_c2w[0], poses_c2w[-1]]
        poses_q2d = [poses2to1[0], poses2to1[1]]

        if mode == 'seq' and fid > 1: 
            poses_db.append(poses_c2w[fid-1])
            poses_q2d.append(poses2to1[2])

        pose = reloc3r_visloc.motion_averaging(poses_db, poses_q2d)

        pose_end = poses_c2w.pop()
        poses_c2w.append(pose)
        poses_c2w.append(pose_end)

    # save poses to file
    for pid in range(len(poses_c2w)):
        pose = poses_c2w[pid]
        np.savetxt('{}/pose_{:04d}.txt'.format(pose_folder, pid), pose)
    print('Poses saved to {}'.format(pose_folder))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer absolute poses from video')
    parser.add_argument('--img_reso', type=str, default='512')
    # parser.add_argument('--img_reso', type=str, default='224')
    parser.add_argument('--video_path', type=str, default='data/wild_video/desk.MOV')
    parser.add_argument('--output_folder', type=str, default='data/wild_video/')
    parser.add_argument('--mode', type=str, default='seq')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    args = parser.parse_args()

    wild_visloc(img_reso=args.img_reso, video_path=args.video_path, output_folder=args.output_folder, mode=args.mode, use_amp = args.amp)

