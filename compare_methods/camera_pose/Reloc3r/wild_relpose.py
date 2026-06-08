import os
import numpy as np
import torch
from reloc3r.utils.image import parse_video, load_images, check_images_shape_format
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.utils.device import to_numpy


def wild_relpose(img_reso, v1_path, v2_path, output_folder=None):
    if output_folder is None:
        output_folder = v1_path[0:v1_path.rfind('/')]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load Reloc3r
    print('Loading Reloc3r...')
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device)
    
    # load images
    print('Loading images...')
    images = load_images([v1_path, v2_path], size=int(img_reso))
    images = check_images_shape_format(images, device)

    # relpose
    print('Running relative pose estimation...')
    batch = [images[0], images[1]]
    pose2to1 = to_numpy(inference_relpose(batch, reloc3r_relpose, device)[0])
    pose2to1[0:3,3] = pose2to1[0:3,3] / np.linalg.norm(pose2to1[0:3,3])  # normalize the scale to 1 meter

    # save poses to file
    np.savetxt('{}/pose2to1.txt'.format(output_folder), pose2to1)
    print('Pose saved to {}'.format(output_folder))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer relative pose from wild images')
    parser.add_argument('--img_reso', type=str, default='512')
    # parser.add_argument('--img_reso', type=str, default='224')
    parser.add_argument('--v1_path', type=str, default='data/wild_images/sun0.jpg')
    parser.add_argument('--v2_path', type=str, default='data/wild_images/sun1.png')
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    wild_relpose(img_reso=args.img_reso, v1_path=args.v1_path, v2_path=args.v2_path, output_folder=args.output_folder)

