from __future__ import absolute_import, division, print_function

import argparse
import os
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_matterport3d/segnet_v1.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    if args.smoke:
        config['epoch_max'] = 1
        config['epoch_save'] = 1
        config['log_frequency'] = 1
        config['max_steps'] = args.max_steps or 2
        config['max_val_steps'] = 1
        for key in ('train_dataset', 'val_dataset'):
            config[key]['batch_size'] = 1
            config[key]['num_workers'] = 0
            config[key]['args']['height'] = 224
            config[key]['args']['width'] = 448
            config[key]['args']['repeat'] = 1
            config[key]['args']['split'] = 'smoke'
            config[key]['args']['max_samples'] = 8 if key == 'train_dataset' else 4
        
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = args.output_dir or os.path.join('./tmp', save_name)

    from trainer import Trainer
    trainer = Trainer(config, save_path)
    trainer.train()
