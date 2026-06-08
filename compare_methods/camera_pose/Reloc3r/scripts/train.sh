# full-scale training on 8 H800 GPUs

torchrun --nproc_per_node=8 train.py \
    --train_dataset "50_000 @ Co3d(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) \
                   + 50_000 @ ScanNetpp(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) \
                   + 50_000 @ ARKitScenes(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) \
                   + 50_000 @ BlendedMVS(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) \
                   + 50_000 @ MegaDepth(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) \
                   + 50_000 @ DL3DV(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) \
                   + 50_000 @ RealEstate(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "1_000 @ ScanNet1500(resolution=(512, 384), seed=777) \
                  + 1_000 @ ARKitScenes(split='test', resolution=(512, 384), seed=777) \
                  + 1_000 @ MegaDepth_valid(split='test', resolution=(512, 384), seed=777) \
                  + 1_000 @ DL3DV(split='test', resolution=(512, 288), seed=777) " \
    --model "Reloc3rRelpose(img_size=512)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 32 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_full-scale_"  

