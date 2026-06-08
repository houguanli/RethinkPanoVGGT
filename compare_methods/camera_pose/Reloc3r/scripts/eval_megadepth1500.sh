# Running evaluation for megadepth1500

CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --test_dataset "MegaDepth_valid(resolution=(512,384), seed=777)" \

