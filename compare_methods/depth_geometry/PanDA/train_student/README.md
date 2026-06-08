# Second stage: Fine-tune a student model using both labeled and unlabeled data

We provide a codebase to fine-tune the Depth Anything v2 model using both synthetic labeled data and real-world unlabeled data. In our practice, the ViT encoder is frozen with LoRA layers added, while the DPT decoder is fully trained. For the synthetic labeled data, we utilize an indoor synthetic dataset (Structured3D) and an outdoor synthetic dataset (Deep360). For the real-world unlabeled data, we extract from [ZInD](https://github.com/zillow/zind) and [360+x](https://x360dataset.github.io/) datasets.

## Usage

### Checkpoint Prepraration

Prepare the checkpoint of teacher model in the first stage.

### Dataset Prepraration

Download the [Structured3D](https://structured3d-dataset.org/) and [Deep360](https://github.com/nju-ee/MODE-2022) datasets.

Download the [ZInD](https://github.com/zillow/zind) and [360+x](https://x360dataset.github.io/) datasets. You can also utilize other unlabeled 360 datasets. 

*`Note`*: We generate pseudo depth labels in an *`offline`* manner. To address the sky region in outdoor scenes which often contains noise, we employ [SegFormer](https://github.com/NVlabs/SegFormer) to generate sky masks.

Then, modify the dataset path in the corresponding yaml files.

```bash
./config/student/train_[small, base, large].yaml
```

### Training

We take the `vitl` encoder as an example. You can also use `vitb` or `vits` encoders by modifying the model config in the training yaml files.

*`Note`*: We utilize a NVIDIA A800 GPU to train with `vits` encoder. Instead, `vitl` might require four NVIDIA A800 GPUs for 2~4 days.

```bash
python train_student/semi_train.py \
       --config ./config/student/train_large.yaml
```

### Evaluation

We evaluate on two real-world datasets (Matterport3D and Stanford2D3D) under different spherical transformations.

```bash
python train_student/eval_mobius.py \
       --config ./config/student/test_matterport.yaml
```