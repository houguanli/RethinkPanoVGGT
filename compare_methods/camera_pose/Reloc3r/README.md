<p align="center">
  <h2 align="center">[CVPR 2025] Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization</h2>
 <p align="center">
    <a href="https://siyandong.github.io/">Siyan Dong*</a>
    ·
    <a href="https://ffrivera0.github.io/">Shuzhe Wang*</a>
    ·
    <a href="http://b1ueber2y.me/">Shaohui Liu</a>
    ·
    <a href="">Lulu Cai</a>
    ·
    <a href="https://fqnchina.github.io/">Qingnan Fan</a>
    ·
    <a href="https://users.aalto.fi/~kannalj1/">Juho Kannala</a>
    ·
    <a href="https://yanchaoyang.github.io/">Yanchao Yang</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2412.08376">Paper</a> | <a href="https://www.youtube.com/watch?v=uHcEZRPOE5s">Video</a> | <a href="https://drive.google.com/file/d/1-siUbcn0N3jdr1IjJ4BBxKLSOafPDCFP/view?usp=drive_link">Poster</a> | <a href="https://huggingface.co/spaces/siyan824/reloc3r-512_relpose">Demo (Online)</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./media/overview.png" alt="Teaser" width="100%">
  </a>
</p>

<p align="center">
<strong>Reloc3r</strong> is a simple yet effective camera pose estimation framework that combines a pre-trained two-view relative camera pose regression network with a multi-view motion averaging module.
</p>
<br>

<p align="center">
  <a href="">
    <img src="./media/wild_visloc.png" alt="Teaser" width="100%">
  </a>
</p>

<p align="center">
Trained on approximately 8 million posed image pairs, <strong>Reloc3r</strong> achieves surprisingly good performance and generalization ability, producing high-quality camera pose estimates in real-time.
</p>
<be>


## Table of Contents

- [TODO List](#todo-list)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation on Relative Camera Pose Estimation](#evaluation-on-relative-camera-pose-estimation)
- [Evaluation on Visual Localization](#evaluation-on-visual-localization)
- [Training](#training)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)


## TODO List

- [x] Release pre-trained weights and inference code. 
- [x] Release evaluation code for ScanNet1500 and MegaDepth1500 datasets. 
- [x] Release evaluation code for 7Scenes and Cambridge datasets. 
- [x] Release sample code for self-captured images and videos.
- [x] Release training code and data.
- [x] Gradio demo for relative camera pose regression, and its visualization.  
- [ ] Evaluation code for other datasets. 
- [ ] Accelerated version for visual localization. 


## Installation

1. Clone Reloc3r
```bash
git clone --recursive https://github.com/ffrivera0/reloc3r.git
cd reloc3r
# if you have already cloned reloc3r:
# git submodule update --init --recursive
```

2. Create the environment using conda
```bash
conda create -n reloc3r python=3.11 cmake=3.14.0
conda activate reloc3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# optional: you can also install additional packages to:
# - add support for HEIC images
pip install -r requirements_optional.txt
```

3. Optional: Compile the cuda kernels for RoPE 
```bash
# Reloc3r relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

4. Optional: Download the checkpoints [Reloc3r-224](https://huggingface.co/siyan824/reloc3r-224)/[Reloc3r-512](https://huggingface.co/siyan824/reloc3r-512). The pre-trained model weights will automatically download when running the evaluation and demo code below. 


## Usage

Using Reloc3r, you can estimate camera poses for images and videos you captured. 

For relative pose estimation, try the demo code in `wild_relpose.py`. We provide some [image pairs](https://drive.google.com/drive/folders/1TmoSKrtxR50SlFoXOwC4a9aGr18h00yy?usp=sharing) used in our paper.  

```bash
# replace the args with your paths
python wild_relpose.py --v1_path ./data/wild_images/zurich0.jpg --v2_path ./data/wild_images/zurich1.jpg --output_folder ./data/wild_images/
```

Visualize the relative pose
```bash
# replace the args with your paths
python visualization.py --mode relpose --pose_path ./data/wild_images/pose2to1.txt
```

For visual localization, the demo code in `wild_visloc.py` estimates absolute camera poses from sampled frames in self-captured videos. 

> [!IMPORTANT]
> The demo simply uses the first and last frames as the database, which <strong>requires</strong> overlapping regions among all images. This demo does <strong>not</strong> support linear motion. We provide some [videos](https://drive.google.com/drive/folders/1sbXiXScts5OjESAfSZQwLrAQ5Dta1ibS?usp=sharing) as examples. 

```bash
# replace the args with your paths
python wild_visloc.py --video_path ./data/wild_video/ids.MOV --output_folder ./data/wild_video
```

Visualize the absolute poses
```bash
# replace the args with your paths
python visualization.py --mode visloc --pose_folder ./data/wild_video/ids_poses/
```


## Evaluation on Relative Camera Pose Estimation 

To reproduce our evaluation on ScanNet1500, download the dataset [here](https://drive.google.com/drive/folders/16g--OfRHb26bT6DvOlj3xhwsb1kV58fT?usp=sharing) and unzip it to `./data/scannet1500`.
Then run the following script. 
```bash
bash scripts/eval_scannet1500.sh
```

To reproduce our evaluation on MegaDepth1500, download the dataset [here](https://drive.google.com/drive/folders/16g--OfRHb26bT6DvOlj3xhwsb1kV58fT?usp=sharing) and unzip it to `./data/megadepth1500`.
Then run the following script. 
```bash
bash scripts/eval_megadepth1500.sh
```

> [!NOTE]
> To achieve faster inference speed, set `--amp=1`. This enables evaluation with `fp16`, which increases speed from <strong>24 FPS</strong> to <strong>40 FPS</strong> on an RTX 4090 with Reloc3r-512, without any accuracy loss.


## Evaluation on Visual Localization 

To reproduce our evaluation on 7Scenes, download the dataset [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and unzip it to `./data/7scenes`.
Then run the following script. 
```bash
bash scripts/eval_7scenes.sh
```

To reproduce our evaluation on Cambridge, download the dataset [here](https://drive.google.com/file/d/1XcJIVRMma4_IClJdRq6rwBKX3ZPet5az/view?usp=sharing) and unzip it to `./data/cambridge`.
Then run the following script. 
```bash
bash scripts/eval_cambridge.sh
```


## Training

We follow [DUSt3R](https://github.com/naver/dust3r) to process the training data. Download the datasets: [CO3Dv2](https://github.com/facebookresearch/co3d), [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), [ARKitScenes](https://github.com/apple/ARKitScenes), [BlendedMVS](https://github.com/YoYo000/BlendedMVS), [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/), [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/), [RealEstate10K](https://google.github.io/realestate10k/). 

For each dataset, we provide a preprocessing script in the `datasets_preprocess` directory and an archive containing the list of [pairs](https://drive.google.com/drive/folders/193Lv5YB-2OVkqK3k6vnZJi36-FRPcAuu?usp=sharing) when needed. You have to download the datasets yourself from their official sources, agree to their license, and run the preprocessing script.

We provide a sample script to train Reloc3r with ScanNet++ on an RTX 3090 GPU
```bash
bash scripts/train_small.sh
```

To reproduce our training for Reloc3r-512 with 8 H800 GPUs, run the following script
```bash
bash scripts/train.sh
```

> [!NOTE]
> They are not strictly equivalent to what was used to train Reloc3r, but they should be close enough.


## Citation

If you find our work helpful in your research, please consider citing: 
```
@article{reloc3r,
  title={Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization},
  author={Dong, Siyan and Wang, Shuzhe and Liu, Shaohui and Cai, Lulu and Fan, Qingnan and Kannala, Juho and Yang, Yanchao},
  journal={arXiv preprint arXiv:2412.08376},
  year={2024}
}
```


## Acknowledgments

Thanks to these great repositories: [Croco](https://github.com/naver/croco), [DUSt3R](https://github.com/naver/dust3r), [Marepo](https://github.com/nianticlabs/marepo), and many other inspiring works in the community.

