# PanDA for Metric Depth Estimation

We provide training and evaluation codes to fine-tune PanDA on real-world datasets, i.e., Matterport3D and Stanford2D3D datasets.

![teaser](../assets/compare.png)

# Pre-trained Models

We provide **six metric depth models** of three scales for Matterport3D and Stanford2D3D scenes, respectively.

| Base Model              | Params |                      Matterport3D                      |                  Stanford2D3D                   |
| :---------------------- | -----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| PanDA-Small |  24.8M | [Download](https://huggingface.co/ZidongC/PanDA/resolve/main/panda_matterport_small.pth?download=true) | [Download](https://huggingface.co/ZidongC/PanDA/resolve/main/panda_stanford_small.pth?download=true) |
| PanDA-Base  |  97.5M | [Download](https://huggingface.co/ZidongC/PanDA/resolve/main/panda_matterport_base.pth?download=true) | [Download](https://huggingface.co/ZidongC/PanDA/resolve/main/panda_stanford_base.pth?download=true) |
| PanDA-Large | 335.3M | [Download](https://huggingface.co/ZidongC/PanDA/resolve/main/panda_matterport_large.pth?download=true) | [Download](https://huggingface.co/ZidongC/PanDA/resolve/main/panda_stanford_large.pth?download=true) |

## Usage

### Prepraration

(i) Download the checkpoints listed [here](#pre-trained-models) and put them under the `tmp` directory.

(ii) Modify the checkpoint path in yaml file at "load_weights_dir".

(ii) Download the datasets: [Matterport3D](https://niessner.github.io/Matterport/) and [Stanford2D3D](https://github.com/alexsax/2D-3D-Semantics). Please be noted the data pre-processing. For example, the stitching process in Matterport3D (See [here](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/tree/main/UniFuse/Matterport3D)) and filling process in Stanford2D3D (See [here](https://github.com/HalleyJiang/UniFuse-Unidirectional-Fusion/blob/main/UniFuse/preprocess_s2d3d.py)).

### Training

```bash
python train_metric_depth/train.py --config \
./config/metric_depth/train_matterport3d.yaml
```

### Evaluation

```bash
python train_metric_depth/eval.py \
--config ./config/metric_depth/test_matterport.yaml
```