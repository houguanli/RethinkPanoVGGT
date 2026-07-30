# P2P-VGGT supplementary code

This repository contains the cleaned implementation used for panorama-to-
panorama camera and depth reconstruction with the VGGT-Omega backbone and
LUNA residual adapters. The release includes the model, panorama sampler,
training objectives, dataset readers, official PanoCity split indexes,
evaluation utilities, and unit tests.

Experiment outputs and machine-specific material are intentionally excluded:
there are no logs, checkpoints, TensorBoard files, generated reports, cached
Python files, or server launch scripts in this branch.

## Method overview

An equirectangular panorama is sampled into calibrated virtual pinhole views.
Each view carries spherical patch metadata and known virtual-camera metadata.
Zero-initialized LUNA adapters inject two residual signals into VGGT-Omega:

- a spherical patch-bank residual that communicates across overlapping views;
- a virtual-camera residual conditioned on yaw, pitch, and field of view.

Multi-panorama samples are flattened into one view sequence. The training
objective combines depth reconstruction with panorama-relative camera
supervision and optional shared-frame point consistency. Fresh LUNA adapters
are identity mappings at initialization, so a released VGGT-Omega checkpoint
can be loaded with non-strict state-dict matching.

## Repository layout

```text
configs/train_multipano.yaml       reference 3-to-9 panorama curriculum
evaluation_common/                ERP depth splatting shared by evaluation/tests
scripts/                          data indexing, evaluation, and reconstruction
tests/                            CPU-oriented unit and geometry tests
training/                         dataset readers and training entry point
vggt_omega/                       backbone, LUNA adapters, sampler, and heads
launch.py                         convenience training entry point
```

Only one training configuration is retained. Hardware, dataset, checkpoint,
and output paths can be overridden from the command line.

## Environment

Python 3.10 or newer is required. A CUDA build of PyTorch is recommended for
training and full-resolution evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[evaluation]"
```

The equivalent non-editable dependency set is listed in `requirements.txt`.

## Checkpoints

Model weights are not committed to this repository. Place the released
VGGT-Omega foundation checkpoint at
`checkpoints/vggt_omega_1b_512.pt`, or pass another path with
`--checkpoint`. Training produces delta checkpoints by default.

If a moved delta checkpoint contains an absolute reference to its original
foundation checkpoint, set an explicit replacement before evaluation:

```bash
export P2P_VGGT_CHECKPOINT_OVERRIDE=/path/to/vggt_omega_1b_512.pt
```

## Data

The reference configuration expects the four-dataset minimal bundle below a
single root:

```text
<dataset-root>/
  Panocity/<city>/<block>/
    pano_images/
    panodepth_images/
    *_poses.json
  Matterport3D/<scan>/
    pano_skybox_color/
    pano_depth/
    pano_poses/
  Stanford2D3DS/<area>/pano/
    rgb/
    depth/
    pose/
  Structured3D/<scene>/2D_rendering/<camera>/panorama/full/
    rgb_rawlight.png
    depth.png
```

Build or refresh the cache indexes with:

```bash
python scripts/build_mixed4_official_indexes.py \
  --root /path/to/mixed4 \
  --bad-scene-list configs/structured3d_bad_scenes.txt
```

The official PanoCity train/validation/test split indexes used by the reader
are included under `training/data/splits/panocity/`. The datasets and their
licenses are not redistributed here.

## Sanity checks

The generated-data smoke run does not require a dataset or checkpoint:

```bash
python training/launch.py --smoke --no-tensorboard --no-progress-bar
```

Run the full unit suite from the repository root:

```bash
python tests/run_all.py
```

## Training

Single-process training:

```bash
python training/launch.py \
  --config configs/train_multipano.yaml \
  --dataset-root /path/to/mixed4 \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --output-dir outputs/train_multipano
```

Four-GPU distributed training:

```bash
torchrun --standalone --nproc_per_node=4 training/launch.py \
  --config configs/train_multipano.yaml \
  --dataset-root /path/to/mixed4 \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --output-dir outputs/train_multipano
```

Explicit command-line arguments override values from the YAML file. The
reference schedule uses 1024x512 ERP inputs, eight 384x384 virtual views per
panorama, and a 3-to-6-to-9 panorama curriculum. Reduce the window or panorama
count for lower-memory hardware.

## Evaluation

Evaluate a trained checkpoint on the four held-out dataset splits:

```bash
python scripts/evaluate_mixed4_depth_checkpoint.py \
  --config configs/train_multipano.yaml \
  --checkpoint /path/to/last.pt \
  --dataset-root /path/to/mixed4 \
  --output outputs/eval/summary.json \
  --per-sample-csv outputs/eval/per_sample.csv
```

Use `--limit-per-dataset 0` for the complete split, or a positive value for a
bounded validation run. Multi-worker evaluation shards can be merged with
`scripts/merge_mixed4_eval_shards.py`.

## Reconstruction export

Export depth previews and a point cloud for a panorama:

```bash
python scripts/reconstruct_pano_omega.py \
  --checkpoint /path/to/last.pt \
  --pano-path /path/to/panorama.png \
  --output-dir outputs/reconstruction
```

Pass `--depth-path` when ground-truth depth is available. Dataset-indexed
export is also supported through `--dataset-root`, `--dataset-format`, and
`--sample-index`.

## License

See `LICENSE` for the code license. Dataset and checkpoint terms remain those
of their respective providers.
