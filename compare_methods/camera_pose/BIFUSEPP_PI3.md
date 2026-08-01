# BiFuse++ and Pi3 camera-side methods

## Scope

BiFuse++ is primarily a monocular 360 depth method. Its supervised checkpoint
contains DepthNet only; the self-supervised checkpoint also contains PoseNet,
which estimates relative motion for a center panorama and two adjacent frames.
It is therefore useful for panoramic geometry and relative-motion experiments,
but it is not equivalent to a general multi-view camera reconstruction model.

Pi3 is a general multi-view visual geometry model. It predicts camera-to-world
poses, local point maps, global points, and confidence without selecting a
reference frame.

## Upstream sources

| Method | Paper | Official code | Pinned submodule commit | Code license |
| --- | --- | --- | --- | --- |
| BiFuse++ | https://arxiv.org/abs/2209.02952 | https://github.com/fuenwang/BiFusev2 | `6a0f4ba3772a9290b4363c97b8cbc2432d05b327` | MIT |
| Pi3 | https://arxiv.org/abs/2507.13347 | https://github.com/yyfz/Pi3 | `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` | BSD-3-Clause |

Pi3 model weights are CC BY-NC 4.0 and are restricted to non-commercial use.
BiFuse++ removed PanoSUNCG in May 2025 for legal and licensing reasons. Do not
download or use that dataset; the released self-supervised checkpoint remains
available from the authors.

## Setup

From the repository root:

```bash
bash compare_methods/scripts/create_method_env.sh bifusepp
bash compare_methods/scripts/create_method_env.sh pi3

conda run --no-capture-output -n cmp_bifusepp \
  bash compare_methods/scripts/download_camera_method_ckpts.sh bifusepp
conda run --no-capture-output -n cmp_pi3 \
  bash compare_methods/scripts/download_camera_method_ckpts.sh pi3
```

The environment script discovers `conda.sh` from `which conda`/`CONDA_EXE`,
initializes both Git submodules, installs the pinned inference dependencies,
and installs Pi3 editable. BiFuse++ full training additionally requires
PyTorch Lightning and PyTorch3D as documented upstream; neither is needed by
the released supervised inference path tested here.

Downloaded checkpoint records:

| Checkpoint | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| BiFuse++ supervised | 213175513 | `644acfe1e1068ce3c8b166b841f894a02ec6a40ef2b1cd13f8220b85bb7c02e0` |
| BiFuse++ self-supervised | 269882615 | `415f70aa8867e6f221f1da23a448825398ee391d3a091431dd5c4fff4c3108ff` |
| Pi3 | 3834909248 | `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d` |

## Sample inference

BiFuse++ runs two released 512x1024 panoramas and writes metric depth arrays,
16-bit millimeter PNGs, and a JSON summary:

```bash
conda run --no-capture-output -n cmp_bifusepp \
  python compare_methods/camera_pose/bifusepp_inference.py
```

For the released self-supervised checkpoint, pass three temporally adjacent
panoramas. The adapter writes one depth map per frame and the two relative
6-DoF poses predicted with respect to the center frame:

```bash
conda run --no-capture-output -n cmp_bifusepp \
  python compare_methods/camera_pose/bifusepp_inference.py \
    --mode selfsupervised \
    --input /path/to/previous.png \
    --input /path/to/reference.png \
    --input /path/to/next.png
```

Pi3 runs two official multi-view examples with two frames each, writes PLY
point clouds and camera-pose NPZ files, and records peak GPU memory:

```bash
conda run --no-capture-output -n cmp_pi3 \
  python compare_methods/camera_pose/pi3_inference.py
```

The default Pi3 smoke resolution is intentionally bounded by
`--pixel-limit 50176`. Increase it only after checking available GPU memory.
Use `STAGE=evaluate` with `run_panocity_pipeline.sh` to invoke the same
checkpoint-explicit sample adapters through the method registry.

## Verified smoke results

The following checks were run on CUDA on 2026-08-01, not as dry runs:

- BiFuse++ supervised: strict learned-parameter loading and forward inference
  on both official 512x1024 samples; metric depth NPY and 16-bit PNG outputs.
- BiFuse++ self-supervised: strict learned-parameter loading on three adjacent
  PanoCity frames; three depth outputs and a `(2, 6)` relative-pose array.
- Pi3: strict safetensors loading and two-frame inference on both official
  `house` and `parkour` examples; finite `(2, 4, 4)` camera poses and binary PLY
  point clouds with 54573 and 28827 points, respectively. Peak allocated CUDA
  memory was about 5.38 GiB at the bounded smoke resolution.
- Both environments pass `pip check`; the registry pipeline activates
  `cmp_bifusepp` and `cmp_pi3` independently and prints the resolved Python and
  checkpoint paths before execution.

Pi3 may report that the compiled RoPE2D extension is unavailable. Its tested
PyTorch fallback is functionally valid but slower.

## PanoSUNCG zero-shot benchmark

The shared evaluator uses the official DA-2 split (3944 depth samples) and the
same 118 complete trajectories, five evenly spaced frames, and Sim(3) camera
alignment as the existing comparison table. Depth alignment is fitted on the
full valid ERP with 100-step AbsRel IRLS scale+shift; AbsRel, RMSE, delta1, and
delta2 are then accumulated pixel-micro over latitude -15 to +60 degrees.

```bash
bash compare_methods/scripts/run_bifuse_pi3_panosuncg.sh \
  /path/to/PanoSUNCG_zeroshot \
  /path/to/results

python compare_methods/camera_pose/update_panosuncg_zeroshot_table.py \
  --results-root /path/to/results \
  --table /path/to/zero_shot_single_column.tex
```

The table updater refuses partial results and supports both the older
three-subtable camera layout and the newer unified 11-column layout. The run
script is resumable by default; set `RESUME=0` to replace an existing run.

Pi3 is a perspective model. For depth, each ERP is converted to six 196x196
90-degree cube faces and the predicted local Z-depth is reconstructed to a
full ERP before alignment. For camera evaluation, each trajectory contributes
one canonical front-facing 196x196 perspective per panorama. BiFuse++ uses its
native 512x1024 ERP path. Its camera result is marked with a caveat because the
released self-supervised camera checkpoint was trained on PanoSUNCG and is
therefore in-domain rather than strict zero-shot; its supervised Matterport3D
depth checkpoint is a valid zero-shot depth evaluation. The supervised depth
path uses the normalization in the training/validation combined forward and
then the release demo's `[0, 10]` output clipping.

Complete local results from 2026-08-01:

| Method | Depth AbsRel | Depth RMSE | Depth delta1 | Depth delta2 | Camera AUC@30 | Camera direction mean/median | Camera ATE/nATE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiFuse++ | 0.3277 | 1.0401 | 0.4124 | 0.6583 | 0.5147 | 20.0101 / 12.8581 | 0.6520 / 0.3523 |
| Pi3 | 0.1272 | 0.3845 | 0.8432 | 0.9598 | 0.7553 | 11.7066 / 3.2082 | 0.6470 / 0.2189 |
