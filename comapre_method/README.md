# Comparison Methods

This directory stores external baselines used for local pano reconstruction comparisons.

## Layout

- `vggt_omega_baseline/`: official VGGT-Omega baseline code. Its default checkpoint resolves to `ckpt/vggt_omega_1b_512.pt` from the project root.
- `PANOVGGT_baseline/`: PanoVGGT baseline code, copied without its `.venv`, example data, outputs, or checkpoint payload.

PanoVGGT's external checkpoint is kept with the other project checkpoints:

- `/home/aoki/RethinkPanoVGGT_omega/ckpt/panovggt/model.pt`

PanoVGGT example images are kept outside the method code at:

- `/home/aoki/RethinkPanoVGGT_omega/dataset/panovggt_examples`

The shared one-pano comparison input is kept at:

- `/home/aoki/RethinkPanoVGGT_omega/dataset/panovggt_compare_sample`

## PanoVGGT Sample Run

```bash
cd /home/aoki/RethinkPanoVGGT_omega/comapre_method/PANOVGGT_baseline
/home/aoki/PANOVGGT/.venv/bin/python scripts/run_rethink_pano_sample.py --device cuda
```

Default output:

- `/home/aoki/RethinkPanoVGGT_omega/Rethink_pano_new_exp_omega/outputs/panovggt_baseline_sample0`

## PanoVGGT Sim(3) Alignment To GT

PanoVGGT may produce a point cloud in a different rotation/translation/scale frame.
Use the built-in robust Sim(3) trimmed ICP alignment before visual comparison:

```bash
cd /home/aoki/RethinkPanoVGGT_omega/comapre_method/PANOVGGT_baseline
/home/aoki/PANOVGGT/.venv/bin/python scripts/align_pointcloud_to_gt.py
```

Default output:

- `/home/aoki/RethinkPanoVGGT_omega/Rethink_pano_new_exp_omega/outputs/panovggt_baseline_sample0/aligned_to_gt`

For the one-pano comparison sample, prefer pixel-correspondence alignment over
blind ICP. It uses the same ERP pixel in the PanoVGGT output and GT depth as an
automatic point pair, estimates a robust scale+orientation+translation
transform, and also writes a residual-filtered inlier cloud:

```bash
cd /home/aoki/RethinkPanoVGGT_omega/comapre_method/PANOVGGT_baseline
/home/aoki/PANOVGGT/.venv/bin/python scripts/align_by_pixel_correspondence.py --source-space world
```

Default output:

- `/home/aoki/RethinkPanoVGGT_omega/Rethink_pano_new_exp_omega/outputs/panovggt_baseline_sample0/pixel_aligned_to_gt`
