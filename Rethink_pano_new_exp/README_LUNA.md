# Rethink Pano VGGT LUNA

This experiment starts from `VGGT_pano_b2_baseline` and adds an MVP LUNA path:

- pano equirectangular image to virtual pinhole windows;
- token metadata with `global_patch_id`, spherical direction, seam flags, and local patch ids;
- zero-initialized `LunaPatchAdapter` for global patch-bank sharing;
- zero-initialized `LunaCameraAdapter` for known virtual-camera metadata injection;
- optional point, seam, and shared-camera consistency losses.

The default training config uses `vggt.models.vggt_luna.VGGT_LUNA`, keeps VGGT's local RoPE unchanged, and inserts LUNA adapters in the second half of the aggregator blocks.

## Pano Data Chain

Remake raw MatrixCity-style panoramas into the numbered pano format:

```bash
python tools/pano_data_remaker.py \
  --input_dir E:/pano_rl4/YOUR_RAW_PANO_FOLDER \
  --output_root E:/pano_rl4/YOUR_PANO_LUNA_NUMBERED \
  --depth_suffix _depth.exr \
  --input_depth_scale 0.01 \
  --output_depth_scale 100.0
```

The generated dataset has `00000/`, `00001/`, ... folders. Each item contains whole-pano `rgb.png`, `depth.png`, optional `normal.png`, `camera_6dof.txt`, `pose_c2w.txt`, and `meta.json`.

Train with the pano loader config after setting `Pano_DIR` in `training/config/pano_luna.yaml`:

```bash
cd training
torchrun --nproc_per_node=1 launch.py --config pano_luna
```

The checkpoint path defaults to `ckpt/model.pt` under this experiment folder.

Useful smoke commands from this directory:

```bash
python tests/test_pano_sampler_luna.py
python tests/test_luna_loss.py
python tests/test_pano_geometry.py
python tests/test_pano_loss.py
```
