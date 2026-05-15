# Rethink Pano VGGT LUNA

This experiment starts from `VGGT_pano_b2_baseline` and adds an MVP LUNA path:

- pano equirectangular image to virtual pinhole windows;
- token metadata with `global_patch_id`, spherical direction, seam flags, and local patch ids;
- zero-initialized `LunaPatchAdapter` for global patch-bank sharing;
- zero-initialized `LunaCameraAdapter` for known virtual-camera metadata injection;
- optional point, seam, and shared-camera consistency losses.

The default training config uses `vggt.models.vggt_luna.VGGT_LUNA`, keeps VGGT's local RoPE unchanged, and inserts LUNA adapters in the second half of the aggregator blocks.

Useful smoke commands from this directory:

```bash
python tests/test_pano_sampler_luna.py
python tests/test_luna_loss.py
python tests/test_pano_geometry.py
python tests/test_pano_loss.py
```

