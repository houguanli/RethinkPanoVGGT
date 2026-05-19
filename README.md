# rethinvggt-omega

This folder is the organized VGGT-Omega branch for the pano experiments.

## Layout

- `Rethink_pano_new_exp_omega/`: current LUNA + panorama metadata port on top of VGGT-Omega.
- `baseline01_vggt_omega/`: baseline01 panorama geometry baseline migrated from the original `vggt` package layout to the newer `vggt_omega` framework.

## Baseline01 Omega Notes

`baseline01_vggt_omega` starts from the released VGGT-Omega package structure and adds the baseline01 training/loss surface:

- `training/pano_loss.py` now imports `vggt_omega.utils.*`.
- `training/loss.py` only computes camera/depth/point losses when their config blocks are enabled, so pano-only runs are valid.
- `vggt_omega.models.VGGTOmega.forward` exposes `pose_enc_list = [pose_enc]` for compatibility with the original baseline01 multi-stage loss API.
- `training/config/default.yaml` instantiates `vggt_omega.models.vggt_omega.VGGTOmega`.

Smoke test:

```bash
cd rethinvggt-omega/baseline01_vggt_omega
python tests/test_pano_loss.py
```

The LUNA omega experiment can be tested separately:

```bash
cd rethinvggt-omega/Rethink_pano_new_exp_omega
python tests/test_pano_sampler_luna_omega.py
```
