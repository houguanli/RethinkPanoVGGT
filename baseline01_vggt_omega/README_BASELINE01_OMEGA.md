# Baseline01 VGGT-Omega

This package is the baseline01 panorama geometry baseline ported from the old
`vggt` code layout to the newer `vggt_omega` framework.

The model remains the plain VGGT-Omega backbone: no LUNA adapters and no pano
metadata injection. The panorama-specific part is the baseline01 geometry loss,
which constrains crops from the same panorama with zero-translation,
absolute-rotation, and relative-rotation terms.

## Main Changes From Baseline01

- Package imports use `vggt_omega.*` instead of `vggt.*`.
- The training config instantiates `vggt_omega.models.vggt_omega.VGGTOmega`.
- Omega's single camera prediction is wrapped as `pose_enc_list = [pose_enc]`
  so the old baseline01 staged-loss code still works.
- Pano-only loss runs are supported by setting `camera: null`, `depth: null`,
  and `point: null` in the loss config.

## Quick Check

```bash
python tests/test_pano_loss.py
```
