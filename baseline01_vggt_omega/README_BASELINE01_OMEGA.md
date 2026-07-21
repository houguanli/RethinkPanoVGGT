# Baseline01 VGGT-Omega

This package is the baseline01 panorama geometry baseline ported from the old
`vggt` code layout to the newer `vggt_omega` framework.

The model remains the plain VGGT-Omega backbone: no LUNA adapters and no pano
metadata injection. The panorama-specific part is the baseline01 geometry loss,
which constrains crops from the same panorama with zero-translation,
absolute-rotation, and relative-rotation terms.

## Baseline Taxonomy

- `naive_vggt_omega / full_erp_no_window_split`: one resized equirectangular
  panorama is one Omega frame. There is no perspective projection or crop. The
  full ERP radial depth is supervised after the same scene normalization and
  per-sample log-median depth alignment used by the existing experiments.
- `window_split_ablation / perspective_window_split`: each panorama is split
  into perspective windows before Omega. All legacy `mixed4_pano_vggtomega_*`
  configs belong to this ablation, not to the completely naive baseline.

The naive baseline uses spherical rays to construct point maps for scene-scale
normalization. Its focal target is necessarily a pseudo-intrinsic because a
complete ERP is not a pinhole image; this limitation is part of the naive
baseline rather than hidden preprocessing.

Future checkpoints include `experiment_variant`, `input_representation`, and
`checkpoint_label`. The same fields are written to `experiment_manifest.json`
beside each `loss.csv`.

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
python tests/test_full_erp_dataset.py
python scripts/check_naive_fullerp_batch.py
```

Run the progressive local short experiment with:

```bash
bash scripts/run_naive_fullerp_vggtomega_short_4090.sh
```

It tests full-ERP widths `384,512,1024,2048` in sequence and stops at the first
CUDA OOM while preserving the last successful checkpoint.

On the four-GPU server, activate the `vggt-omega` conda environment and start
the production 12-hour schedule with:

```bash
bash scripts/start_naive_fullerp_vggtomega_12h_server.sh
```

The server launcher runs four three-hour stages at ERP widths
`384,512,1024,2048`, keeps two complete panoramas per sample, rebuilds the
mixed4 index once, resumes model/optimizer/scaler between stages, and writes a
PID plus the nohup log under `logs/`.
