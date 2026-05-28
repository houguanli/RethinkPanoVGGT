# Pano Training Settings

## Single Pano

Use `configs/single_pano_depthonly_4090_16h.yaml`.

For a single panorama, all virtual perspective windows are sampled from the
same known pano center. The training target therefore disables camera
supervision and trains reconstruction only:

- `pano_sample_mode: single`
- `camera_supervision_mode: none`
- `camera_loss_weight: 0.0`
- `trainable: luna_dense`

Evaluation should use the known-window projection path when comparing point
clouds, because the window geometry is deterministic from yaw, pitch and FoV.

## Multi Pano

Use `configs/multipano_pano_relative_4090.yaml`.

For multiple panoramas, each pano still has deterministic intra-pano window
geometry. The camera objective is pano-level relative translation, not
independent local-window pose:

- `pano_sample_mode: variable_neighborhood`
- `pano_min_count: 2`
- `pano_max_count: 8`
- `camera_supervision_mode: pano_relative`
- `camera_position_mode: relative_anchor`

The loss converts each predicted window translation back to a pano center using
the known sampled window rotation, averages centers per pano, and compares
those centers against GT positions relative to pano0. A small consistency term
keeps windows sampled from the same pano from disagreeing about the pano center.

## 4090 Capacity Check

With `window_size=512` and `num_yaw=2`, the local RTX 4090 completed one train
step for fixed neighborhoods of 4, 5, 6, 7 and 8 panos. The measured one-step
times were approximately 1.4, 4.2, 8.1, 10.9 and 13.7 minutes respectively.
`pano_max_count=8` is the verified upper setting for this configuration, but
lower caps are more practical for iterative experiments.
