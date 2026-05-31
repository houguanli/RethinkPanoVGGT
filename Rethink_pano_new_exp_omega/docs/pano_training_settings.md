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
- `view_sampling_mode: cyclic`
- `views_per_pano: 2`

The loss converts each predicted window translation back to a pano center using
the known sampled window rotation, averages centers per pano, and compares
those centers against GT positions relative to pano0. A small consistency term
keeps windows sampled from the same pano from disagreeing about the pano center.

## 4090 Capacity Check

The current 4090 multipano profile keeps the source panorama at its converted
resolution, but uses `window_size=384` and samples only two perspective windows
per pano per step from an `8 yaw x 3 pitch` candidate grid. This preserves
long-run spherical coverage without placing all candidate windows into the
Omega aggregator at once.

The older `window_size=512`, `num_yaw=2`, `pitch_degrees=0` setting was memory
fragile for 6-pano batches and only observed the equatorial band. Prefer the
dynamic 384 profile for local multipano experiments, then use denser eval
windows or a short higher-resolution fine-tune after the loss behavior is
validated.

Smoke checks on the local RTX 4090 passed for fixed worst-case batches of
6 panos and 8 panos with `views_per_pano=2`; the 8-pano step is usable but
noticeably slower, so reduce `pano_max_count` to 6 for faster iteration.
