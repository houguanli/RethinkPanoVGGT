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

Use `configs/multipano_equator_ring_4090.yaml` for local 4090 experiments when
single-step pano context matters. Use `configs/multipano_pano_relative_4090.yaml`
as the lower-memory stochastic spherical fallback.

For multiple panoramas, each pano still has deterministic intra-pano window
geometry. The camera objective is pano-level relative translation, not
independent local-window pose:

- `pano_sample_mode: variable_neighborhood`
- `pano_min_count: 2`
- equator-ring profile: `pano_max_count: 3`
- stochastic fallback: `pano_max_count: 8`
- `camera_supervision_mode: pano_relative`
- `camera_position_mode: relative_anchor`
- equator-ring profile: `window_size: 256`, `num_yaw: 8`, `pitch_degrees: "0"`, `view_sampling_mode: fixed`
- stochastic fallback: `window_size: 384`, `num_yaw: 8`, `pitch_degrees: "-35,0,35"`, `view_sampling_mode: cyclic`, `views_per_pano: 2`

The loss converts each predicted window translation back to a pano center using
the known sampled window rotation, averages centers per pano, and compares
those centers against GT positions relative to pano0. A small consistency term
keeps windows sampled from the same pano from disagreeing about the pano center.

## 4090 Capacity Check

The preferred 4090 multipano profile keeps the source panorama at its converted
resolution and sends the full equatorial horizon ring into each forward pass:
`8 yaw x 1 pitch` windows at `window_size=256`. This gives each pano-global
token access to the whole street-view ring instead of isolated local windows.
With the current Omega aggregator this is practical for 2..3 neighboring panos
on a 24GB 4090.

The stochastic fallback uses `window_size=384` and samples two perspective
windows per pano per step from an `8 yaw x 3 pitch` candidate grid. It preserves
long-run spherical coverage without placing all candidate windows into the
Omega aggregator at once, but each single forward pass has weaker pano-level
context.

The older `window_size=512`, `num_yaw=2`, `pitch_degrees=0` setting was memory
fragile for 6-pano batches and only observed the equatorial band. Prefer the
dynamic 384 profile for local multipano experiments, then use denser eval
windows or a short higher-resolution fine-tune after the loss behavior is
validated.

Smoke checks on the local RTX 4090:

- equator ring, `2 panos x 8 windows x 256`: passed in about 14 seconds.
- equator ring, `3 panos x 8 windows x 256`: passed in about 15 seconds.
- equator ring, `4 panos x 8 windows x 256`: hit the 24GB memory cliff and did
  not complete a step in 10 minutes.
- equator ring, `6 panos x 8 windows x 256`: hit the 24GB memory cliff and did
  not complete a step in 15 minutes.
- stochastic fallback, fixed worst-case 6 and 8 pano batches with
  `views_per_pano=2`: passed.

Use the equator-ring config when full pano horizon context is more important
than pano count. Use the stochastic spherical fallback when the experiment needs
up to 8 neighboring panos on the single 4090.
