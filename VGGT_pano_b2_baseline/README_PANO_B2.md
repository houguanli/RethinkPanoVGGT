# VGGT Pano Baseline B2

This baseline builds on B1 and adds panorama-aware structure to VGGT.

## Added components

- All B1 pano loss and optional LoRA pieces.
- `Aggregator(enable_pano_global_token=True)`: adds a panorama global camera
  token before register tokens while keeping the original camera token at index
  0 for `CameraHead` compatibility.
- Geometry injection from view parameters into the camera token and pano global
  token.

## Pano geometry input

`VGGT.forward` accepts either:

- `pano_view_params`: `[B, S, 3+]` with `(theta, phi, fov)` or
  `(theta, phi, fov_h, fov_w)`.
- `pano_angles` plus optional `pano_fov`.

The model encodes geometry as:

`[sin(theta), cos(theta), sin(phi), cos(phi), fov_h, fov_w]`.

The trainer forwards these keys automatically when they exist in the batch.
