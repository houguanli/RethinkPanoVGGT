# Learned full-ERP completion v1

## Geometry and knowledge transfer

VGGT-Omega keeps the canonical M1 input distribution: pitch `-15`, FoV
`75x75`, and four yaw views. Its window depth is converted to radial depth and
splatted into ERP coordinates. A 1.15M-parameter circular U-Net receives:

- full ERP RGB;
- Omega splatted log depth;
- Omega coverage mask and boundary distance;
- sine/cosine latitude encoding.

The trusted Omega core is copied exactly. The head predicts a bounded log-depth
residual only for missing pixels, with a narrow outside-boundary blend. Thus the
completion cannot regress the validated core reconstruction or camera head.

## Loss

The training objective is:

`L = L_remaining + wb L_boundary + wd L_Omega_distill + ws L_edge_smooth`.

`L_remaining` and `L_boundary` are scale-invariant log-Huber losses weighted by
`max(cos(latitude), 0.05)`, so polar ERP rows do not dominate despite their
small spherical area. Distillation teaches the completion decoder Omega's local
depth convention on observed pixels. Horizontal derivatives use circular ERP
seams; smoothness is suppressed at RGB edges.

## Twelve-hour schedule

1. 2h full Omega warm-up from the M1 A checkpoint, canonical FoV only.
2. 8h frozen-Omega remaining-band training (`main`).
3. 2h lower-LR boundary/polar refinement (`refine`).
4. Quick 20-set-per-dataset learned-full-ERP evaluation and preview.
5. Formal anchor-policy evaluation over exactly 8,833 sets.

The launcher is `scripts/run_full_erp_completion_v1_local.sh`. Every stage is
checkpoint guarded and resumable. Formal evaluation reports learned fill and
evaluated GT fractions; a healthy full-ERP run should approach 1.0 evaluated
fraction rather than the previous 42.6% canonical-window coverage.
