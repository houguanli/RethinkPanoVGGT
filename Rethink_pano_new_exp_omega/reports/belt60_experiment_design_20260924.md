# Belt60 crop / polar completion pilot

This is a controlled pilot, not a claim of improved reconstruction quality.
The existing server launcher and branch remain unchanged.

## Observational audit

Reproduce with:

```bash
python scripts/audit_latitude_information.py --dataset-root /mnt/e/PanoVGGT_minimal_datasets/datasets --output reports/latitude_audit_20260924.json
```

24 training panoramas per dataset; seed 57; nine pitches (-60 to +60,
15-degree spacing), four yaws, identical 75-degree perspective probes at
128x128. Average horizontal/vertical RGB gradients and clipped log-depth
gradients on valid neighboring depth pixels. Validity uses the existing
RGB-depth common mask and range (0,80] metres. Source paths are retained
in the JSON. These are descriptive proxies, not a proof of optimal
reconstruction, correspondence quality, or test-set performance.

The negative/mid-latitude preference is supported, but a universal optimum
at -15 degrees is not. In particular, outdoor positive high latitudes often
lack finite depth labels; indoor ceilings can still contain valid geometry.
Missing/invalid depth is NOT automatically a sky semantic label.

| Dataset (24 panoramas each) | Valid-texture peak pitch | Valid log-depth-gradient peak pitch | Valid fraction at +60 probe |
|---|---|---|---|
| Stanford2D3DS | -30 | -15 | 73.92% |
| Matterport3D | -30 | -15 | 61.91% |
| Structured3D | -45 | -15 | 98.97% |
| PanoCity | -45 | -15 | 3.26% |

The table reports *probe center pitches*, not disjoint latitude bands.
Each 75-degree probe overlaps its neighbors. For example a +60 probe
contains more than the polar cap, so its valid fraction is not the cap's
label density. Negative30 vs positive30 valid-texture ratios are
1.36/1.39/1.51/3.70 respectively. Finite depth availability alone is not
texture or multi-view matchability. No confidence claim is made from 24
samples or from selecting the maximum among nine pitches.

## Controlled change

| Arm | Pitches | Yaws per ring | FoV | Total windows | Trusted Omega ERP |
|---|---|---|---|---|---|
| A | -15 | 4 | 75x75 | 4 | Native window splat |
| B | -25,+25 | 6 | 75x75 | 12 | Window splat intersected with abs(latitude)<=60 |

Geometry-only nearest-splat measurement at windows384 / ERP512x1024:
A direct coverage 34.49% ERP pixels / 48.76% spherical area;
B direct coverage 66.72% ERP pixels / 86.64% spherical area.
B covers 99.879% of the +/-60 belt; tiny raster holes remain.
These are raw direct-splat fractions, NOT valid-GT fractions or low-resolution
head-input coverage. The polar caps occupy about 33.3% ERP pixels but 13.4%
spherical area. The existing small U-Net still runs on the full low-resolution
ERP; its prediction responsibility shrinks, not its tensor dimensions.

Both arms use the same frozen 2-hour Omega warm-up checkpoint, same freshly
initialized width32 completion head, seed, data ordering, optimizer and
300 updates (120-minute safety ceiling; unequal updates fail the pipeline).
No new full-model warm-up is needed for this pilot. This tests whether the
existing Omega knowledge transfers to the wider crop arrangement. It is
not a converged 12-hour end-to-end training experiment.

Common prerequisites, applied to BOTH arms:

- Honor config crop geometry; persist/restore it in head checkpoints.
- Fix scale fitting: predicted remainder cannot determine its own scale.
  One scale per pano set, on the shared canonical -15/75/yaw4 domain.
  Training uses median log ratio; reporting retains existing IRLS fitting.
- Enforce the requested learning rate after optimizer resume.
- New eval metadata uses the common RGB/depth validity mask, excludes
  unlabeled caps, and reports missing regional metrics as NaN plus counts.
- Exceptions write interrupted.pt and failed status, never a successful last.pt.
- Fix single-panorama preview interpreting RGB channels as panorama count.

The head architecture, global-median base fill, boundary blend and existing
loss coefficients are otherwise unchanged, to isolate the crop hypothesis.
Known limitations: unsupervised confidence output; no explicit sky classifier;
global-median fill near boundaries can still create artifacts.

## Execution and interpretation

```bash
bash scripts/run_belt60_completion_ab_local.sh
```

Output: logs/belt60_completion_ab_20260924. Stages: B single-update
preflight; A300/B300; smoothed loss plots; diagnostic80_two_panos per arm;
same Panocity preview; B 10-pano capacity probe; formal full8833 per arm.
No overwrite or automatic restart of interrupted training. The capacity
probe must pass without reducing official pano counts.

Diagnostic evaluation is explicitly 20 unseen test sets per dataset,
2 panoramas per set. It is not a paper-protocol benchmark. Formal evaluation
must pass cardinality validation: Stanford216 / Matterport891 /
Structured1662 / Panocity6064 = 8833 sets.

Compare full ERP and fixed canonical-region AbsRel/delta separately, and
cap60 AbsRel with valid-pixel counts. Do not compare each arm's own
remaining-region error as if the domains were identical. Report dataset
results individually and include GPU memory, throughput and time.
More coverage alone is not evidence of better reconstruction; reject B if
it damages the common core or is too expensive for the intended hardware.
