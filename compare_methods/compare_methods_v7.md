# Compare Methods for V7

This note follows the v7 quantitative scope: arXiv-only rows marked with # in the v6 tables are removed. The remaining quantitative comparison is split into two groups.

## Group 1: Camera Pose

### Metrics to report

| Metric family | Concrete metrics | Notes |
|---|---|---|
| Relative pose | RRE, RTE, relative translation-direction error | Use the same panorama pair / sequence split for every method. |
| Trajectory / global pose | ATE, global RRE / RTE, multi-view averaging success rate | Needed for Reloc3r motion averaging and sequence-level panorama evaluation. |
| Threshold accuracy | pose accuracy at 5 / 10 / 20 deg or matched angular / translation thresholds | Report both mean / median and threshold success if possible. |
| Calibration / reprojection | reprojection residual, intrinsic / extrinsic residual where available | Useful for VGGT-family outputs and failure diagnosis. |

### Methods

| Method | Venue in PPT | Why compare | Code / repo path |
|---|---|---|---|
| PanoVGGT | CVPR 2026 | Direct panorama feed-forward 3D reconstruction baseline; predicts camera and geometry. | https://github.com/YijingGuo-June/PanoVGGT |
| VGGT-Omega | CVPR 2026 Oral | Backbone-family baseline closest to our implementation. | https://github.com/facebookresearch/vggt-omega |
| Reloc3r | CVPR 2025 | Modern open-source relative-pose baseline with multi-view motion averaging; run on the same ERP-to-window/cubemap protocol as VGGT-Omega and Ours. | https://github.com/ffrivera0/reloc3r |
| Ours | Current implementation | VGGT-Omega + PanoWindowSampler + spherical metadata + zero-init LUNA adapters. | D:\\github_aoki\\RethinkPanoVGGT_omega\\Rethink_pano_new_exp_omega |

## Group 2: Depth + Geometry Reconstruction Quality

### Metrics to report

| Metric family | Concrete metrics | Notes |
|---|---|---|
| Depth accuracy | AbsRel, SqRel, RMSE, RMSElog, delta1 / delta2 / delta3 | Report metric-depth and scale-aligned variants when scale conventions differ. |
| Depth robustness | valid-pixel ratio, failure rate, seam / pole stability | Important for ERP panoramas and cubemap / virtual-window methods. |
| Derived geometry | Chamfer distance, F-score, normal consistency, completeness / accuracy | Generate point clouds or meshes from predicted depth / point maps under the same camera convention. |
| Efficiency | runtime, peak memory, input resolution, number of windows / views | Keep secondary, but useful for feed-forward claims. |

### Methods

| Method | Venue in PPT | Why compare | Code / repo path |
|---|---|---|---|
| PanoVGGT | CVPR 2026 | Direct upper-bound style panorama reconstruction baseline for depth, camera, and point cloud. | https://github.com/YijingGuo-June/PanoVGGT |
| VGGT-360 | CVPR 2026 | Training-free panoramic depth via VGGT-style 3D consistency; strong depth + geometry reference. | https://github.com/Yuanjiayii/VGGT-360 |
| DAP | CVPR 2026 | Panoramic depth foundation model; strong forward depth baseline. | https://github.com/Insta360-Research-Team/DAP |
| PanDA | CVPR 2025 | Panoramic Depth Anything baseline with unlabeled panoramas and Mobius spatial augmentation. | Project page: https://caozidong.github.io/PanDA_Depth/ ; no official GitHub repo found as of 2026-06-08. |
| Ours | Current implementation | Camera + depth-derived geometry from VGGT-Omega panorama adaptation. | D:\\github_aoki\\RethinkPanoVGGT_omega\\Rethink_pano_new_exp_omega |

## Source Links Checked

- PanoVGGT GitHub: https://github.com/YijingGuo-June/PanoVGGT
- VGGT-Omega GitHub: https://github.com/facebookresearch/vggt-omega
- VGGT-360 CVPR page / code link: https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_VGGT-360_Geometry-Consistent_Zero-Shot_Panoramic_Depth_Estimation_CVPR_2026_paper.html
- DAP GitHub: https://github.com/Insta360-Research-Team/DAP
- PanDA project page: https://caozidong.github.io/PanDA_Depth/
- Reloc3r GitHub: https://github.com/ffrivera0/reloc3r
