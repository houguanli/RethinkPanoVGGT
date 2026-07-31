# Quantitative comparison methods

Created on 2026-06-08.

This folder mirrors the comparison groups in `compare_methods_v7.md`.
Repeated methods are intentionally cloned into each comparison group when they
appear in more than one group.

## camera_pose

| Method | Code status | Dependency audit |
| --- | --- | --- |
| PanoVGGT | Cloned from `https://github.com/YijingGuo-June/PanoVGGT` | `requirements.txt` updated to match README PyTorch/xFormers versions and add `iopath`, `wcmatch`. |
| VGGT-Omega | Cloned from `https://github.com/facebookresearch/vggt-omega` | Existing `requirements.txt` plus `requirements_demo.txt` cover inference/demo imports; no training code in public repo. |
| Reloc3r | Cloned from `https://github.com/ffrivera0/reloc3r` | CroCo submodule initialized; requirements expanded for inference, evaluation, training, and dataset preprocessing. |
| BiFuse++ | Official `fuenwang/BiFusev2` submodule | Pinned inference environment and headless two-panorama checkpoint test; supervised release is DepthNet-only. |
| Pi3 | Official `yyfz/Pi3` submodule | Pinned PyTorch 2.5.1 environment, local safetensors routing, camera-pose and PLY sample tests. |

BiFuse++ and Pi3 source, checkpoint, license, and sample commands are documented
in [`camera_pose/BIFUSEPP_PI3.md`](camera_pose/BIFUSEPP_PI3.md).

## depth_geometry

| Method | Code status | Dependency audit |
| --- | --- | --- |
| PanoVGGT | Cloned from `https://github.com/YijingGuo-June/PanoVGGT` | Independent copy; same dependency updates as the camera-pose copy. |
| VGGT-Omega | Copied from the `camera_pose` group clone of `https://github.com/facebookresearch/vggt-omega` | Existing `requirements.txt` plus `requirements_demo.txt` cover inference/demo imports; no training code in public repo. |
| VGGT-360 | Cloned from `https://github.com/Yuanjiayii/VGGT-360` | Public repo currently contains only a README; see `VGGT-360/DEPENDENCY_AUDIT.md`. |
| DAP | Cloned from `https://github.com/Insta360-Research-Team/DAP` | `requirements.txt` updated for README environment and source imports used by inference/evaluation/utilities. |
| PanDA | Cloned from `https://github.com/caozidong/PanDA` | `requirements.txt` updated for README PyTorch 2.0.0 environment and source imports used by inference/training/evaluation. |

## local method

The current method is not cloned here because it is a local implementation:

`D:\github_aoki\RethinkPanoVGGT_omega\Rethink_pano_new_exp_omega`
