#!/usr/bin/env python3
"""Compare latitude information proxies on reproducible TRAIN panoramas.

Equal-FoV perspective probes avoid comparing stretched ERP pixel gradients.
Texture and valid depth are proxies, not proof of reconstruction accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.data.pano_minimal import _read_rgb_tensor, _read_depth_tensor, _rgb_depth_common_valid_mask
from training.data import pano_minimal as pm
from vggt_omega.data.pano_sampler import PanoWindowSampler
from vggt_omega.models.erp_completion import splat_omega_window_depth_to_erp


def audit_paths(root, name):
    folder, prefix, scale = {
        "stanford2d3ds": ("Stanford2D3DS", "2d3ds", pm._STANFORD2D3DS_DEPTH_SCALE),
        "matterport3d": ("Matterport3D", "matterport3d", pm._MATTERPORT3D_DEPTH_SCALE),
        "structured3d": ("Structured3D", "structured3d", pm._STRUCTURED3D_DEPTH_SCALE),
        "panocity": ("Panocity", "panocity", pm._PANOCITY_DEPTH_SCALE),
    }[name]
    base = root/folder
    rows = json.loads((base/"cache"/(prefix+"_train_index.json")).read_text())
    paths = []
    bad = pm._load_bad_samples(ROOT/"configs/structured3d_bad_scenes.txt",root)
    for r in rows:
        if name == "panocity":
            paths.append((pm._resolve_cached_path(base,r["rgb_path"]),pm._resolve_cached_path(base,r["depth_path"]),scale))
            continue
        for pid in r[1] if name == "structured3d" else r[3]:
            if name == "stanford2d3ds":
                rgb = base/str(r[0])/"pano/rgb"/f"camera_{pid}_*_rgb.png"
                dep = base/str(r[0])/"pano/depth"/f"camera_{pid}_*_depth.png"
            elif name == "matterport3d":
                rgb = base/str(r[0])/"pano_skybox_color"/f"{pid}.jpg"
                dep = base/str(r[0])/"pano_depth"/f"{pid}.png"
            else:
                rgb = base/str(r[0])/"2D_rendering"/str(pid)/"panorama/full/rgb_rawlight.png"
                dep = rgb.with_name("depth.png")
            if not pm._is_bad_sample(str(rgb),str(dep),str(r[0]),bad):
                paths.append((rgb,dep,scale))
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples-per-dataset", type=int, default=24)
    p.add_argument("--seed", type=int, default=57)
    args = p.parse_args()
    torch.set_num_threads(2)
    pitches = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
    probe = PanoWindowSampler(window_size=128, num_yaw=4, pitch_degrees=pitches, fov_degrees=75)
    results = {"split": "train", "seed": args.seed, "fov_degrees": 75,
               "note": "Mean horizontal/vertical gradients in equal-FoV views. Descriptive proxies, not reconstruction accuracy; training split only.",
               "datasets": {}, "geometry": {}}
    lat = 90 - (torch.arange(512) + .5) * 180 / 512
    area = torch.cos(torch.deg2rad(lat)).view(1, 1, 512, 1).expand(1, 1, 512, 1024)
    for name, ps, ny in [("canonical", [-15], 4), ("belt60", [-25, 25], 6)]:
        s = PanoWindowSampler(window_size=384, num_yaw=ny, pitch_degrees=ps, fov_degrees=75)
        with torch.no_grad():
            sampled = s(torch.zeros(1, 3, 512, 1024))
            splat = splat_omega_window_depth_to_erp(torch.ones(1, len(ps)*ny, 384, 384, 1),
                sampled.camera_meta, num_panos=1, erp_height=512, erp_width=1024)
        m = splat.valid_mask
        band = (lat.abs() <= 60).view(1, 1, 512, 1).expand_as(m)
        if name == "belt60":
            m = m & band
        results["geometry"][name] = {"views": len(ps)*ny, "erp_coverage": m.float().mean().item(),
            "sphere_coverage": ((m*area).sum()/area.sum()).item(),
            "within_60_coverage": (m&band).sum().item()/band.sum().item()}
    for ds_idx, dataset_name in enumerate(["stanford2d3ds", "matterport3d", "structured3d", "panocity"]):
        rng = np.random.default_rng(args.seed+ds_idx)
        paths = audit_paths(args.dataset_root, dataset_name)
        indices = rng.choice(len(paths), min(len(paths), args.samples_per_dataset), replace=False)
        records = []
        for idx in indices:
            rgb_path, depth_path, scale = paths[int(idx)]
            if "*" in rgb_path.name:
                rgb_path = next(rgb_path.parent.glob(rgb_path.name))
                depth_path = next(depth_path.parent.glob(depth_path.name))
            image = F.interpolate(_read_rgb_tensor(rgb_path)[None],size=(512,1024),mode="bilinear",align_corners=False)[0]
            dep = F.interpolate(_read_depth_tensor(depth_path,scale,65535)[None],size=(512,1024),mode="nearest")[0]
            sample = {"pano_image":image,"pano_depth":dep,"sequence_name":str(rgb_path),
                      "pano_rgb_depth_common_mask":_rgb_depth_common_valid_mask(image,dep,dataset_name)}
            rgb, depth = sample["pano_image"], sample["pano_depth"]
            valid = sample["pano_rgb_depth_common_mask"] & torch.isfinite(depth) & (depth > 0) & (depth <= 80)
            with torch.no_grad():
                images = probe(rgb[None]).windows[0]
                masks = probe(valid.float().expand(3, -1, -1)[None], interpolation_mode="nearest").windows[0, :, :1] > .5
                log_depth = torch.log(torch.where(valid, depth, torch.ones_like(depth)))
                depths = probe(log_depth.expand(3, -1, -1)[None], interpolation_mode="nearest").windows[0, :, :1]
                gray = images.mean(1, keepdim=True)
                tex = [(gray[..., 1:]-gray[..., :-1]).abs(),
                       (gray[..., 1:, :]-gray[..., :-1, :]).abs()]
                pair = [masks[..., 1:] & masks[..., :-1],
                        masks[..., 1:, :] & masks[..., :-1, :]]
                dz = [(depths[..., 1:]-depths[..., :-1]).abs().clamp_max(1),
                      (depths[..., 1:, :]-depths[..., :-1, :]).abs().clamp_max(1)]
                # make_default_view_grid orders yaw first, pitch second.
                pitch_rows = []
                for j, pitch in enumerate(pitches):
                    sl = slice(j, None, len(pitches))
                    pitch_rows.append({"pitch": pitch, "valid_fraction": masks[sl].float().mean().item(),
                        "rgb_gradient": sum(t[sl].mean().item() for t in tex)/2,
                        "valid_texture_energy": sum((t[sl]*m[sl]).mean().item() for t,m in zip(tex,pair))/2,
                        "valid_logdepth_gradient": sum((d[sl]*m[sl]).mean().item() for d,m in zip(dz,pair))/2})
                band_rows = []
                for lo in range(-90,90,15):
                    band = (lat >= lo) & (lat < lo+15)
                    band_rows.append({"lo":lo, "hi":lo+15,
                        "valid_fraction":valid[:, band].float().mean().item()})
            records.append({"index":int(idx), "sequence":str(sample.get("sequence_name")),
                            "pitches":pitch_rows, "bands":band_rows})
            print(f"[AUDIT] {dataset_name} {len(records)}/{len(indices)}",flush=True)
        averages = []
        for j, pitch in enumerate(pitches):
            averages.append({"pitch":pitch, **{k:float(np.mean([r["pitches"][j][k] for r in records]))
                for k in ["valid_fraction","rgb_gradient","valid_texture_energy","valid_logdepth_gradient"]}})
        results["datasets"][dataset_name] = {"samples":len(records), "mean_by_pitch":averages,"records":records}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2),encoding="utf-8")
        print("[RESULT]",dataset_name,json.dumps(averages),flush=True)


if __name__ == "__main__":
    main()
