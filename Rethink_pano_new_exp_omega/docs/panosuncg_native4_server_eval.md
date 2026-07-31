# PanoSUNCG 0716 native-four-window zero-shot evaluation

This entry point reproduces the PanoSUNCG protocol used for the local 0716
full-pipeline result:

- one complete ERP input per depth sample;
- checkpoint-native four `384 x 384` pinhole windows;
- four yaw angles, pitch `-15` degrees, FOV `75` degrees;
- ground truth sampled onto the same rays and converted from radial depth to
  pinhole Z-depth;
- solid-angle weighting and overlap de-duplication;
- repository-default 10-step scale-only IRLS;
- direct window-space metrics, with no ERP fusion and no latitude mask.

The launcher shards depth evaluation across the requested GPUs, verifies and
merges all per-sample CSVs, then evaluates camera-center trajectories on the
first GPU. All stages are resumable.

## Dataset layout

Pass either the `PanoSUNCG_zeroshot` directory below or its parent `datasets`
directory:

```text
PanoSUNCG_zeroshot/
├── panosuncg_da2_split.txt
└── PanoSUNCG/
    ├── rotated/
    │   └── <scene>/<trajectory>/<frame>_color.png
    │                              └── <frame>_depth.png
    └── labels/
        └── <scene>/<trajectory>
```

The server dataset is expected at:

```text
/AOKI/whitehole/PanoVGGT_minimal_datasets/datasets/PanoSUNCG_zeroshot
```

## Preflight

From `Rethink_pano_new_exp_omega`:

```bash
bash scripts/run_panosuncg_0716_native4_4gpu.sh \
  --checkpoint /ABSOLUTE/PATH/TO/last.pt \
  --dataset-root /AOKI/whitehole/PanoVGGT_minimal_datasets/datasets/PanoSUNCG_zeroshot \
  --output-dir /ABSOLUTE/PATH/TO/eval_panosuncg_0716_native4 \
  --gpus 0,1,2,3 \
  --check-only
```

The checkpoint preflight rejects a sampler other than `384 / 4 / -15 / 75`.

## Full depth and camera evaluation

```bash
nohup bash scripts/run_panosuncg_0716_native4_4gpu.sh \
  --checkpoint /ABSOLUTE/PATH/TO/last.pt \
  --dataset-root /AOKI/whitehole/PanoVGGT_minimal_datasets/datasets/PanoSUNCG_zeroshot \
  --output-dir /ABSOLUTE/PATH/TO/eval_panosuncg_0716_native4 \
  --gpus 0,1,2,3 \
  > /ABSOLUTE/PATH/TO/eval_panosuncg_0716_native4/launcher.log 2>&1 &
```

Use `--python /absolute/path/to/python` if the environment is not in one of
the launcher's standard locations. For a relocated foundation checkpoint, add:

```bash
--foundation-checkpoint /absolute/path/to/vggt_omega_1b_512.pt
```

The checkpoint's referenced warmup delta must also exist in the mirrored
repository/log layout.

## Smoke test

```bash
bash scripts/run_panosuncg_0716_native4_4gpu.sh \
  --checkpoint /ABSOLUTE/PATH/TO/last.pt \
  --dataset-root /AOKI/whitehole/PanoVGGT_minimal_datasets/datasets/PanoSUNCG_zeroshot \
  --output-dir /tmp/panosuncg_native4_smoke \
  --gpus 0,1,2,3 \
  --limit 8 \
  --max-trajectories 1
```

## Outputs

```text
<output-dir>/
├── depth_native4/
│   ├── metrics_summary.json
│   ├── per_sample_metrics.csv
│   ├── progress.json
│   ├── run_config.json
│   ├── merge.log
│   └── shards/shard_00..03/
└── camera_pose/
    ├── camera_center_summary.json
    ├── trajectory_metrics.csv
    ├── frame_metrics.csv
    ├── pair_metrics.csv
    ├── progress.json
    ├── run_config.json
    └── run.log
```

Re-run the identical command to resume. The launcher and each evaluator use
non-blocking file locks to reject duplicate writers.
