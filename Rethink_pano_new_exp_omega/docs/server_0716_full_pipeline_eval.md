# 0716 full-pipeline evaluation on the 4xRTX5000 server

The launcher mirrors the local workstation layout below `/whitehole/AOKI` and
evaluates the complete held-out Mixed4 splits with four shards.

Required files:

- Dataset root: `/whitehole/AOKI/PanoVGGT_minimal_datasets/datasets`
- LUNA delta: `RethinkPanoVGGT_omega_multipano_work/Rethink_pano_new_exp_omega/logs/local_full_warmup_no_depthres_20260716_002_luna_9h/last.pt`
- Warmup delta: `RethinkPanoVGGT_omega_multipano_work/Rethink_pano_new_exp_omega/logs/local_full_warmup_no_depthres_20260716_002_warmup_3h/last.pt`
- Foundation: `RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt`

After pulling branch `omega-multipano-work`, run a path/checkpoint preflight:

```bash
cd /whitehole/AOKI/RethinkPanoVGGT_omega_multipano_work/Rethink_pano_new_exp_omega
bash scripts/run_server_0716_full_pipeline_eval_4xrtx5000.sh --check-only
```

Start the full evaluation:

```bash
bash scripts/run_server_0716_full_pipeline_eval_4xrtx5000.sh
```

The defaults are `GPUS=0,1,2,3`, the full four-dataset test splits, anchor
sampling, PanoVGGT comparison pano counts, 384-pixel windows, four yaw windows,
and camera metrics from up to three panoramas. All settings and paths can be
overridden with environment variables. For example:

```bash
GPUS=2,3 DATASET_ROOT=/alternate/datasets \
  EVAL_OUT=/alternate/results/0716_eval \
  bash scripts/run_server_0716_full_pipeline_eval_4xrtx5000.sh
```

`AOKI_STORAGE_ROOT` controls recursive checkpoint relocation. With its default
value `/whitehole/AOKI`, checkpoint references saved under `/home/aoki/...`
are resolved under the server mirror without rewriting the multi-gigabyte
checkpoint files.
