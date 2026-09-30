# Learned full-ERP completion v1

## Current four-GPU server entry point

The server launcher now uses B geometry (pitch `-25,25`, yaw6 per ring,
FoV75x75, 12 windows/pano) from the start of warm-up. Training uses Panocity=6
panos and indoor=3, ERP input 1024x512. The original canonical v1 design below
is historical, not the current server sampler configuration.

From an activated training environment:

```bash
cd /whitehole/AOKI/RethinkPanoVGGT_omega_multipano_work_pro/Rethink_pano_new_exp_omega
git pull --ff-only origin codex/multipano-work-pro
bash scripts/run_full_erp_completion_v1_4xrtx5000.sh
```

Default paths are built in. Logs are created automatically. The pipeline runs
2h warm-up + 8h completion + 2h refinement, optional loss analysis, quick20,
formal full evaluation, then optional preview. Evaluation and preview inherit
the saved crop geometry; they must not force the legacy yaw4/pitch=-15 sampler.
Plot/preview errors are warned about but do not block the formal evaluation.
Evaluator failures remain fatal and their shard log tails appear in the console.

### Recover evaluation after training has finished

Use the **same RUN_NAME as the completed training** (below is the default):

```bash
EVAL_ONLY=1 RUN_NAME=full_erp_completion_v1_4xrtx5000_2h8h2h \
  bash scripts/run_full_erp_completion_v1_4xrtx5000.sh
```

This skips all training and quick20. It requires nonempty
`logs/${RUN_NAME}_omega_warmup_2h/last.pt` and
`logs/${RUN_NAME}_completion_refine_2h/last.pt`; missing files cause an error,
never a training restart. Existing compatible shard CSV rows are resumed and
an existing full summary is retained and validated. Do not reuse a run/output
directory for a different checkpoint or evaluation protocol. CSVs containing
Panocity=6 from the old training-cap bug cannot be resumed as Panocity=10.

Formal evaluation remains `anchor`, `panovggt`, limit=0:
Stanford2D3DS=216, Matterport3D=891, Structured3D=1662, Panocity=6064
(8,833 sets). Evaluation pano counts remain Panocity=10 and indoor=3,
independent of the training cap of 6. An OOM must be reported, not worked
around by silently reducing inputs.

The result is
`logs/${RUN_NAME}_eval_full8833_anchor_full_erp_4gpu/validation_mixed4_by_dataset_valtestfull_summary.json`.
Failures can be inspected in `logs/${RUN_NAME}_pipeline.log` and the eval
directory's `shards/shard_*.log`.

## Final report and selected examples (server pipeline)

After the four shards merge, the launcher now writes and prints one
`EVAL_REPORT.txt` in the eval output directory. It contains all four datasets,
actual set/pano counts, crop and ERP depth metrics, canonical/polar diagnostics,
camera AUC and valid-pair counts, and clearly separated overall aggregations
(equal per set, equal per dataset, and pooled crop valid-pixel/solid-angle
weighting). Missing camera GT is N/A, never a perfect zero-error score.
Partial/smoke evaluations are explicitly marked as not full8833.

`good_cases.json` selects up to three examples per dataset by default:
crop IRLS AbsRel <= 0.15, delta1.25 >= 0.85, valid fraction >= 0.1,
and positive valid-pixel count. Nonfinite scores are excluded, identical pano
groups deduplicated, and thresholds are never silently relaxed. These are
depth-selected good examples, not representative performance or camera-selected
examples. The formal metrics still include every evaluated set.

Selected sets are replayed with the same ordered multi-pano input (Panocity=10,
indoor=3), checkpoint geometry and sample index. Each case under `examples/`
contains `input_panos.jpg`, `comparison.png` (anchor-pano window RGB / GT /
IRLS-aligned prediction / AbsRel), and `metrics.json` (original and replayed
scores). GT and predictions share the depth color scale. Invalid GT is masked;
these panels do not claim full-ERP completion quality. Examples never switch
the model to single-pano inference. Original benchmark CSVs are not modified.

The server start command is unchanged. With `EVAL_ONLY=1`, an existing complete
summary is reused and acquires these artifacts without rerunning all 8,833 sets.
Only selected examples require new inference; completed matching examples are
cached. `EXAMPLES_PER_DATASET=5` changes the count; `EXPORT_EXAMPLES=0` writes
the report/selection manifest without rendering images.

To regenerate artifacts directly from an already merged result:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/summarize_mixed4_eval.py \
  --eval-dir logs/full_erp_completion_v1_4xrtx5000_2h8h2h_eval_full8833_anchor_full_erp_4gpu \
  --export-examples
```

The standalone command also accepts `--examples-per-dataset`, `--max-absrel`,
`--min-delta`, and `--min-valid-fraction`. Failed example export preserves the
benchmark and reports its error log; it is not reported as successful export.

## Historical v1 geometry and knowledge transfer

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
