# Independent ERP completion ablations — 2026-09-29

Based on `codex/multipano-work-pro@3ea1294d82d702279719a744fa3d5d26016f964a`.
Selected recipe: **4 x RTX 5000, 2h warmup + 8h completion + 2h refinement**.

| Directory | configs/train.yaml: geora_ablation | Intervention |
|---|---|---|
| A1_no_geora | no_geora | Patch AND Camera GeoRA: all parameters zero/frozen |
| A2_no_camera_geora | no_camera_geora | Camera GeoRA: all parameters zero/frozen |
| A3_no_patch_geora | no_patch_geora | ENTIRE Patch GeoRA zero/frozen, user-confirmed new w/o Patch Bank |
| A4_random_patch_bank | random_patch_bank | Seeded N(0,1) bank vectors, not feature shuffle |
| Rethink_pano_new_exp_omega | full | Full reference and retained historical logs |

All five directories contain physical, independent copies of source, config,
evaluation helpers and tests. No cross-experiment imports or symlinks.
Only the dataset and foundation weights are externally shared.

## Run on the server

Activate the existing conda environment and enter ONE experiment directory:

```bash
cd /whitehole/AOKI/RethinkPanoVGGT_omega_ablation/A1_no_geora
bash run.sh --run-dir logs/run_001
# Skip 2h warmup: directly run completion 8h + refinement 2h.
bash run.sh --warmup-checkpoint /absolute/path/warmup/last.pt --run-dir logs/run_002
bash run.sh --no-auto-eval --run-dir logs/run_003
bash run.sh --dry-run
# Independent evaluation: only checkpoint and output directory.
python scripts/run_ablation_full_eval.py logs/run_001/completion_refine/last.pt logs/run_001/eval_full
```

Each experiment has only `configs/train.yaml` (complete training config) and
`configs/pipeline.yaml` (paths, GPUs, duration, evaluation) as run configs,
plus the original bad-scene list. Edit server hyperparameters there.
Defaults: dataset `/whitehole/AOKI/panovggt`, foundation
`/whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt`.
Do not run four arms concurrently on the same four GPUs.
Nonempty output directories are rejected rather than overwritten.

Training success immediately triggers full evaluation; training failure never
triggers it, and eval failure returns nonzero. Disable with `--no-auto-eval`
or YAML `auto_eval: false`. Run state is saved in `pipeline_status.json`.
The completion checkpoint identifies its Omega teacher; keep the warmup and
completion output folders together when relocating results. Evaluating a plain
Omega checkpoint remains supported, without learned-completion metrics.

## Controls and interpretation

Common upstream geometry: 384-square windows, 6 yaw x pitches [-25,+25],
FoV 75 x 75, input ERP 1024x512. Training: 6 PanoCity panoramas, 3 indoor.
All arms retain the same depth/camera heads, pano-global tokens, losses and
2h/8h/2h schedule. Omega is frozen in the completion and refinement stages.
Eval retains upstream anchor/cardinality settings and camera 3-pano cap.
Native geometry overrides are zero: the old launcher yaw=4 default must not
override a 12-window completion checkpoint.

Disabled branches remain instantiated with identical tensor shapes and total
parameter count. ALL their parameters, including LayerNorm and residual alpha,
are zero and requires_grad=False. They return the input unchanged. A checkpoint
post-load hook reapplies zero/freeze, as does trainability selection before
optimizer construction. Trainable parameter counts intentionally differ.
Camera GeoRA off does not disable camera prediction or supervision.

A4 replaces bank contents with independent standard-normal vectors per occupied
(panorama, ERP cell). Same-cell windows share the same vector. Seed 43 plus a
stable layer offset and a private RNG make training/eval/checkpoint recomputation
deterministic. No bank features are pooled or shuffled; positional encoding stays.
This is a random-context control, NOT the old correspondence-shuffle experiment.

Shared warmup reuse is supported and recorded. Disabled parameters are re-zeroed,
and completion saves its intervention for eval even if the warmup came from
another arm. For controlled comparisons, train all arms from the same foundation
with equal budgets, or explicitly disclose shared warmup reuse. **Old A1–A5
numbers are not results of these new configurations.**

## Tests and recovery

Eval handoff fixes from upstream `8665b56` are included in all five directories:
the explicit evaluation pano count overrides the training dataset cap (PanoCity
10, not silently clamped to 6); incompatible old per-sample rows are rejected
before resume; shard errors are surfaced and logs appended. The checkpoint-native
window/yaw defaults were already zero in this branch, and the training pipeline
does not run the upstream's formerly blocking preview step.

No retraining is needed for this evaluation-only fix. If an older eval used the
wrong pano count, retain that output and re-evaluate the same refined checkpoint
into a NEW output directory:

```bash
python scripts/run_ablation_full_eval.py logs/RUN/completion_refine/last.pt logs/RUN/eval_full_fixed
```

```bash
python tests/run_tests.py
python tests/smoke_ablation.py --device cuda --output-dir logs/smoke_check
```

The smoke check uses a small backbone and synthetic data while keeping 12 windows.
It performs optimizer updates, saves/reloads weights, trains main/refine completion
steps and writes ERP metrics. It is not a benchmark or a four-GPU performance test.
Historical test fixtures are under tests/fixtures, not exposed as run recipes.

Removed old b1/b2, compare-method and non-Omega pipelines, old 3h+9h launchers,
memory probes and unrelated eval launchers/configs. Historical logs/checkpoints
are untouched. Pre-change code is recoverable at `40016be` and local backup
branch `codex/ablation-pre-pro-20260929`. This is a scoped port of the selected
upstream pipeline, not a whole-branch merge.
