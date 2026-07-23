# Omega Multipano Patch Bank and GeoRA Ablations

This is the authoritative ablation pipeline. It uses the Omega multipano code
and inherits the complete schedule from:

`configs/multipano_rtx5000x4_mixed4_pano_low_to_high_luna_after_full_warmup_9h.yaml`

All variants share the original mixed4 proportions, 2-to-8-pano curriculum,
384-to-512 window schedule, losses, optimizer, and 4xRTX5000 runtime. The
derived configs override only the ablation switch, output paths, and the
concrete server dataset path.

## Conditions

| Run | Config | Patch GeoRA | Camera GeoRA | Patch Bank |
| --- | --- | ---: | ---: | --- |
| Full reference | `ablation_rtx5000x4_full.yaml` | on | on | aligned |
| No Patch Bank | `ablation_rtx5000x4_no_patch_bank.yaml` | on | on | zero context |
| No GeoRA | `ablation_rtx5000x4_no_geora.yaml` | off | off | absent |
| Shuffled Patch Bank | `ablation_rtx5000x4_shuffle_patch_bank.yaml` | on | on | wrong ERP lookup |

The No Patch Bank control keeps the complete Patch GeoRA MLP and input width;
only `m_pi(i)` is replaced by zeros. The shuffled control aggregates the same
bank features and then permutes occupied ERP cells separately inside each
panorama. Features are never shuffled across panorama camera centers.

## Clone

```bash
git clone --branch codex/geora-patch-bank-ablation \
  git@github.com:houguanli/RethinkPanoVGGT.git \
  RethinkPanoVGGT_omega_ablation

cd RethinkPanoVGGT_omega_ablation/Rethink_pano_new_exp_omega
```

For an existing clone:

```bash
git fetch origin codex/geora-patch-bank-ablation
git switch codex/geora-patch-bank-ablation
git pull --ff-only
cd Rethink_pano_new_exp_omega
```

## Data preflight

The committed 4xRTX5000 ablation configs use:

`/mnt/e/PanoVGGT_minimal_datasets/datasets`

Override it with `PANOVGGT_ROOT` in the wrapper or `--dataset-root` on the
training command.

```bash
PYTHONPATH=. python scripts/check_ablation_data.py \
  --config configs/ablation_rtx5000x4_full.yaml
```

## Recommended launch

The wrapper uses one shared 3-hour Omega warmup checkpoint for all conditions.
The first launch creates it; later launches reuse it.

```bash
export PYTHON=/path/to/environment/bin/python
export NPROC_PER_NODE=4
export PANOVGGT_ROOT=/mnt/e/PanoVGGT_minimal_datasets/datasets
export BASE_CHECKPOINT=/path/to/vggt_omega_1b_512.pt

bash scripts/run_ablation_4xrtx5000.sh full
bash scripts/run_ablation_4xrtx5000.sh no_patch_bank
bash scripts/run_ablation_4xrtx5000.sh no_geora
bash scripts/run_ablation_4xrtx5000.sh shuffle_patch_bank
```

If a shared warmup already exists:

```bash
export WARMUP_CHECKPOINT=/path/to/shared_warmup/last.pt
export SKIP_WARMUP=1
bash scripts/run_ablation_4xrtx5000.sh no_patch_bank
```

Useful optional overrides:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ABLATION_MAX_DURATION_MINUTES=540
export EXTRA_TRAIN_ARGS="--save-every-steps 1000"
```

## Direct torchrun

```bash
PYTHONPATH=. python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  training/train_pano_omega.py \
  --config configs/ablation_rtx5000x4_no_patch_bank.yaml \
  --dataset-root /mnt/e/PanoVGGT_minimal_datasets/datasets \
  --base-checkpoint /path/to/vggt_omega_1b_512.pt \
  --checkpoint /path/to/shared_warmup/last.pt \
  --no-inherit-checkpoint-training-defaults
```

Replace the config filename with the Full, No GeoRA, or shuffled config listed
above. Use the same warmup checkpoint and `pred_depth_scale` for every run.

## Local verification

```bash
PYTHONPATH=. python tests/test_ablation_omega.py
PYTHONPATH=. python training/train_pano_omega.py \
  --config configs/ablation_rtx5000x4_shuffle_patch_bank.yaml \
  --smoke
```
