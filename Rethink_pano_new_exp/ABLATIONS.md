# Patch Bank and GeoRA Ablations

All experiments inherit the data, sampler, optimization, loss, and model-head
settings from `training/config/pano_luna.yaml`. The aligned full-model reference
is therefore:

```bash
cd training
torchrun --nproc_per_node=1 launch.py --config pano_luna
```

The three requested ablations are:

| Condition | Config | Patch GeoRA | Camera GeoRA | Patch Bank lookup |
| --- | --- | ---: | ---: | --- |
| No Patch Bank | `pano_luna_ablate_patch_bank` | on | on | zero context |
| No GeoRA | `pano_luna_ablate_geora` | off | off | not used |
| Shuffled Patch Bank | `pano_luna_ablate_patch_bank_shuffle` | on | on | wrong ERP correspondence |

Run them with:

```bash
cd training
torchrun --nproc_per_node=1 launch.py --config pano_luna_ablate_patch_bank
torchrun --nproc_per_node=1 launch.py --config pano_luna_ablate_geora
torchrun --nproc_per_node=1 launch.py --config pano_luna_ablate_patch_bank_shuffle
```

## Fairness guarantees

- `luna_patch_bank_mode: none` keeps every Patch GeoRA parameter and concatenated
  input dimension unchanged. It substitutes an all-zero tensor for
  `m_pi(i)`, so the adapter still receives the local token and spherical
  position encoding.
- `luna_patch_bank_mode: shuffled` first computes the same occupied Patch Bank
  cells as the full model, then permutes those cells within each panorama. It
  preserves the bank feature multiset while breaking the ERP-aligned lookup.
- `enable_luna: false` removes both Patch and Camera GeoRA adapters. It does not
  change the split-window sampler or the pano-global-token condition.

`luna_patch_bank_shuffle_seed` makes the wrong correspondence reproducible.
Each insertion layer receives a distinct seed offset so different GeoRA layers
do not all use the same permutation.

## Smoke tests

From `Rethink_pano_new_exp`:

```bash
PYTHONPATH=.:training python tests/test_ablation_configs.py
PYTHONPATH=.:training python tests/test_patch_bank_ablation.py
PYTHONPATH=.:training python tests/test_pano_sampler_luna.py
```
