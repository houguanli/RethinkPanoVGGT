# VGGT Pano Baseline B1

This baseline keeps the original VGGT architecture and adds panorama-specific
training supervision.

## Added components

- `training/pano_loss.py`: same-center translation consistency, absolute
  quaternion supervision, and relative rotation supervision.
- `training/loss.py`: optional `pano:` loss block in `MultitaskLoss`.
- `vggt/utils/lora.py`: optional LoRA adapters for selected `nn.Linear`
  modules.
- `training/trainer.py`: applies LoRA when `model.lora.enabled: true`.

## Expected pano batch keys

- `is_pano` or `pano_is_pano`: bool or batch bool mask.
- `pano_rotations`: optional `[B, S, 3, 3]` ground-truth crop rotations.
  If absent, rotations are derived from `extrinsics`.
- `pano_pairs`: optional pair list for relative rotation loss.
- `pano_valid_mask`: optional `[B, S]` valid view mask.

The default config leaves LoRA disabled and enables pano loss only when pano
supervision keys are present.
