# Rethink Pano VGGT-Omega + LUNA

Patch Bank and GeoRA ablation configs are documented in
[`ABLATIONS.md`](ABLATIONS.md).

This experiment ports the b1-baseline LUNA augmentation to the **VGGT-Omega**
backbone (DINOv3-style ViT with K-bias mask, sparse layer caching, register
attention, bf16 autocast). The high-level idea is unchanged from the b1 work
in `Rethink_pano_new_exp/` — we keep VGGT's local pinhole reasoning and bolt on
lightweight residual adapters that:

- pano equirectangular image → virtual pinhole windows;
- per-token metadata (`global_patch_id`, spherical direction, seam flags,
  local patch ids);
- zero-initialized `LunaPatchAdapter` for global patch-bank sharing;
- zero-initialized `LunaCameraAdapter` for known virtual-camera metadata
  injection;
- optional pano-global token (sits between camera and register tokens) driven
  by per-view sin/cos(yaw, pitch) + FoV features.

## What changed vs the released VGGT-Omega

Everything is opt-in: with `enable_luna=False, enable_pano_global_token=False`
the model is numerically identical to `vggt-omega/vggt_omega/`. New knobs:

- `Aggregator(... enable_pano_global_token, pano_geom_dim, enable_luna,
  luna_patch_layers, luna_camera_layers, luna_sphere_dim,
  luna_camera_meta_dim, luna_hidden_dim)`
- `VGGTOmega.forward(images, pano_view_params=..., pano_angles=..., pano_fov=...,
  pano_token_meta=..., pano_camera_meta=...)`
- `VGGTOmega_LUNA` — pano-aware wrapper that bundles the sampler and forwards
  either `images` or `pano_images`.

### Where LUNA is injected

The omega aggregator runs `frame_block + inter_frame_block` per layer
(`depth=24`). LUNA is applied **after** each layer's full pair, just before the
sparse cache decision. This matches the b1 ordering ("LUNA after a complete
attention pair, then package the cached features") and is compatible with both
`inter_frame_attention_types` (`"global"` and `"register"`):

- For `"global"` layers, the inter-frame block has already mixed all tokens
  across frames; LUNA runs on the full `[B, S, P, C]` tensor.
- For `"register"` layers (omega indices `[2, 6, 9, 14, 20]`), patch tokens
  were sliced out for the cross-frame attention then reconstituted. LUNA runs
  on the recombined `[B, S, P, C]` tensor afterwards, so the adapter signature
  is identical.

By default LUNA inserts into the **second half** of the backbone (layers
`12..23`). To get DenseHead / CameraHead to see corrected features, the LUNA
patch layers should overlap the `cached_layer_indices`
(`[4, 11, 17, 23]` in the released omega). Layer 23 is the most important
target since it feeds CameraHead and TextAlignmentHead. The default
"second_half" policy includes 17 and 23, which is enough for an MVP.

You can also pass:

```python
VGGTOmega_LUNA(
    luna_patch_layers=[17, 23],   # only inject where cached layers will see it
    luna_camera_layers=[23],
)
```

or use the b1-style string shortcuts: `"second_half"`, `"last"`, or a
comma-separated list like `"17,23"`.

### Pano-global token

When `enable_pano_global_token=True`, an extra learnable special token is
inserted **between camera and register tokens**, so the patch start index
becomes `1 + 1 + num_register_tokens = 18` (instead of the default `17`).
Its initial state is `~N(0, 1e-3)`, and a 2-layer MLP turns the per-view
`[sin θ, cos θ, sin φ, cos φ, fov_h, fov_w]` feature into a residual added to
both the pano-global and the camera tokens. The omega `"register"` inter-frame
blocks naturally include the pano-global token in their cross-frame slice.

## Quickstart

```python
import torch
from vggt_omega.models.vggt_omega_luna import VGGTOmega_LUNA

model = VGGTOmega_LUNA(
    patch_size=16,
    embed_dim=1024,
    enable_camera=True,
    enable_depth=True,
    enable_pano_global_token=True,
    enable_luna=True,
    luna_patch_layers="second_half",   # or [17, 23]
    luna_camera_layers=[23],
)

# Pano mode
pano = torch.rand(1, 3, 1024, 2048)
preds = model(pano_images=pano)
print(preds["depth"].shape, preds["pose_enc"].shape)

# Regular multi-view mode still works
images = torch.rand(1, 4, 3, 512, 512)
preds = model(images=images)
```

## Camera Supervision For Single vs Multi Pano

For a single panorama, all sampled pinhole windows are virtual crops from the
same camera center. In that setting the absolute UE/world panorama position is
not a useful target for the VGGT camera head. The correct local single-pano
target is that every virtual camera shares the same origin, so the single-pano
config uses:

```yaml
camera_position_mode: local_zero
camera_translation_weight: 1.0
```

This trains all virtual camera translations to zero while rotation/FoV are
supervised from the known yaw/pitch/FoV window sampler. Depth remains the
metric reconstruction signal.

For multi-pano training, the dataset can return neighboring panoramas as one
sample:

```yaml
panos_per_sample: 2
pano_grouping: nearest
camera_position_mode: relative_anchor
camera_translation_weight: 1.0
```

The model accepts `[B, N, 3, H, W]` pano batches, samples each pano into pinhole
windows, and flattens them to a single `[B, N*S, 3, window, window]` VGGT view
sequence. `relative_anchor` subtracts the first pano position in the group, so
translation supervision is local to the multi-pano sample rather than tied to
the UE/global coordinate origin.

### Loading the released VGGT-Omega weights

Because all new parameters (`LunaPatchAdapter.alpha`, `LunaCameraAdapter.alpha`,
their MLPs, the pano-global token and geometry MLP) are zero-init residuals
that resolve to the identity at construction time, you can load the released
omega state-dict with `strict=False` and finetune only the LUNA bits:

```python
ckpt = torch.load("vggt_omega.pt", map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt, strict=False)
# `missing` will list the new LUNA / pano params — that's expected.
```

## Folder layout

```text
Rethink_pano_new_exp_omega/
  vggt_omega/
    __init__.py
    data/
      pano_sampler.py
      pano_token_meta.py
    models/
      aggregator.py          # extended with pano_global token + LUNA hooks
      vggt_omega.py          # forward accepts pano metadata
      vggt_omega_luna.py     # top-level wrapper with PanoWindowSampler
      luna_adapter.py        # LunaConfig dataclass
      heads/                 # CameraHead / DenseHead / TextAlignmentHead (unchanged)
      layers/
        attention.py / block.py / ...      # unchanged omega layers
        luna_patch.py        # NEW — global patch-bank residual MLP
        luna_camera.py       # NEW — known-camera residual MLP
        pano_position.py     # NEW — spherical / pinhole geometry helpers
  tests/
    test_pano_sampler_luna_omega.py
  README_LUNA_OMEGA.md
```

## Tests

A self-contained smoke test mirrors the b1 one:

```bash
cd Rethink_pano_new_exp_omega
python tests/test_pano_sampler_luna_omega.py
```

It checks:

1. `PanoWindowSampler` output shapes (`windows`, `token_meta`, `camera_meta`).
2. Both LUNA adapters are zero-init residuals (output equals input on a fresh
   init).
3. The LUNA-augmented aggregator runs end-to-end with a tiny conv patch embed
   (avoids loading the real DINOv3-style backbone) and produces the expected
   `[B, S, P, 2C]` cached outputs and `patch_token_start = 1 + 1 + R` indexing.

## Migration notes (b1 → omega)

A complete code-level comparison of the b1 vs omega backbones is in
`outputs/vggt_omega_vs_b1_网络结构对比.md`. The b1 → omega LUNA port follows
these rules:

- **Hook point**: b1 inserts LUNA after a `["frame","global"]` pair using
  `global_idx - 1` as the layer index; omega inserts after `frame_block +
  inter_frame_block` using `block_idx` directly.
- **Inter-frame `"register"` blocks** in omega slice off patch tokens; LUNA
  still runs on the full reconstructed `[B, S, P, C]` afterwards, so the
  adapter interface does not need to change.
- **Sparse caching**: omega caches only `[4, 11, 17, 23]`. LUNA layers should
  overlap these to actually affect head outputs.
- **bf16 autocast**: aggregator runs under `torch.autocast(bf16)`. LUNA's
  `alpha.to(dtype=tokens.dtype)` keeps the residual in the right dtype; the
  `LayerNorm + Linear` MLPs are autocast-safe.
- **`patch_token_start` vs `patch_start_idx`**: omega's name is kept;
  b1-compatible heads aren't migrated here because omega's
  `CameraHead`/`DenseHead`/`TextAlignmentHead` already use `patch_token_start`.

That's the entire surface area required to recover the b1 LUNA behaviour on
the omega backbone.
