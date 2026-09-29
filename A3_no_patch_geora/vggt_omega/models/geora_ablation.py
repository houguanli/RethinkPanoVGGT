"""Parameter-preserving GeoRA controls, reapplied after checkpoint loading."""

import torch

from .layers.luna_camera import LunaCameraAdapter
from .layers.luna_patch import LunaPatchAdapter

VARIANTS = ("full", "no_geora", "no_camera_geora", "no_patch_geora", "random_patch_bank")


def enforce_zero_frozen(model, _incompatible_keys=None):
    """Never let a warmup checkpoint or a trainability stage revive disabled branches."""
    with torch.no_grad():
        for module in model.modules():
            if getattr(module, "zero_frozen", False):
                for parameter in module.parameters():
                    parameter.zero_()
                    parameter.requires_grad_(False)
                    parameter.grad = None


def configure_geora_ablation(model, variant="full", seed=43):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown GeoRA ablation: {variant}")
    model.geora_ablation = variant
    for index, module in enumerate(model.modules()):
        if isinstance(module, LunaPatchAdapter):
            module.zero_frozen = variant in ("no_geora", "no_patch_geora")
            module.random_bank_seed = seed + index if variant == "random_patch_bank" else None
        elif isinstance(module, LunaCameraAdapter):
            module.zero_frozen = variant in ("no_geora", "no_camera_geora")
    if not getattr(model, "_geora_load_hook_registered", False):
        model.register_load_state_dict_post_hook(enforce_zero_frozen)
        model._geora_load_hook_registered = True
    enforce_zero_frozen(model)
    return model
