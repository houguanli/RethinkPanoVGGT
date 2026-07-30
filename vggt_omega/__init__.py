# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT-Omega inference package (LUNA experimental extension)."""

from .models import LunaConfig, VGGTOmega, VGGTOmega_LUNA, default_luna_config

__version__ = "0.0.1+luna"

__all__ = [
    "LunaConfig",
    "VGGTOmega",
    "VGGTOmega_LUNA",
    "default_luna_config",
    "__version__",
]
