"""Local subset of utils3d used by DAP.

The PyPI utils3d package pins obsolete Open3D versions, which conflicts with
modern Python environments. DAP only uses image_uv and points_to_normals, so we
provide those functions locally and avoid the external dependency.
"""

from . import numpy, torch

__all__ = ["numpy", "torch"]
