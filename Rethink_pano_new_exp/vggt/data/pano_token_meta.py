from dataclasses import dataclass
from typing import Tuple


@dataclass
class PanoTokenMeta:
    pano_id: int
    window_id: int
    local_patch_x: int
    local_patch_y: int
    pano_u: float
    pano_v: float
    theta: float
    phi: float
    sphere_dir: Tuple[float, float, float]
    global_patch_id: int
    is_seam_region: bool = False
