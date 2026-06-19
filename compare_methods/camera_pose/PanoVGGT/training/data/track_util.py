"""Track helpers for PanoVGGT training.

The compare-method PanoCity configs keep ``load_track`` disabled. This module
exists so the shared composed dataset can import cleanly; enabling tracks still
requires a real track generator.
"""


def build_tracks_by_depth_pano(*args, **kwargs):
    raise NotImplementedError(
        "PanoCity compare-method configs set load_track=False. "
        "A real track generator is required before enabling load_track."
    )
