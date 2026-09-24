"""Fail-closed provenance checks for the foundation-started belt60 comparison."""
from pathlib import Path
import math


def interrupt_on_term(signum, frame):
    raise KeyboardInterrupt("Training received SIGTERM; preserving interrupted state")

GEOMETRY = {"A": ("-15", 4), "B": ("-25,25", 6)}
SAMPLER_KEYS = ("pitch_degrees", "num_yaw", "window_size", "patch_size",
                "fov_degrees", "fov_x_degrees", "fov_y_degrees")


def validate_geometry(values, arm):
    pitch, yaw = GEOMETRY[arm]
    expected = dict(pitch_degrees=pitch, num_yaw=yaw, window_size=384, patch_size=16,
                    fov_degrees=75, fov_x_degrees=75, fov_y_degrees=75)
    for key, value in expected.items():
        actual = values.get(key)
        if key == "pitch_degrees":
            actual = tuple(float(x) for x in str(actual).split(","))
            value = tuple(float(x) for x in value.split(","))
        if actual != value:
            raise ValueError(f"Matched warm-up {arm}: {key}={actual}, required {value}")


def validate_start(args, payload):
    arm = getattr(args, "matched_warmup_arm", None)
    if arm is None:
        return
    if args.inherit_checkpoint_training_defaults:
        raise ValueError("Matched warm-up cannot inherit checkpoint training defaults")
    if args.base_checkpoint is None or args.checkpoint is None or Path(args.base_checkpoint).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("Matched warm-up must start directly from the same foundation/base checkpoint")
    if any(key in payload for key in ("model_delta", "step", "args", "completion_head")):
        raise ValueError("Matched warm-up requires an unadapted foundation, not a prior training checkpoint")
    validate_geometry(vars(args), arm)
    for stage in args.training_stages or [{}]:
        values = {**vars(args), **stage}
        validate_geometry(values, arm)
        for key, expected in (("pano_min_count", 2), ("pano_max_count", 2), ("trainable", "all")):
            if values[key] != expected:
                raise ValueError(f"Matched warm-up {arm} requires {key}={expected}")


def validate_teacher(args, payload, arm):
    validate_geometry(vars(args), arm)
    parent_args = payload.get("args", {})
    if parent_args.get("matched_warmup_arm") != arm:
        raise ValueError(f"Arm {arm} needs its own matched foundation warm-up; legacy/cross-arm teacher rejected")
    validate_geometry(parent_args, arm)
    status = payload.get("training_status", {})
    if status.get("state") != "completed" or status.get("stop_reason") != "max_duration" or status.get("elapsed_seconds", 0) < 7200:
        raise ValueError("Formal completion requires a completed 120-minute matched warm-up")


def input_metadata(predictions, pano_images, args):
    """Inspect actual sampled model input, including rays, after the first forward."""
    windows = predictions["pano_windows"]
    meta = predictions["pano_camera_meta"]
    values = {key: sorted({round(float(v) * 180 / math.pi, 4) for v in meta[key].detach().cpu().flatten()})
              for key in ("pitch", "yaw", "fov_x", "fov_y")}
    panos = int(pano_images.shape[1]) if pano_images.ndim == 5 else 1
    result = {"pano_count": panos, "window_shape": list(windows.shape), "degrees": values}
    arm = getattr(args, "matched_warmup_arm", None)
    if arm:
        pitch, yaw = GEOMETRY[arm]
        pitches = [float(x) for x in pitch.split(",")]
        assert panos == 2 and windows.shape[1] == panos * yaw * len(pitches), result
        assert list(windows.shape[-2:]) == [384, 384], result
        assert values["pitch"] == pitches and values["fov_x"] == [75.0] and values["fov_y"] == [75.0], result
    return result
