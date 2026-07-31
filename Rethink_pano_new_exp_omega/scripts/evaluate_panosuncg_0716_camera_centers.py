#!/usr/bin/env python3
"""Evaluate 0716 full-pipeline camera-center trajectories on PanoSUNCG."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_ROOT = PROJECT_ROOT.parent
for root in (PROJECT_ROOT, GIT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.evaluate_depth_checkpoint import (  # noqa: E402
    apply_checkpoint_eval_defaults,
    build_eval_model,
    load_checkpoint_payload,
    normalize_args_for_eval,
)
from scripts.evaluate_panosuncg_zeroshot import (  # noqa: E402
    atomic_write_json,
    resolve_dataset,
)
from training.train_pano_omega import parse_args as parse_training_args  # noqa: E402


TRAJECTORY_FIELDS = (
    "scene",
    "trajectory",
    "num_available_frames",
    "num_evaluated_frames",
    "selected_indices",
    "pair_count",
    "sim3_scale",
    "ate_rmse",
    "ate_mean",
    "ate_median",
    "ate_normalized_rmse",
    "direction_deg_mean",
    "direction_deg_median",
    "direction_auc3",
    "direction_auc5",
    "direction_auc15",
    "direction_auc30",
    "relative_length_error_mean",
    "relative_length_error_median",
    "inference_seconds",
    "trajectory_seconds",
)
FRAME_FIELDS = (
    "scene",
    "trajectory",
    "frame_index",
    "gt_x",
    "gt_y",
    "gt_z",
    "pred_aligned_x",
    "pred_aligned_y",
    "pred_aligned_z",
    "ate",
)
PAIR_FIELDS = (
    "scene",
    "trajectory",
    "frame_i",
    "frame_j",
    "gt_baseline",
    "pred_aligned_baseline",
    "direction_error_deg",
    "relative_length_error",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="datasets root, PanoSUNCG_zeroshot root, or PanoSUNCG/rotated root",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-trajectory", type=int, default=5)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def read_split(split_file: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for line_number, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{split_file}:{line_number}: expected RGB and depth paths")
        result.append((fields[0], fields[1]))
    return result


def build_trajectories(split_file: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rgb_rel, _ in read_split(split_file):
        parts = Path(rgb_rel).parts
        if len(parts) != 3:
            raise ValueError(f"Unexpected PanoSUNCG path: {rgb_rel}")
        scene, trajectory, filename = parts
        groups[(scene, trajectory)].append(int(filename.split("_", 1)[0]))
    return [
        {
            "scene": scene,
            "trajectory": trajectory,
            "indices": sorted(set(indices)),
        }
        for (scene, trajectory), indices in sorted(groups.items())
    ]


def evenly_spaced_indices(indices: Sequence[int], count: int) -> list[int]:
    if len(indices) <= count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=count)
    return list(dict.fromkeys(indices[int(round(position))] for position in positions))


def load_gt_centers(labels_path: Path, indices: Sequence[int]) -> np.ndarray:
    centers = []
    for line_number, raw in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if len(fields) != 3:
            raise ValueError(f"{labels_path}:{line_number}: expected XYZ")
        centers.append([float(value) for value in fields])
    array = np.asarray(centers, dtype=np.float64)
    if max(indices) >= len(array):
        raise IndexError(f"Frame index exceeds labels in {labels_path}")
    return array[np.asarray(indices, dtype=np.int64)]


def load_images(
    rotated_root: Path,
    scene: str,
    trajectory: str,
    indices: Sequence[int],
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    images = []
    for index in indices:
        path = rotated_root / scene / trajectory / f"{index}_color.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        interpolation = cv2.INTER_AREA if bgr.shape[0] >= height else cv2.INTER_LINEAR
        rgb = cv2.resize(bgr, (width, height), interpolation=interpolation)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
    return torch.stack(images).unsqueeze(0).to(device, non_blocking=True)


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    train_args = parse_training_args(["--config", str(args.config)])
    train_args.device = args.device
    train_args.distributed = "none"
    train_args.batch_size = 1
    train_args.num_workers = 0
    train_args.checkpoint = args.checkpoint
    normalize_args_for_eval(train_args)
    payload = load_checkpoint_payload(args.checkpoint)
    apply_checkpoint_eval_defaults(train_args, payload)
    model = build_eval_model(
        train_args,
        args.checkpoint,
        payload,
        torch.device(args.device),
    )
    model.eval()
    return model, payload, train_args


def infer_camera_centers(
    model: torch.nn.Module,
    images: torch.Tensor,
    amp_dtype: str,
) -> tuple[np.ndarray, float]:
    if images.device.type == "cuda" and amp_dtype != "none":
        dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        amp_context = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        amp_context = contextlib.nullcontext()
    if images.device.type == "cuda":
        torch.cuda.synchronize(images.device)
    started = time.perf_counter()
    with torch.inference_mode(), amp_context:
        predictions = model(
            pano_images=images,
            return_sampler_output=False,
            return_window_pose=False,
        )
        centers = predictions["pano_camera_center"][0].float()
    if images.device.type == "cuda":
        torch.cuda.synchronize(images.device)
    return centers.cpu().numpy().astype(np.float64), time.perf_counter() - started


def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance < 1e-12:
        raise ValueError("Predicted camera centers are degenerate")
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def direction_auc(errors: np.ndarray, threshold: int) -> float:
    hist, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(hist.astype(np.float64) / max(len(errors), 1))))


def evaluate_centers(
    predicted: np.ndarray,
    target: np.ndarray,
    selected_indices: Sequence[int],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    scale, rotation, translation = umeyama_sim3(predicted, target)
    aligned = (scale * (rotation @ predicted.T)).T + translation
    frame_errors = np.linalg.norm(aligned - target, axis=1)
    target_centered = target - target.mean(axis=0)
    trajectory_scale = float(np.sqrt(np.mean(np.sum(target_centered**2, axis=1))))

    frame_rows = [
        {
            "frame_index": int(frame_index),
            "gt_x": float(target[local_index, 0]),
            "gt_y": float(target[local_index, 1]),
            "gt_z": float(target[local_index, 2]),
            "pred_aligned_x": float(aligned[local_index, 0]),
            "pred_aligned_y": float(aligned[local_index, 1]),
            "pred_aligned_z": float(aligned[local_index, 2]),
            "ate": float(frame_errors[local_index]),
        }
        for local_index, frame_index in enumerate(selected_indices)
    ]

    pair_rows: list[dict[str, Any]] = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            gt_delta = target[j] - target[i]
            pred_delta = aligned[j] - aligned[i]
            gt_norm = float(np.linalg.norm(gt_delta))
            pred_norm = float(np.linalg.norm(pred_delta))
            if gt_norm < 1e-8 or pred_norm < 1e-8:
                continue
            cosine = float(
                np.clip(np.dot(gt_delta, pred_delta) / (gt_norm * pred_norm), -1.0, 1.0)
            )
            pair_rows.append(
                {
                    "frame_i": int(selected_indices[i]),
                    "frame_j": int(selected_indices[j]),
                    "gt_baseline": gt_norm,
                    "pred_aligned_baseline": pred_norm,
                    "direction_error_deg": float(np.degrees(np.arccos(cosine))),
                    "relative_length_error": float(abs(pred_norm - gt_norm) / gt_norm),
                }
            )
    if not pair_rows:
        raise ValueError("No valid camera-center pairs")
    directions = np.asarray([row["direction_error_deg"] for row in pair_rows])
    lengths = np.asarray([row["relative_length_error"] for row in pair_rows])
    metrics = {
        "pair_count": float(len(pair_rows)),
        "sim3_scale": scale,
        "ate_rmse": float(np.sqrt(np.mean(frame_errors**2))),
        "ate_mean": float(np.mean(frame_errors)),
        "ate_median": float(np.median(frame_errors)),
        "ate_normalized_rmse": float(
            np.sqrt(np.mean(frame_errors**2)) / max(trajectory_scale, 1e-12)
        ),
        "direction_deg_mean": float(np.mean(directions)),
        "direction_deg_median": float(np.median(directions)),
        "direction_auc3": direction_auc(directions, 3),
        "direction_auc5": direction_auc(directions, 5),
        "direction_auc15": direction_auc(directions, 15),
        "direction_auc30": direction_auc(directions, 30),
        "relative_length_error_mean": float(np.mean(lengths)),
        "relative_length_error_median": float(np.median(lengths)),
    }
    return metrics, frame_rows, pair_rows


def append_rows(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    rows = list(rows)
    if not rows:
        return
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path, resume: bool) -> list[dict[str, str]]:
    if not resume or not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def summarize(
    trajectory_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frame_errors = np.asarray([float(row["ate"]) for row in frame_rows], dtype=np.float64)
    directions = np.asarray(
        [float(row["direction_error_deg"]) for row in pair_rows], dtype=np.float64
    )
    lengths = np.asarray(
        [float(row["relative_length_error"]) for row in pair_rows], dtype=np.float64
    )
    macro_keys = (
        "ate_rmse",
        "ate_mean",
        "ate_median",
        "ate_normalized_rmse",
        "direction_deg_mean",
        "direction_deg_median",
        "direction_auc3",
        "direction_auc5",
        "direction_auc15",
        "direction_auc30",
        "relative_length_error_mean",
        "relative_length_error_median",
    )
    macro = {key: finite_mean(trajectory_rows, key) for key in macro_keys}
    micro = {
        "ate_rmse": float(np.sqrt(np.mean(frame_errors**2))),
        "ate_mean": float(np.mean(frame_errors)),
        "ate_median": float(np.median(frame_errors)),
        "direction_deg_mean": float(np.mean(directions)),
        "direction_deg_median": float(np.median(directions)),
        "direction_auc3": direction_auc(directions, 3),
        "direction_auc5": direction_auc(directions, 5),
        "direction_auc15": direction_auc(directions, 15),
        "direction_auc30": direction_auc(directions, 30),
        "relative_length_error_mean": float(np.mean(lengths)),
        "relative_length_error_median": float(np.median(lengths)),
    }
    return {
        "trajectory_count": len(trajectory_rows),
        "frame_count": len(frame_rows),
        "pair_count": len(pair_rows),
        "macro_by_trajectory": macro,
        "micro": micro,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.frames_per_trajectory < 3:
        raise ValueError("--frames-per-trajectory must be at least 3")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    rotated_root, split_file = resolve_dataset(args.dataset_root)
    labels_root = rotated_root.parent / "labels"
    if not labels_root.is_dir():
        raise FileNotFoundError(labels_root)
    trajectories = build_trajectories(split_file)
    if args.max_trajectories > 0:
        trajectories = trajectories[: args.max_trajectories]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_dir / ".eval.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another evaluation is writing {args.output_dir}") from exc

    trajectory_csv = args.output_dir / "trajectory_metrics.csv"
    frame_csv = args.output_dir / "frame_metrics.csv"
    pair_csv = args.output_dir / "pair_metrics.csv"
    trajectory_rows = read_rows(trajectory_csv, args.resume)
    frame_rows = read_rows(frame_csv, args.resume)
    pair_rows = read_rows(pair_csv, args.resume)
    completed = {(row["scene"], row["trajectory"]) for row in trajectory_rows}

    model, payload, train_args = load_model(args)
    native_sampler = {
        "window_size": int(train_args.window_size),
        "num_yaw": int(train_args.num_yaw),
        "pitch_degrees": str(train_args.pitch_degrees),
        "fov_degrees": float(train_args.fov_degrees),
    }
    run_config = {
        "dataset": "PanoSUNCG",
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "method": "0716 full pipeline",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_format": payload.get("checkpoint_format"),
        "checkpoint_step": payload.get("step"),
        "config": str(args.config.resolve()),
        "native_sampler": native_sampler,
        "rotated_root": str(rotated_root),
        "labels_root": str(labels_root),
        "trajectory_count": len(trajectories),
        "frames_per_trajectory": args.frames_per_trajectory,
        "frame_selection": "deterministic evenly spaced, including endpoints",
        "input_height": args.input_height,
        "input_width": args.input_width,
        "alignment": "per-trajectory Umeyama Sim(3), predicted centers to GT centers",
        "rotation_ground_truth_available": False,
        "device": args.device,
        "amp_dtype": args.amp_dtype,
    }
    atomic_write_json(args.output_dir / "run_config.json", run_config)
    print(
        f"[camera] trajectories={len(trajectories)} resumed={len(completed)}",
        flush=True,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True

    started = time.time()
    session_count = 0
    for item in trajectories:
        scene = item["scene"]
        trajectory = item["trajectory"]
        if (scene, trajectory) in completed:
            continue
        trajectory_started = time.perf_counter()
        try:
            selected = evenly_spaced_indices(item["indices"], args.frames_per_trajectory)
            target = load_gt_centers(labels_root / scene / trajectory, selected)
            images = load_images(
                rotated_root,
                scene,
                trajectory,
                selected,
                args.input_height,
                args.input_width,
                args.device,
            )
            predicted, inference_seconds = infer_camera_centers(
                model, images, args.amp_dtype
            )
            metrics, new_frame_rows, new_pair_rows = evaluate_centers(
                predicted, target, selected
            )
            trajectory_row = {
                "scene": scene,
                "trajectory": trajectory,
                "num_available_frames": len(item["indices"]),
                "num_evaluated_frames": len(selected),
                "selected_indices": ",".join(str(index) for index in selected),
                **metrics,
                "pair_count": int(metrics["pair_count"]),
                "inference_seconds": inference_seconds,
                "trajectory_seconds": time.perf_counter() - trajectory_started,
            }
            for row in new_frame_rows:
                row.update({"scene": scene, "trajectory": trajectory})
            for row in new_pair_rows:
                row.update({"scene": scene, "trajectory": trajectory})
            append_rows(trajectory_csv, TRAJECTORY_FIELDS, [trajectory_row])
            append_rows(frame_csv, FRAME_FIELDS, new_frame_rows)
            append_rows(pair_csv, PAIR_FIELDS, new_pair_rows)
            trajectory_rows.append({key: str(value) for key, value in trajectory_row.items()})
            frame_rows.extend({key: str(value) for key, value in row.items()} for row in new_frame_rows)
            pair_rows.extend({key: str(value) for key, value in row.items()} for row in new_pair_rows)
            completed.add((scene, trajectory))
            session_count += 1
            if (
                session_count == 1
                or session_count % max(args.progress_every, 1) == 0
                or len(completed) == len(trajectories)
            ):
                current = summarize(trajectory_rows, frame_rows, pair_rows)
                elapsed = time.time() - started
                rate = session_count / max(elapsed, 1e-9)
                remaining = len(trajectories) - len(completed)
                atomic_write_json(
                    args.output_dir / "progress.json",
                    {
                        "status": "completed" if remaining == 0 else "running",
                        "completed_trajectories": len(completed),
                        "total_trajectories": len(trajectories),
                        "eta_seconds": remaining / rate if rate > 0 else None,
                        "running_summary": current,
                        "last_trajectory": f"{scene}/{trajectory}",
                    },
                )
                print(
                    f"[camera] {len(completed)}/{len(trajectories)} "
                    f"ATE={current['micro']['ate_rmse']:.4f} "
                    f"T={current['micro']['direction_deg_mean']:.3f}deg",
                    flush=True,
                )
        except Exception as exc:
            with (args.output_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "scene": scene,
                            "trajectory": trajectory,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if not args.continue_on_error:
                raise
            print(f"[camera][error] {scene}/{trajectory}: {exc}", flush=True)

    result = {
        "dataset": "PanoSUNCG",
        "method": "0716 full pipeline",
        "protocol": "camera-center trajectory estimate on DA2 complete trajectories",
        "limitations": {
            "rotation_metrics": "not available: distributed labels contain XYZ centers only",
            "alignment": "Sim(3) removes global translation, rotation, and scale gauge",
        },
        **summarize(trajectory_rows, frame_rows, pair_rows),
        "run_config": run_config,
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(args.output_dir / "camera_center_summary.json", result)
    atomic_write_json(
        args.output_dir / "progress.json",
        {
            "status": "completed",
            "completed_trajectories": len(completed),
            "total_trajectories": len(trajectories),
            "running_summary": result,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
