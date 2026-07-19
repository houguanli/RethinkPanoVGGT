#!/usr/bin/env python3
"""Join baseline warmup/LoRA CSVs and plot robust two-stage loss curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("loss_objective", "loss_reg_depth", "loss_camera", "loss_T", "loss_R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--lora", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=401)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def centered_median(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    result = np.empty_like(arr)
    half = max(int(window), 1) // 2
    for index in range(len(arr)):
        start = max(0, index - half)
        end = min(len(arr), index + half + 1)
        result[index] = np.median(arr[start:end])
    return result


def append_segment(
    merged: list[dict[str, str]],
    rows: list[dict[str, str]],
    phase: str,
    substage: str,
    offset_seconds: float,
) -> float:
    elapsed = [finite_float(row.get("elapsed_seconds")) for row in rows]
    valid_elapsed = [value for value in elapsed if value is not None]
    if not valid_elapsed:
        raise ValueError(f"No finite elapsed_seconds in {substage}")
    origin = valid_elapsed[0]
    last_relative = 0.0
    for row, value in zip(rows, elapsed):
        if value is None:
            continue
        last_relative = max(last_relative, value - origin)
        output = dict(row)
        output["phase"] = phase
        output["substage"] = substage
        output["timeline_seconds"] = f"{offset_seconds + value - origin:.9f}"
        output["timeline_hours"] = f"{(offset_seconds + value - origin) / 3600.0:.9f}"
        merged.append(output)
    return offset_seconds + last_relative


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "early_mean": None, "recent_mean": None}
    arr = np.asarray(values, dtype=np.float64)
    edge = max(1, len(arr) // 5)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "early_mean": float(np.mean(arr[:edge])),
        "early_median": float(np.median(arr[:edge])),
        "recent_mean": float(np.mean(arr[-edge:])),
        "recent_median": float(np.median(arr[-edge:])),
    }


def metric_values(rows: list[dict[str, str]], metric: str, camera_valid_only: bool = False) -> list[float]:
    values: list[float] = []
    for row in rows:
        if camera_valid_only and (finite_float(row.get("camera_valid_fraction")) or 0.0) <= 0.0:
            continue
        value = finite_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def plot_primary(rows: list[dict[str, str]], boundary: float, internal: list[float], out: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 6), dpi=150)
    styles = {
        "warmup": ("#1f77b4", "#ff7f0e"),
        "lora": ("#2ca02c", "#d62728"),
    }
    for phase in ("warmup", "lora"):
        selected = [row for row in rows if row["phase"] == phase]
        x = [float(row["timeline_hours"]) for row in selected]
        y = [finite_float(row.get("loss_reg_depth")) for row in selected]
        pairs = [(xv, yv) for xv, yv in zip(x, y) if yv is not None]
        x = [pair[0] for pair in pairs]
        y = [float(pair[1]) for pair in pairs]
        raw_color, smooth_color = styles[phase]
        axis.plot(x, y, color=raw_color, linewidth=0.55, alpha=0.12, label=f"{phase} raw")
        axis.plot(x, centered_median(y, window), color=smooth_color, linewidth=2.2, label=f"{phase} rolling_median{window}")
    axis.axvline(boundary, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8)
    axis.text(boundary, 0.98, " LoRA starts", transform=axis.get_xaxis_transform(), va="top", fontsize=9)
    for position in internal:
        axis.axvline(position, color="#777777", linestyle=":", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("elapsed hours")
    axis.set_ylabel("loss_reg_depth")
    axis.set_title("VGGT-Omega baseline: full warmup + LoRA robust smoothed depth loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_components(rows: list[dict[str, str]], boundary: float, out: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=150, sharex=True)
    specs = (
        ("loss_objective", False, "Omega objective (includes confidence term)"),
        ("loss_reg_depth", False, "Scale-aligned depth regression"),
        ("loss_camera", True, "Camera loss (camera-valid samples only)"),
    )
    colors = {"warmup": "#ff7f0e", "lora": "#d62728"}
    for axis, (metric, camera_valid_only, title) in zip(axes, specs):
        for phase in ("warmup", "lora"):
            selected = [row for row in rows if row["phase"] == phase]
            pairs: list[tuple[float, float]] = []
            for row in selected:
                if camera_valid_only and (finite_float(row.get("camera_valid_fraction")) or 0.0) <= 0.0:
                    continue
                value = finite_float(row.get(metric))
                if value is not None:
                    pairs.append((float(row["timeline_hours"]), value))
            if not pairs:
                continue
            x = [pair[0] for pair in pairs]
            y = [pair[1] for pair in pairs]
            axis.plot(x, y, linewidth=0.5, alpha=0.10, color=colors[phase])
            axis.plot(x, centered_median(y, window), linewidth=2.0, color=colors[phase], label=phase)
        axis.axvline(boundary, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_ylabel(metric)
        axis.set_title(title, fontsize=10)
        axis.grid(True, alpha=0.22)
        axis.legend()
    axes[-1].set_xlabel("elapsed hours")
    fig.suptitle("VGGT-Omega baseline two-stage training losses", fontsize=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, str]] = []
    fieldnames: list[str] = []

    warmup_fields, warmup_rows = read_csv(args.warmup)
    fieldnames.extend(warmup_fields)
    offset = append_segment(merged, warmup_rows, "warmup", args.warmup.parent.name, 0.0)
    phase_boundary_hours = offset / 3600.0

    internal_boundaries: list[float] = []
    for index, path in enumerate(args.lora):
        lora_fields, lora_rows = read_csv(path)
        for field in lora_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        if index > 0:
            internal_boundaries.append(offset / 3600.0)
        offset = append_segment(merged, lora_rows, "lora", path.parent.name, offset)

    output_fields = ["phase", "substage", "timeline_seconds", "timeline_hours", *fieldnames]
    with (args.out_dir / "loss_two_stage_merged.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    summary: dict[str, object] = {
        "warmup_csv": str(args.warmup),
        "lora_csvs": [str(path) for path in args.lora],
        "phase_boundary_hours": phase_boundary_hours,
        "total_elapsed_hours": offset / 3600.0,
        "rolling_window": int(args.rolling_window),
        "phases": {},
        "substages": {},
    }
    for phase in ("warmup", "lora"):
        selected = [row for row in merged if row["phase"] == phase]
        summary["phases"][phase] = {
            metric: summarize(metric_values(selected, metric, camera_valid_only=metric in ("loss_camera", "loss_T", "loss_R")))
            for metric in METRICS
        }
    for substage in dict.fromkeys(row["substage"] for row in merged):
        selected = [row for row in merged if row["substage"] == substage]
        summary["substages"][substage] = {
            metric: summarize(metric_values(selected, metric, camera_valid_only=metric in ("loss_camera", "loss_T", "loss_R")))
            for metric in METRICS
        }
    (args.out_dir / "loss_two_stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plot_primary(
        merged,
        phase_boundary_hours,
        internal_boundaries,
        args.out_dir / "loss_curve_smoothed_robust.png",
        args.rolling_window,
    )
    plot_components(
        merged,
        phase_boundary_hours,
        args.out_dir / "loss_components_smoothed_robust.png",
        args.rolling_window,
    )
    print(f"[INFO] rows={len(merged)} total_hours={offset / 3600.0:.3f}")
    print(f"[INFO] report={args.out_dir}")


if __name__ == "__main__":
    main()
