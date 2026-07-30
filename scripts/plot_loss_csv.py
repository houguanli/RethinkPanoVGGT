#!/usr/bin/env python3
"""Plot one or more training loss.csv files with robust smoothing."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", type=Path, help="loss.csv files to plot.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Named run to plot. Can be repeated. Example: --run luna=outputs/run/loss.csv. "
            "Use LABEL=PATH@HOURS to offset elapsed-time x values."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("loss_csv_plot.png"))
    parser.add_argument("--metric", default="loss", help="CSV column to plot, or auto for loss/loss_objective.")
    parser.add_argument("--x", choices=["step", "elapsed_hours", "elapsed_minutes"], default="step")
    parser.add_argument("--smooth", type=float, default=0.95, help="EMA smoothing factor in [0, 1).")
    parser.add_argument(
        "--smooth-method",
        choices=["ema", "rolling_mean", "rolling_median", "none"],
        default="ema",
    )
    parser.add_argument("--rolling-window", type=int, default=401, help="Odd/even window size for rolling smoothing.")
    parser.add_argument(
        "--resample-seconds",
        type=float,
        default=0.0,
        help="Resample elapsed-time series into fixed-second bins using median y and mean x.",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.14,
        help="Alpha for raw points. Use 0 to hide raw values.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.995,
        help="Clip y-axis to this quantile for readability. Use 1.0 to disable.",
    )
    parser.add_argument("--title", default=None)
    return parser


def parse_runs(args: argparse.Namespace) -> list[tuple[str, Path, float]]:
    runs: list[tuple[str, Path, float]] = []
    for raw in args.run:
        if "=" not in raw:
            raise ValueError(f"--run must be LABEL=PATH, got: {raw}")
        label, path = raw.split("=", 1)
        path_text, offset_hours = parse_path_offset(path)
        runs.append((label.strip(), Path(path_text), offset_hours))
    for path in args.csv:
        runs.append((path.parent.name or path.stem, path, 0.0))
    if not runs:
        raise ValueError("Provide at least one loss.csv path or --run LABEL=PATH.")
    return runs


def parse_path_offset(raw: str) -> tuple[str, float]:
    if "@" not in raw:
        return raw, 0.0
    path, offset = raw.rsplit("@", 1)
    offset = offset.strip().lower().removesuffix("h").removesuffix("hr").removesuffix("hours")
    return path, float(offset)


def read_series(path: Path, metric: str, x_axis: str, offset_hours: float = 0.0) -> tuple[list[float], list[float], str]:
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        metric_key = resolve_metric(reader.fieldnames, metric)
        required = {"step", metric_key}
        if x_axis != "step":
            required.add("elapsed_seconds")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        for row in reader:
            try:
                y = float(row[metric_key])
                if x_axis == "step":
                    x = float(row["step"])
                elif x_axis == "elapsed_hours":
                    x = float(row["elapsed_seconds"]) / 3600.0 + offset_hours
                else:
                    x = float(row["elapsed_seconds"]) / 60.0 + offset_hours * 60.0
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                x_values.append(x)
                y_values.append(y)
    if not x_values:
        raise ValueError(f"No finite values for {metric!r} in {path}")
    return x_values, y_values, metric_key


def resolve_metric(fieldnames: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    for candidate in ("loss", "loss_objective", "total_loss", "loss_depth", "loss_reg_depth"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"Could not auto-detect loss column from {fieldnames}")


def ema(values: Iterable[float], alpha: float) -> list[float]:
    smoothed: list[float] = []
    current = None
    for value in values:
        current = value if current is None else alpha * current + (1.0 - alpha) * value
        smoothed.append(current)
    return smoothed


def rolling(values: list[float], window: int, reducer: str) -> list[float]:
    if not values:
        return []
    window = max(1, int(window))
    half = window // 2
    arr = np.asarray(values, dtype=np.float64)
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        segment = arr[start:end]
        if reducer == "median":
            result.append(float(np.median(segment)))
        else:
            result.append(float(np.mean(segment)))
    return result


def smooth_values(values: list[float], method: str, alpha: float, window: int) -> list[float]:
    if method == "none":
        return list(values)
    if method == "ema":
        return ema(values, alpha)
    if method == "rolling_median":
        return rolling(values, window, "median")
    if method == "rolling_mean":
        return rolling(values, window, "mean")
    raise ValueError(f"Unknown smoothing method: {method}")


def resample_series(x_values: list[float], y_values: list[float], x_axis: str, seconds: float) -> tuple[list[float], list[float]]:
    if seconds <= 0 or x_axis == "step":
        return x_values, y_values
    factor = 3600.0 if x_axis == "elapsed_hours" else 60.0
    bin_width = float(seconds) / factor
    if bin_width <= 0:
        return x_values, y_values
    bins: dict[int, list[tuple[float, float]]] = {}
    origin = x_values[0]
    for x, y in zip(x_values, y_values):
        key = int(math.floor((x - origin) / bin_width))
        bins.setdefault(key, []).append((x, y))
    new_x: list[float] = []
    new_y: list[float] = []
    for key in sorted(bins):
        pairs = bins[key]
        new_x.append(float(np.mean([item[0] for item in pairs])))
        new_y.append(float(np.median([item[1] for item in pairs])))
    return new_x, new_y


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def main() -> None:
    args = build_parser().parse_args()
    runs = parse_runs(args)
    if not 0.0 <= args.smooth < 1.0:
        raise ValueError("--smooth must be in [0, 1).")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    all_raw_values: list[float] = []
    plt.figure(figsize=(11, 6), dpi=150)
    metric_labels: set[str] = set()
    for label, path, offset_hours in runs:
        x_values, y_values, metric_key = read_series(path, args.metric, args.x, offset_hours=offset_hours)
        x_values, y_values = resample_series(x_values, y_values, args.x, args.resample_seconds)
        metric_labels.add(metric_key)
        all_raw_values.extend(y_values)
        if args.raw_alpha > 0:
            plt.plot(x_values, y_values, linewidth=0.6, alpha=args.raw_alpha, label=f"{label} raw")
        smoothed = smooth_values(y_values, args.smooth_method, args.smooth, args.rolling_window)
        smooth_label = args.smooth_method
        if args.smooth_method == "ema":
            smooth_label = f"ema{args.smooth:g}"
        elif args.smooth_method.startswith("rolling"):
            smooth_label = f"{args.smooth_method}{args.rolling_window}"
        plt.plot(x_values, smoothed, linewidth=2.0, label=f"{label} {smooth_label}")

    if 0.0 < args.clip_quantile < 1.0:
        y_max = quantile(all_raw_values, args.clip_quantile)
        if y_max > 0:
            plt.ylim(bottom=0.0, top=y_max * 1.05)
    xlabel = {"step": "step", "elapsed_hours": "elapsed hours", "elapsed_minutes": "elapsed minutes"}[args.x]
    plt.xlabel(xlabel)
    ylabel = args.metric if args.metric != "auto" else "/".join(sorted(metric_labels))
    plt.ylabel(ylabel)
    plt.title(args.title or f"{ylabel} from loss.csv")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"[INFO] saved plot = {args.out}")


if __name__ == "__main__":
    main()
