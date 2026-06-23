#!/usr/bin/env python3
"""Plot one or more training loss.csv files with optional EMA smoothing."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", type=Path, help="loss.csv files to plot.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Named run to plot. Can be repeated. Example: --run luna=logs/run/loss.csv",
    )
    parser.add_argument("--out", type=Path, default=Path("loss_csv_plot.png"))
    parser.add_argument("--metric", default="loss", help="CSV column to plot, default: loss.")
    parser.add_argument("--x", choices=["step", "elapsed_hours", "elapsed_minutes"], default="step")
    parser.add_argument("--smooth", type=float, default=0.95, help="EMA smoothing factor in [0, 1).")
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=1.0,
        help="Clip y-axis to this quantile for readability. Use 1.0 to disable.",
    )
    parser.add_argument("--title", default=None)
    return parser


def parse_runs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for raw in args.run:
        if "=" not in raw:
            raise ValueError(f"--run must be LABEL=PATH, got: {raw}")
        label, path = raw.split("=", 1)
        runs.append((label.strip(), Path(path)))
    for path in args.csv:
        runs.append((path.parent.name or path.stem, path))
    if not runs:
        raise ValueError("Provide at least one loss.csv path or --run LABEL=PATH.")
    return runs


def read_series(path: Path, metric: str, x_axis: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"step", metric}
        if x_axis != "step":
            required.add("elapsed_seconds")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        for row in reader:
            try:
                y = float(row[metric])
                if x_axis == "step":
                    x = float(row["step"])
                elif x_axis == "elapsed_hours":
                    x = float(row["elapsed_seconds"]) / 3600.0
                else:
                    x = float(row["elapsed_seconds"]) / 60.0
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                x_values.append(x)
                y_values.append(y)
    if not x_values:
        raise ValueError(f"No finite values for {metric!r} in {path}")
    return x_values, y_values


def ema(values: Iterable[float], alpha: float) -> list[float]:
    smoothed: list[float] = []
    current = None
    for value in values:
        current = value if current is None else alpha * current + (1.0 - alpha) * value
        smoothed.append(current)
    return smoothed


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
    for label, path in runs:
        x_values, y_values = read_series(path, args.metric, args.x)
        all_raw_values.extend(y_values)
        plt.plot(x_values, y_values, linewidth=0.6, alpha=0.18, label=f"{label} raw")
        plt.plot(x_values, ema(y_values, args.smooth), linewidth=1.8, label=f"{label} ema{args.smooth:g}")

    if 0.0 < args.clip_quantile < 1.0:
        y_max = quantile(all_raw_values, args.clip_quantile)
        if y_max > 0:
            plt.ylim(bottom=0.0, top=y_max * 1.05)
    xlabel = {"step": "step", "elapsed_hours": "elapsed hours", "elapsed_minutes": "elapsed minutes"}[args.x]
    plt.xlabel(xlabel)
    plt.ylabel(args.metric)
    plt.title(args.title or f"{args.metric} from loss.csv")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"[INFO] saved plot = {args.out}")


if __name__ == "__main__":
    main()
