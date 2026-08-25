#!/usr/bin/env python3
"""Plot smoothed training losses and emit a compact numerical health report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFERRED_METRICS = (
    "loss",
    "loss_depth",
    "loss_camera",
    "loss_remaining",
    "loss_boundary",
    "loss_distill",
    "loss_smooth",
)
FINITE_METRICS = ("pred_depth_finite_ratio", "pred_finite_ratio")
VALID_METRICS = ("depth_valid_ratio", "depth_loss_valid_ratio", "remaining_valid_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Named loss CSV. Repeat once per training stage.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_paths = [parse_named_path(item) for item in args.series]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    figure, axes = plt.subplots(len(named_paths), 1, figsize=(13, 4.2 * len(named_paths)), squeeze=False)
    for axis, (name, path) in zip(axes[:, 0], named_paths):
        rows = read_rows(path)
        report = analyze_stage(name, path, rows)
        reports[name] = report
        plot_stage(axis, name, rows, report["smoothing_window"])
    figure.tight_layout()
    plot_path = args.output_dir / "training_loss_curves_smoothed.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    overall = aggregate_status(reports)
    payload = {"overall_status": overall, "stages": reports, "plot": str(plot_path)}
    report_path = args.output_dir / "training_loss_health.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LOSS-ANALYSIS] status={overall} plot={plot_path} report={report_path}", flush=True)
    for name, report in reports.items():
        print(
            f"[LOSS-ANALYSIS] {name}: status={report['status']} rows={report['rows']} "
            f"loss_change={report.get('loss_relative_change')} nonfinite={report['nonfinite_values']} "
            f"spikes={report.get('loss_spike_count', 0)}",
            flush=True,
        )


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CSV, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError(f"Expected NAME=CSV, got {value!r}")
    return name, Path(raw_path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "nan")))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def smoothing_window(length: int) -> int:
    if length <= 5:
        return max(1, length)
    window = max(11, min(501, length // 50))
    return window + 1 if window % 2 == 0 else window


def ema(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full_like(values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return output
    alpha = 2.0 / (window + 1.0)
    running = float(values[finite_indices[0]])
    for index, value in enumerate(values):
        if math.isfinite(value):
            running = alpha * float(value) + (1.0 - alpha) * running
            output[index] = running
    return output


def analyze_stage(name: str, path: Path, rows: list[dict[str, str]]) -> dict:
    if not rows:
        return {"path": str(path), "rows": 0, "status": "critical", "warnings": ["empty CSV"], "smoothing_window": 1, "nonfinite_values": 0}
    available = [key for key in PREFERRED_METRICS if key in rows[0]]
    window = smoothing_window(len(rows))
    warnings: list[str] = []
    nonfinite = 0
    metrics: dict[str, dict[str, float | int | None]] = {}
    for key in available:
        values = numeric_column(rows, key)
        nonfinite += int((~np.isfinite(values)).sum())
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            warnings.append(f"{key} has no finite values")
            continue
        smooth = ema(values, window)
        smooth_finite = smooth[np.isfinite(smooth)]
        edge = max(1, smooth_finite.size // 10)
        start = float(np.median(smooth_finite[:edge]))
        end = float(np.median(smooth_finite[-edge:]))
        relative_change = (end - start) / max(abs(start), 1e-12)
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        spike_threshold = median + max(8.0 * mad, 1e-12)
        spike_count = int((finite > spike_threshold).sum())
        metrics[key] = {
            "start_smoothed_median": start,
            "end_smoothed_median": end,
            "relative_change": relative_change,
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
            "spike_count": spike_count,
        }
    for key in FINITE_METRICS:
        if key in rows[0]:
            values = numeric_column(rows, key)
            finite = values[np.isfinite(values)]
            if finite.size and float(finite.min()) < 0.99:
                warnings.append(f"{key} minimum {float(finite.min()):.6f} < 0.99")
    for key in VALID_METRICS:
        if key in rows[0]:
            values = numeric_column(rows, key)
            finite = values[np.isfinite(values)]
            if finite.size == 0 or float(finite.max()) <= 0.0:
                warnings.append(f"{key} has no positive valid samples")
    loss_stats = metrics.get("loss", {})
    loss_change = loss_stats.get("relative_change")
    spike_count = int(loss_stats.get("spike_count", 0))
    if nonfinite:
        warnings.append(f"found {nonfinite} non-finite loss values")
    if isinstance(loss_change, float) and loss_change > 0.25:
        warnings.append(f"smoothed total loss increased by {loss_change:.1%}")
    if spike_count > max(20, int(len(rows) * 0.10)):
        warnings.append(f"frequent total-loss spikes: {spike_count}/{len(rows)}")
    status = "healthy" if not warnings else ("critical" if nonfinite or not available else "warning")
    return {
        "path": str(path),
        "rows": len(rows),
        "status": status,
        "warnings": warnings,
        "smoothing_window": window,
        "nonfinite_values": nonfinite,
        "loss_relative_change": loss_change,
        "loss_spike_count": spike_count,
        "metrics": metrics,
    }


def plot_stage(axis, name: str, rows: list[dict[str, str]], window: int) -> None:
    if not rows:
        axis.set_title(f"{name}: empty")
        return
    if "step" in rows[0]:
        steps = numeric_column(rows, "step")
    else:
        steps = np.arange(1, len(rows) + 1, dtype=np.float64)
    plotted = 0
    for key in PREFERRED_METRICS:
        if key not in rows[0]:
            continue
        values = numeric_column(rows, key)
        color = axis._get_lines.get_next_color()
        axis.plot(steps, values, color=color, alpha=0.12, linewidth=0.7)
        axis.plot(steps, ema(values, window), color=color, linewidth=1.8, label=f"{key} (EMA {window})")
        plotted += 1
    axis.set_title(f"{name} — raw (faint) and smoothed")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(loc="best", fontsize=8, ncol=2)


def aggregate_status(reports: dict[str, dict]) -> str:
    statuses = {report["status"] for report in reports.values()}
    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "healthy"


if __name__ == "__main__":
    main()
