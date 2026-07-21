#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def stats(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--window", type=int, default=70)
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    window = max(1, min(args.window, len(rows)))
    groups = {
        "all": rows,
        "early": rows[:window],
        "recent": rows[-window:],
        "camera_valid": [row for row in rows if float(row.get("camera_valid_fraction", 0.0)) > 0.5],
        "camera_invalid": [row for row in rows if float(row.get("camera_valid_fraction", 0.0)) <= 0.5],
    }
    metrics = [
        "loss_objective",
        "loss_reg_depth",
        "loss_log_l1_depth",
        "loss_overlap_depth",
        "loss_conf_depth",
        "loss_grad_depth",
        "loss_camera",
        "loss_T",
        "loss_R",
        "loss_FL",
        "camera_valid_fraction",
        "valid_fraction",
        "lr",
    ]
    result = {
        "path": str(args.csv_path),
        "rows": len(rows),
        "groups": {
            group_name: {
                "rows": len(group_rows),
                "metrics": {key: value for key in metrics if (value := stats(group_rows, key)) is not None},
            }
            for group_name, group_rows in groups.items()
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
