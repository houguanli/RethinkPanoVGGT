#!/usr/bin/env python3
"""Plot baseline01 training loss curves from loss.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=200)
    args = parser.parse_args()

    rows = []
    with args.loss_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {args.loss_csv}")

    steps = [int(float(row["step"])) for row in rows]
    losses = [float(row["loss_objective"]) for row in rows]
    smooth = rolling_mean(losses, max(1, int(args.smooth)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(steps, losses, color="#7b8794", linewidth=0.45, alpha=0.45, label="step loss")
    ax.plot(steps, smooth, color="#005cc5", linewidth=1.8, label=f"rolling mean ({args.smooth})")
    ax.set_title("baseline01 full VGGT-Omega finetune on PanoCity-100")
    ax.set_xlabel("step")
    ax.set_ylabel("clipped log-depth loss")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output)
    print(args.output)


def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


if __name__ == "__main__":
    main()
