#!/usr/bin/env python3
"""Append completed BiFuse++ and Pi3 PanoSUNCG results to the paper table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METHODS = ("bifusepp", "pi3")
EXPECTED_DEPTH_SAMPLES = 3944
EXPECTED_CAMERA_TRAJECTORIES = 118
EXPECTED_CAMERA_FRAMES = EXPECTED_CAMERA_TRAJECTORIES * 5
EXPECTED_CAMERA_PAIRS = EXPECTED_CAMERA_TRAJECTORIES * 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(root: Path) -> dict[str, dict]:
    results = {}
    for method in METHODS:
        depth = load_json(root / method / "depth" / "metrics_summary.json")
        camera = load_json(root / method / "camera" / "camera_center_summary.json")
        if depth.get("completed_samples") != EXPECTED_DEPTH_SAMPLES or depth.get(
            "expected_full_split_samples"
        ) != EXPECTED_DEPTH_SAMPLES:
            raise RuntimeError(f"{method} depth is not the complete 3944-sample split")
        if depth.get("metric_region_latitude_degrees") != [-15.0, 60.0]:
            raise RuntimeError(f"{method} depth latitude band does not match the table protocol")
        if depth.get("aggregation") != "pixel micro":
            raise RuntimeError(f"{method} depth aggregation does not match the table protocol")
        if "100-step IRLS scale+shift" not in depth.get("alignment", ""):
            raise RuntimeError(f"{method} depth alignment does not match the table protocol")
        camera_config = camera.get("run_config", {})
        if camera.get("trajectory_count") != EXPECTED_CAMERA_TRAJECTORIES or camera_config.get(
            "expected_full_split_trajectories"
        ) != EXPECTED_CAMERA_TRAJECTORIES:
            raise RuntimeError(f"{method} camera is not the complete 118-trajectory split")
        if camera.get("frame_count") != EXPECTED_CAMERA_FRAMES:
            raise RuntimeError(f"{method} camera does not contain all 590 evaluated frames")
        if camera.get("pair_count") != EXPECTED_CAMERA_PAIRS:
            raise RuntimeError(f"{method} camera does not contain all 1180 frame pairs")
        if camera_config.get("frames_per_trajectory") != 5:
            raise RuntimeError(f"{method} camera frame selection does not match the table protocol")
        if camera_config.get("alignment") != "per-trajectory Umeyama Sim(3)":
            raise RuntimeError(f"{method} camera alignment does not match the table protocol")
        results[method] = {"depth": depth, "camera": camera}
    return results


def insert_before_bottomrule(text: str, start: int, rows: list[str]) -> tuple[str, int]:
    position = text.index("\\bottomrule", start)
    insertion = "\n".join(rows) + "\n"
    return text[:position] + insertion + text[position:], position + len(insertion)


def main() -> int:
    args = parse_args()
    results = load_results(args.results_root.expanduser().resolve())
    table = args.table.expanduser().resolve()
    if not table.is_file():
        raise FileNotFoundError(table)
    text = table.read_text(encoding="utf-8-sig")
    text = re.sub(r"(?m)^\s*(?:BiFuse\+\+|Pi3).*?\\\\\s*$\n?", "", text)
    text = text.replace(
        "\\caption*{$^\\dagger$ BiFuse++ camera uses the authors' PanoSUNCG-trained "
        "self-supervised checkpoint and is therefore in-domain, not strict zero-shot.}\n",
        "",
    )
    text = text.replace("The four baselines use", "The baselines use")

    depth_rows = []
    for method, label in (("bifusepp", "BiFuse++"), ("pi3", "Pi3")):
        metrics = results[method]["depth"]["metrics"]["pixel_micro"]
        depth_rows.append(
            f"    {label} & {metrics['absrel']:.4f} & {metrics['rmse']:.4f} "
            f"& {metrics['delta1']:.4f} & {metrics['delta2']:.4f} \\\\"
        )
    depth_start = text.index("\\label{tab:panosuncg-depth-final}")
    text, _ = insert_before_bottomrule(text, depth_start, depth_rows)

    camera_rows = []
    for method, label in (("bifusepp", "BiFuse++$^\\dagger$"), ("pi3", "Pi3")):
        camera = results[method]["camera"]
        micro = camera["micro"]
        macro = camera["macro_by_trajectory"]
        camera_rows.append(
            {
                "unified": (
                    f"    {label} & {micro['direction_auc3']:.4f} "
                    f"& {micro['direction_auc5']:.4f} & {micro['direction_auc15']:.4f} "
                    f"& {micro['direction_auc30']:.4f} "
                    f"& {micro['direction_deg_mean']:.4f} & {micro['direction_deg_median']:.4f} "
                    f"& {micro['ate_rmse']:.4f} & {macro['ate_normalized_rmse']:.4f} "
                    f"& {micro['relative_length_error_mean']:.4f} "
                    f"& {micro['relative_length_error_median']:.4f} \\\\"
                ),
                "standard": (
                    f"    {label} & {micro['direction_auc30']:.4f} & -- & -- "
                    f"& {micro['direction_deg_mean']:.4f} & {micro['direction_deg_median']:.4f} \\\\"
                ),
                "auc": (
                    f"    {label} & {micro['direction_auc3']:.4f} "
                    f"& {micro['direction_auc5']:.4f} & {micro['direction_auc15']:.4f} \\\\"
                ),
                "trajectory": (
                    f"    {label} & {micro['ate_rmse']:.4f} "
                    f"& {macro['ate_normalized_rmse']:.4f} "
                    f"& {micro['relative_length_error_mean']:.4f} "
                    f"& {micro['relative_length_error_median']:.4f} \\\\"
                ),
            }
        )
    camera_start = text.index("\\label{tab:panosuncg-camera-final}")
    camera_end = text.index("\\end{table}", camera_start)
    camera_section = text[camera_start:camera_end]
    if "\\begin{tabular}{lrrrrrrrrrr}" in camera_section:
        text, cursor = insert_before_bottomrule(
            text, camera_start, [row["unified"] for row in camera_rows]
        )
    else:
        text, cursor = insert_before_bottomrule(
            text, camera_start, [row["standard"] for row in camera_rows]
        )
        text, cursor = insert_before_bottomrule(
            text, cursor, [row["auc"] for row in camera_rows]
        )
        text, cursor = insert_before_bottomrule(
            text, cursor, [row["trajectory"] for row in camera_rows]
        )
    table_end = text.index("\\end{table}", cursor)
    caveat = (
        "\\caption*{$^\\dagger$ BiFuse++ camera uses the authors' PanoSUNCG-trained "
        "self-supervised checkpoint and is therefore in-domain, not strict zero-shot.}\n"
    )
    text = text[:table_end] + caveat + text[table_end:]
    temporary = table.with_suffix(table.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(table)
    print(f"Updated complete PanoSUNCG results in {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
