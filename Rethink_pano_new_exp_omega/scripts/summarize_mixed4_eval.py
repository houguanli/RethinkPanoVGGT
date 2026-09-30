#!/usr/bin/env python3
"""Human-readable mixed4 report and explicitly depth-selected, exact-input examples."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.validate_eval_cardinality import EXPECTED_FULL_ANCHOR_SETS

SUMMARY_NAME = "validation_mixed4_by_dataset_valtestfull_summary.json"
CSV_NAME = "validation_mixed4_by_dataset_valtestfull_per_sample.csv"
CROP = ("depth_irls_abs_rel", "depth_irls_delta_1p25")
ERP = ("erp_prior_depth_irls_abs_rel", "erp_prior_depth_irls_delta_1p25")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def fmt(value):
    value = number(value)
    return "N/A" if value is None else f"{value:.4f}"


def mean(rows, key):
    values = [number(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def select_examples(rows, top_k=3, max_absrel=0.15, min_delta=0.85, min_valid=0.1):
    """Rank within each dataset; no silent relaxation when nothing qualifies."""
    selected = []
    for dataset in EXPECTED_FULL_ANCHOR_SETS:
        eligible = []
        for row in rows:
            absrel, delta = number(row.get(CROP[0])), number(row.get(CROP[1]))
            valid, pixels = number(row.get("valid_fraction")), number(row.get("depth_valid_pixels"))
            if (row.get("dataset") == dataset and absrel is not None and 0 <= absrel <= max_absrel
                    and delta is not None and min_delta <= delta <= 1
                    and valid is not None and min_valid <= valid <= 1
                    and pixels is not None and pixels > 0):
                eligible.append(row)
        eligible.sort(key=lambda r: (float(r[CROP[0]]), -float(r[CROP[1]]), int(r["dataset_index"])))
        groups = set()
        for row in eligible:
            if len(groups) >= top_k:
                break
            group = tuple(sorted(str(row.get("seq_name", "")).split("|")))
            if group == ("",):
                group = (str(row["dataset_index"]),)
            if group in groups:
                continue
            groups.add(group)
            selected.append({"dataset": dataset, "dataset_index": int(row["dataset_index"]),
                             "rank": len(groups), "source": dict(row)})
    return selected


def validate_sources(summary, rows):
    expected = Counter()
    for run in summary.get("runs", []):
        expected[run["dataset"]] += int(run["evaluated_samples"])
    actual = Counter(row["dataset"] for row in rows)
    if actual != expected:
        raise ValueError(f"Summary/CSV set counts disagree: summary={dict(expected)}, CSV={dict(actual)}")
    keys = [(r["dataset"], r["run"], r["dataset_index"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate evaluated sets in merged CSV")
    for row in rows:
        count = (summary.get("dataset_pano_counts") or {}).get(row["dataset"].lower())
        if count is not None and number(row.get("input_pano_count")) != int(count):
            raise ValueError(f"Actual pano count disagrees with evaluation policy: {row['dataset']}")


def build_report(summary, rows, cases, settings, example_status):
    counts = Counter(row["dataset"] for row in rows)
    full = (dict(counts) == EXPECTED_FULL_ANCHOR_SETS and summary.get("sample_policy") == "anchor"
            and int(summary.get("limit_per_dataset", -1)) == 0)
    lines = [
        "MIXED4 FINAL EVALUATION REPORT",
        f"Status: {'FULL 8833 anchor sets' if full else 'PARTIAL / DIAGNOSTIC (not formal full8833)'}",
        f"Evaluated sets: {len(rows)}; expected full split: 8833; shards: {summary.get('num_shards', 'N/A')}",
        f"Checkpoint: {summary.get('checkpoint')}",
        f"Completion checkpoint: {summary.get('erp_completion_checkpoint')}",
        f"Geometry: pitch={summary.get('pitch_degrees')}, yaw/ring={summary.get('num_yaw')}, "
        f"window={summary.get('window_size')}, FoV={summary.get('fov_x_degrees') or summary.get('fov_degrees')}"
        f"x{summary.get('fov_y_degrees') or summary.get('fov_degrees')}",
        f"Input policy: {summary.get('pano_count_policy')}; sample policy: {summary.get('sample_policy')}",
        f"Depth protocol: {summary.get('depth_metric_protocol')}",
        f"ERP mode: {summary.get('erp_completion_mode')}",
        "",
        "PER DATASET (depth = arithmetic mean across sets; IRLS scale aligned; delta is a fraction)",
        "Dataset        Sets/full   Actual panos/set  Crop AbsRel / delta1.25  ERP AbsRel / delta1.25",
    ]
    dataset_means = []
    for dataset, expected in EXPECTED_FULL_ANCHOR_SETS.items():
        subset = [r for r in rows if r["dataset"] == dataset]
        pano_counts = sorted({int(float(r["input_pano_count"])) for r in subset})
        lines.append(f"{dataset:<14} {len(subset):>4}/{expected:<4}   {str(pano_counts):<16} "
                     f"{fmt(mean(subset, CROP[0]))} / {fmt(mean(subset, CROP[1]))}        "
                     f"{fmt(mean(subset, ERP[0]))} / {fmt(mean(subset, ERP[1]))}")
        if subset:
            dataset_means.append({key: mean(subset, key) for key in CROP + ERP})
        lines.append("  ERP diagnostics (set means): canonical AbsRel="
                     f"{fmt(mean(subset, 'erp_canonical_depth_irls_abs_rel'))}; "
                     f"cap60 AbsRel={fmt(mean(subset, 'erp_cap60_abs_rel'))}; "
                     f"coverage={fmt(mean(subset, 'erp_window_coverage_fraction'))}; "
                     f"evaluated GT fraction={fmt(mean(subset, 'erp_evaluated_gt_fraction'))}")
    lines += ["", "OVERALL (different aggregations, do not interchange)"]
    for label, subset in (("Equal weight per set", rows),
                          (f"Equal weight per dataset ({len(dataset_means)}/4 available)", dataset_means)):
        lines.append(f"{label}: crop AbsRel={fmt(mean(subset, CROP[0]))}, delta={fmt(mean(subset, CROP[1]))}; "
                     f"ERP AbsRel={fmt(mean(subset, ERP[0]))}, delta={fmt(mean(subset, ERP[1]))}")
    micro = summary.get("panovggt_depth_benchmark", {}).get("overall_micro_irls_scale_aligned", {})
    lines.append(f"Pooled crop valid-pixel/solid-angle weighting: AbsRel={fmt(micro.get(CROP[0]))}, "
                 f"delta={fmt(micro.get(CROP[1]))}")
    camera = summary.get("panovggt_camera_benchmark", {})
    lines += ["", "CAMERA (pooled valid pose pairs; AUC values are fractions)",
              "Dataset        Pairs      AUC@3     AUC@5     AUC@15    AUC@30"]
    for dataset in [*EXPECTED_FULL_ANCHOR_SETS, "Overall"]:
        pose = (camera.get("overall", {}) if dataset == "Overall"
                else camera.get("per_dataset", {}).get(dataset, {}).get("pose", {}))
        count = int(pose.get("pair_count", 0) or 0)
        auc = [fmt(pose.get(f"auc@{angle}")) if count else "N/A" for angle in (3, 5, 15, 30)]
        lines.append(f"{dataset:<14} {count:<10} " + "   ".join(f"{v:>7}" for v in auc))
    lines += [
        "N/A means unavailable/no valid GT, not zero error. Camera metrics are not used for case selection.",
        "Depth means omit nonfinite/unavailable values; a cap without valid GT is not counted as zero error.",
        "Crop metrics cover sampled windows, not full ERP. ERP completion metrics are separate diagnostics.",
        "PanoVGGT reference numbers are not protocol-identical to these crop/solid-angle metrics.",
        "",
        f"SELECTED EXAMPLES: {example_status}",
        f"At most {settings['top_k']} per dataset; crop IRLS AbsRel <= {settings['max_absrel']}, "
        f"delta >= {settings['min_delta']}, valid fraction >= {settings['min_valid']}, valid pixels > 0.",
        "Rank: lowest crop AbsRel, then highest delta; identical pano groups deduplicated.",
        "These are deliberately selected good-depth cases, NOT representative aggregate performance.",
        "Images rerun the same ordered multi-pano set; crop panels show the anchor pano only.",
        "Prediction/GT share a depth color scale; prediction uses the set's IRLS GT scale alignment.",
    ]
    for case in cases:
        source = case["source"]
        lines.append(f"  {case['dataset']} #{case['rank']} index={case['dataset_index']} "
                     f"panos={source['input_pano_count']} AbsRel={fmt(source[CROP[0]])} "
                     f"delta={fmt(source[CROP[1]])} -> {case['example_dir']}")
    for dataset in EXPECTED_FULL_ANCHOR_SETS:
        if not any(c["dataset"] == dataset for c in cases):
            lines.append(f"  {dataset}: no qualifying case (thresholds not relaxed).")
    return "\n".join(lines) + "\n"


def export_depth_example(case, row, batch, predictions, pred_depth_base, target_depth, target_valid):
    """Render the evaluated anchor windows; NEVER change multi-pano inference to single-pano."""
    import numpy as np
    from PIL import Image, ImageDraw
    import torch.nn.functional as F
    from scripts.reconstruct_pano_omega import colorize_depth

    source = case["source"]
    if (row["seq_name"] != source["seq_name"]
            or int(row["input_pano_count"]) != int(float(source["input_pano_count"]))):
        raise ValueError("Example replay pano set/order differs from the ranked evaluation; refusing misleading images")
    output = Path(case["example_dir"])
    output.mkdir(parents=True, exist_ok=True)
    panos = int(row["input_pano_count"])
    views = predictions["pano_windows"].shape[1] // panos
    rgb = predictions["pano_windows"][0, :views].detach().float().cpu().numpy()
    pred = pred_depth_base[0, :views, ..., 0].detach().float().cpu().numpy() * float(row["depth_irls_scale"])
    gt = target_depth[0, :views, ..., 0].detach().float().cpu().numpy()
    valid = target_valid[0, :views, ..., 0].detach().cpu().numpy().astype(bool)
    valid &= np.isfinite(gt) & (gt > 0)
    depth_max = max(float(np.percentile(gt[valid], 98)), 1e-6) if valid.any() else 1.0
    size = 192
    canvas = Image.new("RGB", (size * 4, 52 + views * (size + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), f"{row['dataset']} | {panos} input panos | anchor windows | selected example", fill="black")
    for column, label in enumerate(("RGB", f"GT z (0..{depth_max:.2f})", "Pred z (IRLS aligned)", "AbsRel (0..0.5)")):
        draw.text((column * size + 4, 27), label, fill="black")
    for view in range(views):
        common = valid[view] & np.isfinite(pred[view]) & (pred[view] > 0)
        error = np.abs(pred[view] - gt[view]) / np.maximum(gt[view], 1e-6)
        panels = [(np.clip(rgb[view].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8),
                  colorize_depth(gt[view], valid[view], depth_max),
                  colorize_depth(pred[view], common, depth_max),
                  colorize_depth(np.maximum(error, 1e-6), common, 0.5)]
        y = 52 + view * (size + 18)
        draw.text((4, y), f"window {view}", fill="black")
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel).resize((size, size)), (column * size, y + 18))
    canvas.save(output / "comparison.png")
    inputs = F.interpolate(batch["pano_image"][0].detach().float(), size=(80, 160), mode="bilinear",
                           align_corners=False).cpu().numpy()
    contact = Image.new("RGB", (160 * min(panos, 5), 100 * ((panos + 4) // 5)), "white")
    draw = ImageDraw.Draw(contact)
    for i, pano in enumerate(inputs):
        x, y = (i % 5) * 160, (i // 5) * 100
        draw.text((x + 3, y + 2), f"input {i}" + (" (anchor)" if i == 0 else ""), fill="black")
        contact.paste(Image.fromarray((np.clip(pano.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)), (x, y + 20))
    contact.save(output / "input_panos.jpg")
    (output / "metrics.json").write_text(json.dumps(
        {"selected_source": source, "replayed_metrics": row, "depth_color_max": depth_max,
         "display": "anchor windows, shared GT/pred scale, GT-IRLS aligned, invalid pixels black",
         "error_color_max": 0.5}, indent=2, ensure_ascii=False), encoding="utf-8")


def replay_command(summary, manifest, output, device):
    cmd = [sys.executable, str(ROOT / "scripts/evaluate_mixed4_depth_checkpoint.py"),
           "--config", summary["config"], "--checkpoint", summary["checkpoint"],
           "--dataset-root", summary["dataset_root"], "--output", str(output / "summary.json"),
           "--per-sample-csv", str(output / "per_sample.csv"), "--example-manifest", str(manifest),
           "--datasets", ",".join(sorted({c["dataset"].lower() for c in json.loads(manifest.read_text())["cases"]})),
           "--pano-count-policy", summary.get("pano_count_policy", "panovggt"), "--sample-policy", "anchor",
           "--dataset-pano-counts", ",".join(f"{k}:{v}" for k, v in (summary.get("dataset_pano_counts") or {}).items()),
           "--limit-per-dataset", "0", "--num-workers", "0", "--device", device, "--amp-dtype", "bfloat16",
           "--seed", str(summary.get("seed", 123)),
           "--camera-eval-max-panos", str(summary.get("camera_eval_max_panos", 3))]
    if summary.get("erp_completion_checkpoint"):
        cmd += ["--erp-completion-checkpoint", summary["erp_completion_checkpoint"]]
    for key in ("num_yaw", "window_size", "pitch_degrees", "fov_degrees", "fov_x_degrees", "fov_y_degrees"):
        if summary.get(key) is not None:
            cmd.append(f"--{key.replace('_', '-')}={summary[key]}")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--examples-per-dataset", type=int, default=3)
    parser.add_argument("--max-absrel", type=float, default=0.15)
    parser.add_argument("--min-delta", type=float, default=0.85)
    parser.add_argument("--min-valid-fraction", type=float, default=0.1)
    parser.add_argument("--export-examples", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.examples_per_dataset < 0 or args.max_absrel < 0 or not (0 <= args.min_delta <= 1 and 0 <= args.min_valid_fraction <= 1):
        parser.error("Invalid example selection thresholds")
    output = args.eval_dir.resolve()
    summary = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
    with (output / CSV_NAME).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    validate_sources(summary, rows)
    settings = dict(top_k=args.examples_per_dataset, max_absrel=args.max_absrel,
                    min_delta=args.min_delta, min_valid=args.min_valid_fraction)
    cases = select_examples(rows, **settings)
    identity = {key: summary.get(key) for key in ("config", "checkpoint", "erp_completion_checkpoint",
                "dataset_root", "seed", "num_yaw", "pitch_degrees", "window_size", "fov_degrees",
                "fov_x_degrees", "fov_y_degrees", "depth_metric_protocol", "dataset_pano_counts")}
    identity["checkpoint_stats"] = []
    for key in ("checkpoint", "erp_completion_checkpoint"):
        path = Path(summary[key]) if summary.get(key) else None
        identity["checkpoint_stats"].append(
            [path.stat().st_size, path.stat().st_mtime_ns] if path and path.is_file() else None)
    signature = hashlib.sha256(json.dumps([identity, cases], sort_keys=True).encode()).hexdigest()[:16]
    example_root = output / "examples" / signature
    for case in cases:
        case["example_dir"] = str(example_root / f"{case['dataset'].lower()}_{case['dataset_index']:06d}")
    manifest = output / "good_cases.json"
    manifest.write_text(json.dumps({"selection": settings, "evaluation": identity, "cases": cases},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
    status = "pending exact-input replay" if args.export_examples else "manifest only; images not requested"
    report = output / "EVAL_REPORT.txt"
    report.write_text(build_report(summary, rows, cases, settings, status), encoding="utf-8")
    failed = False
    if args.export_examples and cases:
        ready = all(all((Path(c["example_dir"]) / name).is_file()
                        for name in ("comparison.png", "input_panos.jpg", "metrics.json")) for c in cases)
        if not ready:
            example_root.mkdir(parents=True, exist_ok=True)
            log_path = example_root / "export.log"
            with log_path.open("a", encoding="utf-8") as log:
                print(f"[EXAMPLES] replaying {len(cases)} exact multi-pano sets; log={log_path}", flush=True)
                result = subprocess.run(replay_command(summary, manifest, example_root, args.device),
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            failed = result.returncode != 0 or not all(
                (Path(c["example_dir"]) / "metrics.json").is_file() for c in cases)
            status = f"FAILED; metrics preserved; see {log_path}" if failed else "exported"
        else:
            status = "existing exact-input images reused"
    elif not cases:
        status = "no qualifying cases"
    report.write_text(build_report(summary, rows, cases, settings, status), encoding="utf-8")
    print(report.read_text(encoding="utf-8"), flush=True)
    print(f"[REPORT] {report}\n[CASES] {manifest}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
