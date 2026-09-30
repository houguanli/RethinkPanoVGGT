import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mixed4_eval import (
    CROP, ERP, build_report, select_examples, validate_sources, export_depth_example,
)


def sample(dataset="Panocity", index=0, absrel=0.05, delta=0.95, seq=None):
    return dict(dataset=dataset, run=dataset + "_test_0", dataset_index=str(index),
                seq_name=seq or f"a{index}|b{index}", input_pano_count="10" if dataset == "Panocity" else "3",
                depth_irls_abs_rel=str(absrel), depth_irls_delta_1p25=str(delta),
                erp_prior_depth_irls_abs_rel="0.25", erp_prior_depth_irls_delta_1p25="0.75",
                valid_fraction="0.8", depth_valid_pixels="100", depth_irls_scale="1.0")


class EvalReportTest(unittest.TestCase):
    def test_four_shards_merge_into_one_report(self):
        from scripts.evaluate_depth_checkpoint import PER_SAMPLE_CSV_FIELDS, CAMERA_PAIR_CSV_FIELDS, DEPTH_METRIC_PROTOCOL
        from scripts.summarize_mixed4_eval import ROOT, SUMMARY_NAME, CSV_NAME
        datasets = ("Stanford2D3DS", "Matterport3D", "Structured3D", "Panocity")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            jsons, csvs, cameras = [], [], []
            for rank, dataset in enumerate(datasets):
                row = {key: "0" for key in PER_SAMPLE_CSV_FIELDS}
                row.update(sample(dataset, rank), depth_metric_protocol=DEPTH_METRIC_PROTOCOL)
                payload = dict(depth_metric_protocol=DEPTH_METRIC_PROTOCOL, num_shards=4,
                               sample_policy="anchor", limit_per_dataset=1, num_yaw=6,
                               pitch_degrees="-25,25", fov_degrees=75, fov_x_degrees=75, fov_y_degrees=75,
                               runs=[dict(name=row["run"], dataset=dataset, split="test")])
                shard = output / f"shard{rank}.json"
                shard.write_text(json.dumps(payload))
                data = output / f"shard{rank}.csv"
                with data.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=PER_SAMPLE_CSV_FIELDS)
                    writer.writeheader()
                    writer.writerow(row)
                camera = output / f"camera{rank}.csv"
                with camera.open("w", newline="") as f:
                    csv.writer(f).writerow(CAMERA_PAIR_CSV_FIELDS)
                jsons.append(str(shard))
                csvs.append(str(data))
                cameras.append(str(camera))
            merged = subprocess.run(
                [sys.executable, str(ROOT / "scripts/merge_mixed4_eval_shards.py"),
                 "--shard-json", *jsons, "--shard-csv", *csvs, "--shard-camera-csv", *cameras,
                 "--output", str(output / SUMMARY_NAME), "--per-sample-csv", str(output / CSV_NAME)],
                capture_output=True, text=True)
            self.assertEqual(merged.returncode, 0, merged.stderr)
            reported = subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize_mixed4_eval.py"), "--eval-dir", temp],
                capture_output=True, text=True)
            self.assertEqual(reported.returncode, 0, reported.stderr)
            report = (output / "EVAL_REPORT.txt").read_text()
            self.assertIn("Evaluated sets: 4", report)
            self.assertIn("shards: 4", report)
            for dataset in datasets:
                self.assertIn(dataset, report)
            self.assertIn("OVERALL", report)
            self.assertEqual(len(json.loads((output / "good_cases.json").read_text())["cases"]), 4)

    def test_selection_is_per_dataset_finite_quality_gated_and_deduplicated(self):
        rows = [sample(index=0, seq="a|b"), sample(index=1, absrel=0.02, seq="b|a"),
                sample(index=2, absrel="nan"), sample(index=3, absrel=0.3),
                sample(index=4, delta=0.5), sample(index=5),
                sample("Matterport3D", index=7)]
        rows[5]["valid_fraction"] = "0.001"
        selected = select_examples(rows)
        self.assertEqual([(c["dataset"], c["dataset_index"]) for c in selected],
                         [("Matterport3D", 7), ("Panocity", 1)])
        self.assertEqual(select_examples(rows, top_k=0), [])
        self.assertEqual(select_examples(rows, max_absrel=0.001), [])

    def test_report_distinguishes_means_partial_results_and_missing_camera_gt(self):
        rows = [sample("Matterport3D", 0, 0.1), sample(index=0, absrel=0.01),
                sample(index=1, absrel=0.03)]
        summary = {"sample_policy": "anchor", "limit_per_dataset": 1,
                   "runs": [{"dataset": "Matterport3D", "evaluated_samples": 1},
                            {"dataset": "Panocity", "evaluated_samples": 2}]}
        validate_sources(summary, rows)
        report = build_report(summary, rows, [], dict(top_k=3, max_absrel=.15, min_delta=.85, min_valid=.1), "test")
        self.assertIn("PARTIAL / DIAGNOSTIC", report)
        self.assertIn("Equal weight per set: crop AbsRel=0.0467", report)
        self.assertIn("Equal weight per dataset (2/4 available): crop AbsRel=0.0600", report)
        self.assertIn("N/A", report)
        self.assertIn("not zero error", report)
        self.assertIn("NOT representative", report)
        self.assertIn("no qualifying case", report)

    def test_rejects_missing_duplicate_or_wrong_pano_results(self):
        row = sample()
        summary = {"runs": [{"dataset": "Panocity", "evaluated_samples": 1}],
                   "dataset_pano_counts": {"panocity": 10}}
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            validate_sources(summary, [])
        wrong = dict(row, input_pano_count="6")
        with self.assertRaisesRegex(ValueError, "pano count"):
            validate_sources(summary, [wrong])
        summary["runs"][0]["evaluated_samples"] = 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_sources(summary, [row, row])

    def test_same_multiview_renderer_and_mismatch_guard(self):
        import torch
        from PIL import Image
        torch.set_num_threads(2)
        row = sample("Matterport3D")
        row["input_pano_count"] = 3
        with tempfile.TemporaryDirectory() as temp:
            case = {"source": dict(row), "example_dir": temp}
            batch = {"pano_image": torch.rand(1, 3, 3, 16, 32)}
            predictions = {"pano_windows": torch.rand(1, 12, 3, 16, 16)}
            depth = torch.ones(1, 12, 16, 16, 1)
            valid = torch.ones_like(depth, dtype=torch.bool)
            export_depth_example(case, row, batch, predictions, depth, depth, valid)
            self.assertTrue((Path(temp) / "metrics.json").is_file())
            with Image.open(Path(temp) / "comparison.png") as image:
                self.assertEqual(image.width, 4 * 192)
            with Image.open(Path(temp) / "input_panos.jpg") as image:
                self.assertEqual(image.width, 3 * 160)
            wrong = dict(row, seq_name="different")
            with self.assertRaisesRegex(ValueError, "set/order differs"):
                export_depth_example(case, wrong, batch, predictions, depth, depth, valid)


if __name__ == "__main__":
    unittest.main()
