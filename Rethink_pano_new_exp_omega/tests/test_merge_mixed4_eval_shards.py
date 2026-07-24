import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_depth_checkpoint import CAMERA_PAIR_CSV_FIELDS, PER_SAMPLE_CSV_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = PROJECT_ROOT / "scripts" / "merge_mixed4_eval_shards.py"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_merge_accepts_blank_optional_numeric_metrics():
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        shard_json = root / "shard_0.json"
        shard_csv = root / "shard_0.csv"
        shard_camera_csv = root / "shard_0_camera_pairs.csv"
        output = root / "summary.json"
        per_sample = root / "per_sample.csv"
        camera_pairs = root / "camera_pairs.csv"
        shard_json.write_text(
            json.dumps(
                {
                    "config": "configs/test.yaml",
                    "checkpoint": "last.pt",
                    "num_shards": 1,
                    "runs": [
                        {
                            "name": "Panocity_test_0",
                            "split": "test",
                            "dataset_size": 1,
                            "candidate_samples": 1,
                            "dataset": "Panocity",
                            "minimal_dataset": "panocity",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        write_csv(
            shard_csv,
            PER_SAMPLE_CSV_FIELDS,
            [
                {
                    "dataset": "Panocity",
                    "split": "test",
                    "run": "Panocity_test_0",
                    "dataset_index": 0,
                    "quality_bin": "all",
                    "loss": 1.0,
                    "loss_depth": 1.0,
                    "loss_overlap": 0.0,
                    "valid_fraction": 1.0,
                    "depth_irls_abs_rel": "",
                    "depth_irls_delta_1p25": "",
                    "erp_depth_irls_abs_rel": "",
                    "erp_depth_irls_delta_1p25": "not-a-number",
                }
            ],
        )
        write_csv(shard_camera_csv, CAMERA_PAIR_CSV_FIELDS, [])

        subprocess.run(
            [
                sys.executable,
                str(MERGE_SCRIPT),
                "--shard-json",
                str(shard_json),
                "--shard-csv",
                str(shard_csv),
                "--shard-camera-csv",
                str(shard_camera_csv),
                "--output",
                str(output),
                "--per-sample-csv",
                str(per_sample),
                "--camera-pair-csv",
                str(camera_pairs),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["runs"][0]["evaluated_samples"] == 1
        assert payload["case_rankings"]["best_by_depth_irls_abs_rel"] == []
        assert payload["case_rankings"]["best_by_erp_depth_irls_abs_rel"] == []
        assert per_sample.is_file()
        assert camera_pairs.is_file()


if __name__ == "__main__":
    test_merge_accepts_blank_optional_numeric_metrics()
    print("mixed4 shard merge missing-metric tests ok")
