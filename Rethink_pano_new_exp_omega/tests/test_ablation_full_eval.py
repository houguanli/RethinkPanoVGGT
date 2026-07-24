from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_ablation_full_eval import (
    build_environment,
    resolve_dataset_root,
    resolve_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_full_eval_recovers_machine_paths_from_checkpoint():
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        dataset_root = tmp_path / "datasets"
        dataset_root.mkdir()
        payload = {
            "args": {
                "config": "configs/ablation_4090_local0716_no_camera_geora.yaml",
                "dataset_root": dataset_root,
            }
        }
        eval_config = {
            "model": {"training_config": "checkpoint"},
            "data": {
                "dataset_root": "checkpoint",
                "datasets": "all",
                "limit_per_dataset": 0,
                "sample_policy": "anchor",
            },
            "geometry": {
                "pano_count_policy": "panovggt",
                "dataset_pano_counts": "",
                "camera_eval_max_panos": 3,
                "num_yaw": 8,
                "erp_latitude_limit_deg": 75,
            },
            "runtime": {"gpus": "0,1,2,3", "num_workers_per_gpu": 0},
        }
        checkpoint = tmp_path / "last.pt"
        checkpoint.touch()

        training_config = resolve_training_config(eval_config, payload)
        resolved_dataset_root = resolve_dataset_root(eval_config, payload)
        env = build_environment(
            eval_config,
            checkpoint,
            tmp_path / "eval",
            training_config,
            resolved_dataset_root,
        )

        assert training_config == (
            PROJECT_ROOT / "configs" / "ablation_4090_local0716_no_camera_geora.yaml"
        ).resolve()
        assert resolved_dataset_root == dataset_root.resolve()
        assert env["GPUS"] == "0,1,2,3"
        assert env["LIMIT_PER_DATASET"] == "0"
        assert env["EVAL_DATASETS"] == "all"
        assert env["NUM_WORKERS_PER_GPU"] == "0"
        assert env["PANO_COUNT_POLICY"] == "panovggt"
        assert env["NUM_YAW"] == "8"


if __name__ == "__main__":
    test_full_eval_recovers_machine_paths_from_checkpoint()
    print("ablation full eval launcher tests ok")
