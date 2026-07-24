import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = PROJECT_ROOT / "scripts" / "run_ablation_train_then_eval.py"


TRAIN_PROGRAM = """
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--exit-code", type=int, default=0)
args = parser.parse_args()
print("short training started", flush=True)
if args.exit_code == 0:
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.write_bytes(b"short-train-checkpoint")
print("short training finished", flush=True)
sys.exit(args.exit_code)
"""


EVAL_PROGRAM = """
import json
import os
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "received_checkpoint.txt").write_text(str(checkpoint), encoding="utf-8")
exit_code = int(os.environ.get("FAKE_EVAL_EXIT_CODE", "0"))
if exit_code == 0:
    (output_dir / "validation_mixed4_by_dataset_valtestfull_summary.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "state": "done"}),
        encoding="utf-8",
    )
sys.exit(exit_code)
"""


def prepare_programs(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_program = tmp_path / "short_train.py"
    eval_program = tmp_path / "short_eval.py"
    eval_config = tmp_path / "eval.yaml"
    train_program.write_text(TRAIN_PROGRAM, encoding="utf-8")
    eval_program.write_text(EVAL_PROGRAM, encoding="utf-8")
    eval_config.write_text("runtime: {}\n", encoding="utf-8")
    return train_program, eval_program, eval_config


def run_lifecycle(
    tmp_path: Path,
    *,
    auto_eval: bool = True,
    train_exit_code: int = 0,
    eval_exit_code: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    train_program, eval_program, eval_config = prepare_programs(tmp_path)
    checkpoint = tmp_path / "train" / "last.pt"
    eval_output = tmp_path / "eval"
    status_json = tmp_path / "train" / "post_training_eval_status.json"
    environment = os.environ.copy()
    environment["FAKE_EVAL_EXIT_CODE"] = str(eval_exit_code)
    command = [
        sys.executable,
        str(LIFECYCLE),
        "--checkpoint",
        str(checkpoint),
        "--eval-output",
        str(eval_output),
        "--train-log",
        str(tmp_path / "train" / "train.log"),
        "--status-json",
        str(status_json),
        "--auto-eval",
        "1" if auto_eval else "0",
        "--eval-launcher",
        str(eval_program),
        "--eval-config",
        str(eval_config),
        "--python",
        sys.executable,
        "--",
        sys.executable,
        str(train_program),
        "--checkpoint",
        str(checkpoint),
        "--exit-code",
        str(train_exit_code),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed, checkpoint, eval_output, status_json


def test_success_starts_eval_with_checkpoint_and_writes_results():
    with TemporaryDirectory(prefix="ablation_short_train_test_") as temporary_directory:
        tmp_path = Path(temporary_directory)
        completed, checkpoint, eval_output, status_json = run_lifecycle(tmp_path)
        status = json.loads(status_json.read_text(encoding="utf-8"))

        assert completed.returncode == 0, completed.stdout
        assert checkpoint.is_file()
        assert (eval_output / "received_checkpoint.txt").read_text(encoding="utf-8") == str(
            checkpoint.resolve()
        )
        assert (
            eval_output / "validation_mixed4_by_dataset_valtestfull_summary.json"
        ).is_file()
        assert status["state"] == "completed"
        assert status["eval_started"] is True


def test_explicit_disable_skips_eval():
    with TemporaryDirectory(prefix="ablation_eval_disabled_") as temporary_directory:
        tmp_path = Path(temporary_directory)
        completed, checkpoint, eval_output, status_json = run_lifecycle(
            tmp_path,
            auto_eval=False,
        )
        status = json.loads(status_json.read_text(encoding="utf-8"))

        assert completed.returncode == 0, completed.stdout
        assert checkpoint.is_file()
        assert not eval_output.exists()
        assert status["state"] == "eval_disabled"
        assert status["eval_started"] is False


def test_training_failure_never_starts_eval():
    with TemporaryDirectory(prefix="ablation_train_failure_") as temporary_directory:
        tmp_path = Path(temporary_directory)
        completed, checkpoint, eval_output, status_json = run_lifecycle(
            tmp_path,
            train_exit_code=17,
        )
        status = json.loads(status_json.read_text(encoding="utf-8"))

        assert completed.returncode == 17, completed.stdout
        assert not checkpoint.exists()
        assert not eval_output.exists()
        assert status["state"] == "training_failed"
        assert status["eval_started"] is False


def test_eval_failure_is_returned_to_caller():
    with TemporaryDirectory(prefix="ablation_eval_failure_") as temporary_directory:
        tmp_path = Path(temporary_directory)
        completed, checkpoint, eval_output, status_json = run_lifecycle(
            tmp_path,
            eval_exit_code=23,
        )
        status = json.loads(status_json.read_text(encoding="utf-8"))

        assert completed.returncode == 23, completed.stdout
        assert checkpoint.is_file()
        assert (eval_output / "received_checkpoint.txt").is_file()
        assert status["state"] == "eval_failed"
        assert status["eval_returncode"] == 23


if __name__ == "__main__":
    test_success_starts_eval_with_checkpoint_and_writes_results()
    test_explicit_disable_skips_eval()
    test_training_failure_never_starts_eval()
    test_eval_failure_is_returned_to_caller()
    print("ablation train-then-eval lifecycle tests ok")
