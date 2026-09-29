"""Exercise the real server shell launchers with a tiny fake Python workload."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = "run_full_erp_completion_v1_4xrtx5000.sh"
EVAL_LAUNCHER = "run_multipano_mixed4_eval_4gpu.sh"
SUMMARY = "validation_mixed4_by_dataset_valtestfull_summary.json"

FAKE_PYTHON = r"""
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == '-':
    sys.argv = a
    exec(sys.stdin.read())
    sys.exit(0)
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(a) + '\n')
def value(flag):
    return a[a.index(flag) + 1]
def save(path, text='fixture'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
if a[0] == '-m':
    save(Path(value('--output-dir')) / 'last.pt')
elif a[0].endswith('evaluate_mixed4_depth_checkpoint.py'):
    if os.environ.get('FAIL_EVAL') == '1':
        sys.exit('ValueError: intentional evaluator failure')
    save(value('--output'), '{}')
    csv = Path(value('--per-sample-csv'))
    save(csv, 'run,dataset_index\n')
    save(csv.with_name(csv.stem + '_camera_pairs.csv'), 'run,pair_index\n')
elif a[0].endswith('merge_mixed4_eval_shards.py'):
    save(value('--output'), '{}')
    save(value('--per-sample-csv'), 'run,dataset_index\n')
    save(value('--camera-pair-csv'), 'run,pair_index\n')
elif a[0].endswith(('analyze_full_erp_training_losses.py', 'reconstruct_pano_omega.py')):
    sys.exit('intentional optional artifact failure')
elif not a[0].endswith('validate_eval_cardinality.py'):
    sys.exit('unexpected call: ' + repr(a))
"""


class ServerEvalHandoffTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "bin").mkdir()
        (self.root / "data").mkdir()
        for name in (LAUNCHER, EVAL_LAUNCHER):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.python = self.root / "bin" / "python"
        self.python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
        self.python.chmod(0o755)
        smi = self.root / "bin" / "nvidia-smi"
        smi.write_text("#!/bin/sh\nprintf 'GPU 0: test\\nGPU 1: test\\n'\n")
        smi.chmod(0o755)
        for name in ("foundation.pt", "config.yaml"):
            (self.root / name).write_text("fixture")
        self.env = dict(os.environ)
        for key in ("WINDOW_SIZE", "NUM_YAW", "DATASET_PANO_COUNTS", "BASE_CHECKPOINT_OVERRIDE"):
            self.env.pop(key, None)
        self.env.update(
            PROJECT_ROOT=str(self.root), PYTHON_BIN=str(self.python),
            PANOVGGT_ROOT=str(self.root / "data"),
            FOUNDATION_CHECKPOINT=str(self.root / "foundation.pt"),
            WARMUP_INIT_CHECKPOINT=str(self.root / "foundation.pt"),
            CONFIG=str(self.root / "config.yaml"), RUN_NAME="test",
            CUDA_VISIBLE_DEVICES="0,1", NPROC_PER_NODE="2", EVAL_ONLY="1",
            SHOW_PROGRESS="0", PROGRESS_INTERVAL_SECONDS="0.02",
            CALLS=str(self.root / "calls.jsonl"),
            PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
        )

    def checkpoint(self, stage):
        path = self.root / "logs" / f"test_{stage}" / "last.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve existing checkpoint")
        return path

    def run_launcher(self):
        return subprocess.run(
            ["bash", str(self.root / "scripts" / LAUNCHER)],
            env=self.env, cwd=self.root.parent, capture_output=True, text=True, timeout=30,
        )

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def assert_eval_geometry(self, call):
        self.assertEqual(call[call.index("--num-yaw") + 1], "0")
        self.assertEqual(call[call.index("--window-size") + 1], "0")
        self.assertEqual(call[call.index("--pano-count-policy") + 1], "panovggt")
        self.assertEqual(call[call.index("--sample-policy") + 1], "anchor")
        self.assertEqual(call[call.index("--dataset-pano-counts") + 1], "")
        self.assertIn("--resume", call)

    def test_eval_only_preserves_checkpoints_and_survives_optional_failures(self):
        checkpoints = [self.checkpoint("omega_warmup_2h"), self.checkpoint("completion_refine_2h")]
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertFalse(any(call[0] == "-m" for call in calls))
        evals = [call for call in calls if call[0].endswith("evaluate_mixed4_depth_checkpoint.py")]
        self.assertEqual(len(evals), 2)  # one formal invocation per GPU; no quick20
        for call in evals:
            self.assert_eval_geometry(call)
            self.assertEqual(call[call.index("--limit-per-dataset") + 1], "0")
        preview = next(call for call in calls if call[0].endswith("reconstruct_pano_omega.py"))
        for flag in ("--num-yaw", "--pitch-degrees", "--fov-degrees", "--window-size"):
            self.assertNotIn(flag, preview)
        self.assertIn("[WARN] loss analysis failed", result.stdout)
        self.assertIn("[WARN] preview failed", result.stdout)
        self.assertTrue((self.root / "logs/test_eval_full8833_anchor_full_erp_4gpu" / SUMMARY).is_file())
        for path in checkpoints:
            self.assertEqual(path.read_bytes(), b"preserve existing checkpoint")
        # An existing full summary is not overwritten or re-evaluated.
        again = self.run_launcher()
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertEqual(sum(c[0].endswith("evaluate_mixed4_depth_checkpoint.py") for c in self.calls()), 2)

    def test_eval_only_missing_refined_checkpoint_never_trains(self):
        self.checkpoint("omega_warmup_2h")
        result = self.run_launcher()
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing/empty file", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_normal_training_hands_off_to_quick_and_full_eval(self):
        self.env["EVAL_ONLY"] = "0"
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(sum(c[0] == "-m" for c in calls), 3)
        evals = [c for c in calls if c[0].endswith("evaluate_mixed4_depth_checkpoint.py")]
        self.assertEqual(len(evals), 4)
        self.assertEqual(sorted(c[c.index("--limit-per-dataset") + 1] for c in evals), ["0", "0", "20", "20"])
        for call in evals:
            self.assert_eval_geometry(call)

    def test_evaluator_failure_is_nonzero_and_traceback_is_visible(self):
        self.checkpoint("omega_warmup_2h")
        self.checkpoint("completion_refine_2h")
        self.env["FAIL_EVAL"] = "1"
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: intentional evaluator failure", result.stdout)
        self.assertFalse(any(c[0].endswith("merge_mixed4_eval_shards.py") for c in self.calls()))
        self.assertNotIn("[COMPLETE]", result.stdout)


if __name__ == "__main__":
    unittest.main()
