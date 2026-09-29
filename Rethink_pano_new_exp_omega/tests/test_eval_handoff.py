"""Regression checks for checkpoint-native geometry and the actual shard launcher."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from training.train_pano_omega import parse_args as parse_training_args
from scripts.evaluate_depth_checkpoint import apply_checkpoint_eval_defaults, apply_eval_sampler_overrides
from scripts.evaluate_mixed4_depth_checkpoint import build_parser
from vggt_omega.models.erp_completion import (
    SAMPLER_KEYS, apply_completion_sampler_args, validate_completion_sampler_args,
)

ROOT = Path(__file__).resolve().parents[1]
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
if a[0].endswith('evaluate_mixed4_depth_checkpoint.py'):
    assert value('--num-yaw') == value('--window-size') == '0'
    assert value('--pano-count-policy') == 'panovggt'
    assert value('--dataset-pano-counts') == ''
    if os.environ.get('FAIL_EVAL') == '1':
        print('ValueError: intentional sampler mismatch', file=sys.stderr, flush=True)
        sys.exit(7)
    save(value('--output'), '{}')
    csv = Path(value('--per-sample-csv'))
    save(csv, 'run,dataset_index\n')
    save(csv.with_name(csv.stem + '_camera_pairs.csv'), 'run,pair_index\n')
elif a[0].endswith('merge_mixed4_eval_shards.py'):
    save(value('--output'), '{}')
    save(value('--per-sample-csv'), 'run,dataset_index\n')
    save(value('--camera-pair-csv'), 'run,pair_index\n')
elif not a[0].endswith('validate_eval_cardinality.py'):
    sys.exit('unexpected call: ' + repr(a))
"""


class EvalHandoffTest(unittest.TestCase):
    def test_checkpoint_native_domain_and_explicit_conflict(self):
        train_args = parse_training_args(["--config", str(ROOT / "configs/train.yaml")])
        completion = {"sampler_args": {key: getattr(train_args, key) for key in SAMPLER_KEYS}}
        # Even a reused old Omega teacher must not override completion's crop domain.
        apply_checkpoint_eval_defaults(train_args, {"args": {"num_yaw": 4, "pitch_degrees": "-15"}})
        apply_completion_sampler_args(train_args, completion)
        eval_args = build_parser().parse_args([
            "--config", "unused.yaml", "--checkpoint", "unused.pt", "--output", "unused.json",
        ])
        apply_eval_sampler_overrides(train_args, eval_args)
        validate_completion_sampler_args(train_args, completion)
        self.assertEqual((train_args.num_yaw, train_args.pitch_degrees), (6, "-25,25"))
        # Keep the safety check: deliberately overriding the domain is still an error.
        eval_args.num_yaw = 4
        apply_eval_sampler_overrides(train_args, eval_args)
        with self.assertRaisesRegex(ValueError, "requires num_yaw=6, got 4"):
            validate_completion_sampler_args(train_args, completion)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ablation_eval_handoff_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "bin").mkdir()
        self.launcher = self.root / "scripts/run_multipano_mixed4_eval_4gpu.sh"
        shutil.copyfile(ROOT / "scripts" / self.launcher.name, self.launcher)
        python = self.root / "bin/python"
        python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
        python.chmod(0o755)
        smi = self.root / "bin/nvidia-smi"
        smi.write_text("#!/bin/sh\nprintf 'GPU 0: test\\nGPU 1: test\\n'\n")
        smi.chmod(0o755)
        self.env = dict(os.environ)
        for key in ("NUM_YAW", "WINDOW_SIZE", "DATASET_PANO_COUNTS", "BASE_CHECKPOINT_OVERRIDE"):
            self.env.pop(key, None)
        self.env.update(
            PYTHON=str(python), GPUS="0,1", CONFIG="config.yaml",
            CHECKPOINT="teacher.pt", ERP_COMPLETION_CHECKPOINT="completion.pt",
            EVAL_OUT="eval", EVAL_DATASETS="all", LIMIT_PER_DATASET="0",
            PANO_COUNT_POLICY="panovggt", SAMPLE_POLICY="anchor",
            SHOW_PROGRESS="0", PROGRESS_INTERVAL_SECONDS="0.02", RESUME="1",
            CALLS=str(self.root / "calls.jsonl"),
            PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
        )

    def run_launcher(self):
        return subprocess.run(["bash", str(self.launcher)], cwd=self.root.parent,
                              env=self.env, capture_output=True, text=True, timeout=30)

    def test_real_shell_uses_native_geometry_and_preserves_logs(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.root / "eval" / SUMMARY).is_file())
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(sum(c[0].endswith("evaluate_mixed4_depth_checkpoint.py") for c in calls), 2)
        for name in ("eval_4gpu.log", "shards/shard_0.log", "shards/shard_1.log"):
            with (self.root / "eval" / name).open("a") as handle:
                handle.write("KEEP_EXISTING_LOG\n")
        again = self.run_launcher()
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        for name in ("eval_4gpu.log", "shards/shard_0.log", "shards/shard_1.log"):
            self.assertIn("KEEP_EXISTING_LOG", (self.root / "eval" / name).read_text())

    def test_real_shell_surfaces_eval_failure_without_merging(self):
        self.env["FAIL_EVAL"] = "1"
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exit=7", result.stdout)
        self.assertIn("ValueError: intentional sampler mismatch", result.stdout)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertFalse(any(c[0].endswith("merge_mixed4_eval_shards.py") for c in calls))
        self.assertFalse((self.root / "eval" / SUMMARY).exists())


if __name__ == "__main__":
    unittest.main()
