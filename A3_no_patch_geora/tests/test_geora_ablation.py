import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from training.train_pano_omega import (
    configure_trainable, configure_trainable_for_stage, parse_args, build_model,
)
from vggt_omega.models.geora_ablation import configure_geora_ablation
from vggt_omega.models.layers.luna_patch import LunaPatchAdapter, random_bank_by_global_id
from vggt_omega.models.layers.luna_camera import LunaCameraAdapter
from scripts import run_ablation as pipeline
from scripts import run_ablation_full_eval as evaluation
from scripts.evaluate_depth_checkpoint import apply_checkpoint_eval_defaults


class GeoRAAblationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_frozen_after_load_stage_and_optimizer_step(self):
        for variant, disabled in (
            ("no_geora", (True, True)),
            ("no_camera_geora", (False, True)),
            ("no_patch_geora", (True, False)),
        ):
            model = nn.Module()
            model.patch = LunaPatchAdapter(8)
            model.camera = LunaCameraAdapter(8)
            model.head = nn.Linear(8, 1)
            state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
            count = sum(p.numel() for p in model.parameters())
            configure_geora_ablation(model, variant)
            model.load_state_dict(state)
            args = SimpleNamespace(trainable="all", camera_supervision_mode="pano_relative")
            for _ in range(2):
                configure_trainable_for_stage(model, args, {"trainable": "all"})
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
                tokens = torch.randn(1, 2, 3, 8)
                meta = {"global_patch_id": torch.zeros(1, 2, 3, dtype=torch.long)}
                output = model.patch(tokens, meta)
                cameras = model.camera(output.mean(2), torch.randn(1, 2, 16))
                model.head(cameras).square().mean().backward()
                optimizer.step()
                for module, off in zip((model.patch, model.camera), disabled):
                    if off:
                        for param in module.parameters():
                            self.assertFalse(param.requires_grad)
                            self.assertIsNone(param.grad)
                            self.assertEqual(torch.count_nonzero(param).item(), 0)
                        source = tokens if module is model.patch else cameras
                        self.assertTrue(torch.equal(module(source, None), source))
                optimizer.zero_grad(set_to_none=True)
            self.assertEqual(count, sum(p.numel() for p in model.parameters()))

    def test_random_bank_is_not_shuffle_or_learnable_context(self):
        x = torch.ones(1, 2, 3, 8, requires_grad=True)
        ids = torch.tensor([[[1, 1, -1], [1, 1, 2]]])
        panos = torch.tensor([[[0, 0, 0], [1, 1, 1]]])
        before = torch.random.get_rng_state()
        noise = random_bank_by_global_id(x, ids, pano_ids=panos, seed=43)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(noise.requires_grad)
        self.assertTrue(torch.equal(noise[0, 0, 0], noise[0, 0, 1]))
        self.assertFalse(torch.equal(noise[0, 0, 0], noise[0, 1, 0]))
        self.assertEqual(torch.count_nonzero(noise[0, 0, 2]).item(), 0)
        self.assertTrue(torch.equal(noise, random_bank_by_global_id(x * 100, ids, pano_ids=panos, seed=43)))
        self.assertFalse(torch.equal(noise, random_bank_by_global_id(x, ids, pano_ids=panos, seed=44)))
        self.assertFalse(torch.equal(noise, torch.ones_like(noise)))

    def test_config_and_real_small_model_build(self):
        args = parse_args(["--config", "configs/train.yaml", "--smoke", "--window-size", "32"])
        self.assertEqual((args.num_yaw, args.pitch_degrees), (6, "-25,25"))
        self.assertEqual(args.max_duration_minutes, 120)
        for variant in ("no_geora", "no_camera_geora", "no_patch_geora", "random_patch_bank"):
            args.geora_ablation = variant
            model = build_model(args)
            self.assertEqual(model.geora_ablation, variant)
            configure_trainable(model, "all")
            for module in model.modules():
                if getattr(module, "zero_frozen", False):
                    self.assertTrue(all(not p.requires_grad and not p.count_nonzero() for p in module.parameters()))
        restored = copy.copy(args)
        apply_checkpoint_eval_defaults(restored, {"args": {"geora_ablation": "no_geora", "random_bank_seed": 7}})
        self.assertEqual((restored.geora_ablation, restored.random_bank_seed), ("no_geora", 7))


class PipelineTest(unittest.TestCase):
    def settings(self, tmp):
        settings = yaml.safe_load((pipeline.ROOT / "configs/pipeline.yaml").read_text())
        (tmp / "base.pt").touch()
        settings.update(base_checkpoint=str(tmp / "base.pt"), dataset_root=str(tmp), nproc_per_node=1, gpus="0")
        return settings

    def args(self, tmp, **extra):
        args = dict(run_dir=tmp / "run", warmup_checkpoint=None, base_checkpoint=None,
                    dataset_root=None, nproc_per_node=None, no_auto_eval=False, dry_run=False)
        args.update(extra)
        return SimpleNamespace(**args)

    def fake_train(self, command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        torch.save({"test": True}, output / "last.pt")

    def test_success_checkpoint_chain_and_eval(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            args, settings = self.args(tmp), self.settings(tmp)
            with patch.object(pipeline.subprocess, "run", side_effect=self.fake_train) as run, \
                 patch.object(pipeline, "run_eval", return_value=tmp / "summary.json") as evaluate:
                pipeline.run_pipeline(args, settings)
            self.assertEqual(run.call_count, 3)
            main, refine = (call.args[0] for call in run.call_args_list[1:])
            self.assertEqual(main[main.index("--omega-checkpoint") + 1], str(tmp / "run/warmup/last.pt"))
            self.assertEqual(refine[refine.index("--resume") + 1], str(tmp / "run/completion_main/last.pt"))
            self.assertEqual(evaluate.call_args.args[0], tmp / "run/completion_refine/last.pt")
            self.assertEqual(json.loads((tmp / "run/pipeline_status.json").read_text())["state"], "completed")

    def test_failure_propagation_and_disable_eval(self):
        for failure in ("train", "eval", "disabled"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                tmp = Path(folder)
                settings = self.settings(tmp)
                args = self.args(tmp, no_auto_eval=failure == "disabled")
                train_effect = subprocess.CalledProcessError(7, ["train"]) if failure == "train" else self.fake_train
                with patch.object(pipeline.subprocess, "run", side_effect=train_effect), \
                     patch.object(pipeline, "run_eval", side_effect=subprocess.CalledProcessError(9, ["eval"])) as evaluate:
                    if failure == "disabled":
                        pipeline.run_pipeline(args, settings)
                    else:
                        with self.assertRaises(subprocess.CalledProcessError) as caught:
                            pipeline.run_pipeline(args, settings)
                        self.assertEqual(caught.exception.returncode, 7 if failure == "train" else 9)
                    self.assertEqual(evaluate.call_count, int(failure == "eval"))

    def test_existing_warmup_skips_training_and_two_arg_eval(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            settings = self.settings(tmp)
            teacher = tmp / "teacher.pt"
            torch.save({"model": {}}, teacher)
            plan = pipeline.build_plan(self.args(tmp, warmup_checkpoint=teacher), settings)
            self.assertEqual(len(plan), 2)
            self.assertTrue(all("training/train_erp_completion.py" in command for _, command, _ in plan))
            checkpoint = tmp / "completion.pt"
            torch.save({"completion_head": {}, "omega_checkpoint": str(teacher), "geora_ablation": "no_geora"}, checkpoint)
            output = tmp / "eval"
            def fake_eval(command, **kwargs):
                self.assertEqual(kwargs["env"]["CHECKPOINT"], str(teacher))
                self.assertEqual(kwargs["env"]["ERP_COMPLETION_CHECKPOINT"], str(checkpoint))
                self.assertEqual(kwargs["env"]["NUM_YAW"], "0")
                (output / "validation_mixed4_by_dataset_valtestfull_summary.json").write_text("{}")
            with patch.object(evaluation.subprocess, "run", side_effect=fake_eval):
                evaluation.run_eval(checkpoint, output, settings)


if __name__ == "__main__":
    unittest.main()
